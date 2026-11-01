package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/mlxrunner"
	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

type asrWindowTestBackend struct {
	*asrFakeBackend
	seed   func(context.Context, []float32) (*asrstream.Update, error)
	config asrstream.Config
}

func (b *asrWindowTestBackend) SeedASR(ctx context.Context, _ string, pcm []float32) (*asrstream.Update, error) {
	return b.seed(ctx, pcm)
}

func (b *asrWindowTestBackend) BeginASR(ctx context.Context, id string, config asrstream.Config) error {
	b.config = config
	return b.asrFakeBackend.BeginASR(ctx, id, config)
}

func asrWindowMultipart(t *testing.T, fields [][2]string, pcm []byte) ([]byte, string) {
	t.Helper()
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	for _, field := range fields {
		if err := w.WriteField(field[0], field[1]); err != nil {
			t.Fatal(err)
		}
	}
	if pcm != nil {
		part, err := w.CreateFormFile("file", "window.pcm")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := part.Write(pcm); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return body.Bytes(), w.FormDataContentType()
}

func asrWindowRouter(t *testing.T, acquire asrAcquire) (*gin.Engine, *asrSessionManager) {
	t.Helper()
	m := newASRSessionManager(t.Context(), acquire)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if err := m.shutdown(ctx); err != nil {
			t.Error(err)
		}
	})
	r := gin.New()
	registerASRSessionRoutes(r, m)
	return r, m
}

func TestQwen3ASRWindowHTTPFreshCodecAndCoverage(t *testing.T) {
	pcm := make([]byte, 2*asrstream.MaxSeedSamples)
	for i := range len(pcm) / 2 {
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(int16(i%65536-32768)))
	}
	seeds, finishes := 0, 0
	b := &asrWindowTestBackend{asrFakeBackend: &asrFakeBackend{exited: make(chan struct{})}}
	b.seed = func(_ context.Context, samples []float32) (*asrstream.Update, error) {
		seeds++
		if len(samples) != len(pcm)/2 {
			t.Fatal("window was packetized", len(samples))
		}
		for i, sample := range samples {
			if sample != float32(int16(binary.LittleEndian.Uint16(pcm[2*i:])))/32768 {
				t.Fatal("PCM codec changed a sample", i)
			}
		}
		return &asrstream.Update{Revision: 1, AudioEnd: len(samples), Text: "fresh", Language: "German"}, nil
	}
	b.finish = func(context.Context) (asrstream.Update, error) {
		finishes++
		return asrstream.Update{Revision: 2, AudioEnd: len(pcm) / 2, Text: "fresh", Language: "German", Final: true}, nil
	}
	var lease context.Context
	r, m := asrWindowRouter(t, func(ctx context.Context, name string) (asrBackend, error) {
		if name != "test" {
			t.Fatal(name)
		}
		lease = ctx
		return b, nil
	})
	body, content := asrWindowMultipart(t, [][2]string{{"model", "test"}, {"language", "de"}, {"prompt", "context"}, {"start_sample", "12345"}}, pcm)
	result := asrIntervalHTTP(t, r, "POST", asrWindowPath, content, body, http.StatusOK)
	if seeds != 1 || finishes != 1 || b.config.Language != "German" || b.config.Context != "context" || b.config.UpdateSamples != 32000 || result["text"] != "fresh" || result["fresh"] != true || result["source_start_sample"] != float64(12345) || result["source_end_sample"] != float64(12345+len(pcm)/2) || result["audio_samples"] != float64(len(pcm)/2) || result["sample_rate"] != float64(16000) {
		t.Fatal(result, b.config, seeds, finishes)
	}
	if _, present := result["final"]; present {
		t.Fatal("window claimed public commitment")
	}
	if lease.Err() == nil {
		t.Fatal("successful window returned before lease release")
	}
	m.mu.Lock()
	active := len(m.active)
	m.mu.Unlock()
	if active != 0 {
		t.Fatal("stateless window left active session", active)
	}
}

func TestQwen3ASRWindowHTTPValidationBeforeAcquisition(t *testing.T) {
	for _, test := range []struct {
		name   string
		fields [][2]string
		pcm    []byte
		status int
	}{
		{"missing-model", nil, make([]byte, 402), 400},
		{"missing-file", [][2]string{{"model", "test"}}, nil, 400},
		{"short", [][2]string{{"model", "test"}}, make([]byte, 400), 400},
		{"odd", [][2]string{{"model", "test"}}, make([]byte, 403), 400},
		{"oversized-file", [][2]string{{"model", "test"}}, make([]byte, 2*asrstream.MaxSeedSamples+2), 413},
		{"bad-language", [][2]string{{"model", "test"}, {"language", "fr"}}, make([]byte, 402), 400},
		{"duplicate", [][2]string{{"model", "test"}, {"model", "test"}}, make([]byte, 402), 400},
		{"unknown", [][2]string{{"model", "test"}, {"previous_text", "never seed"}}, make([]byte, 402), 400},
		{"long-prompt", [][2]string{{"model", "test"}, {"prompt", strings.Repeat("x", 4097)}}, make([]byte, 402), 400},
		{"invalid-utf8", [][2]string{{"model", "test"}, {"prompt", "\xff"}}, make([]byte, 402), 400},
		{"negative-start", [][2]string{{"model", "test"}, {"start_sample", "-1"}}, make([]byte, 402), 400},
		{"noncanonical-start", [][2]string{{"model", "test"}, {"start_sample", "01"}}, make([]byte, 402), 400},
		{"overflow-start", [][2]string{{"model", "test"}, {"start_sample", "9223372036854775807"}}, make([]byte, 402), 400},
	} {
		t.Run(test.name, func(t *testing.T) {
			r, _ := asrWindowRouter(t, func(context.Context, string) (asrBackend, error) {
				t.Fatal("invalid window acquired model")
				return nil, nil
			})
			body, content := asrWindowMultipart(t, test.fields, test.pcm)
			asrIntervalHTTP(t, r, "POST", asrWindowPath, content, body, test.status)
		})
	}
}

func TestQwen3ASRWindowHTTPWaitsForPositiveDrain(t *testing.T) {
	for _, generationLimit := range []bool{false, true} {
		t.Run(strconv.FormatBool(generationLimit), func(t *testing.T) {
			closing, release := make(chan struct{}), make(chan struct{})
			var closeOnce, releaseOnce sync.Once
			defer releaseOnce.Do(func() { close(release) })
			b := &asrWindowTestBackend{asrFakeBackend: &asrFakeBackend{exited: make(chan struct{})}}
			b.seed = func(context.Context, []float32) (*asrstream.Update, error) {
				if generationLimit {
					return nil, fmt.Errorf("native: %w", mlxrunner.ErrASRGenerationLimit)
				}
				return &asrstream.Update{Revision: 1, AudioEnd: 201, Text: "fresh"}, nil
			}
			b.finish = func(context.Context) (asrstream.Update, error) {
				return asrstream.Update{Revision: 2, AudioEnd: 201, Text: "fresh", Final: true}, nil
			}
			b.close = func(ctx context.Context) error {
				closeOnce.Do(func() { close(closing) })
				select {
				case <-release:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			var lease context.Context
			r, m := asrWindowRouter(t, func(ctx context.Context, _ string) (asrBackend, error) { lease = ctx; return b, nil })
			body, content := asrWindowMultipart(t, [][2]string{{"model", "test"}}, make([]byte, 402))
			result := make(chan *httptest.ResponseRecorder, 1)
			go func() {
				w := httptest.NewRecorder()
				req := httptest.NewRequest("POST", asrWindowPath, bytes.NewReader(body))
				req.Header.Set("Content-Type", content)
				r.ServeHTTP(w, req)
				result <- w
			}()
			asrAwait(t, closing)
			select {
			case w := <-result:
				t.Fatal("returned before drain", w.Code)
			default:
			}
			if lease.Err() != nil {
				t.Fatal("released undrained model lease")
			}
			if _, err := m.create(t.Context(), "other", asrstream.Config{}); !errors.Is(err, errASRSessionCapacity) {
				t.Fatal("window bypassed admission", err)
			}
			releaseOnce.Do(func() { close(release) })
			select {
			case w := <-result:
				want := 200
				if generationLimit {
					want = 422
				}
				if w.Code != want {
					t.Fatal(w.Code, w.Body)
				}
				if generationLimit && !strings.Contains(w.Body.String(), `"code":"asr_generation_limit"`) {
					t.Fatal(w.Body)
				}
			case <-time.After(time.Second):
				t.Fatal("window did not return after drain")
			}
		})
	}
}

func TestQwen3ASRWindowHTTPDisconnectRetainsLease(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	b := &asrWindowTestBackend{asrFakeBackend: &asrFakeBackend{exited: make(chan struct{})}}
	b.seed = func(ctx context.Context, _ []float32) (*asrstream.Update, error) {
		close(entered)
		<-ctx.Done()
		return nil, ctx.Err()
	}
	b.close = func(ctx context.Context) error {
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	var lease context.Context
	r, m := asrWindowRouter(t, func(ctx context.Context, _ string) (asrBackend, error) { lease = ctx; return b, nil })
	m.policy.cleanupAttempt, m.policy.retry = 20*time.Millisecond, time.Millisecond
	server := httptest.NewServer(r)
	defer server.Close()
	body, content := asrWindowMultipart(t, [][2]string{{"model", "test"}}, make([]byte, 402))
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST", server.URL+asrWindowPath, bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", content)
	result := make(chan error, 1)
	go func() {
		response, err := server.Client().Do(req)
		if response != nil {
			io.Copy(io.Discard, response.Body)
			response.Body.Close()
		}
		result <- err
	}()
	asrAwait(t, entered)
	m.mu.Lock()
	var p *asrPublicSession
	for _, session := range m.active {
		p = session
	}
	m.mu.Unlock()
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if lease.Err() != nil {
		t.Fatal("disconnect released model before drain")
	}
	if _, err := m.create(t.Context(), "other", asrstream.Config{}); !errors.Is(err, errASRSessionCapacity) {
		t.Fatal(err)
	}
	releaseOnce.Do(func() { close(release) })
	asrAwait(t, p.done)
	asrAwait(t, lease.Done())
}

func TestQwen3ASRWindowHTTPDoesNotClassifyBeforeDrainOnTimeout(t *testing.T) {
	closing, release := make(chan struct{}), make(chan struct{})
	var releaseOnce, closeOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	b := &asrWindowTestBackend{asrFakeBackend: &asrFakeBackend{exited: make(chan struct{})}}
	b.seed = func(context.Context, []float32) (*asrstream.Update, error) {
		return nil, mlxrunner.ErrASRGenerationLimit
	}
	b.close = func(ctx context.Context) error {
		closeOnce.Do(func() { close(closing) })
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	r, _ := asrWindowRouter(t, func(context.Context, string) (asrBackend, error) { return b, nil })
	body, content := asrWindowMultipart(t, [][2]string{{"model", "test"}}, make([]byte, 402))
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	w := httptest.NewRecorder()
	req := httptest.NewRequest("POST", asrWindowPath, bytes.NewReader(body)).WithContext(ctx)
	req.Header.Set("Content-Type", content)
	done := make(chan struct{})
	go func() { r.ServeHTTP(w, req); close(done) }()
	asrAwait(t, closing)
	cancel()
	asrAwait(t, done)
	var result map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if w.Code != http.StatusRequestTimeout || result["code"] != nil {
		t.Fatal(w.Code, result)
	}
	releaseOnce.Do(func() { close(release) })
}
