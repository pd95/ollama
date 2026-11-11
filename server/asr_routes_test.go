package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner"
	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

func TestQwen3ASRSessionExperimentalGate(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Setenv("OLLAMA_EXPERIMENT", map[bool]string{true: "asr-streaming", false: ""}[enabled])
		s := &Server{}
		r := gin.New()
		s.registerASRSessionRoutes(r)
		if (len(r.Routes()) != 0) != enabled {
			t.Fatal("unexpected experimental routes", enabled)
		}
		s.closeASRSessions()
	}
}

func TestQwen3ASRSessionHTTP(t *testing.T) {
	accepted, revision := 0, 0
	f := &asrFakeBackend{exited: make(chan struct{}), push: func(_ context.Context, offset int, pcm []float32) (*asrstream.Update, error) {
		if offset != accepted {
			t.Fatal("offset lost")
		}
		accepted += len(pcm)
		if accepted >= 32000 && revision == 0 {
			revision++
			return &asrstream.Update{Revision: revision, AudioEnd: 32000, Text: "Hallo", Language: "German"}, nil
		}
		return nil, nil
	}, finish: func(context.Context) (asrstream.Update, error) {
		revision++
		return asrstream.Update{Revision: revision, AudioEnd: accepted, Text: "Hallo Welt", Language: "German", Final: true}, nil
	}}
	m := newASRSessionManager(t.Context(), func(context.Context, string) (asrBackend, error) { return f, nil })
	defer func() {
		if err := m.shutdown(t.Context()); err != nil {
			t.Error(err)
		}
	}()
	r := gin.New()
	registerASRSessionRoutes(r, m)
	call := func(method, path, content string, body []byte, status int) map[string]any {
		t.Helper()
		req := httptest.NewRequest(method, path, bytes.NewReader(body))
		req.Header.Set("Content-Type", content)
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		if w.Code != status {
			t.Fatalf("%s %s: %d %s", method, path, w.Code, w.Body.String())
		}
		var result map[string]any
		if w.Body.Len() > 0 {
			if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
		}
		return result
	}
	call("POST", asrSessionPath, "application/json", []byte(`{"model":"test","language":"fr"}`), 400)
	call("POST", asrSessionPath, "application/json", []byte(`{"model":"test","other":1}`), 400)
	created := call("POST", asrSessionPath, "application/json", []byte(`{"model":"test","language":"de"}`), 201)
	id := created["id"].(string)
	base := asrSessionPath + "/" + id
	call("POST", asrSessionPath, "application/json", []byte(`{"model":"test"}`), 429)
	call("POST", base+"/audio?offset_samples=0", "audio/wav", make([]byte, 10), 400)
	call("POST", base+"/audio?offset_samples=00", "application/octet-stream", make([]byte, 10), 400)
	call("POST", base+"/audio?offset_samples=0", "application/octet-stream", make([]byte, 32002), 413)
	call("POST", base+"/audio?offset_samples=0", "application/octet-stream", make([]byte, 3), 400)
	call("POST", base+"/audio?offset_samples=0", "application/octet-stream", make([]byte, 32000), 200)
	call("POST", base+"/audio?offset_samples=16000", "application/octet-stream", make([]byte, 24000), 200)
	update := call("POST", base+"/audio?offset_samples=28000", "application/octet-stream", make([]byte, 32000), 200)
	if update["accepted_samples"] != float64(44000) || update["audio_end_sample"] != float64(32000) {
		t.Fatal(update)
	}
	call("POST", base+"/audio?offset_samples=32000", "application/octet-stream", make([]byte, 2), 400)
	final := call("POST", base+"/finish", "", nil, 200)
	if final["final"] != true || final["audio_end_sample"] != float64(44000) {
		t.Fatal(final)
	}
	call("DELETE", base, "", nil, 204)
	call("POST", base+"/finish", "", nil, 404)
	call("DELETE", asrSessionPath+"/"+strings.Repeat("f", 32), "", nil, 404)
}

func TestQwen3ASRSessionUploadDeadlineExcludesInference(t *testing.T) {
	f := &asrFakeBackend{exited: make(chan struct{}), push: func(ctx context.Context, _ int, _ []float32) (*asrstream.Update, error) {
		timer := time.NewTimer(150 * time.Millisecond)
		defer timer.Stop()
		select {
		case <-timer.C:
			return nil, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}}
	m := newASRSessionManager(t.Context(), func(context.Context, string) (asrBackend, error) { return f, nil })
	m.policy.idle = 50 * time.Millisecond
	defer func() {
		if err := m.shutdown(t.Context()); err != nil {
			t.Error(err)
		}
	}()
	r := gin.New()
	registerASRSessionRoutes(r, m)
	srv := httptest.NewServer(r)
	defer srv.Close()
	response, err := http.Post(srv.URL+asrSessionPath, "application/json", strings.NewReader(`{"model":"test"}`))
	if err != nil {
		t.Fatal(err)
	}
	var created struct{ ID string }
	if err := json.NewDecoder(response.Body).Decode(&created); err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	f.finish = func(ctx context.Context) (asrstream.Update, error) {
		timer := time.NewTimer(150 * time.Millisecond)
		defer timer.Stop()
		select {
		case <-timer.C:
			return asrstream.Update{Revision: 1, AudioEnd: 500, Final: true}, nil
		case <-ctx.Done():
			return asrstream.Update{}, ctx.Err()
		}
	}
	response, err = http.Post(srv.URL+asrSessionPath+"/"+created.ID+"/audio?offset_samples=0", "application/octet-stream", bytes.NewReader(make([]byte, 1000)))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(response.Body)
	response.Body.Close()
	if response.StatusCode != 200 {
		t.Fatalf("upload deadline canceled inference: %d %s", response.StatusCode, body)
	}
	response, err = http.Post(srv.URL+asrSessionPath+"/"+created.ID+"/finish", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	body, _ = io.ReadAll(response.Body)
	response.Body.Close()
	if response.StatusCode != 200 {
		t.Fatalf("bodyless finish canceled by read deadline: %d %s", response.StatusCode, body)
	}
}

func TestQwen3ASRSessionAcquisitionHTTPFailures(t *testing.T) {
	for _, test := range []struct {
		err    error
		status int
	}{{asrInputError("bad model"), 400}, {os.ErrNotExist, 404}, {ErrMaxQueue, 503}, {errors.New("load failed"), 500}} {
		m := newASRSessionManager(t.Context(), func(context.Context, string) (asrBackend, error) { return nil, test.err })
		r := gin.New()
		registerASRSessionRoutes(r, m)
		w := httptest.NewRecorder()
		r.ServeHTTP(w, httptest.NewRequest("POST", asrSessionPath, strings.NewReader(`{"model":"test"}`)))
		if w.Code != test.status {
			t.Fatalf("%v: %d %s", test.err, w.Code, w.Body.String())
		}
		if err := m.shutdown(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
}

func TestQwen3ASRSessionOverlappingUploads(t *testing.T) {
	f := &asrFakeBackend{exited: make(chan struct{})}
	m := newASRSessionManager(t.Context(), func(context.Context, string) (asrBackend, error) { return f, nil })
	p, err := m.create(t.Context(), "test", asrstream.Config{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := m.shutdown(t.Context()); err != nil {
			t.Error(err)
		}
	}()
	r := gin.New()
	registerASRSessionRoutes(r, m)
	srv := httptest.NewServer(r)
	defer srv.Close()
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	result := make(chan error, 1)
	go func() {
		response, err := http.Post(srv.URL+asrSessionPath+"/"+p.id+"/audio?offset_samples=0", "application/octet-stream", reader)
		if err == nil {
			io.Copy(io.Discard, response.Body)
			response.Body.Close()
			if response.StatusCode != 200 {
				err = errors.New("first upload failed")
			}
		}
		result <- err
	}()
	deadline := time.Now().Add(time.Second)
	for p.transport.TryLock() {
		p.transport.Unlock()
		if time.Now().After(deadline) {
			t.Fatal("upload did not acquire request slot")
		}
		time.Sleep(time.Millisecond)
	}
	response, err := http.Post(srv.URL+asrSessionPath+"/"+p.id+"/audio?offset_samples=0", "application/octet-stream", bytes.NewReader([]byte{0, 0}))
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != 429 {
		t.Fatal("overlapping body readers admitted", response.StatusCode)
	}
	if _, err := writer.Write([]byte{0, 0}); err != nil {
		t.Fatal(err)
	}
	writer.Close()
	if err := <-result; err != nil {
		t.Fatal(err)
	}
}

func TestQwen3ASRSessionHTTPDisconnectRetainsLease(t *testing.T) {
	entered, closing, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var closeOnce sync.Once
	var lease context.Context
	f := &asrFakeBackend{exited: make(chan struct{}), push: func(ctx context.Context, _ int, _ []float32) (*asrstream.Update, error) {
		close(entered)
		<-ctx.Done()
		return nil, ctx.Err()
	}, close: func(ctx context.Context) error {
		closeOnce.Do(func() { close(closing) })
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}}
	m := newASRSessionManager(t.Context(), func(ctx context.Context, _ string) (asrBackend, error) { lease = ctx; return f, nil })
	m.policy.cleanupAttempt = 20 * time.Millisecond
	m.policy.retry = time.Millisecond
	p, err := m.create(t.Context(), "test", asrstream.Config{})
	if err != nil {
		t.Fatal(err)
	}
	var releaseOnce sync.Once
	defer func() {
		releaseOnce.Do(func() { close(release) })
		if err := m.shutdown(t.Context()); err != nil {
			t.Error(err)
		}
	}()
	r := gin.New()
	registerASRSessionRoutes(r, m)
	server := httptest.NewServer(r)
	defer server.Close()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST", server.URL+asrSessionPath+"/"+p.id+"/audio?offset_samples=0", bytes.NewReader(make([]byte, 1000)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	result := make(chan error, 1)
	go func() {
		response, err := server.Client().Do(req)
		if response != nil {
			response.Body.Close()
		}
		result <- err
	}()
	asrAwait(t, entered)
	cancel()
	asrAwait(t, closing)
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal("client did not disconnect", err)
	}
	if lease.Err() != nil {
		t.Fatal("HTTP disconnect released model before drain")
	}
	if _, err := m.create(t.Context(), "second", asrstream.Config{}); !errors.Is(err, errASRSessionCapacity) {
		t.Fatal("disconnected closing session released admission", err)
	}
	releaseOnce.Do(func() { close(release) })
	asrAwait(t, p.done)
	asrAwait(t, lease.Done())
}

func TestQwen3ASRSessionHTTPGenerationLimit(t *testing.T) {
	for _, action := range []string{"audio", "finish"} {
		t.Run(action, func(t *testing.T) {
			failure := fmt.Errorf("ASR replay: %w", mlxrunner.ErrASRGenerationLimit)
			f := &asrFakeBackend{exited: make(chan struct{}), push: func(context.Context, int, []float32) (*asrstream.Update, error) {
				if action == "audio" {
					return nil, failure
				}
				return nil, nil
			}, finish: func(context.Context) (asrstream.Update, error) { return asrstream.Update{}, failure }}
			m, lease := asrTestManager(t, f)
			p := asrOnlySession(t, m)
			r := gin.New()
			registerASRSessionRoutes(r, m)
			path := asrSessionPath + "/" + p.id
			if action == "finish" {
				w := httptest.NewRecorder()
				req := httptest.NewRequest("POST", path+"/audio?offset_samples=0", bytes.NewReader(make([]byte, 402)))
				req.Header.Set("Content-Type", "application/octet-stream")
				r.ServeHTTP(w, req)
				if w.Code != http.StatusOK {
					t.Fatalf("initial packet: %d %s", w.Code, w.Body)
				}
			}
			w := httptest.NewRecorder()
			req := httptest.NewRequest("POST", path+"/"+action, nil)
			if action == "audio" {
				req = httptest.NewRequest("POST", path+"/audio?offset_samples=0", bytes.NewReader(make([]byte, 402)))
				req.Header.Set("Content-Type", "application/octet-stream")
			}
			r.ServeHTTP(w, req)
			var body map[string]string
			if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if w.Code != http.StatusUnprocessableEntity || len(body) != 2 || body["code"] != "asr_generation_limit" || body["error"] != failure.Error() {
				t.Fatalf("%d %s", w.Code, w.Body)
			}
			asrAwait(t, p.done)
			asrAwait(t, lease)
		})
	}
}

func TestQwen3ASRSessionHTTPUnclassifiedErrorsUnchanged(t *testing.T) {
	for _, test := range []struct {
		err    error
		status int
	}{
		{asrInputError("invalid packet"), http.StatusBadRequest},
		{errors.New(mlxrunner.ErrASRGenerationLimit.Error()), http.StatusInternalServerError},
		{&api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "backend failure"}, http.StatusBadRequest},
		{&api.StatusError{StatusCode: http.StatusUnprocessableEntity, ErrorMessage: "asr_generation_limit"}, http.StatusUnprocessableEntity},
	} {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		writeASRSessionError(c, test.err)
		var body map[string]string
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		if w.Code != test.status || len(body) != 1 || body["error"] != test.err.Error() {
			t.Fatalf("%v: %d %s", test.err, w.Code, w.Body)
		}
	}
}
