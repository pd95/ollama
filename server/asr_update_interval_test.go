package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

type asrConfigBackend struct {
	*asrFakeBackend
	config asrstream.Config
}

func (b *asrConfigBackend) BeginASR(ctx context.Context, id string, config asrstream.Config) error {
	b.config = config
	return b.asrFakeBackend.BeginASR(ctx, id, config)
}

func asrIntervalHTTP(t *testing.T, router *gin.Engine, method, path, content string, body []byte, status int) map[string]any {
	t.Helper()
	request := httptest.NewRequest(method, path, bytes.NewReader(body))
	request.Header.Set("Content-Type", content)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != status {
		t.Fatalf("%s %s: %d %s", method, path, response.Code, response.Body.String())
	}
	var fields map[string]any
	if response.Body.Len() > 0 {
		if err := json.Unmarshal(response.Body.Bytes(), &fields); err != nil {
			t.Fatal(err)
		}
	}
	return fields
}

func TestQwen3ASRSessionUpdateIntervalValidation(t *testing.T) {
	for _, test := range []struct {
		value   string
		samples int
	}{
		{"", 32000},
		{`,"update_interval_ms":500`, 8000},
		{`,"update_interval_ms":1000`, 16000},
		{`,"update_interval_ms":2000`, 32000},
		{`,"update_interval_ms":0`, 0},
		{`,"update_interval_ms":null`, 0},
		{`,"update_interval_ms":250`, 0},
		{`,"update_interval_ms":-1000`, 0},
		{`,"update_interval_ms":3000`, 0},
		{`,"update_interval_ms":1000.5`, 0},
		{`,"update_interval_ms":"1000"`, 0},
	} {
		t.Run(test.value, func(t *testing.T) {
			backend := &asrConfigBackend{asrFakeBackend: &asrFakeBackend{exited: make(chan struct{})}}
			acquired := false
			manager := newASRSessionManager(t.Context(), func(context.Context, string) (asrBackend, error) {
				acquired = true
				return backend, nil
			})
			defer func() {
				if err := manager.shutdown(t.Context()); err != nil {
					t.Error(err)
				}
			}()
			router := gin.New()
			registerASRSessionRoutes(router, manager)
			status := http.StatusBadRequest
			if test.samples != 0 {
				status = http.StatusCreated
			}
			result := asrIntervalHTTP(t, router, "POST", asrSessionPath, "application/json", []byte(`{"model":"test"`+test.value+`}`), status)
			if test.samples == 0 {
				if acquired {
					t.Fatal("invalid interval reached model acquisition")
				}
				return
			}
			if result["update_samples"] != float64(test.samples) || result["update_interval_ms"] != float64(test.samples*1000/16000) || result["max_packet_samples"] != float64(min(16000, test.samples)) || backend.config.UpdateSamples != test.samples {
				t.Fatal("negotiated interval was not propagated", result, backend.config)
			}
			if err := manager.closeAndWait(t.Context(), result["id"].(string)); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestQwen3ASRSessionOneSecondUpdatesAndStability(t *testing.T) {
	texts := []string{"hello world one", "hello world two", "hello world three", "hello brave four"}
	backend := &asrConfigBackend{asrFakeBackend: &asrFakeBackend{exited: make(chan struct{})}}
	accepted, processed, revision := 0, 0, 0
	backend.push = func(_ context.Context, offset int, pcm []float32) (*asrstream.Update, error) {
		if offset != accepted {
			t.Fatal("offset lost", offset, accepted)
		}
		accepted += len(pcm)
		if accepted < processed+backend.config.UpdateSamples {
			return nil, nil
		}
		processed += backend.config.UpdateSamples
		revision++
		return &asrstream.Update{Revision: revision, AudioEnd: processed, Text: texts[revision-1], Language: "English"}, nil
	}
	backend.finish = func(context.Context) (asrstream.Update, error) {
		return asrstream.Update{Revision: revision + 1, AudioEnd: accepted, Text: texts[revision-1], Language: "English", Final: true}, nil
	}
	manager := newASRSessionManager(t.Context(), func(context.Context, string) (asrBackend, error) { return backend, nil })
	defer func() {
		if err := manager.shutdown(t.Context()); err != nil {
			t.Error(err)
		}
	}()
	router := gin.New()
	registerASRSessionRoutes(router, manager)
	created := asrIntervalHTTP(t, router, "POST", asrSessionPath, "application/json", []byte(`{"model":"test","update_interval_ms":1000}`), 201)
	base := asrSessionPath + "/" + created["id"].(string)
	previous, observed := 0, 0
	for _, count := range []int{12000, 16000, 16000, 16000, 4000} {
		result := asrIntervalHTTP(t, router, "POST", fmt.Sprintf("%s/audio?offset_samples=%d", base, previous), "application/octet-stream", make([]byte, count*2), 200)
		previous += count
		if result["accepted_samples"] != float64(previous) {
			t.Fatal("incorrect accepted offset", result)
		}
		if _, update := result["revision"]; !update {
			if _, present := result["stable_text"]; present {
				t.Fatal("stability added to acceptance-only response", result)
			}
			if _, present := result["draft_text"]; present {
				t.Fatal("draft added to acceptance-only response", result)
			}
			continue
		}
		observed++
		stable := []string{"", "", "hello world ", "hello "}[observed-1]
		if result["audio_end_sample"] != float64(observed*16000) || result["stable_text"] != stable || result["stable_text"].(string)+result["draft_text"].(string) != result["text"] {
			t.Fatal("incorrect replacement hypothesis or stability", result)
		}
	}
	if observed != 4 {
		t.Fatal("incorrect one-second cadence", observed)
	}
	final := asrIntervalHTTP(t, router, "POST", base+"/finish", "", nil, 200)
	if final["final"] != true || final["stable_text"] != texts[3] || final["draft_text"] != "" || final["accepted_samples"] != float64(64000) {
		t.Fatal("incorrect final replacement", final)
	}
}

func TestQwen3ASRSessionRejectsUnsupportedSampleInterval(t *testing.T) {
	called := false
	manager := newASRSessionManager(t.Context(), func(context.Context, string) (asrBackend, error) { called = true; return nil, nil })
	defer func() {
		if err := manager.shutdown(t.Context()); err != nil {
			t.Error(err)
		}
	}()
	if _, err := manager.create(t.Context(), "test", asrstream.Config{UpdateSamples: 4000}); err == nil || called {
		t.Fatal("unsupported interval reached acquisition", err)
	}
}

func TestQwen3ASRSessionHalfSecondPacketCapAndOffsets(t *testing.T) {
	backend := &asrConfigBackend{asrFakeBackend: &asrFakeBackend{exited: make(chan struct{})}}
	accepted, processed, revision, calls := 0, 0, 0, 0
	backend.push = func(_ context.Context, offset int, pcm []float32) (*asrstream.Update, error) {
		calls++
		if offset != accepted || len(pcm) > 8000 {
			t.Fatal("oversized or reordered packet reached backend")
		}
		accepted += len(pcm)
		if accepted < processed+8000 {
			return nil, nil
		}
		processed += 8000
		revision++
		return &asrstream.Update{Revision: revision, AudioEnd: processed, Text: "hello world", Language: "English"}, nil
	}
	backend.finish = func(context.Context) (asrstream.Update, error) {
		return asrstream.Update{Revision: revision + 1, AudioEnd: accepted, Text: "hello world", Language: "English", Final: true}, nil
	}
	manager := newASRSessionManager(t.Context(), func(context.Context, string) (asrBackend, error) { return backend, nil })
	defer func() {
		if err := manager.shutdown(t.Context()); err != nil {
			t.Error(err)
		}
	}()
	router := gin.New()
	registerASRSessionRoutes(router, manager)
	created := asrIntervalHTTP(t, router, "POST", asrSessionPath, "application/json", []byte(`{"model":"test","update_interval_ms":500}`), 201)
	base := asrSessionPath + "/" + created["id"].(string)
	if created["max_packet_samples"] != float64(8000) {
		t.Fatal("half-second packet cap not advertised", created)
	}
	asrIntervalHTTP(t, router, "POST", base+"/audio?offset_samples=0", "application/octet-stream", make([]byte, 6000), 200)
	// Both transport and direct coordinator admission reject a one-second packet.
	asrIntervalHTTP(t, router, "POST", base+"/audio?offset_samples=3000", "application/octet-stream", make([]byte, 32000), 413)
	p := asrOnlySession(t, manager)
	if _, _, err := p.operation(t.Context(), 3000, make([]float32, 16000), false); err == nil {
		t.Fatal("coordinator accepted a one-second packet")
	}
	if accepted != 3000 || calls != 1 {
		t.Fatal("rejected packet consumed audio", accepted, calls)
	}
	offset, observed := 3000, 0
	for _, count := range []int{8000, 8000, 5000, 17} {
		result := asrIntervalHTTP(t, router, "POST", fmt.Sprintf("%s/audio?offset_samples=%d", base, offset), "application/octet-stream", make([]byte, count*2), 200)
		offset += count
		if result["accepted_samples"] != float64(offset) {
			t.Fatal("rejected packet changed next accepted offset", result)
		}
		if _, present := result["revision"]; present {
			observed++
			if result["audio_end_sample"] != float64(observed*8000) || result["stable_text"].(string)+result["draft_text"].(string) != result["text"] {
				t.Fatal("incorrect half-second boundary acknowledgment", result)
			}
		}
	}
	if observed != 3 {
		t.Fatal("half-second update count", observed)
	}
	final := asrIntervalHTTP(t, router, "POST", base+"/finish", "", nil, 200)
	if final["final"] != true || final["accepted_samples"] != float64(24017) || final["audio_end_sample"] != float64(24017) || final["stable_text"] != final["text"] || final["draft_text"] != "" {
		t.Fatal("incorrect unpadded half-second final tail", final)
	}
}
