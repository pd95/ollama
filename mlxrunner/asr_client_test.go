package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

func asrTestClient(t *testing.T, handler http.Handler) *Client {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	_, port, err := net.SplitHostPort(strings.TrimPrefix(srv.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	number, err := strconv.Atoi(port)
	if err != nil {
		t.Fatal(err)
	}
	return &Client{port: number, client: srv.Client(), done: make(chan struct{})}
}

func TestQwen3ASRClientAcknowledgments(t *testing.T) {
	id := strings.Repeat("a", 48)
	c := asrTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var cmd struct {
			ID     string        `json:"id"`
			Config stream.Config `json:"config"`
			Offset int           `json:"offset"`
			PCM    []float32     `json:"pcm"`
		}
		if err := json.NewDecoder(r.Body).Decode(&cmd); err != nil {
			t.Error(err)
		}
		switch strings.TrimPrefix(r.URL.Path, "/v1/asr/replay/") {
		case "reserve":
			fmt.Fprintf(w, `{"id":%q,"begin_deadline_ms":%d}`, id, time.Now().Add(time.Second).UnixMilli())
		case "begin":
			if cmd.ID != id || cmd.Config.Language != "German" {
				t.Error("lost begin configuration")
			}
			fmt.Fprint(w, `{"ok":true}`)
		case "push":
			if cmd.Offset != 123 || len(cmd.PCM) != 2 {
				t.Error("lost packet")
			}
			fmt.Fprint(w, `null`)
		case "finish":
			fmt.Fprint(w, `{"revision":1,"audio_end_sample":125,"text":"Hallo","language":"German","final":true}`)
		case "close":
			fmt.Fprint(w, `{"drained":true}`)
		default:
			t.Error("unexpected action")
		}
	}))
	got, _, err := c.ReserveASR(t.Context())
	if err != nil || got != id {
		t.Fatal(got, err)
	}
	if err := c.BeginASR(t.Context(), id, stream.Config{Language: "German"}); err != nil {
		t.Fatal(err)
	}
	if u, err := c.PushASR(t.Context(), id, 123, []float32{0, .5}); err != nil || u != nil {
		t.Fatal(u, err)
	}
	if u, err := c.FinishASR(t.Context(), id); err != nil || u.Text != "Hallo" {
		t.Fatal(u, err)
	}
	if err := c.CloseASR(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	select {
	case <-c.ASRProcessDone():
		t.Fatal("HTTP success is not process exit")
	default:
	}
	close(c.done)
	select {
	case <-c.ASRProcessDone():
	default:
		t.Fatal("process exit not observed")
	}
}

func TestQwen3ASRClientRejectsFalseDrain(t *testing.T) {
	for _, body := range []string{`{}`, `null`, `{"drained":false}`, `{"drained":true} {}`, `{"drained":true,"other":1}`, `bad`, strings.Repeat("x", (1<<20)+1)} {
		t.Run(fmt.Sprint(len(body), body[:min(len(body), 20)]), func(t *testing.T) {
			c := asrTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { fmt.Fprint(w, body) }))
			if err := c.CloseASR(t.Context(), "id"); err == nil {
				t.Fatal("invalid acknowledgment passed")
			}
		})
	}
	c := asrTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { http.Error(w, "wrong epoch", http.StatusConflict) }))
	if err := c.CloseASR(context.Background(), "id"); err == nil {
		t.Fatal("HTTP error became drain")
	}
}

func TestQwen3ASRClientRegistryRoundtrip(t *testing.T) {
	registry := testReplayRegistry(t, func(context.Context, stream.Request) (string, error) { return "Hallo", nil })
	mux := http.NewServeMux()
	registerASRReplayRoutes(mux, registry)
	c := asrTestClient(t, mux)
	id, _, err := c.ReserveASR(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := c.BeginASR(t.Context(), id, stream.Config{Language: "German"}); err != nil {
		t.Fatal(err)
	}
	if u, err := c.PushASR(t.Context(), id, 0, make([]float32, 16000)); err != nil || u != nil {
		t.Fatal(u, err)
	}
	if u, err := c.PushASR(t.Context(), id, 16000, make([]float32, 16000)); err != nil || u == nil || u.Text != "Hallo" || u.AudioEnd != 32000 {
		t.Fatal(u, err)
	}
	if u, err := c.FinishASR(t.Context(), id); err != nil || !u.Final || u.Revision != 2 {
		t.Fatal(u, err)
	}
	if err := c.CloseASR(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	if err := c.BeginASR(t.Context(), id, stream.Config{Language: "German"}); err == nil {
		t.Fatal("late begin recreated session")
	}
}

func TestQwen3ASRClientGenerationLimitValidation(t *testing.T) {
	valid := `{"error":"native replay reached its length bound","code":"asr_generation_limit"}`
	for _, test := range []struct {
		name       string
		body       string
		status     int
		classified bool
	}{
		{"valid", valid, 422, true},
		{"valid-code-first", `{"code":"asr_generation_limit","error":"failure"}`, 422, true},
		{"wrong-status", valid, 400, false},
		{"other-server-status", valid, 500, false},
		{"other-code", `{"error":"failure","code":"other"}`, 422, false},
		{"missing-code", `{"error":"asr_generation_limit"}`, 422, false},
		{"missing-error", `{"code":"asr_generation_limit"}`, 422, false},
		{"empty-error", `{"error":" ","code":"asr_generation_limit"}`, 422, false},
		{"null-error", `{"error":null,"code":"asr_generation_limit"}`, 422, false},
		{"null-code", `{"error":"failure","code":null}`, 422, false},
		{"duplicate-code-last-limit", `{"error":"failure","code":"other","code":"asr_generation_limit"}`, 422, false},
		{"duplicate-code-first-limit", `{"error":"failure","code":"asr_generation_limit","code":"other"}`, 422, false},
		{"duplicate-code-same", `{"error":"failure","code":"asr_generation_limit","code":"asr_generation_limit"}`, 422, false},
		{"duplicate-error", `{"error":"failure","error":"other failure","code":"asr_generation_limit"}`, 422, false},
		{"duplicate-error-null-first", `{"error":null,"error":"failure","code":"asr_generation_limit"}`, 422, false},
		{"duplicate-error-null-last", `{"error":"failure","error":null,"code":"asr_generation_limit"}`, 422, false},
		{"uppercase-error", `{"Error":"failure","code":"asr_generation_limit"}`, 422, false},
		{"uppercase-code", `{"error":"failure","CODE":"asr_generation_limit"}`, 422, false},
		{"error-case-alias", `{"Error":"other failure","error":"failure","code":"asr_generation_limit"}`, 422, false},
		{"code-case-alias", `{"error":"failure","CODE":"other","code":"asr_generation_limit"}`, 422, false},
		{"unknown-field", `{"error":"failure","code":"asr_generation_limit","retry":true}`, 422, false},
		{"wrong-type", `{"error":true,"code":"asr_generation_limit"}`, 422, false},
		{"extra-object", valid + `{}`, 422, false},
		{"truncated", valid[:len(valid)-1], 422, false},
		{"plain-text", "asr_generation_limit", 422, false},
		{"null", `null`, 422, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			c := asrTestClient(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(test.status)
				fmt.Fprint(w, test.body)
			}))
			u, err := c.PushASR(t.Context(), "id", 0, []float32{0})
			if u != nil || err == nil || errors.Is(err, ErrASRGenerationLimit) != test.classified {
				t.Fatalf("update=%+v error=%v", u, err)
			}
			if !test.classified {
				var statusErr *api.StatusError
				if !errors.As(err, &statusErr) || statusErr.StatusCode != test.status || statusErr.ErrorMessage != strings.TrimSpace(test.body) {
					t.Fatalf("unclassified response changed: %v", err)
				}
			}
		})
	}
}

func TestQwen3ASRClientRegistryGenerationLimit(t *testing.T) {
	for _, action := range []string{"push", "finish"} {
		t.Run(action, func(t *testing.T) {
			registry := testReplayRegistry(t, func(ctx context.Context, _ stream.Request) (string, error) {
				responses := make(chan CompletionResponse, 1)
				responses <- CompletionResponse{Content: "truncated", Done: true, DoneReason: 1}
				close(responses)
				return collectASRReplay(ctx, responses)
			})
			mux := http.NewServeMux()
			registerASRReplayRoutes(mux, registry)
			c := asrTestClient(t, mux)
			id, _, err := c.ReserveASR(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			if err := c.BeginASR(t.Context(), id, stream.Config{UpdateSamples: 16000}); err != nil {
				t.Fatal(err)
			}
			if action == "push" {
				u, pushErr := c.PushASR(t.Context(), id, 0, make([]float32, 16000))
				if u != nil || !errors.Is(pushErr, ErrASRGenerationLimit) {
					t.Fatalf("update=%+v error=%v", u, pushErr)
				}
			} else {
				if u, err := c.PushASR(t.Context(), id, 0, make([]float32, 201)); err != nil || u != nil {
					t.Fatal(u, err)
				}
				u, finishErr := c.FinishASR(t.Context(), id)
				if u.Final || u.Text != "" || !errors.Is(finishErr, ErrASRGenerationLimit) {
					t.Fatalf("final=%+v error=%v", u, finishErr)
				}
			}
			if err := c.CloseASR(t.Context(), id); err != nil {
				t.Fatal("failed generation did not drain", err)
			}
		})
	}
}
