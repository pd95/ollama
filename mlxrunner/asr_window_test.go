package mlxrunner

import (
	"context"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

func TestQwen3ASRWindowSeedPrivateRoundtrip(t *testing.T) {
	pcm := make([]float32, stream.MaxSeedSamples)
	for i := range pcm {
		pcm[i] = float32(math.Sin(float64(i)))
	}
	calls := 0
	registry := testReplayRegistry(t, func(_ context.Context, request stream.Request) (string, error) {
		calls++
		if request.Prefix != "" || !reflect.DeepEqual(request.Samples, pcm) {
			t.Error("private seed changed PCM or reused a prefix")
		}
		return "fresh window", nil
	})
	mux := http.NewServeMux()
	registerASRReplayRoutes(mux, registry)
	c := asrTestClient(t, mux)
	id, _, err := c.ReserveASR(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := c.BeginASR(t.Context(), id, stream.Config{}); err != nil {
		t.Fatal(err)
	}
	u, err := c.SeedASR(t.Context(), id, pcm)
	if err != nil || u == nil || u.Final || u.AudioEnd != len(pcm) || u.Revision != 1 {
		t.Fatal(u, err)
	}
	if _, err := c.SeedASR(t.Context(), id, pcm); err == nil {
		t.Fatal("second private seed accepted")
	}
	final, err := c.FinishASR(t.Context(), id)
	if err != nil || final.Text != "fresh window" || !final.Final || calls != 1 {
		t.Fatal(final, err, calls)
	}
	if err := c.CloseASR(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	if _, err := c.SeedASR(t.Context(), id, pcm); err == nil {
		t.Fatal("closed source ID was reused")
	}
}

func TestQwen3ASRWindowSeedIssuedIDFence(t *testing.T) {
	registry := testReplayRegistry(t, func(context.Context, stream.Request) (string, error) { return "", nil })
	id, _, err := registry.reserve()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := registry.seed(id, make([]float32, 201)); !errors.Is(err, errReplayNotStarted) {
		t.Fatal("seed bypassed Begin", err)
	}
	other := testReplayRegistry(t, func(context.Context, stream.Request) (string, error) { return "", nil })
	if _, err := other.seed(id, make([]float32, 201)); !errors.Is(err, errReplayInstance) {
		t.Fatal("seed crossed runner epoch", err)
	}
	if _, err := registry.seed(registry.epoch+"ffffffffffffffff", make([]float32, 201)); !errors.Is(err, errReplayUnknown) {
		t.Fatal("seed accepted unissued source ID", err)
	}
	mux := http.NewServeMux()
	registerASRReplayRoutes(mux, registry)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("POST", "/v1/asr/replay/seed", strings.NewReader(strings.Repeat(" ", 4<<20)+`{}`)))
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatal("seed JSON body was not bounded", w.Code)
	}
}
