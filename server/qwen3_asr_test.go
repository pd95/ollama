package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestQwen3ASRCompleteAudio(t *testing.T) {
	s, requests := audioTestServer(t, "qwen3-asr", true, func(_ int, _ string) string { return "language German<asr_text>Hallo." })
	w := serveAudioText(s, audioTextRequest(t, "/v1/audio/transcriptions", map[string]string{"model": "speech-test", "language": "de"}, testSpeech(130, 24000)))
	if w.Code != http.StatusOK {
		t.Fatalf("%d: %s", w.Code, w.Body)
	}
	var got struct{ Text string }
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Text != "Hallo. Hallo. Hallo." || len(*requests) != 3 {
		t.Fatalf("got %q in %d clips", got.Text, len(*requests))
	}
	for _, r := range *requests {
		if !strings.HasSuffix(r.Prompt, "language German<asr_text>") || len(r.Media) != 1 {
			t.Fatalf("incorrect ASR request: %+v", r)
		}
		if r.Options.NumPredict > maxAudioTextTokens(60*time.Second) {
			t.Fatal("clip exceeded ASR token budget")
		}
	}
}

func TestQwen3ASRUploadPreparationCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := splitAudioContext(ctx, []byte("must not decode"), 60*time.Second); !errors.Is(err, context.Canceled) {
		t.Fatalf("outer decoder ignored cancellation: %v", err)
	}
	clip := audioClip{samples: make([]float32, 16000), rate: 16000}
	if _, err := clip.wavContext(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("outer encoder ignored cancellation: %v", err)
	}
	checks := 0
	reader := &audioUploadReader{Reader: strings.NewReader(strings.Repeat("a", 256<<10)), check: func() error {
		checks++
		if checks > 2 {
			return context.Canceled
		}
		return nil
	}}
	if _, err := io.ReadAll(reader); !errors.Is(err, context.Canceled) {
		t.Fatalf("multipart-file read ignored mid-read cancellation: %v", err)
	}
}
