package mlxrunner

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

func TestQwen3ASRReplayInternalPrompt(t *testing.T) {
	prompt, err := asrReplayPrompt(stream.Config{Language: "German", Context: "literal <|audio_pad|> context"}, "vorheriger Text")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(prompt, "[img-0]") != 1 || !strings.Contains(prompt, "system\nliteral <|audio_pad|> context") || !strings.HasSuffix(prompt, "language German<asr_text>vorheriger Text") {
		t.Fatal(prompt)
	}
	if _, err := asrReplayPrompt(stream.Config{Language: "French"}, ""); err == nil {
		t.Fatal("unsupported language accepted")
	}
}

func TestQwen3ASRReplayWaitsForDrain(t *testing.T) {
	for _, cancelWork := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		responses := make(chan CompletionResponse, 1)
		responses <- CompletionResponse{Content: "text", Done: true, DoneReason: 0}
		result := make(chan error, 1)
		go func() { _, err := collectASRReplay(ctx, responses); result <- err }()
		if cancelWork {
			cancel()
		}
		select {
		case err := <-result:
			t.Fatalf("returned before channel cleanup: %v", err)
		case <-time.After(10 * time.Millisecond):
		}
		close(responses)
		err := <-result
		if cancelWork && !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
		if !cancelWork && err != nil {
			t.Fatal(err)
		}
		cancel()
	}
}

func TestQwen3ASRReplayRejectsPartialGeneration(t *testing.T) {
	for _, responses := range [][]CompletionResponse{{{Content: "partial"}}, {{Done: true, DoneReason: 1}}} {
		channel := make(chan CompletionResponse, len(responses))
		for _, r := range responses {
			channel <- r
		}
		close(channel)
		if _, err := collectASRReplay(context.Background(), channel); err == nil {
			t.Fatal("accepted incomplete/EOS-limited replay")
		}
	}
}

func TestQwen3ASRReplayGenerationLimitWaitsForDrain(t *testing.T) {
	responses := make(chan CompletionResponse, 1)
	responses <- CompletionResponse{Content: "partial transcript", Done: true, DoneReason: 1}
	type outcome struct {
		text string
		err  error
	}
	result := make(chan outcome, 1)
	go func() {
		text, err := collectASRReplay(t.Context(), responses)
		result <- outcome{text, err}
	}()
	select {
	case got := <-result:
		t.Fatalf("length stop returned before channel drain: %+v", got)
	case <-time.After(10 * time.Millisecond):
	}
	// A sender must still be able to finish dispatch/cleanup after Done.
	responses <- CompletionResponse{Content: "discarded after stop"}
	close(responses)
	select {
	case got := <-result:
		if got.text != "" || !errors.Is(got.err, ErrASRGenerationLimit) {
			t.Fatalf("published truncated output or lost classification: %+v", got)
		}
	case <-time.After(time.Second):
		t.Fatal("generation-limit response did not drain")
	}
}

func TestQwen3ASRReplayGenerationLimitClassification(t *testing.T) {
	backendErr := &api.StatusError{StatusCode: 500, ErrorMessage: "device failure"}
	for _, test := range []struct {
		name      string
		responses []CompletionResponse
		want      error
	}{
		{"length", []CompletionResponse{{Content: "partial", Done: true, DoneReason: 1}}, ErrASRGenerationLimit},
		{"other-reason", []CompletionResponse{{Done: true, DoneReason: 2}}, nil},
		{"backend-error", []CompletionResponse{{Error: backendErr, Done: true, DoneReason: 1}}, backendErr},
		{"earlier-backend-error", []CompletionResponse{{Error: backendErr}, {Done: true, DoneReason: 1}}, backendErr},
		{"output-bound", []CompletionResponse{{Content: strings.Repeat("x", 32769), Done: true, DoneReason: 1}}, nil},
		{"missing-done", []CompletionResponse{{Content: "partial"}}, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			responses := make(chan CompletionResponse, len(test.responses))
			for _, response := range test.responses {
				responses <- response
			}
			close(responses)
			text, err := collectASRReplay(t.Context(), responses)
			if text != "" || err == nil || errors.Is(err, ErrASRGenerationLimit) != (test.want == ErrASRGenerationLimit) {
				t.Fatalf("text=%q error=%v", text, err)
			}
			if test.want != nil && !errors.Is(err, test.want) {
				t.Fatal(err)
			}
		})
	}
}
