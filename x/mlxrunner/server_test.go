package mlxrunner

import (
	"context"
	"errors"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestPrepareAndQueueCanceledRequest(t *testing.T) {
	runner := mediaTestRunner(t)
	runner.Requests = make(chan Request, 1)
	request := &Request{CompletionRequest: CompletionRequest{
		Prompt: "[img-0]",
		Media:  []llm.MediaData{{ID: 0, Kind: llm.MediaKindImage, Data: []byte("img")}},
	}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := runner.prepareAndQueue(ctx, request); !errors.Is(err, context.Canceled) {
		t.Fatalf("prepareAndQueue() error = %v, want context.Canceled", err)
	}
	if len(runner.Requests) != 0 {
		t.Fatal("canceled request was queued")
	}
	if request.Tokens != nil || request.MediaItems != nil || request.Layout != nil {
		t.Fatalf("canceled request retained partial preparation: %+v", request)
	}
}
