package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

type queueBlockingContext struct {
	done    chan struct{}
	reached chan struct{}
	once    sync.Once
	mu      sync.Mutex
	err     error
}

func newQueueBlockingContext() *queueBlockingContext {
	return &queueBlockingContext{done: make(chan struct{}), reached: make(chan struct{})}
}

func (*queueBlockingContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (*queueBlockingContext) Value(any) any               { return nil }

func (c *queueBlockingContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.reached) })
	return c.done
}

func (c *queueBlockingContext) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}

func (c *queueBlockingContext) cancel() {
	c.mu.Lock()
	c.err = context.Canceled
	c.mu.Unlock()
	close(c.done)
}

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

func TestPrepareAndQueueClosesStructuredFormatOnCancellation(t *testing.T) {
	for _, initialNumPredict := range []int{0, 17} {
		t.Run(fmt.Sprintf("num_predict_%d", initialNumPredict), func(t *testing.T) {
			runner := mediaTestRunner(t)
			runner.grammarEngine = &grammarEngine{}
			runner.Requests = make(chan Request)
			request := &Request{CompletionRequest: CompletionRequest{
				Prompt:  "0",
				Format:  json.RawMessage(`{"type":"object"}`),
				Options: api.Options{NumPredict: initialNumPredict},
			}}
			ctx := newQueueBlockingContext()
			errCh := make(chan error, 1)
			go func() { errCh <- runner.prepareAndQueue(ctx, request) }()

			<-ctx.reached
			ctx.cancel()
			if err := <-errCh; !errors.Is(err, context.Canceled) {
				t.Fatalf("prepareAndQueue() error = %v, want context.Canceled", err)
			}
			if request.Grammar != nil || request.Tokens != nil || request.MediaItems != nil || request.Layout != nil {
				t.Fatalf("canceled queued request retained published state: %+v", request)
			}
			if request.Options.NumPredict != initialNumPredict {
				t.Fatalf("NumPredict = %d, want restored %d", request.Options.NumPredict, initialNumPredict)
			}
			if len(runner.Requests) != 0 {
				t.Fatal("canceled request was queued")
			}
		})
	}
}
