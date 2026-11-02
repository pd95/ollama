package server

import (
	"context"
	"sync"

	"github.com/ollama/ollama/openai"
)

type timestampReply struct {
	response openai.TranscriptionTimestampResponse
	err      error
}
type timestampJob struct {
	ctx    context.Context //nolint:containedctx // Queued HTTP work carries caller cancellation.
	cancel context.CancelFunc
	work   func(context.Context) (openai.TranscriptionTimestampResponse, error)
	done   chan timestampReply
}

// One worker owns a whole ASR->alignment transition, including positive cleanup.
// Waiting jobs carry file paths and metadata, never decoded PCM arrays.
type timestampQueue struct {
	mu      sync.Mutex
	active  *timestampJob
	pending []*timestampJob
	limit   int
	closed  bool
}

func (q *timestampQueue) run(ctx context.Context, work func(context.Context) (openai.TranscriptionTimestampResponse, error)) (openai.TranscriptionTimestampResponse, error) {
	if err := ctx.Err(); err != nil {
		return openai.TranscriptionTimestampResponse{}, err
	}
	jobCtx, cancel := context.WithCancel(ctx)
	j := &timestampJob{ctx: jobCtx, cancel: cancel, work: work, done: make(chan timestampReply, 1)}
	defer cancel()
	q.mu.Lock()
	if q.closed || q.active != nil && len(q.pending) >= q.limit {
		q.mu.Unlock()
		return openai.TranscriptionTimestampResponse{}, ErrMaxQueue
	}
	if q.active == nil {
		q.active = j
		go q.work(j)
	} else {
		q.pending = append(q.pending, j)
	}
	q.mu.Unlock()
	select {
	case reply := <-j.done:
		return reply.response, reply.err
	case <-ctx.Done():
		q.mu.Lock()
		for i, p := range q.pending {
			if p == j {
				copy(q.pending[i:], q.pending[i+1:])
				q.pending[len(q.pending)-1] = nil
				q.pending = q.pending[:len(q.pending)-1]
				break
			}
		}
		q.mu.Unlock()
		return openai.TranscriptionTimestampResponse{}, ctx.Err()
	}
}

func (q *timestampQueue) work(j *timestampJob) {
	for j != nil {
		var reply timestampReply
		if err := j.ctx.Err(); err != nil {
			reply.err = err
		} else {
			reply.response, reply.err = j.work(j.ctx)
		}
		j.done <- reply
		j.cancel()
		q.mu.Lock()
		q.active = nil
		if len(q.pending) > 0 {
			q.active = q.pending[0]
			q.pending[0] = nil
			q.pending = q.pending[1:]
		}
		j = q.active
		q.mu.Unlock()
	}
}

func (q *timestampQueue) shutdown() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.closed = true
	if q.active != nil {
		q.active.cancel()
	}
	for _, j := range q.pending {
		j.cancel()
		j.done <- timestampReply{err: ErrMaxQueue}
	}
	q.pending = nil
}
