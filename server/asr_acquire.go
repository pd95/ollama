package server

import (
	"context"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

// queueASRRunner always uses the existing bounded scheduler queue. The ordinary
// getRunner shortcut can block on another request's loading refMu before its
// caller reaches a context select. Waiting on buffered request result channels
// instead makes this feature's queued wait cancelable without a waiter goroutine.
// Even an already-loaded model enters the queue and can encounter ErrMaxQueue.
func (s *Server) queueASRRunner(ctx context.Context, m *Model) (llm.LlamaServer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	opts, err := s.modelOptions(m, nil)
	if err != nil {
		return nil, err
	}
	opts.NumCtx = max(opts.NumCtx, 4)
	if m.CheckCapabilities(model.CapabilityVision) == nil {
		opts.NumCtx = max(opts.NumCtx, 2048)
	}
	// These are the normal scheduler admission fields/defaults. ASR validates
	// completion+audio and native MLX before admission; it has no shift/keep-alive
	// overrides, embedding batch policy or GGUF context-shift launch attributes.
	req := &LlmRequest{
		ctx: ctx, model: m, opts: opts,
		successCh: make(chan *runnerRef, 1), errCh: make(chan error, 1),
		numCtxAuto: usesAutomaticNumCtx(m, nil), numBatchAuto: usesAutomaticNumBatch(m, nil),
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case s.sched.pendingReqCh <- req:
	default:
		return nil, ErrMaxQueue
	}
	// The scheduler owns processing/reuse/load. If a success races cancellation,
	// useLoadedRunner/load-success still releases that reference on req.ctx.Done.
	// The coordinator checks its setup context again before Reserve/Begin.
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case err := <-req.errCh:
		return nil, err
	case runner := <-req.successCh:
		return runner.llama, nil
	}
}
