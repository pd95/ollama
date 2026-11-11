package server

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

func TestQwen3ASRAcquireDuringOtherColdLoad(t *testing.T) {
	for _, cancelRequest := range []bool{false, true} {
		t.Run(map[bool]string{false: "deadline", true: "request-cancel"}[cancelRequest], func(t *testing.T) {
			s := &Server{sched: InitScheduler(t.Context()), defaultNumCtx: 2048}
			model := &Model{Name: "asr-cold", Digest: "asr-cold-digest"}
			loading := &runnerRef{loading: true}
			loading.model.Store(model)
			s.sched.loaded[schedulerModelKey(model)] = loading
			// This is the mutex held by another request through WaitUntilRunning. It
			// must remain held until after the new create has returned and freed its slot.
			loading.refMu.Lock()
			released := false
			defer func() {
				if !released {
					loading.refMu.Unlock()
				}
			}()
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			entered := make(chan struct{})
			result := make(chan error, 1)
			m := newASRSessionManager(t.Context(), func(lease context.Context, _ string) (asrBackend, error) {
				close(entered)
				_, err := s.queueASRRunner(lease, model)
				return nil, err
			})
			m.policy.load = 20 * time.Millisecond
			if cancelRequest {
				m.policy.load = time.Second
			}
			go func() { _, err := m.create(ctx, "asr-cold", asrstream.Config{}); result <- err }()
			asrAwait(t, entered)
			if cancelRequest {
				cancel()
			}
			select {
			case err := <-result:
				expected := context.DeadlineExceeded
				if cancelRequest {
					expected = context.Canceled
				}
				if !errors.Is(err, expected) {
					t.Fatal(err)
				}
			case <-time.After(200 * time.Millisecond):
				loading.refMu.Unlock()
				released = true
				select {
				case <-result:
				case <-time.After(time.Second):
					t.Fatal("acquisition did not finish after load released")
				}
				_ = m.shutdown(t.Context())
				t.Fatal("canceled create still waits for unrelated cold load")
			}
			if err := m.shutdown(t.Context()); err != nil {
				t.Fatal(err)
			}
			m.mu.Lock()
			remaining := len(m.active)
			m.mu.Unlock()
			if remaining != 0 {
				t.Fatal("canceled creating slot retained", remaining)
			}
			// Deliberately unlock only after successful cancellation/cleanup assertions.
			loading.refMu.Unlock()
			released = true
		})
	}
}

func TestQwen3ASRAcquireQueueCapacity(t *testing.T) {
	s := &Server{sched: InitScheduler(t.Context()), defaultNumCtx: 2048}
	for range cap(s.sched.pendingReqCh) {
		s.sched.pendingReqCh <- &LlmRequest{}
	}
	_, err := s.queueASRRunner(t.Context(), &Model{})
	if !errors.Is(err, ErrMaxQueue) {
		t.Fatal("full normal scheduler queue was bypassed", err)
	}
}

func TestQwen3ASRAcquireLateSchedulerReference(t *testing.T) {
	schedCtx, stopSched := context.WithCancel(t.Context())
	defer stopSched()
	s := &Server{sched: InitScheduler(schedCtx), defaultNumCtx: 2048}
	model := &Model{Name: "asr-late", Digest: "asr-late-digest"}
	entered := make(chan struct{})
	m := newASRSessionManager(t.Context(), func(lease context.Context, _ string) (asrBackend, error) {
		close(entered)
		_, err := s.queueASRRunner(lease, model)
		return nil, err
	})
	m.policy.load = 20 * time.Millisecond
	result := make(chan error, 1)
	go func() { _, err := m.create(t.Context(), "asr-late", asrstream.Config{}); result <- err }()
	asrAwait(t, entered)
	var pending *LlmRequest
	select {
	case pending = <-s.sched.pendingReqCh:
	case <-time.After(time.Second):
		t.Fatal("not queued")
	}
	if cap(pending.successCh) != 1 || cap(pending.errCh) != 1 {
		t.Fatal("late scheduler results require bounded buffered channels")
	}
	if err := <-result; !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("loading timeout cause lost", err)
	}
	if err := m.shutdown(t.Context()); err != nil {
		t.Fatal(err)
	}
	// The normal scheduler can already have passed its canceled-request check.
	// Exercise its real late-success/refcount path after the caller has returned.
	runner := &runnerRef{modelKey: schedulerModelKey(model), sessionDuration: time.Hour}
	runner.model.Store(model)
	s.sched.loaded[runner.modelKey] = runner
	pending.useLoadedRunner(runner, s.sched.finishedReqCh)
	runner.refMu.Lock()
	refs := runner.refCount
	runner.refMu.Unlock()
	if refs != 1 || len(pending.successCh) != 1 {
		t.Fatal("late success did not retain a single scheduler reference", refs)
	}
	go s.sched.processCompleted(schedCtx)
	deadline := time.Now().Add(time.Second)
	for {
		runner.refMu.Lock()
		refs = runner.refCount
		runner.refMu.Unlock()
		if refs == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("late reference was not released", refs)
		}
		time.Sleep(time.Millisecond)
	}
	runner.refMu.Lock()
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
	}
	runner.refMu.Unlock()
}

func TestQwen3ASRAcquireMatchesNormalAdmission(t *testing.T) {
	s := &Server{sched: InitScheduler(t.Context()), defaultNumCtx: 2048}
	model := &Model{Options: map[string]any{"num_ctx": float64(2), "num_batch": float64(64)}}
	opts, err := s.modelOptions(model, nil)
	if err != nil {
		t.Fatal(err)
	}
	normalCtx, cancelNormal := context.WithCancel(t.Context())
	defer cancelNormal()
	s.sched.getRunner(normalCtx, model, opts, nil, usesAutomaticNumCtx(model, nil), usesAutomaticNumBatch(model, nil), nil)
	normal := <-s.sched.pendingReqCh
	ctx, cancel := context.WithCancel(t.Context())
	result := make(chan error, 1)
	go func() { _, err := s.queueASRRunner(ctx, model); result <- err }()
	feature := <-s.sched.pendingReqCh
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(feature.opts, normal.opts) || feature.numCtxAuto != normal.numCtxAuto || feature.numBatchAuto != normal.numBatchAuto || feature.contextShift != normal.contextShift || feature.sessionDuration != nil || feature.shift != nil {
		t.Fatal("feature admission diverges from normal native scheduler options")
	}
}
