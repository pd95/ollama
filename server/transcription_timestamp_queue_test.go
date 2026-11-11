package server

import (
	"context"
	"errors"
	"math"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
	"github.com/ollama/ollama/openai"
)

func TestTimestampQueueFIFOAndCancellation(t *testing.T) {
	q := &timestampQueue{limit: 3}
	started, release := make(chan struct{}), make(chan struct{})
	first := make(chan error, 1)
	go func() {
		_, err := q.run(t.Context(), func(context.Context) (openai.TranscriptionTimestampResponse, error) {
			close(started)
			<-release
			return openai.TranscriptionTimestampResponse{}, nil
		})
		first <- err
	}()
	<-started
	ctx, cancel := context.WithCancel(t.Context())
	canceled := make(chan error, 1)
	go func() {
		_, err := q.run(ctx, func(context.Context) (openai.TranscriptionTimestampResponse, error) {
			t.Error("canceled queued work executed")
			return openai.TranscriptionTimestampResponse{}, nil
		})
		canceled <- err
	}()
	awaitTimestampPending(t, q, 1)
	second := make(chan error, 1)
	var order []int
	go func() {
		_, err := q.run(t.Context(), func(context.Context) (openai.TranscriptionTimestampResponse, error) {
			order = append(order, 2)
			return openai.TranscriptionTimestampResponse{}, nil
		})
		second <- err
	}()
	awaitTimestampPending(t, q, 2)
	third := make(chan error, 1)
	go func() {
		_, err := q.run(t.Context(), func(context.Context) (openai.TranscriptionTimestampResponse, error) {
			order = append(order, 3)
			return openai.TranscriptionTimestampResponse{}, nil
		})
		third <- err
	}()
	awaitTimestampPending(t, q, 3)
	if _, err := q.run(t.Context(), nil); !errors.Is(err, ErrMaxQueue) {
		t.Fatal("queue capacity bypassed", err)
	}
	cancel()
	if err := <-canceled; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	awaitTimestampPending(t, q, 2)
	close(release)
	if err := <-first; err != nil {
		t.Fatal(err)
	}
	if err := <-second; err != nil {
		t.Fatal(err)
	}
	if err := <-third; err != nil {
		t.Fatal(err)
	}
	if len(order) != 2 || order[0] != 2 || order[1] != 3 {
		t.Fatal(order)
	}
	q.shutdown()
	if _, err := q.run(t.Context(), nil); !errors.Is(err, ErrMaxQueue) {
		t.Fatal("shutdown admitted work")
	}
}

func awaitTimestampPending(t *testing.T, q *timestampQueue, n int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		q.mu.Lock()
		got := len(q.pending)
		q.mu.Unlock()
		if got == n {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("timestamp queue did not reach expected count")
}

func TestTimestampQueueCanceledActiveRetainsCapacityThroughDrain(t *testing.T) {
	q := &timestampQueue{limit: 1}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	started, drain, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() {
		_, _ = q.run(ctx, func(ctx context.Context) (openai.TranscriptionTimestampResponse, error) {
			close(started)
			<-ctx.Done()
			<-drain
			return openai.TranscriptionTimestampResponse{}, ctx.Err()
		})
		close(finished)
	}()
	<-started
	cancel()
	<-finished
	waiter := make(chan error, 1)
	go func() {
		_, err := q.run(t.Context(), func(context.Context) (openai.TranscriptionTimestampResponse, error) {
			return openai.TranscriptionTimestampResponse{}, nil
		})
		waiter <- err
	}()
	awaitTimestampPending(t, q, 1)
	if _, err := q.run(t.Context(), nil); !errors.Is(err, ErrMaxQueue) {
		t.Fatal("HTTP cancellation released active admission before cleanup", err)
	}
	q.shutdown()
	if err := <-waiter; !errors.Is(err, ErrMaxQueue) {
		t.Fatal("shutdown did not release queued caller", err)
	}
	close(drain)
}

type timestampTestBackend struct {
	llm.LlamaServer
	started, closeStarted, allowDrain, processDone chan struct{}
	once                                           sync.Once
}

func (b *timestampTestBackend) ReserveAudioTask(context.Context) (string, error) {
	return strings.Repeat("1", 48), nil
}

func (b *timestampTestBackend) RunAudioTask(ctx context.Context, _ mlxrunner.AudioTaskRequest) (mlxrunner.AudioTaskResult, error) {
	close(b.started)
	<-ctx.Done()
	return mlxrunner.AudioTaskResult{}, ctx.Err()
}

func (b *timestampTestBackend) CloseAudioTask(ctx context.Context, _ string) error {
	b.once.Do(func() { close(b.closeStarted) })
	select {
	case <-b.allowDrain:
		return nil
	case <-b.processDone:
		return errors.New("original runner exited")
	case <-ctx.Done():
		return ctx.Err()
	}
}

func TestTimestampLeaseReleasesAfterOriginalProcessExit(t *testing.T) {
	s := &Server{sched: InitScheduler(t.Context())}
	b := &timestampTestBackend{started: make(chan struct{}), closeStarted: make(chan struct{}), allowDrain: make(chan struct{}), processDone: make(chan struct{})}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	result := make(chan error, 1)
	go func() {
		_, err := s.timestampStage(ctx, &Model{Name: "asr"}, func(llm.LlamaServer) (mlxrunner.AudioTaskRequest, error) {
			return mlxrunner.AudioTaskRequest{Kind: "transcribe"}, nil
		})
		result <- err
	}()
	pending := <-s.sched.pendingReqCh
	pending.successCh <- &runnerRef{llama: b}
	<-b.started
	cancel()
	<-b.closeStarted
	close(b.processDone)
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) || pending.ctx.Err() == nil {
			t.Fatal("original process exit did not release canceled lease", err)
		}
	case <-time.After(time.Second):
		t.Fatal("exact process exit did not finish cleanup")
	}
}

func TestTimestampCanceledAcquisitionFencesLateReference(t *testing.T) {
	s := &Server{sched: InitScheduler(t.Context())}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	result := make(chan error, 1)
	go func() {
		_, err := s.timestampStage(ctx, &Model{Name: "asr"}, func(llm.LlamaServer) (mlxrunner.AudioTaskRequest, error) {
			t.Error("canceled acquisition began work")
			return mlxrunner.AudioTaskRequest{}, nil
		})
		result <- err
	}()
	pending := <-s.sched.pendingReqCh
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if pending.ctx.Err() == nil {
		t.Fatal("canceled acquisition left a live scheduler lease")
	}
	// A scheduler success after the caller has returned must fit its bounded
	// result channel. The existing scheduler releases on this canceled context.
	pending.successCh <- &runnerRef{}
}

func (b *timestampTestBackend) AudioTaskProcessDone() <-chan struct{} { return b.processDone }

func TestTimestampLeaseWaitsForPositiveDrain(t *testing.T) {
	s := &Server{sched: InitScheduler(t.Context())}
	b := &timestampTestBackend{started: make(chan struct{}), closeStarted: make(chan struct{}), allowDrain: make(chan struct{}), processDone: make(chan struct{})}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	result := make(chan error, 1)
	go func() {
		_, err := s.timestampStage(ctx, &Model{Name: "asr"}, func(llm.LlamaServer) (mlxrunner.AudioTaskRequest, error) {
			return mlxrunner.AudioTaskRequest{Kind: "transcribe"}, nil
		})
		result <- err
	}()
	pending := <-s.sched.pendingReqCh
	pending.successCh <- &runnerRef{llama: b}
	<-b.started
	cancel()
	<-b.closeStarted
	select {
	case <-pending.ctx.Done():
		t.Fatal("lease released on HTTP cancellation")
	default:
	}
	select {
	case <-result:
		t.Fatal("stage returned before drain")
	default:
	}
	close(b.allowDrain)
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	select {
	case <-pending.ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("drained stage retained lease")
	}
}

func TestTimestampWordsRejectInvalidPredictions(t *testing.T) {
	units, _, err := qwen3asr.AlignmentPrompt("hello", "English")
	if err != nil {
		t.Fatal(err)
	}
	makeResult := func() *qwen3asr.AlignmentResult {
		return &qwen3asr.AlignmentResult{SourceText: "hello", Classes: []int32{0, 10}, Items: []qwen3asr.AlignmentItem{{AlignmentUnit: units[0], Start: 0, End: .8}}}
	}
	valid := makeResult()
	if words, err := timestampWords("hello", units, valid, 1); err != nil || len(words) != 1 {
		t.Fatal(words, err)
	}
	for _, change := range []func(*qwen3asr.AlignmentResult){
		func(r *qwen3asr.AlignmentResult) { r.Items[0].End = 1.01 },
		func(r *qwen3asr.AlignmentResult) { r.Items[0].End = math.NaN() },
		func(r *qwen3asr.AlignmentResult) { r.Classes = nil },
		func(r *qwen3asr.AlignmentResult) { r.SourceText = "other" },
		func(r *qwen3asr.AlignmentResult) { r.Items[0].Start = -.1 },
	} {
		r := makeResult()
		change(r)
		if _, err := timestampWords("hello", units, r, 1); err == nil {
			t.Fatal("invalid prediction accepted")
		}
	}
}
