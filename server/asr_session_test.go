package server

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

type asrFakeBackend struct {
	exited chan struct{}
	begin  func(context.Context) error
	close  func(context.Context) error
	push   func(context.Context, int, []float32) (*asrstream.Update, error)
	finish func(context.Context) (asrstream.Update, error)
}

func (f *asrFakeBackend) ReserveASR(context.Context) (string, time.Time, error) {
	return "private", time.Now().Add(time.Second), nil
}

func (f *asrFakeBackend) BeginASR(ctx context.Context, _ string, _ asrstream.Config) error {
	if f.begin != nil {
		return f.begin(ctx)
	}
	return nil
}

func (f *asrFakeBackend) PushASR(ctx context.Context, _ string, offset int, pcm []float32) (*asrstream.Update, error) {
	if f.push != nil {
		return f.push(ctx, offset, pcm)
	}
	return nil, nil
}

func (f *asrFakeBackend) FinishASR(ctx context.Context, _ string) (asrstream.Update, error) {
	return f.finish(ctx)
}

func (f *asrFakeBackend) CloseASR(ctx context.Context, _ string) error {
	if f.close != nil {
		return f.close(ctx)
	}
	return nil
}
func (f *asrFakeBackend) ASRProcessDone() <-chan struct{} { return f.exited }

func asrTestManager(t *testing.T, f *asrFakeBackend) (*asrSessionManager, <-chan struct{}) {
	t.Helper()
	var lease context.Context
	m := newASRSessionManager(t.Context(), func(ctx context.Context, _ string) (asrBackend, error) { lease = ctx; return f, nil })
	m.policy.cleanupAttempt = 20 * time.Millisecond
	m.policy.retry = time.Millisecond
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if err := m.shutdown(ctx); err != nil {
			t.Error(err)
		}
	})
	p, err := m.create(t.Context(), "model", asrstream.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if p == nil {
		t.Fatal("no session")
	}
	return m, lease.Done()
}

func asrOnlySession(t *testing.T, m *asrSessionManager) *asrPublicSession {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, p := range m.active {
		return p
	}
	t.Fatal("missing active session")
	return nil
}

func asrAwait(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(time.Second):
		t.Fatal("timed out")
	}
}

func TestQwen3ASRSessionLeaseDrains(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	f := &asrFakeBackend{exited: make(chan struct{}), close: func(ctx context.Context) error {
		once.Do(func() { close(entered) })
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}}
	m, lease := asrTestManager(t, f)
	p := asrOnlySession(t, m)
	p.startClose()
	asrAwait(t, entered)
	select {
	case <-lease:
		t.Fatal("lease released before drain")
	default:
	}
	if _, err := m.create(t.Context(), "other", asrstream.Config{}); !errors.Is(err, errASRSessionCapacity) {
		t.Fatal("closing slot reused", err)
	}
	timeout, cancel := context.WithTimeout(t.Context(), time.Millisecond)
	defer cancel()
	if err := m.closeAndWait(timeout, p.id); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	close(release)
	asrAwait(t, p.done)
	asrAwait(t, lease)
	if err := m.closeAndWait(t.Context(), p.id); err != nil {
		t.Fatal("lost close acknowledgment cannot retry", err)
	}
}

func TestQwen3ASRSessionBeginAcknowledgmentLost(t *testing.T) {
	release := make(chan struct{})
	var lease context.Context
	f := &asrFakeBackend{exited: make(chan struct{}), begin: func(context.Context) error { return errors.New("response lost") }, close: func(ctx context.Context) error {
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}}
	m := newASRSessionManager(t.Context(), func(ctx context.Context, _ string) (asrBackend, error) { lease = ctx; return f, nil })
	m.policy.cleanupAttempt = 10 * time.Millisecond
	m.policy.retry = time.Millisecond
	if _, err := m.create(t.Context(), "model", asrstream.Config{}); err == nil {
		t.Fatal("lost begin passed")
	}
	p := asrOnlySession(t, m)
	select {
	case <-lease.Done():
		t.Fatal("ambiguous begin released lease")
	default:
	}
	close(release)
	asrAwait(t, p.done)
	asrAwait(t, lease.Done())
	if err := m.shutdown(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestQwen3ASRSessionOriginalProcessExit(t *testing.T) {
	f := &asrFakeBackend{exited: make(chan struct{}), close: func(context.Context) error { return errors.New("wrong epoch or transport failure") }}
	m, lease := asrTestManager(t, f)
	p := asrOnlySession(t, m)
	p.startClose()
	select {
	case <-lease:
		t.Fatal("failed cleanup released lease")
	default:
	}
	close(f.exited)
	asrAwait(t, p.done)
	asrAwait(t, lease)
}

func TestQwen3ASRSessionAcquisitionCancellation(t *testing.T) {
	for _, mode := range []string{"timeout", "success-race"} {
		t.Run(mode, func(t *testing.T) {
			started, returned := make(chan struct{}), make(chan struct{})
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			m := newASRSessionManager(t.Context(), func(lease context.Context, _ string) (asrBackend, error) {
				close(started)
				<-lease.Done()
				close(returned)
				if mode == "success-race" {
					return &asrFakeBackend{exited: make(chan struct{})}, nil
				}
				return nil, lease.Err()
			})
			result := make(chan error, 1)
			go func() { _, err := m.create(ctx, "model", asrstream.Config{}); result <- err }()
			asrAwait(t, started)
			cancel()
			asrAwait(t, returned)
			select {
			case err := <-result:
				if !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
			case <-time.After(time.Second):
				t.Fatal("acquisition abandoned")
			}
			if err := m.shutdown(t.Context()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestQwen3ASRAcquisitionCanceledQueue(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	s := &Server{sched: InitScheduler(t.Context()), defaultNumCtx: 2048}
	result := make(chan error, 1)
	go func() { _, err := s.queueASRRunner(ctx, &Model{}); result <- err }()
	// No dispatcher: canceled admission cannot be expected to produce an error.
	cancel()
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("helper ignored canceled acquisition")
	}
}

func TestQwen3ASRSessionCreationContextIsNotLifetime(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	var lease context.Context
	f := &asrFakeBackend{exited: make(chan struct{})}
	m := newASRSessionManager(t.Context(), func(c context.Context, _ string) (asrBackend, error) { lease = c; return f, nil })
	p, err := m.create(ctx, "model", asrstream.Config{})
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case <-lease.Done():
		t.Fatal("create HTTP context owned lease")
	default:
	}
	if err := m.closeAndWait(t.Context(), p.id); err != nil {
		t.Fatal(err)
	}
	if err := m.shutdown(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestQwen3ASRSessionOffsetsAndFinish(t *testing.T) {
	f := &asrFakeBackend{exited: make(chan struct{}), push: func(_ context.Context, offset int, pcm []float32) (*asrstream.Update, error) {
		if offset+len(pcm) >= 32000 {
			return &asrstream.Update{Revision: 1, AudioEnd: 32000, Text: "hello"}, nil
		}
		return nil, nil
	}, finish: func(context.Context) (asrstream.Update, error) {
		return asrstream.Update{Revision: 2, AudioEnd: 44000, Text: "hello world", Final: true}, nil
	}}
	m, _ := asrTestManager(t, f)
	p := asrOnlySession(t, m)
	for _, n := range []int{16000, 12000, 16000} {
		accepted, u, err := p.operation(t.Context(), p.accepted, make([]float32, n), false)
		if err != nil {
			t.Fatal(err)
		}
		if accepted == 44000 && (u == nil || u.AudioEnd != 32000) {
			t.Fatal("accepted/transcribed offsets conflated")
		}
	}
	if _, _, err := p.operation(t.Context(), 32000, make([]float32, 1), false); err == nil {
		t.Fatal("hypothesis offset accepted as packet offset")
	}
	_, u, err := p.operation(t.Context(), 0, nil, true)
	if err != nil || u.AudioEnd != 44000 {
		t.Fatal(u, err)
	}
	asrAwait(t, p.done)
}

func TestQwen3ASRSessionCancellationAndOverlap(t *testing.T) {
	entered, drain := make(chan struct{}), make(chan struct{})
	f := &asrFakeBackend{exited: make(chan struct{}), push: func(ctx context.Context, _ int, _ []float32) (*asrstream.Update, error) {
		close(entered)
		<-ctx.Done()
		return nil, ctx.Err()
	}, close: func(ctx context.Context) error {
		select {
		case <-drain:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}}
	m, lease := asrTestManager(t, f)
	p := asrOnlySession(t, m)
	ctx, cancel := context.WithCancel(t.Context())
	result := make(chan error, 1)
	go func() { _, _, err := p.operation(ctx, 0, make([]float32, 1000), false); result <- err }()
	asrAwait(t, entered)
	if _, _, err := p.operation(t.Context(), 0, make([]float32, 1), false); !errors.Is(err, errASRSessionBusy) {
		t.Fatal(err)
	}
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	select {
	case <-lease:
		t.Fatal("work cancellation released lease")
	default:
	}
	close(drain)
	asrAwait(t, p.done)
}

func TestQwen3ASRSessionExpiryAndCompletedBound(t *testing.T) {
	for _, absolute := range []bool{false, true} {
		t.Run(fmt.Sprint(absolute), func(t *testing.T) {
			f := &asrFakeBackend{exited: make(chan struct{})}
			m := newASRSessionManager(t.Context(), func(context.Context, string) (asrBackend, error) { return f, nil })
			m.policy.idle = time.Hour
			m.policy.absolute = time.Hour
			if absolute {
				m.policy.absolute = time.Millisecond
			} else {
				m.policy.idle = time.Millisecond
			}
			p, err := m.create(t.Context(), "model", asrstream.Config{})
			if err != nil {
				t.Fatal(err)
			}
			asrAwait(t, p.done)
			if err := m.shutdown(t.Context()); err != nil {
				t.Fatal(err)
			}
		})
	}
	f := &asrFakeBackend{exited: make(chan struct{})}
	m := newASRSessionManager(t.Context(), func(context.Context, string) (asrBackend, error) { return f, nil })
	m.policy.completedLimit = 2
	var first string
	for i := range 3 {
		p, err := m.create(t.Context(), "model", asrstream.Config{})
		if err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			first = p.id
		}
		if err := m.closeAndWait(t.Context(), p.id); err != nil {
			t.Fatal(err)
		}
	}
	if len(m.completed) != 2 {
		t.Fatal("unbounded completed IDs")
	}
	if err := m.closeAndWait(t.Context(), first); !errors.Is(err, errASRSessionUnknown) {
		t.Fatal("evicted ID not expired", err)
	}
	if err := m.shutdown(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestQwen3ASRSessionIdleFence(t *testing.T) {
	f := &asrFakeBackend{exited: make(chan struct{})}
	m, _ := asrTestManager(t, f)
	p := asrOnlySession(t, m)
	// A stale timer callback must observe the refreshed deadline.
	p.mu.Lock()
	p.idleDeadline = time.Now().Add(time.Minute)
	p.mu.Unlock()
	p.expireIdle()
	if _, _, err := p.operation(t.Context(), 0, make([]float32, 10), false); err != nil {
		t.Fatal("stale callback closed refreshed session", err)
	}
	// Nor may it cancel work that has already acquired the operation slot.
	p.mu.Lock()
	p.busy = true
	p.idleDeadline = time.Now().Add(-time.Second)
	p.mu.Unlock()
	p.expireIdle()
	p.mu.Lock()
	state := p.state
	p.busy = false
	p.mu.Unlock()
	if state != "active" {
		t.Fatal("idle callback canceled active operation")
	}
	p.expireIdle()
	if _, _, err := p.operation(t.Context(), 10, make([]float32, 1), false); !errors.Is(err, errASRSessionClosed) {
		t.Fatal("expired callback admitted work", err)
	}
	asrAwait(t, p.done)
}

func TestQwen3ASRSessionShutdownDuringBegin(t *testing.T) {
	entered := make(chan struct{})
	var lease context.Context
	f := &asrFakeBackend{exited: make(chan struct{}), begin: func(ctx context.Context) error { close(entered); <-ctx.Done(); return ctx.Err() }}
	m := newASRSessionManager(t.Context(), func(ctx context.Context, _ string) (asrBackend, error) { lease = ctx; return f, nil })
	result := make(chan error, 1)
	go func() { _, err := m.create(t.Context(), "model", asrstream.Config{}); result <- err }()
	asrAwait(t, entered)
	if err := m.shutdown(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	asrAwait(t, lease.Done())
	if _, err := m.create(t.Context(), "model", asrstream.Config{}); !errors.Is(err, errASRSessionCapacity) {
		t.Fatal("shutdown admitted new session", err)
	}
}

func TestQwen3ASRSessionTerminalFence(t *testing.T) {
	for _, final := range []bool{false, true} {
		t.Run(fmt.Sprint(final), func(t *testing.T) {
			release := make(chan struct{})
			var once sync.Once
			f := &asrFakeBackend{exited: make(chan struct{}), close: func(ctx context.Context) error {
				select {
				case <-release:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			}, push: func(context.Context, int, []float32) (*asrstream.Update, error) {
				return nil, errors.New("ambiguous operation")
			}, finish: func(context.Context) (asrstream.Update, error) {
				return asrstream.Update{Revision: 1, AudioEnd: 201, Final: true}, nil
			}}
			m, _ := asrTestManager(t, f)
			p := asrOnlySession(t, m)
			t.Cleanup(func() { once.Do(func() { close(release) }) })
			if final {
				p.mu.Lock()
				p.accepted = 201
				p.mu.Unlock()
			}
			_, _, _ = p.operation(t.Context(), 0, make([]float32, 1), final)
			if _, _, err := p.operation(t.Context(), 0, make([]float32, 1), false); !errors.Is(err, errASRSessionClosed) {
				t.Fatal("terminal operation admitted more audio", err)
			}
			once.Do(func() { close(release) })
			asrAwait(t, p.done)
		})
	}
}
