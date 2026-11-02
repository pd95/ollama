package mlxrunner

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestAudioTaskCloseFencesLateRun(t *testing.T) {
	calls := 0
	r, err := newAudioTaskRegistry(context.Background(), func(context.Context, AudioTaskRequest) (AudioTaskResult, error) {
		calls++
		return AudioTaskResult{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	id, _, err := r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.close(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if _, err := r.run(context.Background(), AudioTaskRequest{ID: id}); err == nil || calls != 0 {
		t.Fatal("closed reservation admitted late native work")
	}
	if err := r.close(context.Background(), strings.Repeat("0", 48)); err == nil {
		t.Fatal("unissued ID acknowledged")
	}
	if err := r.close(context.Background(), id); err != nil {
		t.Fatal("issued closed ID must stay idempotent", err)
	}
}

func TestAudioTaskCancellationWaitsForNativeDrain(t *testing.T) {
	started, canceled, drain := make(chan struct{}), make(chan struct{}), make(chan struct{})
	r, err := newAudioTaskRegistry(context.Background(), func(ctx context.Context, _ AudioTaskRequest) (AudioTaskResult, error) {
		close(started)
		<-ctx.Done()
		close(canceled)
		<-drain
		return AudioTaskResult{}, ctx.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	id, _, err := r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := r.run(context.Background(), AudioTaskRequest{ID: id}); done <- err }()
	<-started
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := r.close(ctx, id); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("close reported drain before worker", err)
	}
	<-canceled
	if _, _, err := r.reserve(); !errors.Is(err, errAudioTaskBusy) {
		t.Fatal("capacity released before native drain", err)
	}
	close(drain)
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := r.close(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if _, _, err := r.reserve(); err != nil {
		t.Fatal("capacity not recovered", err)
	}
	r.shutdown()
}

func TestAudioTaskQueuedCancellationAndShutdown(t *testing.T) {
	r, err := newAudioTaskRegistry(context.Background(), func(ctx context.Context, _ AudioTaskRequest) (AudioTaskResult, error) {
		return AudioTaskResult{}, ctx.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	id, _, err := r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := r.run(ctx, AudioTaskRequest{ID: id}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := r.close(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	r.shutdown()
	if _, _, err := r.reserve(); !errors.Is(err, errAudioTaskStopped) {
		t.Fatal(err)
	}
}

func TestAudioTaskPanicIsNotDrain(t *testing.T) {
	r, err := newAudioTaskRegistry(t.Context(), func(context.Context, AudioTaskRequest) (AudioTaskResult, error) { panic("native failure") })
	if err != nil {
		t.Fatal(err)
	}
	id, _, err := r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("missing injected panic")
			}
		}()
		_, _ = r.run(t.Context(), AudioTaskRequest{ID: id})
	}()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Millisecond)
	defer cancel()
	if err := r.close(ctx, id); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("panic reported successful drain", err)
	}
}

func TestAudioTaskExecutionDeadlineAndReservationExpiry(t *testing.T) {
	r, err := newAudioTaskRegistry(t.Context(), func(ctx context.Context, _ AudioTaskRequest) (AudioTaskResult, error) {
		<-ctx.Done()
		return AudioTaskResult{}, ctx.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	r.reserveTTL = 10 * time.Millisecond
	id, _, err := r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	r.mu.Lock()
	expired := r.entry.done
	r.mu.Unlock()
	select {
	case <-expired:
	case <-time.After(time.Second):
		t.Fatal("reservation did not expire")
	}
	if _, err := r.run(t.Context(), AudioTaskRequest{ID: id}); err == nil {
		t.Fatal("expired task ran")
	}
	r.runTimeout = 10 * time.Millisecond
	id, _, err = r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.run(t.Context(), AudioTaskRequest{ID: id}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	if err := r.close(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	r.shutdown()
}
