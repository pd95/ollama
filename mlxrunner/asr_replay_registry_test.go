package mlxrunner

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

type replayTestTokens struct{}

func (replayTestTokens) Encode(s string, _ bool) []int32 { return []int32(s) }
func (replayTestTokens) Decode(ids []int32) string       { return string(ids) }

func testReplayRegistry(t *testing.T, infer stream.Infer) *asrReplayRegistry {
	t.Helper()
	r, err := newASRReplayRegistry(context.Background(), func(ctx context.Context, c stream.Config) (*stream.Session, error) {
		return stream.New(ctx, c, replayTestTokens{}, infer)
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.shutdown(context.Background()) })
	return r
}

func TestQwen3ASRReplayReservationFence(t *testing.T) {
	calls := 0
	r := testReplayRegistry(t, func(context.Context, stream.Request) (string, error) { calls++; return "ok", nil })
	id, _, err := r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.closeAndWait(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if err := r.begin(id, stream.Config{}); !errors.Is(err, errReplayUnknown) {
		t.Fatal("late begin resurrected closed reservation", err)
	}
	newID, _, err := r.reserve()
	if err != nil || id == newID {
		t.Fatal("reservation ID reused", err)
	}
	if err := r.begin(newID, stream.Config{Language: "English"}); err != nil {
		t.Fatal(err)
	}
	if err := r.begin(newID, stream.Config{Language: "English"}); err != nil {
		t.Fatal("lost-ack retry failed", err)
	}
	if err := r.begin(newID, stream.Config{Language: "German"}); !errors.Is(err, errReplayConflict) {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatal("reservation or begin executed inference")
	}
	other := testReplayRegistry(t, func(context.Context, stream.Request) (string, error) { return "", nil })
	if err := other.closeAndWait(context.Background(), newID); !errors.Is(err, errReplayInstance) {
		t.Fatal("new instance falsely acknowledged old cleanup", err)
	}
	future := r.epoch + "ffffffffffffffff"
	if err := r.closeAndWait(context.Background(), future); !errors.Is(err, errReplayUnknown) {
		t.Fatal("unissued future ID acknowledged", err)
	}
	for range 8 {
		id, _, err := r.reserve()
		if err != nil {
			t.Fatal(err)
		}
		if err := r.closeAndWait(context.Background(), id); err != nil {
			t.Fatal(err)
		}
	}
	if err := r.closeAndWait(context.Background(), r.epoch+"000000000000000A"); !errors.Is(err, errReplayUnknown) {
		t.Fatal("noncanonical ID acknowledged", err)
	}
	if _, err := r.push(newID, 0, make([]float32, 201)); err != nil {
		t.Fatal(err)
	}
	u, err := r.finish(newID)
	if err != nil || u.Text != "ok" || !u.Final {
		t.Fatal(u, err)
	}
	if err := r.closeAndWait(context.Background(), newID); err != nil {
		t.Fatal(err)
	}
	if err := r.begin(newID, stream.Config{}); err == nil {
		t.Fatal("completed ID reused")
	}
}

func TestQwen3ASRReplayShutdownWithoutDispatcher(t *testing.T) {
	queue := make(chan struct{}) // The unbuffered dispatcher has already exited.
	entered := make(chan struct{})
	r := testReplayRegistry(t, func(ctx context.Context, _ stream.Request) (string, error) {
		close(entered)
		select {
		case queue <- struct{}{}:
			return "unexpected", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	id, _, err := r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.begin(id, stream.Config{}); err != nil {
		t.Fatal(err)
	}
	if _, err := r.push(id, 0, make([]float32, 16000)); err != nil {
		t.Fatal(err)
	}
	finished := make(chan error, 1)
	go func() { _, err := r.push(id, 16000, make([]float32, 16000)); finished <- err }()
	<-entered
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := r.shutdown(ctx); err != nil {
		t.Fatal("unadmitted work did not drain after dispatch stopped", err)
	}
	if err := <-finished; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestQwen3ASRReplayRegistryLimits(t *testing.T) {
	r := testReplayRegistry(t, func(context.Context, stream.Request) (string, error) { return "", nil })
	r.capacity = 1
	id, _, err := r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := r.reserve(); !errors.Is(err, errReplayCapacity) {
		t.Fatal(err)
	}
	r.mu.Lock()
	r.entries[id].until = time.Now().Add(-time.Second)
	r.mu.Unlock()
	if err := r.begin(id, stream.Config{}); !errors.Is(err, errReplayUnknown) {
		t.Fatal("expired reservation registered", err)
	}
	r.timeLimit = 20 * time.Millisecond
	id, _, err = r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.begin(id, stream.Config{IdleTimeout: time.Minute}); err != nil {
		t.Fatal(err)
	}
	s, err := r.lookup(id)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-s.Done():
	case <-time.After(time.Second):
		t.Fatal("absolute session limit did not close idle session")
	}
	if err := r.shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, _, err := r.reserve(); !errors.Is(err, errReplayShutdown) {
		t.Fatal(err)
	}
}

func TestQwen3ASRReplayShutdownDrainsOutsideLocks(t *testing.T) {
	entered, allowCleanup := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(allowCleanup) })
	r := testReplayRegistry(t, func(ctx context.Context, _ stream.Request) (string, error) {
		close(entered)
		<-ctx.Done()
		<-allowCleanup
		return "", ctx.Err()
	})
	id, _, err := r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.begin(id, stream.Config{}); err != nil {
		t.Fatal(err)
	}
	if _, err := r.push(id, 0, make([]float32, 16000)); err != nil {
		t.Fatal(err)
	}
	finished := make(chan error, 1)
	go func() { _, err := r.push(id, 16000, make([]float32, 16000)); finished <- err }()
	<-entered
	timeout, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := r.shutdown(timeout); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("shutdown falsely acknowledged pending cleanup", err)
	}
	if _, _, err := r.reserve(); !errors.Is(err, errReplayShutdown) {
		t.Fatal(err)
	}
	releaseOnce.Do(func() { close(allowCleanup) })
	if err := <-finished; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := r.shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := r.closeAndWait(context.Background(), id); err != nil {
		t.Fatal(err)
	}
}

func TestQwen3ASRReplayPrivateHTTP(t *testing.T) {
	r := testReplayRegistry(t, func(context.Context, stream.Request) (string, error) { return "", nil })
	mux := http.NewServeMux()
	registerASRReplayRoutes(mux, r)
	for _, body := range []string{`{"id":""}`, `{"unknown":true}`, `{} {}`, strings.Repeat(" ", 256<<10) + `{}`} {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest("POST", "/v1/asr/replay/close", strings.NewReader(body)))
		if w.Code == 200 {
			t.Fatal("invalid private command accepted")
		}
	}
	id, _, err := r.reserve()
	if err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("POST", "/v1/asr/replay/close", strings.NewReader(`{"id":"`+id+`"}`)))
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"drained":true`) {
		t.Fatal(w.Code, w.Body)
	}
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("POST", "/v1/asr/replay/begin", strings.NewReader(`{"id":"`+id+`"}`)))
	if w.Code != 404 {
		t.Fatal("late begin passed private IPC fence", w.Code)
	}
}
