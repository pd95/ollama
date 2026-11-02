package mlxrunner

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
)

// AudioTaskRequest is private runner IPC, not a public alignment API. Transcribe
// carries the same rendered prompt/media/options as an ordinary completion.
type AudioTaskRequest struct {
	ID         string             `json:"id"`
	Kind       string             `json:"kind"`
	Completion *CompletionRequest `json:"completion,omitempty"`
	Audio      []byte             `json:"audio,omitempty"`
	Text       string             `json:"text,omitempty"`
	Language   string             `json:"language,omitempty"`
}

type AudioTaskResult struct {
	RawText   string                    `json:"raw_text,omitempty"`
	Alignment *qwen3asr.AlignmentResult `json:"alignment,omitempty"`
}

var (
	errAudioTaskBusy    = errors.New("native audio task capacity reached")
	errAudioTaskUnknown = errors.New("unknown or closed native audio task")
	errAudioTaskStopped = errors.New("native audio task runner is shutting down")
)

type audioTaskEntry struct {
	id      string
	running bool
	ctx     context.Context //nolint:containedctx // Work cancellation is independent of transport and lease lifetime.
	cancel  context.CancelFunc
	done    chan struct{}
	timer   *time.Timer
}

// One entry bounds accepted private work. Issued epoch/counter IDs allow old
// close acknowledgments without retaining an unbounded completed-ID map.
type audioTaskRegistry struct {
	mu                     sync.Mutex
	parent                 context.Context //nolint:containedctx // Runner lifetime.
	execute                func(context.Context, AudioTaskRequest) (AudioTaskResult, error)
	epoch                  string
	next                   uint64
	entry                  *audioTaskEntry
	stopped                bool
	reserveTTL, runTimeout time.Duration
}

func newAudioTaskRegistry(parent context.Context, execute func(context.Context, AudioTaskRequest) (AudioTaskResult, error)) (*audioTaskRegistry, error) {
	var epoch [16]byte
	if _, err := rand.Read(epoch[:]); err != nil {
		return nil, err
	}
	return &audioTaskRegistry{parent: parent, execute: execute, epoch: hex.EncodeToString(epoch[:]), reserveTTL: 15 * time.Second, runTimeout: 120 * time.Second}, nil
}

func (r *audioTaskRegistry) reserve() (string, time.Time, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stopped || r.parent.Err() != nil {
		return "", time.Time{}, errAudioTaskStopped
	}
	if r.entry != nil || r.next == math.MaxUint64 {
		return "", time.Time{}, errAudioTaskBusy
	}
	r.next++
	id := fmt.Sprintf("%s%016x", r.epoch, r.next)
	ctx, cancel := context.WithCancel(r.parent)
	e := &audioTaskEntry{id: id, ctx: ctx, cancel: cancel, done: make(chan struct{})}
	r.entry = e
	until := time.Now().Add(r.reserveTTL)
	e.timer = time.AfterFunc(r.reserveTTL, func() {
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.entry == e && !e.running {
			e.cancel()
			close(e.done)
			r.entry = nil
		}
	})
	return id, until, nil
}

func (r *audioTaskRegistry) issued(id string) bool {
	if len(id) != 48 || !strings.HasPrefix(id, r.epoch) {
		return false
	}
	n, err := strconv.ParseUint(id[32:], 16, 64)
	return err == nil && n > 0 && n <= r.next && id[32:] == fmt.Sprintf("%016x", n)
}

func (r *audioTaskRegistry) run(ctx context.Context, input AudioTaskRequest) (result AudioTaskResult, err error) {
	r.mu.Lock()
	e := r.entry
	if r.stopped || e == nil || e.id != input.ID || e.running {
		r.mu.Unlock()
		return result, errAudioTaskUnknown
	}
	e.running = true
	e.timer.Stop()
	work, cancel := context.WithTimeout(e.ctx, r.runTimeout)
	r.mu.Unlock()
	stop := context.AfterFunc(ctx, cancel)
	returned := false
	defer func() {
		stop()
		cancel()
		// A panic is not a positive drain acknowledgment. Keep the operation
		// fenced until the owner observes this exact subprocess exit.
		if !returned {
			return
		}
		r.mu.Lock()
		defer r.mu.Unlock()
		e.cancel()
		// execute returns only after CPU/MLX work is drained, including errors.
		close(e.done)
		if r.entry == e {
			r.entry = nil
		}
	}()
	if err := ctx.Err(); err != nil {
		returned = true
		return result, err
	}
	if err := work.Err(); err != nil {
		returned = true
		return result, err
	}
	result, err = r.execute(work, input)
	returned = true
	return result, err
}

func (r *audioTaskRegistry) close(ctx context.Context, id string) error {
	r.mu.Lock()
	if !r.issued(id) {
		r.mu.Unlock()
		return errAudioTaskUnknown
	}
	e := r.entry
	if e == nil || e.id != id {
		r.mu.Unlock()
		return nil
	}
	e.cancel()
	if !e.running {
		e.timer.Stop()
		close(e.done)
		r.entry = nil
	}
	r.mu.Unlock()
	select {
	case <-e.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (r *audioTaskRegistry) shutdown() {
	r.mu.Lock()
	r.stopped = true
	e := r.entry
	if e != nil {
		e.cancel()
		if !e.running {
			e.timer.Stop()
			close(e.done)
			r.entry = nil
		}
	}
	r.mu.Unlock()
	if e != nil {
		<-e.done
	}
}
