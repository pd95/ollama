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

	"github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

var (
	errReplayUnknown    = errors.New("unknown or expired replay reservation")
	errReplayCapacity   = errors.New("native replay session capacity reached")
	errReplayShutdown   = errors.New("native replay registry is shutting down")
	errReplayNotStarted = errors.New("native replay session has not started")
	errReplayConflict   = errors.New("native replay session configuration conflicts")
	errReplayInstance   = errors.New("replay reservation belongs to another runner instance")
)

type replayReservation struct {
	until   time.Time
	timer   *time.Timer
	session *stream.Session
	config  stream.Config
	closing bool
}

// A reservation is CPU metadata only. Begin can consume an existing reservation
// but can never create one for an unknown ID. Therefore close-before-begin and a
// lost begin acknowledgment are safe without unbounded tombstone storage.
type asrReplayRegistry struct {
	mu                    sync.Mutex
	ctx                   context.Context //nolint:containedctx // Registry lifetime spans its independently cancelable sessions.
	factory               func(context.Context, stream.Config) (*stream.Session, error)
	epoch                 string
	next                  uint64
	entries               map[string]*replayReservation
	closing               bool
	capacity              int
	reserveTTL, timeLimit time.Duration
}

func newASRReplayRegistry(ctx context.Context, factory func(context.Context, stream.Config) (*stream.Session, error)) (*asrReplayRegistry, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if factory == nil {
		return nil, errors.New("replay registry requires a native session factory")
	}
	var epoch [16]byte
	if _, err := rand.Read(epoch[:]); err != nil {
		return nil, err
	}
	return &asrReplayRegistry{ctx: ctx, factory: factory, epoch: hex.EncodeToString(epoch[:]), entries: map[string]*replayReservation{}, capacity: 4, reserveTTL: 15 * time.Second, timeLimit: 2 * time.Minute}, nil
}

func (r *asrReplayRegistry) ready() error {
	if r.closing {
		return errReplayShutdown
	}
	return r.ctx.Err()
}

// Only IDs actually issued by this instance may receive a drained acknowledgment.
// A future counter or another epoch must not be mistaken for a closed old ID.
func (r *asrReplayRegistry) checkIssuedID(id string) error {
	if len(id) != 48 {
		return errReplayUnknown
	}
	if !strings.HasPrefix(id, r.epoch) {
		return errReplayInstance
	}
	sequence, err := strconv.ParseUint(id[32:], 16, 64)
	if err != nil || sequence == 0 || sequence > r.next || id[32:] != fmt.Sprintf("%016x", sequence) {
		return errReplayUnknown
	}
	return nil
}

func (r *asrReplayRegistry) reserve() (string, time.Time, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.ready(); err != nil {
		return "", time.Time{}, err
	}
	if len(r.entries) >= r.capacity || r.next == math.MaxUint64 {
		return "", time.Time{}, errReplayCapacity
	}
	r.next++
	id := fmt.Sprintf("%s%016x", r.epoch, r.next)
	e := &replayReservation{until: time.Now().Add(r.reserveTTL)}
	r.entries[id] = e
	e.timer = time.AfterFunc(r.reserveTTL, func() {
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.entries[id] == e && e.session == nil {
			delete(r.entries, id)
		}
	})
	return id, e.until, nil
}

func (r *asrReplayRegistry) begin(id string, config stream.Config) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.ready(); err != nil {
		return err
	}
	if err := r.checkIssuedID(id); err != nil {
		return err
	}
	e := r.entries[id]
	if e == nil {
		return errReplayUnknown
	}
	if e.closing {
		return stream.ErrClosed
	}
	if e.session != nil {
		select {
		case <-e.session.Done():
			return stream.ErrClosed
		default:
		}
		if e.config != config {
			return errReplayConflict
		}
		return nil // Idempotent retry after an acknowledgment was lost.
	}
	if !time.Now().Before(e.until) {
		e.timer.Stop()
		delete(r.entries, id)
		return errReplayUnknown
	}
	parent, cancel := context.WithTimeout(r.ctx, r.timeLimit)
	s, err := r.factory(parent, config)
	if err != nil {
		cancel()
		e.timer.Stop()
		delete(r.entries, id)
		return err
	}
	e.timer.Stop()
	e.session, e.config = s, config
	go func() {
		<-s.Done()
		cancel()
		r.mu.Lock()
		if r.entries[id] == e {
			delete(r.entries, id)
		}
		r.mu.Unlock()
	}()
	return nil
}

func (r *asrReplayRegistry) lookup(id string) (*stream.Session, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.checkIssuedID(id); err != nil {
		return nil, err
	}
	e := r.entries[id]
	if e == nil {
		return nil, errReplayUnknown
	}
	if e.closing {
		return nil, stream.ErrClosed
	}
	if e.session == nil {
		return nil, errReplayNotStarted
	}
	return e.session, nil
}

func (r *asrReplayRegistry) push(id string, offset int, pcm []float32) (*stream.Update, error) {
	s, err := r.lookup(id)
	if err != nil {
		return nil, err
	}
	return s.Push(offset, pcm)
}

func (r *asrReplayRegistry) finish(id string) (stream.Update, error) {
	s, err := r.lookup(id)
	if err != nil {
		return stream.Update{}, err
	}
	return s.Finish()
}

func (r *asrReplayRegistry) seed(id string, pcm []float32) (*stream.Update, error) {
	s, err := r.lookup(id)
	if err != nil {
		return nil, err
	}
	return s.Seed(pcm)
}

// cancel never waits and is safe for a request-cancellation callback. A model
// lease is released only by a subsequent closeAndWait acknowledgment or exit.
func (r *asrReplayRegistry) cancel(id string) (*stream.Session, error) {
	r.mu.Lock()
	if err := r.checkIssuedID(id); err != nil {
		r.mu.Unlock()
		return nil, err
	}
	e := r.entries[id]
	if e == nil {
		r.mu.Unlock()
		return nil, nil
	}
	e.closing = true
	if e.session == nil {
		e.timer.Stop()
		delete(r.entries, id)
		r.mu.Unlock()
		return nil, nil
	}
	s := e.session
	r.mu.Unlock()
	s.Close()
	return s, nil
}

func (r *asrReplayRegistry) closeAndWait(ctx context.Context, id string) error {
	s, err := r.cancel(id)
	if err != nil {
		return err
	}
	if s == nil {
		return nil
	} // IDs are never recreated; late Begin cannot register.
	select {
	case <-s.Done():
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// shutdown must run outside the MLX worker. Cancel all sessions first, then await
// terminal acknowledgment without locks. It may also run after Run returns:
// accepted jobs have drained then; unadmitted unbuffered sends observe cancellation.
// On timeout the owner must not unload weights as if this returned success.
func (r *asrReplayRegistry) shutdown(ctx context.Context) error {
	r.mu.Lock()
	r.closing = true
	var sessions []*stream.Session
	for id, e := range r.entries {
		e.closing = true
		if e.session == nil {
			e.timer.Stop()
			delete(r.entries, id)
		} else {
			sessions = append(sessions, e.session)
		}
	}
	r.mu.Unlock()
	for _, s := range sessions {
		s.Close()
	}
	for _, s := range sessions {
		select {
		case <-s.Done():
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}
