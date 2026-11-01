// Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
// SPDX-License-Identifier: Apache-2.0
// Native bounded replay adaptation of Qwen3-ASR's streaming prefix policy.

// Package stream implements an internal, transport-independent ASR replay
// prototype. It owns PCM and text only. The caller supplies serialized native
// inference, closes the session on disconnect, and owns model scheduling.
package stream

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const (
	SampleRate       = 16000
	UpdateSamples    = 2 * SampleRate
	MaxSamples       = 60 * SampleRate
	MaxPacketSamples = SampleRate
	MaxSeedSamples   = 16 * SampleRate
	maxTextBytes     = 32768
	maxTextTokens    = 4096
)

var (
	ErrClosed = errors.New("ASR session is closed")
	ErrBusy   = errors.New("ASR session already has an operation in progress")
)

type Tokenizer interface {
	Encode(string, bool) []int32
	Decode([]int32) string
}

type Config struct {
	// UpdateSamples selects half-, one- or two-second replay; zero retains two seconds.
	UpdateSamples int
	// Language is empty for automatic detection, otherwise English or German.
	Language string
	Context  string
	// IdleTimeout applies between operations, never during active inference.
	// Zero selects 15 seconds. The internal prototype permits up to one minute.
	IdleTimeout time.Duration
}

// Request contains the complete audio prefix, without padding, and a controlled
// previous-output prefix. Inference must not retain Samples or reuse encoder/KV
// state across requests. It must enforce bounded, EOS-terminated generation and
// return newly decoded text only, excluding Prefix and EOS.
type Request struct {
	Samples []float32
	Prefix  string
	Config  Config
}

type Infer func(context.Context, Request) (string, error)

// Update replaces the entire previous hypothesis. AudioEnd is a sample offset,
// not a word timestamp. No substring is promised immutable before Final.
type Update struct {
	Revision int    `json:"revision"`
	AudioEnd int    `json:"audio_end_sample"`
	Text     string `json:"text"`
	Language string `json:"language"`
	Final    bool   `json:"final"`
}

type Session struct {
	op           sync.Mutex
	mu           sync.Mutex
	ctx          context.Context //nolint:containedctx // Owns the cancelable lifetime across incremental calls, not an individual request context.
	cancel       context.CancelFunc
	config       Config
	tok          Tokenizer
	infer        Infer
	pcm          []float32
	processed    int
	steps        int
	revision     int
	raw          string
	last         Update
	closed       bool
	active       bool
	drained      bool
	done         chan struct{}
	idle         *time.Timer
	idleDeadline time.Time
	stopParent   func() bool
}

func New(ctx context.Context, config Config, tok Tokenizer, infer Infer) (*Session, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if tok == nil || infer == nil {
		return nil, errors.New("ASR session requires a tokenizer and inference callback")
	}
	if config.Language != "" && config.Language != "English" && config.Language != "German" {
		return nil, errors.New("ASR session language must be English, German or automatic")
	}
	if len(config.Context) > 4096 || !utf8.ValidString(config.Context) {
		return nil, errors.New("ASR session context must be valid UTF-8 within 4096 bytes")
	}
	if config.UpdateSamples == 0 {
		config.UpdateSamples = UpdateSamples
	}
	if config.UpdateSamples != SampleRate/2 && config.UpdateSamples != SampleRate && config.UpdateSamples != UpdateSamples {
		return nil, errors.New("ASR update interval must be 500, 1000 or 2000 milliseconds")
	}
	if config.IdleTimeout == 0 {
		config.IdleTimeout = 15 * time.Second
	}
	if config.IdleTimeout < 0 || config.IdleTimeout > time.Minute {
		return nil, errors.New("ASR idle timeout must be positive and at most one minute")
	}
	parent := ctx
	ctx, cancel := context.WithCancel(ctx)
	s := &Session{ctx: ctx, cancel: cancel, config: config, tok: tok, infer: infer, done: make(chan struct{})}
	s.mu.Lock()
	s.idleDeadline = time.Now().Add(config.IdleTimeout)
	s.idle = time.AfterFunc(config.IdleTimeout, s.expireIdle)
	s.stopParent = context.AfterFunc(parent, s.Close)
	s.mu.Unlock()
	return s, nil
}

// Done closes after Close/Finish/cancellation and after any in-flight inference
// callback has returned. A model lease must remain held until Done, independent
// of the canceled work context. Callbacks must drain their device cleanup before
// returning; Done cannot acknowledge cleanup hidden behind an early callback return.
func (s *Session) Done() <-chan struct{} { return s.done }

func (s *Session) finishOperation() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.active = false
	if s.closed {
		s.signalDrained()
	} else {
		s.idleDeadline = time.Now().Add(s.config.IdleTimeout)
		s.idle.Reset(s.config.IdleTimeout)
	}
}

// signalDrained requires mu; release notification is issued exactly once.
func (s *Session) signalDrained() {
	if !s.active && !s.drained {
		s.drained = true
		close(s.done)
	}
}

// Close cancels in-flight inference and releases retained PCM/text. It may run
// concurrently with Push or Finish; inference must honor its context. It does
// not preempt a GPU kernel or unload a model owned by the caller.
func (s *Session) Close() {
	s.cancel()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closeLocked()
}

// A stopped/reset AfterFunc may already have queued its callback. Check the
// current idle epoch and atomically close under mu so it cannot cancel an active
// operation or a newly refreshed session after dropping the lock.
func (s *Session) expireIdle() {
	s.mu.Lock()
	if s.closed || s.active || time.Now().Before(s.idleDeadline) {
		s.mu.Unlock()
		return
	}
	s.closeLocked()
	s.mu.Unlock()
	s.cancel()
}

func (s *Session) closeLocked() {
	s.closed = true
	if s.idle != nil {
		s.idle.Stop()
	}
	if s.stopParent != nil {
		s.stopParent()
	}
	s.pcm = nil
	s.raw = ""
	s.last = Update{}
	s.signalDrained()
}

// Push accepts finite normalized mono 16 kHz PCM, capped at the lesser of one
// second or the negotiated update interval. Offset must equal the total accepted
// samples: gaps, duplicates and reordering fail without consuming input.
// Backpressure is synchronous, with at most one update and no internal queue.
func (s *Session) Push(offset int, pcm []float32) (*Update, error) {
	if !s.op.TryLock() {
		return nil, ErrBusy
	}
	defer s.op.Unlock()
	defer s.finishOperation()
	s.mu.Lock()
	if err := s.ready(); err != nil {
		s.mu.Unlock()
		s.Close()
		return nil, err
	}
	s.active = true
	s.idle.Stop()
	if offset != len(s.pcm) || len(pcm) == 0 || len(pcm) > min(MaxPacketSamples, s.config.UpdateSamples) || len(s.pcm)+len(pcm) > MaxSamples {
		s.mu.Unlock()
		return nil, errors.New("ASR packet offset, size or utterance limit is invalid")
	}
	for _, v := range pcm {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v < -1 || v > 1 {
			s.mu.Unlock()
			return nil, errors.New("ASR PCM must contain finite samples in [-1,1]")
		}
	}
	s.pcm = append(s.pcm, pcm...)
	end := s.processed + s.config.UpdateSamples
	if len(s.pcm) < end {
		s.mu.Unlock()
		return nil, nil
	}
	s.mu.Unlock()
	u, err := s.decode(end)
	if err != nil {
		s.Close()
		return nil, err
	}
	return &u, nil
}

// Seed performs one fresh decode of a complete bounded window. It is allowed
// only before any audio has been accepted, so decode cannot use a prior prefix.
// Finish then reuses this exact processed hypothesis without another inference.
func (s *Session) Seed(pcm []float32) (*Update, error) {
	if !s.op.TryLock() {
		return nil, ErrBusy
	}
	defer s.op.Unlock()
	defer s.finishOperation()
	s.mu.Lock()
	if err := s.ready(); err != nil {
		s.mu.Unlock()
		s.Close()
		return nil, err
	}
	if len(s.pcm) != 0 || s.steps != 0 || s.processed != 0 || len(pcm) < 201 || len(pcm) > MaxSeedSamples {
		s.mu.Unlock()
		return nil, errors.New("ASR seed requires an empty session and 201 to 256000 samples")
	}
	for _, value := range pcm {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value < -1 || value > 1 {
			s.mu.Unlock()
			return nil, errors.New("ASR seed PCM must contain finite samples in [-1,1]")
		}
	}
	s.active = true
	s.idle.Stop()
	s.pcm = append([]float32(nil), pcm...)
	s.mu.Unlock()
	u, err := s.decode(len(pcm))
	if err != nil {
		s.Close()
		return nil, err
	}
	return &u, nil
}

// Finish decodes the unpadded tail, then emits an immutable final replacement
// and releases the session. At an exact update boundary it reuses the latest
// hypothesis without an extra decode. Fewer than 201 total samples fail.
func (s *Session) Finish() (Update, error) {
	if !s.op.TryLock() {
		return Update{}, ErrBusy
	}
	defer s.op.Unlock()
	defer s.finishOperation()
	defer s.Close()
	s.mu.Lock()
	if err := s.ready(); err != nil {
		s.mu.Unlock()
		return Update{}, err
	}
	s.active = true
	s.idle.Stop()
	end, processed := len(s.pcm), s.processed
	u := s.last
	s.mu.Unlock()
	if end < 201 {
		return Update{}, errors.New("ASR session needs at least 201 samples")
	}
	if end > processed {
		var err error
		u, err = s.decode(end)
		if err != nil {
			return Update{}, err
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return Update{}, err
	}
	if end == processed {
		s.revision++
	}
	u.Revision, u.Final = s.revision, true
	return u, nil
}

// ready requires mu. Cancellation invalidates the session; the operation
// releases its state before returning. Callers also Close idle sessions.
func (s *Session) ready() error {
	if s.closed {
		return ErrClosed
	}
	return s.ctx.Err()
}

func (s *Session) decode(end int) (Update, error) {
	s.mu.Lock()
	if err := s.ready(); err != nil {
		s.mu.Unlock()
		return Update{}, err
	}
	request := Request{Samples: append([]float32(nil), s.pcm[:end]...), Config: s.config}
	raw, steps := s.raw, s.steps
	s.mu.Unlock()
	if steps >= 2 {
		ids := s.tok.Encode(raw, false)
		// Apply the Unicode-safe five-token rollback to both normal updates and
		// final tails. The provider final-tail code has a distinct max(1,...) rule;
		// we deliberately avoid retaining a broken Unicode prefix at that edge.
		for n := max(0, len(ids)-5); n > 0; n-- {
			prefix := s.tok.Decode(ids[:n])
			if utf8.ValidString(prefix) && !strings.ContainsRune(prefix, utf8.RuneError) {
				request.Prefix = prefix
				break
			}
		}
	}
	continuation, err := s.infer(s.ctx, request)
	if err != nil {
		return Update{}, fmt.Errorf("ASR replay: %w", err)
	}
	raw = request.Prefix + continuation
	if len(raw) > maxTextBytes || !utf8.ValidString(raw) || len(s.tok.Encode(raw, false)) > maxTextTokens {
		return Update{}, errors.New("ASR replay output exceeds text bounds or is invalid UTF-8")
	}
	u := Update{AudioEnd: end, Language: s.config.Language, Text: strings.TrimSpace(raw)}
	if s.config.Language == "" && strings.HasPrefix(strings.TrimSpace(raw), "language ") {
		if header, body, found := strings.Cut(strings.TrimSpace(raw), "<asr_text>"); found {
			u.Text = strings.TrimSpace(body)
			u.Language = strings.TrimSpace(strings.TrimPrefix(header, "language "))
			if strings.EqualFold(u.Language, "None") {
				u.Language = ""
			}
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return Update{}, err
	}
	s.steps++
	s.revision++
	u.Revision = s.revision
	s.raw, s.processed, s.last = raw, end, u
	return u, nil
}
