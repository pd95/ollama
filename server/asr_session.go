package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/mlxrunner"
	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
	"github.com/ollama/ollama/types/model"
)

type asrInputError string

func (e asrInputError) Error() string { return string(e) }

var (
	errASRSessionUnknown  = errors.New("unknown or expired audio session")
	errASRSessionCapacity = errors.New("audio session capacity reached")
	errASRSessionClosed   = errors.New("audio session is closing")
	errASRSessionBusy     = errors.New("audio session has an operation in progress")
)

// Optional native extension; other runner implementations stay unchanged.
type asrBackend interface {
	ReserveASR(context.Context) (string, time.Time, error)
	BeginASR(context.Context, string, asrstream.Config) error
	PushASR(context.Context, string, int, []float32) (*asrstream.Update, error)
	FinishASR(context.Context, string) (asrstream.Update, error)
	CloseASR(context.Context, string) error
	ASRProcessDone() <-chan struct{}
}

type asrAcquire func(context.Context, string) (asrBackend, error)

// acquireASRBackend uses the ordinary loader and scheduler, not another cache.
// ctx is the lease lifetime, deliberately independent of HTTP/work contexts.
func (s *Server) acquireASRBackend(ctx context.Context, name string) (asrBackend, error) {
	ref, err := parseAndValidateModelRef(name)
	if err != nil {
		return nil, asrInputError(err.Error())
	}
	if ref.Source == modelSourceCloud {
		return nil, asrInputError("live audio requires a local Qwen3-ASR model")
	}
	m, err := s.getExistingModelForRunner(ref.Name.String(), "")
	if err != nil {
		return nil, err
	}
	if m.Config.RemoteHost != "" || !m.IsMLX() || resolveRendererName(m) != "qwen3-asr" {
		return nil, asrInputError("live audio requires a local Qwen3-ASR model")
	}
	if err := m.CheckCapabilities(model.CapabilityCompletion, model.CapabilityAudio); err != nil {
		return nil, err
	}
	runner, err := s.queueASRRunner(ctx, m)
	if err != nil {
		return nil, err
	}
	backend, ok := runner.(*mlxrunner.Client)
	if !ok {
		return nil, asrInputError("live audio requires the native MLX runner")
	}
	return backend, nil
}

type asrSessionPolicy struct {
	load, idle, absolute, cleanupAttempt, retry, completedTTL time.Duration
	completedLimit                                            int
}

// No public routes are installed by this coordinator. Its owner must keep the
// admission slot until done, including after an operation/network cancellation.
type asrSessionManager struct {
	mu         sync.Mutex
	parent     context.Context //nolint:containedctx // Server lifetime, not any HTTP request.
	acquire    asrAcquire
	policy     asrSessionPolicy
	active     map[string]*asrPublicSession
	completed  map[string]time.Time
	closing    bool
	stopParent func() bool
}

type asrPublicSession struct {
	transport                    sync.Mutex // Serializes an HTTP operation, including upload and reply.
	manager                      *asrSessionManager
	id                           string
	mu                           sync.Mutex
	state                        string
	busy                         bool
	accepted, revision, audioEnd int
	updateSamples                int
	packetSamples                int
	stability                    asrTextStability
	idleDeadline                 time.Time
	idle, absolute               *time.Timer
	backend                      asrBackend
	privateID                    string
	mayBegin                     bool
	cancelLease                  context.CancelFunc
	cancelSetup                  context.CancelFunc
	initialized, done            chan struct{}
	closeOnce                    sync.Once
}

func newASRSessionManager(parent context.Context, acquire asrAcquire) *asrSessionManager {
	m := &asrSessionManager{
		parent: parent, acquire: acquire,
		policy: asrSessionPolicy{load: envconfig.LoadTimeout(), idle: 15 * time.Second, absolute: 2 * time.Minute, cleanupAttempt: 2 * time.Second, retry: 100 * time.Millisecond, completedTTL: time.Minute, completedLimit: 64},
		active: map[string]*asrPublicSession{}, completed: map[string]time.Time{},
	}
	m.stopParent = context.AfterFunc(parent, m.closeAll)
	return m
}

func (m *asrSessionManager) create(ctx context.Context, name string, config asrstream.Config) (*asrPublicSession, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if config.Language != "" && config.Language != "English" && config.Language != "German" {
		return nil, asrInputError("unsupported audio language")
	}
	if len(config.Context) > 4096 || !utf8.ValidString(config.Context) {
		return nil, asrInputError("audio context must be valid UTF-8 within 4096 bytes")
	}
	if config.UpdateSamples == 0 {
		config.UpdateSamples = asrstream.UpdateSamples
	}
	if config.UpdateSamples != asrstream.SampleRate/2 && config.UpdateSamples != asrstream.SampleRate && config.UpdateSamples != asrstream.UpdateSamples {
		return nil, asrInputError("audio update interval must be 500, 1000 or 2000 milliseconds")
	}
	// Public callers cannot override lifecycle policy.
	config.IdleTimeout = m.policy.idle
	var random [16]byte
	if _, err := rand.Read(random[:]); err != nil {
		return nil, err
	}
	id := hex.EncodeToString(random[:])
	leaseCtx, cancelLease := context.WithCancel(context.WithoutCancel(m.parent))
	setupCtx, cancelSetup := context.WithTimeout(ctx, m.policy.load)
	p := &asrPublicSession{manager: m, id: id, state: "creating", updateSamples: config.UpdateSamples, packetSamples: min(asrstream.MaxPacketSamples, config.UpdateSamples), cancelLease: cancelLease, cancelSetup: cancelSetup, initialized: make(chan struct{}), done: make(chan struct{})}
	m.mu.Lock()
	if m.closing || m.parent.Err() != nil || len(m.active) >= 1 {
		m.mu.Unlock()
		cancelLease()
		cancelSetup()
		return nil, errASRSessionCapacity
	}
	m.active[id] = p
	m.mu.Unlock()
	// Closing may now run concurrently, but cleanup waits for initialization.
	defer close(p.initialized)
	defer cancelSetup()
	// Only acquisition cancellation may cancel the lease directly. No backend
	// inference can have been submitted at this point. Stop the callback before
	// reserve/begin so later request cancellation cannot release an acquired lease.
	stopAcquire := context.AfterFunc(setupCtx, cancelLease)
	backend, err := m.acquire(leaseCtx, name)
	stopped := stopAcquire()
	if setupErr := setupCtx.Err(); setupErr != nil {
		err = setupErr
	} else if err == nil && (!stopped || leaseCtx.Err() != nil) {
		err = context.Canceled
	}
	if err != nil {
		p.startClose()
		return nil, asrAcquisitionError(name, err)
	}
	p.backend = backend
	p.privateID, _, err = backend.ReserveASR(setupCtx)
	if err == nil {
		p.mayBegin = true // Even a failed/lost begin response requires reconciliation.
		err = backend.BeginASR(setupCtx, p.privateID, config)
	}
	if err != nil {
		p.startClose()
		return nil, err
	}
	p.mu.Lock()
	if p.state != "creating" || setupCtx.Err() != nil {
		p.mu.Unlock()
		p.startClose()
		return nil, context.Canceled
	}
	p.state = "active"
	p.idleDeadline = time.Now().Add(m.policy.idle)
	p.idle = time.AfterFunc(m.policy.idle, p.expireIdle)
	p.absolute = time.AfterFunc(m.policy.absolute, p.startClose)
	p.mu.Unlock()
	go func() {
		select {
		case <-backend.ASRProcessDone():
			p.startClose()
		case <-p.done:
		}
	}()
	return p, nil
}

func (p *asrPublicSession) expireIdle() {
	p.mu.Lock()
	expire := p.state == "active" && !p.busy && !time.Now().Before(p.idleDeadline)
	if expire {
		p.state = "closing"
	} // Fence new operations before releasing the lock.
	p.mu.Unlock()
	if expire {
		p.startClose()
	}
}

func (p *asrPublicSession) startClose() {
	p.closeOnce.Do(func() {
		p.mu.Lock()
		p.state = "closing"
		if p.idle != nil {
			p.idle.Stop()
		}
		if p.absolute != nil {
			p.absolute.Stop()
		}
		p.mu.Unlock()
		p.cancelSetup()
		go p.cleanup()
	})
}

func (p *asrPublicSession) cleanup() {
	<-p.initialized
	if p.backend != nil && p.privateID != "" {
		for {
			ctx, cancel := context.WithTimeout(context.Background(), p.manager.policy.cleanupAttempt)
			err := p.backend.CloseASR(ctx, p.privateID)
			cancel()
			if err == nil || !p.mayBegin {
				break
			}
			// A timeout/404/epoch error is not evidence of drain. Only the original
			// process exit can replace its terminal acknowledgment.
			timer := time.NewTimer(p.manager.policy.retry)
			select {
			case <-p.backend.ASRProcessDone():
				timer.Stop()
				goto drained
			case <-timer.C:
			}
		}
	}
drained:
	p.cancelLease()
	p.mu.Lock()
	p.state = "closed"
	p.mu.Unlock()
	m := p.manager
	m.mu.Lock()
	delete(m.active, p.id)
	now := time.Now()
	for id, until := range m.completed {
		if !now.Before(until) {
			delete(m.completed, id)
		}
	}
	if len(m.completed) >= m.policy.completedLimit {
		var oldest string
		var until time.Time
		for id, t := range m.completed {
			if oldest == "" || t.Before(until) {
				oldest, until = id, t
			}
		}
		delete(m.completed, oldest)
	}
	m.completed[p.id] = now.Add(m.policy.completedTTL)
	close(p.done)
	m.mu.Unlock()
}

func (m *asrSessionManager) closeAndWait(ctx context.Context, id string) error {
	m.mu.Lock()
	p := m.active[id]
	if p == nil {
		until, ok := m.completed[id]
		if ok && time.Now().Before(until) {
			m.mu.Unlock()
			return nil
		}
		delete(m.completed, id)
		m.mu.Unlock()
		return errASRSessionUnknown
	}
	m.mu.Unlock()
	p.startClose()
	select {
	case <-p.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *asrSessionManager) closeAll() {
	m.mu.Lock()
	m.closing = true
	sessions := make([]*asrPublicSession, 0, len(m.active))
	for _, p := range m.active {
		sessions = append(sessions, p)
	}
	m.mu.Unlock()
	for _, p := range sessions {
		p.startClose()
	}
}

func (m *asrSessionManager) shutdown(ctx context.Context) error {
	m.stopParent()
	m.closeAll()
	m.mu.Lock()
	sessions := make([]*asrPublicSession, 0, len(m.active))
	for _, p := range m.active {
		sessions = append(sessions, p)
	}
	m.mu.Unlock()
	for _, p := range sessions {
		select {
		case <-p.done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func (p *asrPublicSession) operation(ctx context.Context, offset int, pcm []float32, final bool) (int, *asrPublicUpdate, error) {
	if err := ctx.Err(); err != nil {
		p.startClose()
		return 0, nil, err
	}
	p.mu.Lock()
	if p.state != "active" {
		p.mu.Unlock()
		return 0, nil, errASRSessionClosed
	}
	if p.busy {
		p.mu.Unlock()
		return 0, nil, errASRSessionBusy
	}
	if !final && (offset != p.accepted || len(pcm) == 0 || len(pcm) > p.packetSamples || p.accepted+len(pcm) > asrstream.MaxSamples) {
		p.mu.Unlock()
		return 0, nil, asrInputError("invalid audio offset, packet size or duration")
	}
	if final && p.accepted < 201 {
		p.mu.Unlock()
		return 0, nil, asrInputError("audio session needs at least 201 samples")
	}
	p.busy = true
	p.idle.Stop()
	accepted, previousRevision, previousEnd := p.accepted, p.revision, p.audioEnd
	p.mu.Unlock()
	stop := context.AfterFunc(ctx, p.startClose)
	defer stop()
	var u *asrstream.Update
	var err error
	if final {
		var result asrstream.Update
		result, err = p.backend.FinishASR(ctx, p.privateID)
		u = &result
	} else {
		u, err = p.backend.PushASR(ctx, p.privateID, offset, pcm)
		accepted += len(pcm)
	}
	if err == nil && ctx.Err() != nil {
		err = ctx.Err()
	}
	if err == nil {
		expectedEnd := previousEnd + p.updateSamples
		if final {
			expectedEnd = accepted
		}
		due := final || accepted >= expectedEnd
		if due != (u != nil) || (u != nil && (u.AudioEnd != expectedEnd || u.Final != final || u.Revision != previousRevision+1 || !utf8.ValidString(u.Text) || len(u.Text) > 32768)) {
			err = errors.New("invalid ASR update acknowledgment")
		}
	}
	var published *asrPublicUpdate
	p.mu.Lock()
	p.busy = false
	if err == nil && p.state != "active" {
		err = errASRSessionClosed
	}
	if err != nil || final {
		if p.state != "closed" {
			p.state = "closing"
		} // Fence another operation before asynchronous cleanup.
	}
	if err == nil {
		p.accepted = accepted
		if u != nil {
			p.revision, p.audioEnd = u.Revision, u.AudioEnd
			update := p.stability.present(*u)
			published = &update
		}
		if !final {
			p.idleDeadline = time.Now().Add(p.manager.policy.idle)
			p.idle.Reset(p.manager.policy.idle)
		}
	}
	p.mu.Unlock()
	if err != nil || final {
		p.startClose()
	}
	return accepted, published, err
}

func asrAcquisitionError(name string, err error) error {
	var input asrInputError
	var statusError *api.StatusError
	if errors.As(err, &input) || errors.As(err, &statusError) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	status, message := scheduleError(name, err)
	if errors.Is(err, mlxrunner.ErrRuntimeUnavailable) {
		status = 503
	}
	return &api.StatusError{StatusCode: status, ErrorMessage: message}
}
