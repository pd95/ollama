package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner"
	"github.com/ollama/ollama/mlxrunner/model/audio"
)

// This fixed, opt-in qualification uses only the private imported model cache
// and owned test server/processes. It changes no production limits or contracts.
func TestQwen3ASRSessionRecoverySoakNative(t *testing.T) {
	if os.Getenv("OLLAMA_QWEN3_ASR_SESSION_SOAK_NATIVE") != "1" {
		t.Skip("supervised native recovery/soak only")
	}
	t.Setenv("OLLAMA_EXPERIMENT", "asr-streaming")
	ctx, cancel := context.WithTimeout(t.Context(), 170*time.Second)
	defer cancel()
	schedCtx, stopSched := context.WithCancel(context.Background())
	s := &Server{sched: InitScheduler(schedCtx), defaultNumCtx: 2048}
	s.sched.Run(schedCtx)
	defer func() { s.closeASRSessions(); s.sched.unloadAllRunners(); stopSched() }()
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	m := s.asrSessions
	var hookMu sync.Mutex
	var nextHold *asrHeldNativeAck
	// Optional acknowledgment delay is test-owned; native work/drain stays real.
	m.acquire = func(lease context.Context, name string) (asrBackend, error) {
		backend, err := s.acquireASRBackend(lease, name)
		if err != nil {
			return nil, err
		}
		hookMu.Lock()
		hold := nextHold
		nextHold = nil
		hookMu.Unlock()
		if hold != nil {
			hold.asrBackend = backend
			return hold, nil
		}
		return backend, nil
	}
	server := httptest.NewServer(router)
	defer server.Close()
	client := &http.Client{Timeout: 30 * time.Second}
	call := func(method, path, content string, data []byte, status int) map[string]any {
		t.Helper()
		req, err := http.NewRequestWithContext(ctx, method, server.URL+path, bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", content)
		response, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		if err != nil || response.StatusCode != status {
			t.Fatalf("%s %s: %d %s %v", method, path, response.StatusCode, body, err)
		}
		var out map[string]any
		if len(body) > 0 {
			if err := json.Unmarshal(body, &out); err != nil {
				t.Fatal(err)
			}
		}
		return out
	}
	create := func(language string) (string, *asrPublicSession) {
		t.Helper()
		body, _ := json.Marshal(map[string]string{"model": os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "language": language})
		result := call("POST", asrSessionPath, "application/json", body, 201)
		id := result["id"].(string)
		m.mu.Lock()
		p := m.active[id]
		m.mu.Unlock()
		if p == nil {
			t.Fatal("missing created session")
		}
		return id, p
	}
	await := func(done <-chan struct{}) {
		t.Helper()
		select {
		case <-done:
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
	}
	var entries []struct {
		Name     string `json:"name"`
		Language string `json:"language"`
		Prepared string `json:"prepared"`
		SHA      string `json:"prepared_sha256"`
	}
	data, err := os.ReadFile(os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_INPUTS"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatal(err)
	}
	type clip struct {
		name, language string
		pcm            []byte
	}
	wanted := map[string]bool{"corpus-00": true, "corpus-01": true, "corpus-02": true, "corpus-04": true, "corpus-06": true, "corpus-10": true}
	var clips []clip
	for _, entry := range entries {
		if !wanted[entry.Name] {
			continue
		}
		delete(wanted, entry.Name)
		wav, err := os.ReadFile(entry.Prepared)
		if err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256(wav)
		if hex.EncodeToString(hash[:]) != entry.SHA {
			t.Fatal("prepared input hash mismatch")
		}
		pcm, rate, err := audio.Decode(wav)
		if err != nil || rate != 16000 || len(pcm) < 32017 {
			t.Fatal("invalid prepared input", entry.Name, err)
		}
		clips = append(clips, clip{entry.Name, entry.Language, audio.EncodeWAV(pcm[:32017], 16000)[44:]})
	}
	if len(wanted) > 0 || len(clips) != 6 {
		t.Fatal("declared six-speech corpus incomplete")
	}
	clips = append(clips, clip{"silence", "auto", make([]byte, 32000)})
	run := func(c clip, language string) map[string]any {
		t.Helper()
		id, _ := create(language)
		base := asrSessionPath + "/" + id
		total := len(c.pcm) / 2
		// Irregular packets explicitly separate accepted and transcribed offsets.
		for offset := 0; offset < total; {
			n := min([]int{13001, 16000, 3016}[min(offset/13001, 2)], total-offset)
			result := call("POST", fmt.Sprintf("%s/audio?offset_samples=%d", base, offset), "application/octet-stream", c.pcm[offset*2:(offset+n)*2], 200)
			if result["accepted_samples"] != float64(offset+n) {
				t.Fatal("lost PCM samples")
			}
			offset += n
		}
		final := call("POST", base+"/finish", "", nil, 200)
		if final["final"] != true || final["audio_end_sample"] != float64(total) {
			t.Fatal("invalid final acknowledgment", final)
		}
		return map[string]any{"id": id, "case": c.name, "requested_language": language, "final": final}
	}
	baseline := run(clips[0], clips[0].language)
	expected := baseline["final"].(map[string]any)["text"]
	var faultRecords []map[string]any

	// Disconnect a real HTTP request after public operation admission. This does
	// not claim observation/preemption of a particular GPU kernel or first token.
	reached, release := make(chan struct{}, 1), make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	hookMu.Lock()
	nextHold = &asrHeldNativeAck{reached: reached, release: release}
	hookMu.Unlock()
	id, p := create(clips[0].language)
	base := asrSessionPath + "/" + id
	call("POST", base+"/audio?offset_samples=0", "application/octet-stream", clips[0].pcm[:26002], 200)
	call("POST", base+"/audio?offset_samples=13001", "application/octet-stream", clips[0].pcm[26002:58002], 200)
	reqCtx, stopRequest := context.WithCancel(ctx)
	req, err := http.NewRequestWithContext(reqCtx, "POST", server.URL+base+"/audio?offset_samples=29001", bytes.NewReader(clips[0].pcm[58002:]))
	if err != nil {
		stopRequest()
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	result := make(chan error, 1)
	go func() {
		response, err := client.Do(req)
		if response != nil {
			response.Body.Close()
		}
		result <- err
	}()
	for {
		p.mu.Lock()
		busy := p.busy
		p.mu.Unlock()
		if busy {
			break
		}
		timer := time.NewTimer(time.Millisecond)
		select {
		case completedErr := <-result:
			timer.Stop()
			stopRequest()
			t.Fatalf("request completed before admission was observed: %v", completedErr)
		case <-timer.C:
		case <-ctx.Done():
			timer.Stop()
			stopRequest()
			t.Fatal(ctx.Err())
		}
	}
	stopRequest()
	await(reached)
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal("request did not disconnect after admission", err)
	}
	select {
	case <-p.done:
		t.Fatal("terminal acknowledgment was not retained")
	default:
	}
	loaded, err := s.getExistingModelForRunner(os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "")
	if err != nil {
		t.Fatal(err)
	}
	s.sched.loadedMu.Lock()
	runner := s.sched.loaded[schedulerModelKey(loaded)]
	s.sched.loadedMu.Unlock()
	if runner == nil {
		t.Fatal("runner unloaded before terminal acknowledgment")
	}
	runner.refMu.Lock()
	refs := runner.refCount
	runner.refMu.Unlock()
	if refs != 1 {
		t.Fatal("disconnect lost scheduler lease", refs)
	}
	releaseOnce.Do(func() { close(release) })
	await(p.done)
	recovered := run(clips[0], clips[0].language)
	if recovered["final"].(map[string]any)["text"] != expected {
		t.Fatal("disconnect changed recovery text")
	}
	faultRecords = append(faultRecords, map[string]any{"fault": "HTTP disconnect after public admission", "held_terminal_ack_refcount": refs, "recovery_exact": true})

	// Exercise the real default idle duration, without audio or active inference.
	if m.policy.idle != 15*time.Second {
		t.Fatal("expected production default idle duration")
	}
	started := time.Now()
	id, p = create("auto")
	await(p.done)
	if time.Since(started) < 14*time.Second {
		t.Fatal("idle expiry occurred prematurely", time.Since(started))
	}
	call("POST", asrSessionPath+"/"+id+"/audio?offset_samples=0", "application/octet-stream", []byte{0, 0}, 404)
	call("DELETE", asrSessionPath+"/"+id, "", nil, 204)
	faultRecords = append(faultRecords, map[string]any{"fault": "default idle expiry", "seconds": time.Since(started).Seconds(), "configured_timeout_ms": m.policy.idle.Milliseconds()})

	// The absolute-timer mechanism is tested with a short test-only duration,
	// while traffic refreshes idle. Production's two-minute value is unchanged.
	originalAbsolute := m.policy.absolute
	if originalAbsolute != 2*time.Minute {
		t.Fatal("unexpected production absolute timeout")
	}
	m.policy.absolute = 250 * time.Millisecond
	started = time.Now()
	id, p = create("auto")
	offset := 0
	for {
		select {
		case <-p.done:
			goto absoluteDone
		case <-time.After(30 * time.Millisecond):
		}
		req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s%s/%s/audio?offset_samples=%d", server.URL, asrSessionPath, id, offset), bytes.NewReader([]byte{0, 0}))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/octet-stream")
		response, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, response.Body)
		response.Body.Close()
		if response.StatusCode == 200 {
			offset++
		} else if response.StatusCode != 404 {
			t.Fatal("unexpected expiry response", response.StatusCode)
		}
	}
absoluteDone:
	if time.Since(started) < 200*time.Millisecond || offset == 0 {
		t.Fatal("absolute expiry was premature or no traffic was accepted", time.Since(started), offset)
	}
	m.policy.absolute = originalAbsolute
	faultRecords = append(faultRecords, map[string]any{"fault": "absolute expiry during incoming audio", "test_timeout_ms": 250, "production_timeout_ms": originalAbsolute.Milliseconds(), "accepted_samples": offset, "seconds": time.Since(started).Seconds()})

	// Terminate only the exact native subprocess acquired by this owned test.
	// Public cleanup must accept its exit and a following session must reload.
	_, p = create(clips[0].language)
	owned, ok := p.backend.(*mlxrunner.Client)
	if !ok {
		t.Fatal("unexpected owned backend")
	}
	oldPID := owned.Pid()
	oldPrivateID := p.privateID
	if err := owned.Close(); err != nil {
		t.Fatal(err)
	}
	await(owned.ASRProcessDone())
	await(p.done)
	_, replacement := create(clips[0].language)
	fresh, ok := replacement.backend.(*mlxrunner.Client)
	if !ok {
		t.Fatal("unexpected replacement backend")
	}
	if fresh == owned || fresh.Pid() == oldPID || replacement.privateID == oldPrivateID {
		t.Fatal("runner/session identity reused after process loss")
	}
	if err := m.closeAndWait(ctx, replacement.id); err != nil {
		t.Fatal(err)
	}
	recovered = run(clips[0], clips[0].language)
	if recovered["final"].(map[string]any)["text"] != expected {
		t.Fatal("process loss changed recovery text")
	}
	faultRecords = append(faultRecords, map[string]any{"fault": "owned native subprocess exit", "old_pid": oldPID, "new_pid": fresh.Pid(), "recovery_exact": true})

	// Bounded repeated mixed corpus. This is many short sessions, not a long
	// microphone recording or a speech-recognition accuracy evaluation.
	references := map[string]string{}
	var records []map[string]any
	var memorySamples []uint64
	for i := range 100 {
		c := clips[i%len(clips)]
		language := c.language
		if i%2 == 1 {
			language = "auto"
		}
		record := run(c, language)
		key := c.name + "/" + language
		text := record["final"].(map[string]any)["text"].(string)
		if previous, exists := references[key]; exists && previous != text {
			t.Fatal("deterministic session recovery mismatch", key)
		}
		references[key] = text
		if i >= 14 {
			ps := call("GET", "/api/ps", "", nil, 200)
			models, ok := ps["models"].([]any)
			if !ok || len(models) != 1 {
				t.Fatal("missing model memory observation")
			}
			memory := uint64(models[0].(map[string]any)["size_vram"].(float64))
			if memory == 0 {
				t.Fatal("empty model memory observation")
			}
			memorySamples = append(memorySamples, memory)
			record["api_memory_bytes"] = memory
		}
		records = append(records, record)
	}
	lo, hi := memorySamples[0], memorySamples[0]
	for _, n := range memorySamples {
		lo = min(lo, n)
		hi = max(hi, n)
	}
	if hi-lo > 32<<20 {
		t.Fatal("mixed-session warmed memory range exceeds32MiB", hi-lo)
	}
	m.mu.Lock()
	active, completed := len(m.active), len(m.completed)
	m.mu.Unlock()
	if active != 0 || completed > 64 {
		t.Fatal("session registry not bounded", active, completed)
	}
	loaded, err = s.getExistingModelForRunner(os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "")
	if err != nil {
		t.Fatal(err)
	}
	s.sched.expireRunner(loaded)
	for {
		s.sched.loadedMu.Lock()
		count := len(s.sched.loaded)
		s.sched.loadedMu.Unlock()
		if count == 0 {
			break
		}
		timer := time.NewTimer(10 * time.Millisecond)
		select {
		case <-timer.C:
		case <-ctx.Done():
			timer.Stop()
			t.Fatal("normal scheduler unload failed")
		}
	}
	receipt := map[string]any{"passed": true, "faults": faultRecords, "sessions": records, "soak_sessions": 100, "corpus_cases": len(clips), "warmed_memory_samples": memorySamples, "warmed_memory_range_bytes": hi - lo, "completed_ids": completed, "active_sessions": active, "normal_unloaded": true, "accuracy_qualified": false, "public_contract_changed": false}
	encoded, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_OUTPUT"), "session-recovery-soak.json"), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
}
