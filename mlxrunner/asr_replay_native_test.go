package mlxrunner

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxthread"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

func TestQwen3ASRRunnerReplay(t *testing.T) {
	inputs, output := os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_INPUTS"), os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_OUTPUT")
	if inputs == "" || output == "" {
		t.Skip("set supervised native runner replay inputs/output")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	worker, err := mlxthread.Start("asr-replay-test", func() error {
		if err := mlx.CheckInit(); err != nil {
			return err
		}
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Stop(context.Background(), mlx.ClearCache) //nolint:errcheck
	r := &Runner{mlxThread: worker, Requests: make(chan Request)}
	if err := worker.Do(ctx, func() error { return r.Load("qwen3-asr:component") }); err != nil {
		t.Fatal(err)
	}
	weightsClosed := false
	defer func() {
		if weightsClosed {
			return
		}
		_ = worker.Do(context.Background(), func() error { r.Close(); mlx.DefaultStream().Synchronize(); mlx.ClearCache(); return nil })
	}()

	// Endable test dispatcher mirrors Run's exact acceptance/terminal boundary:
	// each request is finite, and Responses closes only after runRequest returns.
	stopPump, pumpDone := make(chan struct{}), make(chan struct{})
	var hookMu sync.Mutex
	var accepted func()
	var onToken, beforeAck func()
	go func() {
		defer close(pumpDone)
		for {
			select {
			case <-stopPump:
				return
			case request := <-r.Requests:
				hookMu.Lock()
				hook := accepted
				accepted = nil
				tokenHook, ackHook := onToken, beforeAck
				onToken, beforeAck = nil, nil
				hookMu.Unlock()
				if hook != nil {
					hook()
				}
				toRun := request
				var relayDone chan struct{}
				if tokenHook != nil {
					proxy := make(chan CompletionResponse)
					relayDone = make(chan struct{})
					toRun.Responses = proxy
					go func() {
						defer close(relayDone)
						observed := false
						for response := range proxy {
							if response.Content != "" && !observed {
								observed = true
								tokenHook()
							}
							request.Responses <- response
						}
					}()
				}
				err := r.runRequest(toRun)
				if relayDone != nil {
					close(toRun.Responses)
					<-relayDone
				}
				if err != nil {
					select {
					case request.Responses <- CompletionResponse{Error: &api.StatusError{StatusCode: 500, ErrorMessage: err.Error()}}:
					case <-request.Ctx.Done():
					}
				}
				if ackHook != nil {
					ackHook()
				}
				close(request.Responses)
			}
		}
	}()
	defer func() { close(stopPump); <-pumpDone }()
	data, err := os.ReadFile(inputs)
	if err != nil {
		t.Fatal(err)
	}
	var entries []struct {
		Name     string `json:"name"`
		Language string `json:"language"`
		Prepared string `json:"prepared"`
		SHA      string `json:"prepared_sha256"`
	}
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatal(err)
	}
	var names, langs []string
	var waves [][]float32
	seen := map[string]bool{}
	for _, entry := range entries {
		if entry.Language != "en" && entry.Language != "de" || seen[entry.Language] {
			continue
		}
		seen[entry.Language] = true
		b, err := os.ReadFile(entry.Prepared)
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.Sum256(b)
		if hex.EncodeToString(h[:]) != entry.SHA {
			t.Fatal("input SHA mismatch")
		}
		pcm, rate, err := audio.Decode(b)
		if err != nil || rate != 16000 {
			t.Fatal("invalid prepared PCM")
		}
		pcm = pcm[:min(len(pcm), 64017)]
		names = append(names, entry.Name)
		langs = append(langs, map[string]string{"en": "English", "de": "German"}[entry.Language])
		waves = append(waves, pcm)
	}
	if len(waves) != 2 {
		t.Fatal("runner replay requires English and German")
	}
	newSession := func(parent context.Context, index int) *stream.Session {
		t.Helper()
		s, err := r.newASRReplaySession(parent, stream.Config{Language: langs[index]})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s
	}
	feed := func(s *stream.Session, pcm []float32) []stream.Update {
		t.Helper()
		var updates []stream.Update
		for offset := 0; offset < len(pcm); {
			n := min(16000, len(pcm)-offset)
			u, err := s.Push(offset, pcm[offset:offset+n])
			if err != nil {
				t.Fatal(err)
			}
			if u != nil {
				updates = append(updates, *u)
			}
			offset += n
		}
		u, err := s.Finish()
		if err != nil {
			t.Fatal(err)
		}
		return append(updates, u)
	}
	var baselines [][]stream.Update
	var records []map[string]any
	for i, pcm := range waves {
		updates := feed(newSession(ctx, i), pcm)
		baselines = append(baselines, updates)
		records = append(records, map[string]any{"case": names[i], "mode": "sequential", "updates": updates})
	}
	sessions := []*stream.Session{newSession(ctx, 0), newSession(ctx, 1)}
	interleaved := make([][]stream.Update, 2)
	for offset := 0; offset < max(len(waves[0]), len(waves[1])); offset += 16000 {
		for i, s := range sessions {
			if offset >= len(waves[i]) {
				continue
			}
			n := min(16000, len(waves[i])-offset)
			u, err := s.Push(offset, waves[i][offset:offset+n])
			if err != nil {
				t.Fatal(err)
			}
			if u != nil {
				interleaved[i] = append(interleaved[i], *u)
			}
		}
	}
	for i, s := range sessions {
		u, err := s.Finish()
		if err != nil {
			t.Fatal(err)
		}
		interleaved[i] = append(interleaved[i], u)
		if !reflect.DeepEqual(interleaved[i], baselines[i]) {
			t.Fatal("interleaved native session changed transcript")
		}
		records = append(records, map[string]any{"case": names[i], "mode": "interleaved", "updates": interleaved[i]})
	}

	// Hold the actual worker with an unrelated finite job, then cancel a replay
	// update queued behind it. Its own terminal acknowledgment must still drain.
	blocked, releaseBlock, blockDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() {
		defer close(blockDone)
		_ = worker.Do(ctx, func() error { close(blocked); <-releaseBlock; return nil })
	}()
	<-blocked
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(releaseBlock) })
	queued, releaseLease := context.WithCancel(context.Background())
	defer releaseLease()
	parent, cancelQueued := context.WithCancel(ctx)
	defer cancelQueued()
	s := newSession(parent, 0)
	go func() { <-s.Done(); releaseLease() }()
	if _, err := s.Push(0, waves[0][:16000]); err != nil {
		t.Fatal(err)
	}
	admitted := make(chan struct{})
	hookMu.Lock()
	accepted = func() { close(admitted) }
	hookMu.Unlock()
	finished := make(chan error, 1)
	go func() { _, err := s.Push(16000, waves[0][16000:32000]); finished <- err }()
	select {
	case <-admitted:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	if queued.Err() != nil {
		t.Fatal("lease dropped before cancellation/drain")
	}
	cancelQueued()
	select {
	case err := <-finished:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	select {
	case <-queued.Done():
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	releaseOnce.Do(func() { close(releaseBlock) })
	<-blockDone
	// The same runner remains usable, without cross-session generation history.
	recovered := feed(newSession(ctx, 0), waves[0])
	if !reflect.DeepEqual(recovered, baselines[0]) {
		t.Fatal("queued cancellation poisoned runner")
	}
	// Cancel only after a real native token has been produced. Hold the test
	// dispatcher's terminal acknowledgment after the pipeline's device cleanup;
	// Session.Done and the independent lease must remain pending until release.
	activeParent, cancelActive := context.WithCancel(ctx)
	defer cancelActive()
	activeSession := newSession(activeParent, 0)
	activeLease, releaseActiveLease := context.WithCancel(context.Background())
	defer releaseActiveLease()
	go func() { <-activeSession.Done(); releaseActiveLease() }()
	tokenSeen, cleanupReached, allowAck := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var ackOnce sync.Once
	defer ackOnce.Do(func() { close(allowAck) })
	hookMu.Lock()
	onToken = func() { cancelActive(); close(tokenSeen) }
	beforeAck = func() { close(cleanupReached); <-allowAck }
	hookMu.Unlock()
	if _, err := activeSession.Push(0, waves[0][:16000]); err != nil {
		t.Fatal(err)
	}
	activeResult := make(chan error, 1)
	go func() { _, err := activeSession.Push(16000, waves[0][16000:32000]); activeResult <- err }()
	select {
	case <-tokenSeen:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	select {
	case <-cleanupReached:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	select {
	case <-activeSession.Done():
		t.Fatal("session finished before native terminal acknowledgment")
	default:
	}
	if activeLease.Err() != nil {
		t.Fatal("active model lease released before drain acknowledgment")
	}
	ackOnce.Do(func() { close(allowAck) })
	select {
	case err := <-activeResult:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	select {
	case <-activeLease.Done():
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	if got := feed(newSession(ctx, 0), waves[0]); !reflect.DeepEqual(got, baselines[0]) {
		t.Fatal("active cancellation poisoned runner")
	}
	// Exercise the actual private IPC mux against the loaded native runner.
	mux := http.NewServeMux()
	closeReplay, err := r.installASRReplayRoutes(ctx, mux)
	if err != nil {
		t.Fatal(err)
	}
	defer closeReplay()
	ipc := httptest.NewServer(mux)
	defer ipc.Close()
	post := func(action string, body any, status int) []byte {
		t.Helper()
		b, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		response, err := http.Post(ipc.URL+"/v1/asr/replay/"+action, "application/json", bytes.NewReader(b))
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		data, err := io.ReadAll(io.LimitReader(response.Body, 65536))
		if err != nil || response.StatusCode != status {
			t.Fatalf("private %s: status=%d error=%v body=%s", action, response.StatusCode, err, data)
		}
		return data
	}
	var reserved struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(post("reserve", nil, 200), &reserved); err != nil {
		t.Fatal(err)
	}
	post("begin", map[string]any{"id": reserved.ID, "config": stream.Config{Language: langs[0]}}, 200)
	post("push", map[string]any{"id": reserved.ID, "offset": 0, "pcm": waves[0][:16000]}, 200)
	var ipcUpdate stream.Update
	if err := json.Unmarshal(post("push", map[string]any{"id": reserved.ID, "offset": 16000, "pcm": waves[0][16000:32000]}, 200), &ipcUpdate); err != nil {
		t.Fatal(err)
	}
	if ipcUpdate.Text != baselines[0][0].Text {
		t.Fatal("private IPC changed native transcript")
	}
	if err := json.Unmarshal(post("finish", map[string]any{"id": reserved.ID}, 200), &ipcUpdate); err != nil || !ipcUpdate.Final {
		t.Fatal("private finish failed", err)
	}
	post("close", map[string]any{"id": reserved.ID}, 200)
	post("begin", map[string]any{"id": reserved.ID, "config": stream.Config{Language: langs[0]}}, 404)
	active, err := mlxthread.Call(ctx, worker, func() (int, error) { return mlx.ActiveMemory(), nil })
	if err != nil {
		t.Fatal(err)
	}
	// Drain the registry outside the worker, then close model weights on it.
	closeReplay()
	ipc.Close()
	var unloaded int
	if err := worker.Do(ctx, func() error {
		r.Close()
		mlx.DefaultStream().Synchronize()
		mlx.ClearCache()
		unloaded = mlx.ActiveMemory()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	weightsClosed = true
	if unloaded > 1<<20 {
		t.Fatalf("native model close retained %d device bytes", unloaded)
	}
	result := map[string]any{"passed": true, "records": records, "queued_cancellation": true, "post_first_token_cancellation": true, "terminal_ack_retains_lease": true, "private_ipc_roundtrip": true, "late_begin_rejected": true, "recovery_exact": true, "active_bytes": active, "unloaded_active_bytes": unloaded, "public_api_added": false, "scheduler_lease": "separate-context contract tested; outer scheduler bridge not integrated"}
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(output, "runner-replay.json"), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
}
