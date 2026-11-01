package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

// Native Client.Load launches the current executable with runner arguments.
// This test-only entrypoint supplies the same Execute used by the CLI without
// adding a production executable override or changing shared TestMain.
func init() {
	if os.Getenv("OLLAMA_QWEN3_ASR_BRIDGE_NATIVE") == "1" && len(os.Args) > 1 && os.Args[1] == "runner" {
		if err := mlxrunner.Execute(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}

type asrHeldNativeAck struct {
	asrBackend
	reached chan struct{}
	release chan struct{}
}

func (b *asrHeldNativeAck) CloseASR(ctx context.Context, id string) error {
	if err := b.asrBackend.CloseASR(ctx, id); err != nil {
		return err
	}
	select {
	case b.reached <- struct{}{}:
	default:
	}
	select {
	case <-b.release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func TestQwen3ASRSessionNativeBridge(t *testing.T) {
	if os.Getenv("OLLAMA_QWEN3_ASR_BRIDGE_NATIVE") != "1" {
		t.Skip("supervised native session bridge only")
	}
	output := os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_OUTPUT")
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
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	schedCtx, stopSched := context.WithCancel(context.Background())
	s := &Server{sched: InitScheduler(schedCtx), defaultNumCtx: 2048}
	s.sched.Run(schedCtx)
	defer func() { s.sched.unloadAllRunners(); stopSched() }()
	reached, release := make(chan struct{}, 1), make(chan struct{})
	m := newASRSessionManager(ctx, func(lease context.Context, name string) (asrBackend, error) {
		backend, err := s.acquireASRBackend(lease, name)
		if err != nil {
			return nil, err
		}
		return &asrHeldNativeAck{asrBackend: backend, reached: reached, release: release}, nil
	})
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
		cleanup, done := context.WithTimeout(context.Background(), 5*time.Second)
		defer done()
		if err := m.shutdown(cleanup); err != nil {
			t.Error(err)
		}
	}()
	seen := map[string]bool{}
	var records []map[string]any
	for _, entry := range entries {
		if (entry.Language != "en" && entry.Language != "de") || seen[entry.Language] {
			continue
		}
		seen[entry.Language] = true
		wav, err := os.ReadFile(entry.Prepared)
		if err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256(wav)
		if hex.EncodeToString(hash[:]) != entry.SHA {
			t.Fatal("prepared WAV hash mismatch")
		}
		pcm, rate, err := audio.Decode(wav)
		if err != nil || rate != 16000 || len(pcm) < 64017 {
			t.Fatal("invalid prepared WAV", err)
		}
		pcm = pcm[:64017]
		createCtx, endCreate := context.WithCancel(ctx)
		p, err := m.create(createCtx, os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), asrstream.Config{Language: map[string]string{"en": "English", "de": "German"}[entry.Language]})
		if err != nil {
			endCreate()
			t.Fatal(err)
		}
		endCreate() // Returning the create request must not drop scheduler ownership.
		var updates []asrstream.Update
		for offset := 0; offset < len(pcm); {
			end := min(offset+16000, len(pcm))
			accepted, u, err := p.operation(ctx, offset, pcm[offset:end], false)
			if err != nil {
				t.Fatal(err)
			}
			if accepted != end {
				t.Fatal("incorrect accepted offset")
			}
			if u != nil {
				updates = append(updates, u.Update)
			}
			offset = end
		}
		_, final, err := p.operation(ctx, 0, nil, true)
		if err != nil || final == nil || !final.Final || final.AudioEnd != 64017 {
			t.Fatal(final, err)
		}
		updates = append(updates, final.Update)
		if len(records) == 0 {
			select {
			case <-reached:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			loaded, err := s.getExistingModelForRunner(os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "")
			if err != nil {
				t.Fatal(err)
			}
			s.sched.expireRunner(loaded)
			s.sched.loadedMu.Lock()
			r := s.sched.loaded[schedulerModelKey(loaded)]
			s.sched.loadedMu.Unlock()
			if r == nil {
				t.Fatal("model unloaded before native drain acknowledgment")
			}
			r.refMu.Lock()
			refs := r.refCount
			r.refMu.Unlock()
			if refs != 1 {
				t.Fatalf("expected one held scheduler lease, got %d", refs)
			}
			select {
			case <-p.done:
				t.Fatal("closed before held acknowledgment")
			default:
			}
			close(release)
		}
		select {
		case <-p.done:
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
		// Force normal scheduler unload after the lease releases, not a direct kill.
		loaded, err := s.getExistingModelForRunner(os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "")
		if err != nil {
			t.Fatal(err)
		}
		s.sched.expireRunner(loaded)
		timer := time.NewTicker(10 * time.Millisecond)
		for {
			s.sched.loadedMu.Lock()
			count := len(s.sched.loaded)
			s.sched.loadedMu.Unlock()
			if count == 0 {
				break
			}
			select {
			case <-timer.C:
			case <-ctx.Done():
				timer.Stop()
				t.Fatal("scheduler did not unload", ctx.Err())
			}
		}
		timer.Stop()
		records = append(records, map[string]any{"case": entry.Name, "language": entry.Language, "updates": updates, "scheduler_unloaded": true})
	}
	if len(records) != 2 {
		t.Fatal("need German and English native fixtures")
	}
	if err := m.shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	receipt := map[string]any{"passed": true, "records": records, "actual_scheduler_lease": true, "held_native_ack_preserves_refcount": true, "create_context_detached": true, "normal_unload_reload": true, "public_routes_added": false}
	encoded, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(output, "session-bridge.json"), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
}
