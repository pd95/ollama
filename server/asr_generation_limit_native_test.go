package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

type asrNativeLimitPlan struct {
	mu                 sync.Mutex
	point              string
	ordinary           bool
	acquired, injected int
	publicDeletes      int
	closed             map[int]bool
	events             []string
	violation          bool
}
type asrNativeLimitFault struct {
	asrBackend
	plan    *asrNativeLimitPlan
	ordinal int
}

func (b *asrNativeLimitFault) fault(point string) error {
	p := b.plan
	p.mu.Lock()
	defer p.mu.Unlock()
	if b.ordinal != 2 || p.point != point || p.injected != 0 {
		return nil
	}
	p.injected++
	p.events = append(p.events, "drained-injected-"+point)
	if p.ordinary {
		return errors.New("controlled unrelated native failure")
	}
	return mlxrunner.ErrASRGenerationLimit
}

func (b *asrNativeLimitFault) PushASR(ctx context.Context, id string, offset int, pcm []float32) (*asrstream.Update, error) {
	u, err := b.asrBackend.PushASR(ctx, id, offset, pcm)
	// Inject only after a real inference update has returned and drained. This
	// test never pretends that a no-update upload executed neural inference.
	if err == nil && u != nil {
		if injected := b.fault("push"); injected != nil {
			u.Text = "UNACCEPTED_FAULT_TEXT"
			return u, injected
		}
	}
	return u, err
}

func (b *asrNativeLimitFault) FinishASR(ctx context.Context, id string) (asrstream.Update, error) {
	u, err := b.asrBackend.FinishASR(ctx, id)
	if err == nil {
		if injected := b.fault("finish"); injected != nil {
			u.Text = "UNACCEPTED_FAULT_TEXT"
			return u, injected
		}
	}
	return u, err
}

func (b *asrNativeLimitFault) CloseASR(ctx context.Context, id string) error {
	err := b.asrBackend.CloseASR(ctx, id)
	if err == nil {
		b.plan.mu.Lock()
		if !b.plan.closed[b.ordinal] {
			b.plan.closed[b.ordinal] = true
			b.plan.events = append(b.plan.events, fmt.Sprintf("close-ack-%d", b.ordinal))
		}
		b.plan.mu.Unlock()
	}
	return err
}

// The failure is an explicit test annotation after actual native inference/drain,
// not evidence that this synthetic audio naturally exhausts a generation budget.
func TestQwen3ASRGenerationLimitNative(t *testing.T) {
	if os.Getenv("OLLAMA_QWEN3_ASR_GENERATION_LIMIT_NATIVE") != "1" {
		t.Skip("supervised native generation-limit policy only")
	}
	t.Setenv("OLLAMA_EXPERIMENT", "asr-streaming")
	ctx, cancel := context.WithTimeout(t.Context(), 160*time.Second)
	defer cancel()
	output := os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_OUTPUT")
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Clean(filepath.Join(cwd, "..", ".."))
	tools := filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools")
	binary := filepath.Join(output, "generation-limit-client")
	args := []string{"build", "-o", binary}
	for _, name := range []string{"main.go", "pcm.go", "render.go", "utterance.go", "utterance_detector.go", "generation_limit.go", "overlap.go", "overlap_text.go"} {
		args = append(args, filepath.Join(tools, "sessionclient", name))
	}
	build := exec.CommandContext(ctx, "go", args...)
	build.Dir = filepath.Join(root, "ollama")
	if log, err := build.CombinedOutput(); err != nil {
		t.Fatal(err, string(log))
	}
	samples := make([]float32, 6*16000)
	for i := range samples {
		v := float32(.20 + .01*float64(i/32000))
		if i%2 != 0 {
			v = -v
		}
		samples[i] = v
	}
	raw := audio.EncodeWAV(samples, 16000)[44:]
	schedCtx, stopSched := context.WithCancel(context.Background())
	s := &Server{sched: InitScheduler(schedCtx), defaultNumCtx: 2048}
	s.sched.Run(schedCtx)
	defer func() { s.closeASRSessions(); s.sched.unloadAllRunners(); stopSched() }()
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	manager := s.asrSessions
	var current *asrNativeLimitPlan
	var currentMu sync.Mutex
	manager.acquire = func(lease context.Context, name string) (asrBackend, error) {
		currentMu.Lock()
		p := current
		currentMu.Unlock()
		p.mu.Lock()
		p.acquired++
		ordinal := p.acquired
		if ordinal > 1 && !p.closed[ordinal-1] {
			p.violation = true
		}
		p.events = append(p.events, fmt.Sprintf("acquire-%d", ordinal))
		p.mu.Unlock()
		backend, err := s.acquireASRBackend(lease, name)
		if err != nil {
			return nil, err
		}
		return &asrNativeLimitFault{asrBackend: backend, plan: p, ordinal: ordinal}, nil
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		router.ServeHTTP(w, r)
		if r.Method == "DELETE" && strings.HasPrefix(r.URL.Path, asrSessionPath+"/") {
			currentMu.Lock()
			p := current
			currentMu.Unlock()
			if p != nil {
				p.mu.Lock()
				p.publicDeletes++
				p.events = append(p.events, fmt.Sprintf("public-delete-returned-%d", p.publicDeletes))
				p.mu.Unlock()
			}
		}
	}))
	defer server.Close()
	defer cancel()
	readJSON := func(path string) map[string]any {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var r map[string]any
		if err := json.Unmarshal(b, &r); err != nil {
			t.Fatal(err)
		}
		return r
	}
	events := func(dir string) []map[string]any {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(dir, "events.ndjson"))
		if os.IsNotExist(err) {
			return nil
		}
		if err != nil {
			t.Fatal(err)
		}
		var out []map[string]any
		lines := bytes.Split(b, []byte{'\n'})
		for _, line := range lines[:len(lines)-1] {
			if len(line) == 0 {
				continue
			}
			var r map[string]any
			if err := json.Unmarshal(line, &r); err != nil {
				t.Fatal(err)
			}
			out = append(out, r)
		}
		return out
	}
	makeCommand := func(dir string, policy string) *exec.Cmd {
		args := []string{"-mode", "pcm", "-segment-seconds", "2", "-url", server.URL, "-model", os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "-language", "en", "-output", dir, "-utterance-detection", "off", "-update-interval-ms", "500"}
		if policy != "" {
			args = append(args, "-on-generation-limit", policy)
		}
		return exec.CommandContext(ctx, binary, args...)
	}
	var results []map[string]any
	for _, c := range []struct {
		name, point, policy string
		frames              int
		ordinary, canceled  bool
	}{
		{name: "continue-push", point: "push", policy: "continue", frames: 96000},
		{name: "continue-finish", point: "finish", policy: "continue", frames: 96000},
		{name: "default-stop", point: "push", frames: 96000},
		{name: "other-error-stop", point: "push", policy: "continue", frames: 96000, ordinary: true},
		{name: "truncated-eof-gap", point: "push", policy: "continue", frames: 3*16000 + 170},
		{name: "cancel-during-skip", point: "push", policy: "continue", frames: 3 * 16000, canceled: true},
	} {
		p := &asrNativeLimitPlan{point: c.point, ordinary: c.ordinary, closed: map[int]bool{}}
		currentMu.Lock()
		current = p
		currentMu.Unlock()
		dir := filepath.Join(output, c.name)
		cmd := makeCommand(dir, c.policy)
		var terminal bytes.Buffer
		cmd.Stdout = &terminal
		cmd.Stderr = &terminal
		var runErr error
		if !c.canceled {
			cmd.Stdin = bytes.NewReader(raw[:c.frames*2])
			runErr = cmd.Run()
		} else {
			writer, err := cmd.StdinPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() { done <- cmd.Wait() }()
			written := make(chan error, 1)
			go func() { _, err := writer.Write(raw[:c.frames*2]); written <- err }()
			select {
			case err := <-written:
				if err != nil {
					t.Fatal(err)
				}
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			until := time.NewTimer(20 * time.Second)
			tick := time.NewTicker(20 * time.Millisecond)
			ready := false
			for !ready {
				p.mu.Lock()
				ready = p.injected == 1 && p.closed[2] && p.publicDeletes >= 2
				p.mu.Unlock()
				if ready {
					break
				}
				select {
				case err := <-done:
					t.Fatal("client ended before skipped-range interruption", err, terminal.String())
				case <-until.C:
					t.Fatal("fault cleanup deadline")
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				case <-tick.C:
				}
			}
			until.Stop()
			tick.Stop()
			if err := cmd.Process.Signal(os.Interrupt); err != nil {
				t.Fatal(err)
			}
			select {
			case runErr = <-done:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			_ = writer.Close()
		}
		summary := readJSON(filepath.Join(dir, "summary.json"))
		records := events(dir)
		p.mu.Lock()
		injected, acquired, violation := p.injected, p.acquired, p.violation
		trace := append([]string(nil), p.events...)
		closed := p.closed[acquired]
		p.mu.Unlock()
		if injected != 1 || violation || !closed || summary["cleanup"] != "acknowledged" {
			t.Fatal("fault/lease ordering", c.name, summary, trace)
		}
		var finals, gaps, failedRequests []map[string]any
		for _, e := range records {
			if e["type"] == "failed_request" {
				failedRequests = append(failedRequests, e)
			}
			if e["type"] == "final" {
				finals = append(finals, e)
			}
			if e["type"] == "generation_gap" {
				gaps = append(gaps, e)
			}
		}
		if len(finals) == 0 || finals[0]["start_sample"] != float64(0) || finals[0]["end_sample"] != float64(32000) {
			t.Fatal("successful first window lost", c.name, finals)
		}
		if !c.ordinary {
			if len(failedRequests) != 1 {
				t.Fatal("missing definite failed-request annotation", c.name, failedRequests)
			}
			failed := failedRequests[0]
			operation := "audio"
			if c.point == "finish" {
				operation = "finish"
			}
			if failed["acknowledged"] != false || failed["reason"] != "asr_generation_limit" || failed["operation"] != operation {
				t.Fatal("false failed-request acknowledgment", failed)
			}
			if operation == "audio" {
				a, okA := failed["start_sample"].(float64)
				b, okB := failed["end_sample"].(float64)
				if !okA || !okB || a < 32000 || b <= a || b > 64000 || failed["pcm_sha256"] != fmt.Sprintf("%x", sha256.Sum256(raw[int64(a)*2:int64(b)*2])) {
					t.Fatal("failed audio range/hash", failed)
				}
			}
		} else if len(failedRequests) != 0 {
			t.Fatal("ordinary error mislabeled generation limit", failedRequests)
		}
		success := c.policy == "continue" && !c.ordinary && !c.canceled
		if success {
			if runErr != nil || summary["status"] != "complete_with_gaps" || summary["transcription_complete"] != false || summary["text_is_final"] != false || len(gaps) != 1 {
				t.Fatal(c.name, runErr, summary, terminal.String())
			}
			expectedEnd := int64(64000)
			if c.frames < 64000 {
				expectedEnd = int64(c.frames)
			}
			gap := gaps[0]
			if gap["start_sample"] != float64(32000) || gap["end_sample"] != float64(expectedEnd) || gap["reason"] != "asr_generation_limit" || gap["planned_resume_sample"] != float64(64000) || gap["coverage"] != "transcription-error-annotation" {
				t.Fatal("incorrect failed window range", c.name, gap)
			}
			if acquired != 3 && c.frames >= 64000 || acquired != 2 && c.frames < 64000 {
				t.Fatal("unexpected recovery sessions", c.name, trace)
			}
			if len(finals) != map[bool]int{true: 2, false: 1}[c.frames >= 64000] {
				t.Fatal("wrong final count", c.name, finals)
			}
			if finals[0]["start_sample"] != float64(0) || finals[0]["end_sample"] != float64(32000) {
				t.Fatal("first successful window lost", finals)
			}
			if c.frames >= 64000 && (finals[1]["start_sample"] != float64(64000) || finals[1]["end_sample"] != float64(c.frames)) {
				t.Fatal("post-gap window coverage", finals)
			}
			// Error annotations overlap captures from the failed window by design. Check
			// only acknowledged packet bytes outside the explicitly failed range.
			type span struct{ start, end int64 }
			var spans []span
			for _, e := range records {
				if e["type"] != "packet" {
					continue
				}
				a, okA := e["packet_start_sample"].(float64)
				b, okB := e["packet_end_sample"].(float64)
				if !okA || !okB {
					t.Fatal(e)
				}
				start, end := int64(a), int64(b)
				if start >= 32000 && end <= expectedEnd {
					continue
				}
				if start < 0 || end <= start || end > int64(c.frames) || start < expectedEnd && end > 32000 {
					t.Fatal("packet straddles generation gap", e)
				}
				if e["pcm_sha256"] != fmt.Sprintf("%x", sha256.Sum256(raw[start*2:end*2])) {
					t.Fatal("non-gap PCM changed", e)
				}
				spans = append(spans, span{start, end})
			}
			sort.Slice(spans, func(i, j int) bool { return spans[i].start < spans[j].start })
			var end int64
			for _, r := range spans {
				if end == 32000 {
					end = expectedEnd
				}
				if r.start != end {
					t.Fatal("non-gap capture gap/overlap", c.name, r, end)
				}
				end = r.end
			}
			if end == 32000 {
				end = expectedEnd
			}
			if end != int64(c.frames) {
				t.Fatal("non-gap tail missing", end, c.frames)
			}
			transcript, err := os.ReadFile(filepath.Join(dir, "transcript.txt"))
			if err != nil {
				t.Fatal(err)
			}
			marker := fmt.Sprintf("[transcription gap %.2f–%.2fs: asr_generation_limit]", 2.0, float64(expectedEnd)/16000)
			if strings.Count(string(transcript), marker) != 1 || strings.Contains(string(transcript), "UNACCEPTED_FAULT_TEXT") {
				t.Fatal("gap marker/provisional output", string(transcript))
			}
		} else if c.canceled {
			if len(finals) != 1 {
				t.Fatal("canceled skip promoted failed window", finals)
			}
			if len(gaps) == 1 {
				a, _ := gaps[0]["start_sample"].(float64)
				b, _ := gaps[0]["end_sample"].(float64)
				if a != 32000 || b < 40000 || b > float64(c.frames) || gaps[0]["planned_resume_sample"] != float64(64000) {
					t.Fatal("canceled skip range", gaps)
				}
			}
			var exit *exec.ExitError
			if !errors.As(runErr, &exit) || exit.ExitCode() != 130 || summary["status"] != "canceled_with_gaps" || summary["transcription_complete"] != false || summary["text_is_final"] != false || len(gaps) != 1 || acquired != 2 {
				t.Fatal("skip interruption", runErr, summary, trace)
			}
		} else {
			if runErr == nil || summary["status"] != "failed" || len(gaps) != 0 || len(finals) != 1 || acquired != 2 {
				t.Fatal("unapproved error recovery", c.name, runErr, summary, trace)
			}
		}
		serialized, _ := json.Marshal(records)
		if bytes.Contains(serialized, []byte("UNACCEPTED_FAULT_TEXT")) {
			t.Fatal("failed output promoted", c.name)
		}
		results = append(results, map[string]any{"case": c.name, "summary": summary, "backend_trace": trace, "real_backend_drained_before_injection": true, "fault_point": c.point, "injections": injected, "finals": len(finals), "generation_gaps": gaps, "failed_requests": failedRequests, "source_pcm_sha256": fmt.Sprintf("%x", sha256.Sum256(raw[:c.frames*2]))})
	}
	manager.mu.Lock()
	active := len(manager.active)
	manager.mu.Unlock()
	if active != 0 {
		t.Fatal("generation-limit session leak", active)
	}
	loaded, err := s.getExistingModelForRunner(os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "")
	if err != nil {
		t.Fatal(err)
	}
	s.sched.expireRunner(loaded)
	for {
		s.sched.loadedMu.Lock()
		remaining := len(s.sched.loaded)
		s.sched.loadedMu.Unlock()
		if remaining == 0 {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("generation-limit unload failed")
		case <-time.After(10 * time.Millisecond):
		}
	}
	receipt := map[string]any{"passed": true, "injected_fault_annotation": true, "natural_generation_exhaustion_qualified": false, "actual_native_inference_and_drain": true, "cleanup_ack_before_next_session": true, "normal_unload": true, "results": results}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(output, "generation-limit.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
