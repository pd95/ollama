package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	pcmencoding "encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/model/audio"
	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

type asrOverlapNativeTrace struct {
	mu         sync.Mutex
	next, open int
	ids        map[string]bool
	raw        []byte
	started    time.Time
	calls      []map[string]any
	violations []string
}
type asrOverlapNativeLease struct {
	asrBackend
	trace                 *asrOverlapNativeTrace
	ordinal, seed, finish int
	mu                    sync.Mutex
	closed                bool
	seedUpdate            asrstream.Update
}

func (b *asrOverlapNativeLease) ReserveASR(ctx context.Context) (string, time.Time, error) {
	id, expiry, err := b.asrBackend.ReserveASR(ctx)
	if err == nil {
		b.trace.mu.Lock()
		if b.trace.ids[id] {
			b.trace.violations = append(b.trace.violations, "private ID reused")
		}
		b.trace.ids[id] = true
		b.trace.mu.Unlock()
	}
	return id, expiry, err
}

func (b *asrOverlapNativeLease) BeginASR(ctx context.Context, id string, cfg asrstream.Config) error {
	if cfg.Context != "" {
		return fmt.Errorf("native qualification expected empty fresh-window context")
	}
	return b.asrBackend.BeginASR(ctx, id, cfg)
}

func (b *asrOverlapNativeLease) PushASR(context.Context, string, int, []float32) (*asrstream.Update, error) {
	return nil, fmt.Errorf("fresh window unexpectedly used session Push")
}

func (b *asrOverlapNativeLease) SeedASR(ctx context.Context, id string, pcm []float32) (*asrstream.Update, error) {
	b.mu.Lock()
	b.seed++
	count := b.seed
	b.mu.Unlock()
	if count != 1 || len(pcm) < 201 || len(pcm) > 256000 {
		return nil, fmt.Errorf("invalid fresh Seed count/size")
	}
	backend, ok := b.asrBackend.(asrWindowBackend)
	if !ok {
		return nil, fmt.Errorf("native backend lacks SeedASR")
	}
	u, err := backend.SeedASR(ctx, id, pcm)
	if err == nil && u != nil {
		b.mu.Lock()
		b.seedUpdate = *u
		b.mu.Unlock()
	}
	return u, err
}

func (b *asrOverlapNativeLease) FinishASR(ctx context.Context, id string) (asrstream.Update, error) {
	b.mu.Lock()
	b.finish++
	b.mu.Unlock()
	u, err := b.asrBackend.FinishASR(ctx, id)
	b.mu.Lock()
	defer b.mu.Unlock()
	if err == nil && (b.seed != 1 || b.finish != 1 || u.Revision != 2 || !u.Final || u.Text != b.seedUpdate.Text || u.AudioEnd != b.seedUpdate.AudioEnd) {
		return u, fmt.Errorf("fresh Finish did not reuse Seed")
	}
	return u, err
}

func (b *asrOverlapNativeLease) CloseASR(ctx context.Context, id string) error {
	err := b.asrBackend.CloseASR(ctx, id)
	b.mu.Lock()
	seed, finish := b.seed, b.finish
	b.mu.Unlock()
	if err == nil {
		b.trace.mu.Lock()
		if !b.closed {
			b.closed = true
			b.trace.open--
			if seed != 1 || finish != 1 {
				b.trace.violations = append(b.trace.violations, "fresh lease lacked exactly one Seed and Finish")
			}
		}
		b.trace.mu.Unlock()
	}
	return err
}

// This native cohort measures all fresh-window requests. A failed half-second
// performance diagnostic must block cadence promotion, not loosen math gates.
func TestQwen3ASROverlapClientNative(t *testing.T) {
	if os.Getenv("OLLAMA_QWEN3_ASR_OVERLAP_CLIENT_NATIVE") != "1" {
		t.Skip("supervised native overlap client only")
	}
	t.Setenv("OLLAMA_EXPERIMENT", "asr-streaming")
	ctx, cancel := context.WithTimeout(t.Context(), 170*time.Second)
	defer cancel()
	output := os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_OUTPUT")
	var trace *asrOverlapNativeTrace
	fail := func(arguments ...any) {
		t.Helper()
		failure := map[string]any{"functional_passed": false, "error": fmt.Sprint(arguments...)}
		if trace != nil {
			trace.mu.Lock()
			failure["raw_requests"] = append([]map[string]any(nil), trace.calls...)
			failure["violations"] = append([]string(nil), trace.violations...)
			failure["open_native_leases"] = trace.open
			trace.mu.Unlock()
		}
		if data, err := json.MarshalIndent(failure, "", "  "); err == nil {
			_ = os.WriteFile(filepath.Join(output, "overlap-failure.json"), data, 0o600)
		}
		t.Fatal(arguments...)
	}

	cwd, err := os.Getwd()
	if err != nil {
		fail(err)
	}
	root := filepath.Clean(filepath.Join(cwd, "..", ".."))
	tools := filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools")
	binary := filepath.Join(output, "overlap-client")
	args := []string{"build", "-o", binary}
	for _, name := range []string{"main.go", "pcm.go", "render.go", "utterance.go", "utterance_detector.go", "generation_limit.go", "overlap.go", "overlap_text.go"} {
		args = append(args, filepath.Join(tools, "sessionclient", name))
	}
	build := exec.CommandContext(ctx, "go", args...)
	build.Dir = filepath.Join(root, "ollama")
	if log, err := build.CombinedOutput(); err != nil {
		fail(err, string(log))
	}
	var inputs []struct {
		Name, Language, Prepared string
		SHA                      string `json:"prepared_sha256"`
	}
	data, err := os.ReadFile(os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_INPUTS"))
	if err != nil {
		fail(err)
	}
	if err := json.Unmarshal(data, &inputs); err != nil {
		fail(err)
	}
	sources := map[string][]float32{}
	sourceSHA := map[string]string{}
	for _, entry := range inputs {
		if entry.Name != "corpus-00" && entry.Name != "corpus-03" {
			continue
		}
		wav, err := os.ReadFile(entry.Prepared)
		if err != nil {
			fail(err)
		}
		sum := sha256.Sum256(wav)
		if hex.EncodeToString(sum[:]) != entry.SHA {
			fail("prepared source hash mismatch")
		}
		pcm, rate, err := audio.Decode(wav)
		if err != nil || rate != 16000 || len(pcm) < 16*16000 {
			fail("short/invalid source", entry.Name, rate, err)
		}
		sources[entry.Language] = pcm
		sourceSHA[entry.Language] = entry.SHA
	}
	if len(sources) != 2 {
		fail("missing long English/German fixture")
	}
	languages := []string{"en", "de"}
	radioInput := os.Getenv("OLLAMA_QWEN3_ASR_OVERLAP_RADIO_INPUT")
	if radioInput != "" {
		data, err := os.ReadFile(radioInput)
		if err != nil {
			t.Fatal(err)
		}
		pcm, rate, err := audio.Decode(data)
		if err != nil || rate != 16000 || len(pcm) != 120*16000 {
			t.Fatal("invalid fixed120s radio fixture", err)
		}
		sources = map[string][]float32{"auto": pcm}
		sourceSHA = map[string]string{"auto": fmt.Sprintf("%x", sha256.Sum256(data))}
		languages = []string{"auto"}
	}

	trace = &asrOverlapNativeTrace{ids: map[string]bool{}}
	schedCtx, stopSched := context.WithCancel(context.Background())
	s := &Server{sched: InitScheduler(schedCtx), defaultNumCtx: 2048}
	s.sched.Run(schedCtx)
	defer func() { s.closeASRSessions(); s.sched.unloadAllRunners(); stopSched() }()
	router, err := s.GenerateRoutes()
	if err != nil {
		fail(err)
	}
	s.asrSessions.acquire = func(lease context.Context, name string) (asrBackend, error) {
		trace.mu.Lock()
		if trace.open != 0 {
			trace.violations = append(trace.violations, "successor acquired before prior native cleanup")
		}
		trace.next++
		ordinal := trace.next
		trace.open++
		trace.mu.Unlock()
		backend, err := s.acquireASRBackend(lease, name)
		if err != nil {
			trace.mu.Lock()
			trace.open--
			trace.mu.Unlock()
			return nil, err
		}
		return &asrOverlapNativeLease{asrBackend: backend, trace: trace, ordinal: ordinal}, nil
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != asrWindowPath {
			router.ServeHTTP(w, r)
			return
		}
		body, err := io.ReadAll(io.LimitReader(r.Body, 700<<10))
		_ = r.Body.Close()
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		clone := r.Clone(r.Context())
		clone.Body = io.NopCloser(bytes.NewReader(body))
		reader, err := clone.MultipartReader()
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		var raw []byte
		start := int64(0)
		for {
			part, err := reader.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				http.Error(w, err.Error(), 500)
				return
			}
			b, err := io.ReadAll(io.LimitReader(part, 600<<10))
			if err != nil {
				http.Error(w, err.Error(), 500)
				return
			}
			if part.FormName() == "file" {
				raw = b
			}
			if part.FormName() == "start_sample" {
				start, err = strconv.ParseInt(string(b), 10, 64)
				if err != nil {
					http.Error(w, "invalid start", 500)
					return
				}
			}
		}
		end := start + int64(len(raw)/2)
		trace.mu.Lock()
		source := trace.raw
		started := trace.started
		valid := len(raw)%2 == 0 && len(raw) >= 402 && len(raw) <= 512000 && start >= 0 && end <= int64(len(source)/2)
		if valid {
			valid = bytes.Equal(raw, source[start*2:end*2])
		}
		if !valid {
			trace.violations = append(trace.violations, "window PCM/global range differs from physical source")
		}
		trace.mu.Unlock()
		if !valid {
			http.Error(w, "invalid qualification window PCM", 500)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		r.ContentLength = int64(len(body))
		sent := time.Now()
		router.ServeHTTP(w, r)
		trace.mu.Lock()
		trace.calls = append(trace.calls, map[string]any{"source_start_sample": start, "source_end_sample": end, "audio_samples": len(raw) / 2, "pcm_sha256": fmt.Sprintf("%x", sha256.Sum256(raw)), "request_seconds": time.Since(sent).Seconds(), "response_lateness_seconds": time.Since(started).Seconds() - float64(end)/16000})
		trace.mu.Unlock()
	}))
	defer server.Close()
	defer cancel()
	var results []map[string]any
	performancePassed := true
	stableJoiningPassed := true
	recoveryExercised := false
	for _, language := range languages {
		source := sources[language]
		sampleCount := 20 * 16000
		if radioInput != "" {
			sampleCount = len(source)
		}
		samples := make([]float32, sampleCount)
		for i := range samples {
			samples[i] = source[i%len(source)]
		}
		raw := audio.EncodeWAV(samples, 16000)[44:]
		if radioInput != "" {
			// Decode normalizes PCM16 by32768; invert it exactly for this
			// capture instead of applying EncodeWAV's32767 rescale.
			raw = make([]byte, len(samples)*2)
			for i, sample := range samples {
				pcmencoding.LittleEndian.PutUint16(raw[2*i:], uint16(int16(math.Round(float64(sample)*32768))))
			}
		}
		if radioInput != "" && fmt.Sprintf("%x", sha256.Sum256(raw)) != os.Getenv("OLLAMA_QWEN3_ASR_OVERLAP_RADIO_PCM_SHA256") {
			fail("radio PCM bytes differ from saved WAV data")
		}
		trace.mu.Lock()
		trace.raw = raw
		trace.calls = nil
		trace.started = time.Now()
		trace.mu.Unlock()
		dir := filepath.Join(output, "paced-"+language)
		cmd := exec.CommandContext(ctx, binary, "-mode", "pcm", "-url", server.URL, "-model", os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "-language", language, "-output", dir, "-update-interval-ms", "500", "-commitment-mode", "overlap", "-line-seconds", "10", "-overlap-seconds", "4", "-on-generation-limit", "continue")
		if radioInput != "" {
			cmd.Args = append(cmd.Args, "-on-overlap-ambiguity", "continue")
		}
		writer, err := cmd.StdinPipe()
		if err != nil {
			fail(err)
		}
		var terminal bytes.Buffer
		cmd.Stdout = &terminal
		cmd.Stderr = &terminal
		if err := cmd.Start(); err != nil {
			fail(err)
		}
		done := make(chan error, 1)
		go func() { done <- cmd.Wait() }()
		sent := make(chan error, 1)
		started := time.Now()
		trace.mu.Lock()
		trace.started = started
		trace.mu.Unlock()
		stopWriter := context.AfterFunc(ctx, func() { _ = writer.Close() })
		go func() {
			defer writer.Close()
			for i := range len(raw) / 16000 {
				timer := time.NewTimer(max(0, time.Until(started.Add(time.Duration(i+1)*500*time.Millisecond))))
				select {
				case <-ctx.Done():
					timer.Stop()
					sent <- ctx.Err()
					return
				case <-timer.C:
				}
				if _, err := writer.Write(raw[i*16000 : (i+1)*16000]); err != nil {
					sent <- err
					return
				}
			}
			sent <- nil
		}()
		select {
		case err := <-done:
			if err != nil {
				fail("native overlap client", language, err, terminal.String())
			}
		case <-ctx.Done():
			fail(ctx.Err())
		}
		stopWriter()
		if err := <-sent; err != nil {
			fail(err)
		}
		summaryBytes, err := os.ReadFile(filepath.Join(dir, "summary.json"))
		if err != nil {
			fail(err)
		}
		var summary map[string]any
		if err := json.Unmarshal(summaryBytes, &summary); err != nil {
			fail(err)
		}
		if summary["status"] != "complete" && summary["status"] != "complete_with_gaps" {
			fail("overlap terminal status", summary)
		}
		events, err := os.ReadFile(filepath.Join(dir, "events.ndjson"))
		if err != nil {
			fail(err)
		}
		committed := ""
		var clientRequests []float64
		maxRetained := 0
		captureEnd := int64(0)
		commitCount, checkpoints := 0, 0
		slideCount := 0
		ambiguityCount := 0
		var ambiguityRecords []map[string]any
		var latestWindow map[string]any
		for _, line := range bytes.Split(events, []byte{'\n'}) {
			if len(line) == 0 {
				continue
			}
			var e map[string]any
			if err := json.Unmarshal(line, &e); err != nil {
				fail(err)
			}
			if retained, ok := e["retained_audio_samples"].(float64); ok {
				if retained < 0 || retained > 256000 || math.IsNaN(retained) || math.IsInf(retained, 0) {
					fail("client audio guard exceeded", e)
				}
				maxRetained = max(maxRetained, int(retained))
			}
			switch e["type"] {
			case "slide":
				slideCount++
			case "window":
				latestWindow = e
				request, ok := e["request_seconds"].(float64)
				if !ok || request < 0 || math.IsNaN(request) || math.IsInf(request, 0) || e["fresh"] != true || e["context_replay"] != true {
					fail("invalid fresh-window timing", e)
				}
				if _, ok := e["retained_audio_samples"].(float64); !ok {
					fail("missing retained audio bound", e)
				}
				clientRequests = append(clientRequests, request)
			case "overlap_gap", "generation_gap":
				annotation, ok := e["annotation_text"].(string)
				if !ok || !strings.Contains(annotation, "[recognition gap processing window ") || e["committed_text_retained"] != true {
					fail("invalid annotation event", e)
				}
				if e["type"] == "overlap_gap" {
					ambiguityCount++
					ambiguityRecords = append(ambiguityRecords, e)
					if e["uncertain_text"] != latestWindow["text"] || e["uncertain_window_start_sample"] != latestWindow["source_start_sample"] || e["uncertain_window_end_sample"] != latestWindow["source_end_sample"] || e["may_overlap_committed_text"] != true {
						fail("uncertain evidence did not retain acknowledged window", e)
					}
				}
				committed += annotation
			case "commit":
				text, ok := e["text"].(string)
				if !ok || !strings.HasPrefix(text, committed) {
					fail("committed prefix retreated", e, committed)
				}
				committed = text
				commitCount++
			case "capture":
				a, okA := e["start_sample"].(float64)
				b, okB := e["end_sample"].(float64)
				if !okA || !okB || int64(a) != captureEnd || b <= a || b > float64(len(samples)) || e["coverage"] != "physical-source" || e["pcm_sha256"] != fmt.Sprintf("%x", sha256.Sum256(raw[int64(a)*2:int64(b)*2])) {
					fail("physical capture duplication/loss", e, captureEnd)
				}
				captureEnd = int64(b)
			case "display_checkpoint":
				if e["source_end_sample"] == float64(160000) {
					checkpoints++
				}
				if n, ok := e["committed_bytes"].(float64); !ok || n > float64(len(committed)) {
					fail("display committed unagreed draft", e)
				}
				if _, ok := e["carried_draft"].(string); !ok {
					fail("display did not retain draft metadata", e)
				}
			}
		}
		if captureEnd != int64(len(samples)) || commitCount == 0 || checkpoints != 1 {
			fail("missing overlap source/commit/display evidence", language, captureEnd, commitCount, checkpoints)
		}
		trace.mu.Lock()
		calls := append([]map[string]any(nil), trace.calls...)
		violations := append([]string(nil), trace.violations...)
		open := trace.open
		trace.mu.Unlock()
		if len(violations) != 0 || open != 0 || len(calls) < 30 {
			fail("fresh window native lifecycle", violations, open, len(calls))
		}
		end := int64(0)
		replayed := int64(0)
		var latencies, lateness []float64
		shiftedWindowObserved := false
		slid := slideCount > 0
		for _, call := range calls {
			a, b := call["source_start_sample"].(int64), call["source_end_sample"].(int64)
			if a > end || b < end || b-a > 256000 {
				fail("window union gap/backward/oversize", call, end)
			}
			end = b
			replayed += b - a
			shiftedWindowObserved = shiftedWindowObserved || a > 0
			lateness = append(lateness, call["response_lateness_seconds"].(float64))
		}
		if end != int64(len(samples)) || (!slid && (radioInput == "" || ambiguityCount == 0)) || replayed <= end {
			fail("bounded overlap window coverage", end, slid, replayed)
		}
		if len(clientRequests) != len(calls) {
			fail("missing client roundtrip timing", len(clientRequests), len(calls))
		}
		if ambiguityCount > 0 {
			evidence, err := os.ReadFile(filepath.Join(dir, "uncertain.ndjson"))
			if err != nil || summary["transcription_complete"] != false {
				fail("uncertainty journal missing", err)
			}
			records := 0
			for _, line := range bytes.Split(evidence, []byte{'\n'}) {
				if len(bytes.TrimSpace(line)) == 0 {
					continue
				}
				var record map[string]any
				if err := json.Unmarshal(line, &record); err != nil {
					fail("malformed uncertainty journal", err)
				}
				if records >= len(ambiguityRecords) || !reflect.DeepEqual(record, ambiguityRecords[records]) {
					fail("uncertainty journal differs from validated event", records)
				}
				records++
			}
			if records != ambiguityCount {
				fail("uncertainty journal lost windows", records, ambiguityCount)
			}
			recoveryExercised = true
		}

		stableJoiningPassed = stableJoiningPassed && slid && summary["status"] == "complete"
		latencies = clientRequests
		sorted := append([]float64(nil), latencies...)
		sort.Float64s(sorted)
		p95 := sorted[int(math.Ceil(.95*float64(len(sorted))))-1]
		mean := func(values []float64) float64 {
			var total float64
			for _, v := range values {
				total += v
			}
			return total / float64(len(values))
		}
		growth := mean(lateness[len(lateness)-5:]) - mean(lateness[:5])
		cadencePass := p95 < .5 && growth <= 1
		performancePassed = performancePassed && cadencePass
		results = append(results, map[string]any{"language": language, "summary": summary, "safe_shift_observed": slid, "accepted_slides": slideCount, "shifted_source_windows_observed": shiftedWindowObserved, "ambiguity_gaps": ambiguityCount, "stable_overlap_joining_passed": slid && summary["status"] == "complete", "source_prepared_sha256": sourceSHA[language], "source_repeated_to_twenty_seconds": len(source) < len(samples), "source_samples": len(samples), "physical_capture_verified": true, "committed_prefix_monotonic": true, "maximum_retained_audio_samples": maxRetained, "display_checkpoint_10s": true, "actual_fresh_lease_per_window": true, "raw_requests": calls, "replayed_samples_not_physical_capture": replayed, "all_requests_p95_seconds": p95, "first5_last5_response_lateness_growth_seconds": growth, "half_second_performance_passed": cadencePass, "recognition_accuracy_qualified": false})
	}
	loaded, err := s.getExistingModelForRunner(os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "")
	if err != nil {
		fail(err)
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
			fail("overlap unload failed")
		case <-time.After(10 * time.Millisecond):
		}
	}
	receipt := map[string]any{"functional_passed": true, "half_second_performance_passed": performancePassed, "performance_diagnostic_includes_all_requests": true, "cadence_promotion_allowed": performancePassed && stableJoiningPassed, "stable_overlap_joining_passed": stableJoiningPassed, "radio_recovery_qualified": radioInput != "" && recoveryExercised, "ambiguity_recovery_exercised": recoveryExercised, "normal_unload": true, "forced_aligner_used": false, "word_timestamps_qualified": false, "results": results}
	data, err = json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		fail(err)
	}
	if err := os.WriteFile(filepath.Join(output, "overlap-client.json"), data, 0o600); err != nil {
		fail(err)
	}
}
