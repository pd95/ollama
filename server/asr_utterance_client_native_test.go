package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/model/audio"
)

// TestQwen3ASRUtteranceClientNative exercises client-only endpointing against the
// real native runner. Recognition accuracy and semantic utterance boundaries are
// deliberately not inferred from energy classification.
func TestQwen3ASRUtteranceClientNative(t *testing.T) {
	if os.Getenv("OLLAMA_QWEN3_ASR_UTTERANCE_CLIENT_NATIVE") != "1" {
		t.Skip("supervised utterance client only")
	}
	t.Setenv("OLLAMA_EXPERIMENT", "asr-streaming")
	ctx, cancel := context.WithTimeout(t.Context(), 160*time.Second)
	defer cancel()
	output := os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_OUTPUT")
	interval := os.Getenv("OLLAMA_QWEN3_ASR_UPDATE_INTERVAL_MS")
	if interval == "" {
		interval = "1000"
	}
	if interval != "500" && interval != "1000" && interval != "2000" {
		t.Fatal("unsupported utterance qualification cadence", interval)
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Clean(filepath.Join(cwd, "..", ".."))
	tools := filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools")
	binary := filepath.Join(output, "utterance-client")
	names := []string{"main.go", "pcm.go", "render.go", "utterance.go", "utterance_detector.go", "generation_limit.go", "overlap.go", "overlap_text.go"}
	args := []string{"build", "-o", binary}
	for _, name := range names {
		args = append(args, filepath.Join(tools, "sessionclient", name))
	}
	build := exec.CommandContext(ctx, "go", args...)
	build.Dir = filepath.Join(root, "ollama")
	if log, err := build.CombinedOutput(); err != nil {
		t.Fatal(err, string(log))
	}
	schedCtx, stopSched := context.WithCancel(context.Background())
	s := &Server{sched: InitScheduler(schedCtx), defaultNumCtx: 2048}
	s.sched.Run(schedCtx)
	defer func() { s.closeASRSessions(); s.sched.unloadAllRunners(); stopSched() }()
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(router)
	defer server.Close()
	var inputs []struct {
		Name     string `json:"name"`
		Prepared string `json:"prepared"`
		SHA      string `json:"prepared_sha256"`
	}
	b, err := os.ReadFile(os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_INPUTS"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &inputs); err != nil {
		t.Fatal(err)
	}
	sources := map[string][]byte{}
	sourceHashes := map[string]string{}
	for _, input := range inputs {
		if input.Name != "corpus-00" && input.Name != "corpus-02" {
			continue
		}
		wav, err := os.ReadFile(input.Prepared)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(wav)
		if hex.EncodeToString(sum[:]) != input.SHA {
			t.Fatal("source hash mismatch", input.Name)
		}
		samples, rate, err := audio.Decode(wav)
		if err != nil || rate != 16000 || len(samples) < 32000 {
			t.Fatal("source decode", input.Name, rate, err)
		}
		// Preserve original files. Normalize only a copied two-second crop so this
		// fixed detector/lifecycle check is not hostage to recording gain.
		crop := append([]float32(nil), samples[:32000]...)
		peak := float32(0)
		for _, v := range crop {
			peak = max(peak, float32(math.Abs(float64(v))))
		}
		if peak == 0 {
			t.Fatal("silent speech fixture", input.Name)
		}
		for i := range crop {
			crop[i] *= .8 / peak
		}
		sources[input.Name] = audio.EncodeWAV(crop, 16000)[44:]
		sourceHashes[input.Name] = input.SHA
	}
	if len(sources) != 2 {
		t.Fatal("missing EN/DE fixtures")
	}
	type running struct {
		cmd    *exec.Cmd
		writer *io.PipeWriter
		reader *io.PipeReader
		done   chan error
		log    *bytes.Buffer
		dir    string
	}
	start := func(name, language string, extra ...string) *running {
		t.Helper()
		dir := filepath.Join(output, name)
		args := []string{"-mode", "pcm", "-url", server.URL, "-model", os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "-language", language, "-output", dir, "-utterance-detection", "energy", "-silence-ms", "1200", "-speech-threshold-db", "-40", "-update-interval-ms", interval}
		cmd := exec.CommandContext(ctx, binary, append(args, extra...)...)
		reader, writer := io.Pipe()
		cmd.Stdin = reader
		log := new(bytes.Buffer)
		cmd.Stdout = log
		cmd.Stderr = log
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		r := &running{cmd: cmd, writer: writer, reader: reader, done: make(chan error, 1), log: log, dir: dir}
		go func() { r.done <- cmd.Wait() }()
		t.Cleanup(func() {
			_ = writer.Close()
			_ = reader.Close()
			_ = cmd.Process.Kill()
		})
		return r
	}
	records := func(r *running) []map[string]any {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(r.dir, "events.ndjson"))
		if os.IsNotExist(err) {
			return nil
		}
		if err != nil {
			t.Fatal(err)
		}
		var result []map[string]any
		lines := bytes.Split(data, []byte{'\n'})
		for _, line := range lines[:len(lines)-1] {
			if len(line) == 0 {
				continue
			}
			var e map[string]any
			if err := json.Unmarshal(line, &e); err != nil {
				t.Fatal(err)
			}
			result = append(result, e)
		}
		return result
	}
	count := func(r *running, kind string) int {
		n := 0
		for _, e := range records(r) {
			if e["type"] == kind {
				n++
			}
		}
		return n
	}
	waitFor := func(r *running, kind string, minimum int) {
		t.Helper()
		deadline := time.NewTimer(20 * time.Second)
		defer deadline.Stop()
		tick := time.NewTicker(20 * time.Millisecond)
		defer tick.Stop()
		for {
			if count(r, kind) >= minimum {
				return
			}
			select {
			case err := <-r.done:
				t.Fatal("client exited before event", kind, err, r.log.String())
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			case <-deadline.C:
				t.Fatal("event deadline", kind, records(r))
			case <-tick.C:
			}
		}
	}
	write := func(r *running, data []byte) {
		t.Helper()
		result := make(chan error, 1)
		go func() {
			for len(data) > 0 {
				n := min(640, len(data))
				if _, err := r.writer.Write(data[:n]); err != nil {
					result <- err
					return
				}
				data = data[n:]
			}
			result <- nil
		}()
		select {
		case err := <-result:
			if err != nil {
				t.Fatal(err)
			}
		case <-ctx.Done():
			_ = r.writer.Close()
			_ = r.reader.Close()
			t.Fatal(ctx.Err())
		}
	}
	finish := func(r *running, want string) map[string]any {
		t.Helper()
		if want == "canceled" {
			// Observe the child's cancellation artifact before EOF. Closing the
			// parent io.Pipe afterward also releases exec's stdin-copy goroutine.
			until := time.NewTimer(20 * time.Second)
			defer until.Stop()
			tick := time.NewTicker(20 * time.Millisecond)
			defer tick.Stop()
		canceled:
			for {
				b, err := os.ReadFile(filepath.Join(r.dir, "summary.json"))
				var summary map[string]any
				if err == nil && json.Unmarshal(b, &summary) == nil && summary["status"] == "canceled" {
					break canceled
				}
				select {
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				case <-until.C:
					t.Fatal("cancellation summary deadline")
				case <-tick.C:
				}
			}
		}
		_ = r.writer.Close()
		select {
		case err := <-r.done:
			if (want == "complete") != (err == nil) {
				t.Fatal("client exit", want, err, r.log.String())
			}
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
		_ = r.writer.Close()
		_ = r.reader.Close()
		data, err := os.ReadFile(filepath.Join(r.dir, "summary.json"))
		if err != nil {
			t.Fatal(err, r.log.String())
		}
		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatal(err)
		}
		if result["status"] != want {
			t.Fatal(result)
		}
		return result
	}
	verifyCoverage := func(r *running, input []byte) map[string]any {
		t.Helper()
		type sampleRange struct {
			start, end int64
			kind       string
		}
		var ranges []sampleRange
		var packetSamples, idleSamples int64
		for _, e := range records(r) {
			kind, _ := e["type"].(string)
			if kind != "packet" && kind != "idle_gap" {
				continue
			}
			startKey, endKey := "start_sample", "end_sample"
			if kind == "packet" {
				startKey, endKey = "packet_start_sample", "packet_end_sample"
			}
			a, okA := e[startKey].(float64)
			b, okB := e[endKey].(float64)
			if !okA || !okB || a != math.Trunc(a) || b != math.Trunc(b) || a < 0 || b <= a || b > float64(len(input)/2) {
				t.Fatal("invalid global sample range", e)
			}
			start, end := int64(a), int64(b)
			ranges = append(ranges, sampleRange{start, end, kind})
			if kind == "packet" {
				expected := fmt.Sprintf("%x", sha256.Sum256(input[start*2:end*2]))
				if e["pcm_sha256"] != expected {
					t.Fatal("sent PCM differs from original sample range", e)
				}
				packetSamples += end - start
			} else {
				idleSamples += end - start
			}
		}
		sort.Slice(ranges, func(i, j int) bool { return ranges[i].start < ranges[j].start })
		var end int64
		for _, r := range ranges {
			if r.start != end {
				t.Fatal("sample gap/overlap", r, end)
			}
			end = r.end
		}
		if end != int64(len(input)/2) {
			t.Fatal("incomplete sample coverage", end, len(input)/2)
		}
		return map[string]any{"input_samples": end, "sent_samples": packetSamples, "skipped_idle_samples": idleSamples, "sent_pcm_ranges_sha256_verified": true, "sample_coverage_exact": true}
	}

	endpointReasons := func(r *running) map[string]int {
		result := map[string]int{}
		for _, e := range records(r) {
			if e["type"] == "endpoint" {
				reason, ok := e["reason"].(string)
				if !ok {
					t.Fatal(e)
				}
				result[reason]++
			}
		}
		return result
	}
	results := map[string]any{}
	for _, fixture := range []struct{ name, language string }{{"corpus-00", "de"}, {"corpus-02", "en"}} {
		speech := sources[fixture.name]
		silence := make([]byte, 64000)
		input := append(append(append([]byte{}, speech...), silence...), speech...)
		r := start("long-pause-"+fixture.language, fixture.language)
		write(r, input[:len(speech)+len(silence)])
		waitFor(r, "final", 1) // Input is still open: EOF cannot cause this endpoint.
		write(r, speech)
		result := finish(r, "complete")
		if count(r, "final") < 2 || result["cleanup"] != "acknowledged" {
			t.Fatal("missing silence/EOF segments", result)
		}
		reasons := endpointReasons(r)
		if reasons["pause"] < 1 || reasons["eof"] < 1 {
			t.Fatal("missing pause/EOF causes", reasons)
		}
		results[fixture.language] = map[string]any{"summary": result, "final_before_eof": true, "source_sha256": sourceHashes[fixture.name], "input_pcm_sha256": fmt.Sprintf("%x", sha256.Sum256(input)), "coverage": verifyCoverage(r, input), "endpoint_reasons": reasons}
	}
	// Deterministic above-threshold frames isolate the detector from real pauses.
	// The model may transcribe/hallucinate this control; no accuracy claim follows.
	tone := make([]float32, 16000)
	for i := range tone {
		if i%2 == 0 {
			tone[i] = .2
		} else {
			tone[i] = -.2
		}
	}
	speech := audio.EncodeWAV(tone, 16000)[44:]
	short := start("short-pause", "en")
	write(short, speech)
	write(short, make([]byte, 16000))
	waitFor(short, "packet", 1)
	if count(short, "final") != 0 {
		t.Fatal("500ms pause ended utterance")
	}
	write(short, speech)
	shortResult := finish(short, "complete")
	if endpointReasons(short)["pause"] != 0 || endpointReasons(short)["eof"] != 1 {
		t.Fatal("short-pause endpoint causes", endpointReasons(short))
	}
	if count(short, "final") != 1 {
		t.Fatal("short pause split utterance")
	}
	shortInput := append(append(append([]byte{}, speech...), make([]byte, 16000)...), speech...)
	results["short_pause"] = map[string]any{"summary": shortResult, "coverage": verifyCoverage(short, shortInput)}
	forcedInput := make([]byte, (6*16000+17)*2)
	for i := range forcedInput {
		forcedInput[i] = speech[i%len(speech)]
	}
	forced := start("forced-cap", "en", "-segment-seconds", "2")
	write(forced, forcedInput)
	forcedResult := finish(forced, "complete")
	forcedReasons := endpointReasons(forced)
	if forcedReasons["max_duration"] != 3 || forcedReasons["eof"] != 1 || count(forced, "final") != 4 {
		t.Fatal("forced-cap/tail endpoint counts", forcedReasons, forcedResult)
	}
	results["forced_cap"] = map[string]any{"summary": forcedResult, "endpoint_reasons": forcedReasons, "coverage": verifyCoverage(forced, forcedInput)}
	silent := start("silence", "auto")
	write(silent, make([]byte, 96000))
	silentResult := finish(silent, "complete")
	if count(silent, "created") != 0 || count(silent, "final") != 0 {
		t.Fatal("pure silence opened inference")
	}
	if silentResult["cleanup"] != "not-needed" {
		t.Fatal(silentResult)
	}
	results["silence"] = map[string]any{"summary": silentResult, "coverage": verifyCoverage(silent, make([]byte, 96000))}
	interrupted := start("interrupted", "en")
	write(interrupted, speech)
	waitFor(interrupted, "created", 1)
	if err := interrupted.cmd.Process.Signal(os.Interrupt); err != nil {
		t.Fatal(err)
	}
	interruptedResult := finish(interrupted, "canceled")
	if interruptedResult["cleanup"] != "acknowledged" {
		t.Fatal("unacknowledged cancellation", interruptedResult)
	}
	results["interrupted"] = interruptedResult
	// Qualify the actual ffmpeg/listener forwarding path without playback.
	pipelineInput := append(append(append([]byte{}, sources["corpus-00"]...), make([]byte, 64000)...), sources["corpus-00"]...)
	wav := append(audio.EncodeWAV(make([]float32, len(pipelineInput)/2), 16000)[:44], pipelineInput...)
	wavPath := filepath.Join(output, "pipeline-input.wav")
	if err := os.WriteFile(wavPath, wav, 0o600); err != nil {
		t.Fatal(err)
	}
	pipelineDir := filepath.Join(output, "pipeline")
	pipeline := exec.CommandContext(ctx, "python3", filepath.Join(tools, "listen.py"), wavPath, "--no-playback", "--url", server.URL, "--model", os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "--language", "de", "--output", pipelineDir, "--timeout", "90", "--utterance-detection", "energy", "--silence-ms", "1200", "--speech-threshold-db", "-40", "--update-interval-ms", interval)
	if log, err := pipeline.CombinedOutput(); err != nil {
		t.Fatal("utterance ffmpeg pipeline", err, string(log))
	}
	pipelineRun := &running{dir: filepath.Join(pipelineDir, "run")}
	data, err := os.ReadFile(filepath.Join(pipelineRun.dir, "summary.json"))
	if err != nil {
		t.Fatal(err)
	}
	var pipelineResult map[string]any
	if err := json.Unmarshal(data, &pipelineResult); err != nil {
		t.Fatal(err)
	}
	if pipelineResult["status"] != "complete" || pipelineResult["cleanup"] != "acknowledged" || count(pipelineRun, "final") < 2 {
		t.Fatal("pipeline endpoints", pipelineResult)
	}
	results["ffmpeg_pipeline"] = map[string]any{"summary": pipelineResult, "coverage": verifyCoverage(pipelineRun, pipelineInput), "endpoint_reasons": endpointReasons(pipelineRun)}
	s.asrSessions.mu.Lock()
	active := len(s.asrSessions.active)
	s.asrSessions.mu.Unlock()
	if active != 0 {
		t.Fatal("session leak", active)
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
			t.Fatal("unload failed")
		case <-time.After(10 * time.Millisecond):
		}
	}
	receipt := map[string]any{"passed": true, "client_only_endpointing": true, "accuracy_qualified": false, "boundary_accuracy_qualified": false, "final_before_eof": true, "normal_unload": true, "results": results, "update_interval_ms": interval}
	data, err = json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(output, "utterance-client.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
