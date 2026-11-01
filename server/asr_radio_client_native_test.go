package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/model/audio"
)

// This fixed network-source check uses only its own loopback WAV endpoint. It
// qualifies pacing, rolling and cleanup, not recognition or external radio URLs.
func TestQwen3ASRRadioClientNative(t *testing.T) {
	if os.Getenv("OLLAMA_QWEN3_ASR_RADIO_CLIENT_NATIVE") != "1" {
		t.Skip("supervised loopback radio only")
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
		t.Fatal("unsupported radio cadence", interval)
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Clean(filepath.Join(cwd, "..", ".."))
	tools := filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools")
	// Continuous deterministic above-threshold PCM prevents pause endpoints from
	// hiding the managed network input's default ten-second duration cap.
	samples := make([]float32, 12*16000+17)
	for i := range samples {
		if i%2 == 0 {
			samples[i] = .2
		} else {
			samples[i] = -.2
		}
	}
	wav := audio.EncodeWAV(samples, 16000)
	raw := wav[44:]
	network := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/radio.wav" || (r.Method != "GET" && r.Method != "HEAD") {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "audio/wav")
		http.ServeContent(w, r, "radio.wav", time.Time{}, bytes.NewReader(wav))
	}))
	defer network.Close()
	schedCtx, stopSched := context.WithCancel(context.Background())
	s := &Server{sched: InitScheduler(schedCtx), defaultNumCtx: 2048}
	s.sched.Run(schedCtx)
	defer func() { s.closeASRSessions(); s.sched.unloadAllRunners(); stopSched() }()
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	native := httptest.NewServer(router)
	defer native.Close()
	// Cancel owned commands before deferred HTTP shutdown on an early assertion.
	defer cancel()
	type record struct {
		Type        string  `json:"type"`
		Start       int64   `json:"start_sample"`
		End         int64   `json:"end_sample"`
		PacketStart int64   `json:"packet_start_sample"`
		PacketEnd   int64   `json:"packet_end_sample"`
		SHA         string  `json:"pcm_sha256"`
		Reason      string  `json:"reason"`
		Elapsed     float64 `json:"elapsed_seconds"`
	}
	readEvents := func(dir string) []record {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(dir, "run/events.ndjson"))
		if os.IsNotExist(err) {
			return nil
		}
		if err != nil {
			t.Fatal(err)
		}
		lines := bytes.Split(data, []byte{'\n'})
		var records []record
		for _, line := range lines[:len(lines)-1] {
			if len(line) == 0 {
				continue
			}
			var e record
			if err := json.Unmarshal(line, &e); err != nil {
				t.Fatal(err)
			}
			records = append(records, e)
		}
		return records
	}
	readSummary := func(dir string) map[string]any {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(dir, "run/summary.json"))
		if err != nil {
			t.Fatal(err)
		}
		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	command := func(dir string) *exec.Cmd {
		return exec.CommandContext(ctx, "python3", filepath.Join(tools, "listen.py"), "--stream-url", network.URL+"/radio.wav", "--no-playback", "--url", native.URL, "--model", os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "--language", "en", "--output", dir, "--timeout", "90", "--update-interval-ms", interval)
	}
	dir := filepath.Join(output, "paced")
	started := time.Now()
	cmd := command(dir)
	log, err := cmd.CombinedOutput()
	elapsed := time.Since(started).Seconds()
	if err != nil {
		t.Fatal("loopback radio", err, string(log))
	}
	duration := float64(len(samples)) / 16000
	summary := readSummary(dir)
	if summary["status"] != "complete" || summary["cleanup"] != "acknowledged" || summary["segments"] != float64(2) {
		t.Fatal("default network rolling", summary)
	}
	records := readEvents(dir)
	var ranges []record
	var finalEnds []int64
	reasons := map[string]int{}
	var sent, idle int64
	var firstPacket, lastPacket *record
	for _, e := range records {
		switch e.Type {
		case "packet":
			if math.IsNaN(e.Elapsed) || math.IsInf(e.Elapsed, 0) || e.Elapsed < 0 {
				t.Fatal("invalid packet elapsed time", e)
			}
			packet := e
			if firstPacket == nil {
				firstPacket = &packet
			}
			lastPacket = &packet
			if e.PacketStart < 0 || e.PacketEnd <= e.PacketStart || e.PacketEnd > int64(len(samples)) {
				t.Fatal("invalid PCM range", e)
			}
			expected := fmt.Sprintf("%x", sha256.Sum256(raw[2*e.PacketStart:2*e.PacketEnd]))
			if e.SHA != expected {
				t.Fatal("network PCM changed", e)
			}
			e.Start, e.End = e.PacketStart, e.PacketEnd
			ranges = append(ranges, e)
			sent += e.End - e.Start
		case "idle_gap":
			ranges = append(ranges, e)
			idle += e.End - e.Start
		case "final":
			finalEnds = append(finalEnds, e.End)
		case "endpoint":
			reasons[e.Reason]++
		}
	}
	if firstPacket == nil || lastPacket == nil {
		t.Fatal("missing paced packet events")
	}
	packetSourceSpan := float64(lastPacket.PacketEnd-firstPacket.PacketEnd) / 16000
	packetElapsedSpan := lastPacket.Elapsed - firstPacket.Elapsed
	// Client-relative packet timings cancel build, process startup and first
	// model acquisition. A slow build cannot make an unpaced source pass.
	if packetSourceSpan < 10 || packetElapsedSpan < packetSourceSpan-1 {
		t.Fatal("network packet span ran faster than real time", packetElapsedSpan, packetSourceSpan)
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].Start < ranges[j].Start })
	var end int64
	for _, r := range ranges {
		if r.Start != end || r.End <= r.Start {
			t.Fatal("network sample gap/overlap", r, end)
		}
		end = r.End
	}
	if end != int64(len(samples)) || sent != int64(len(samples)) || idle != 0 {
		t.Fatal("network sample coverage", end, sent, idle)
	}
	if len(finalEnds) != 2 || finalEnds[0] != 160000 || finalEnds[1] != int64(len(samples)) || reasons["max_duration"] != 1 || reasons["eof"] != 1 {
		t.Fatal("default ten-second cap/tail", finalEnds, reasons)
	}
	// Signal only the Python supervisor. It must stop its own Go/ffmpeg children
	// and receive acknowledged DELETE without the test signaling those children.
	cancelDir := filepath.Join(output, "interrupted")
	interrupt := command(cancelDir)
	var interruptedLog bytes.Buffer
	interrupt.Stdout = &interruptedLog
	interrupt.Stderr = &interruptedLog
	if err := interrupt.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- interrupt.Wait() }()
	t.Cleanup(func() { _ = interrupt.Process.Kill() })
	until := time.NewTimer(20 * time.Second)
	defer until.Stop()
	tick := time.NewTicker(20 * time.Millisecond)
	defer tick.Stop()
	ready := false
	for !ready {
		for _, e := range readEvents(cancelDir) {
			if e.Type == "packet" {
				ready = true
				break
			}
		}
		if ready {
			break
		}
		select {
		case err := <-done:
			t.Fatal("radio ended before cancellation", err, interruptedLog.String())
		case <-until.C:
			t.Fatal("no radio packet before cancellation")
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-tick.C:
		}
	}
	if err := interrupt.Process.Signal(os.Interrupt); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		var exit *exec.ExitError
		if !errors.As(err, &exit) || exit.ExitCode() != 130 {
			t.Fatal("supervisor interrupt status", err, interruptedLog.String())
		}
	case <-ctx.Done():
		t.Fatal("radio interrupt did not drain", ctx.Err())
	}
	interrupted := readSummary(cancelDir)
	if interrupted["status"] != "canceled" || interrupted["cleanup"] != "acknowledged" || interrupted["text_is_final"] != false {
		t.Fatal("radio cancellation cleanup", interrupted)
	}
	terminal := interruptedLog.String()
	for _, bad := range []string{"Post \"http", "context canceled", "Broken pipe", "BrokenPipeError", "Traceback"} {
		if strings.Contains(terminal, bad) {
			t.Fatal("noisy radio interruption", bad, terminal)
		}
	}
	if !strings.Contains(terminal, "Stopped.") {
		t.Fatal("missing quiet interrupt acknowledgment", terminal)
	}
	s.asrSessions.mu.Lock()
	active := len(s.asrSessions.active)
	s.asrSessions.mu.Unlock()
	if active != 0 {
		t.Fatal("radio session leak", active)
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
			t.Fatal("radio unload failed")
		case <-time.After(10 * time.Millisecond):
		}
	}
	receipt := map[string]any{"passed": true, "owned_loopback_source": true, "external_network_requested": false, "source_wav_sha256": fmt.Sprintf("%x", sha256.Sum256(wav)), "source_samples": len(samples), "source_seconds": duration, "wall_seconds": elapsed, "packet_source_span_seconds": packetSourceSpan, "packet_elapsed_span_seconds": packetElapsedSpan, "default_network_segment_seconds": 10, "final_ends": finalEnds, "endpoint_reasons": reasons, "sent_samples": sent, "idle_samples": idle, "exact_sample_coverage": true, "packet_pcm_hashes_verified": true, "paced_summary": summary, "interrupted_summary": interrupted, "only_parent_python_signaled": true, "interrupt_exit_status": 130, "quiet_operator_interrupt": true, "normal_unload": true, "accuracy_qualified": false, "update_interval_ms": interval}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(output, "radio-client.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
