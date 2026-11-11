package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/model/audio"
)

// Only digital silence is sent to the real CoreAudio player. Timestamp checks
// qualify delivery to the player pipe, not acoustic timing at the output device.
func TestQwen3ASRPlaybackDelayNative(t *testing.T) {
	if os.Getenv("OLLAMA_QWEN3_ASR_PLAYBACK_DELAY_NATIVE") != "1" {
		t.Skip("supervised silent playback delay only")
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
		t.Fatal("unsupported playback delay cadence", interval)
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Clean(filepath.Join(cwd, "..", ".."))
	tools := filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools")
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
	defer cancel()
	shortPath := filepath.Join(output, "silence-1s.wav")
	longPath := filepath.Join(output, "silence-3s.wav")
	for path, n := range map[string]int{shortPath: 16000, longPath: 48000} {
		if err := os.WriteFile(path, audio.EncodeWAV(make([]float32, n), 16000), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	readJSON := func(path string) map[string]any {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	number := func(record map[string]any, key string) float64 {
		t.Helper()
		n, ok := record[key].(float64)
		if !ok || math.IsNaN(n) || math.IsInf(n, 0) || n < 0 {
			t.Fatal("invalid timing/byte field", key, record)
		}
		return n
	}
	command := func(input, dir string, extra ...string) *exec.Cmd {
		args := []string{filepath.Join(tools, "listen.py"), input, "--url", server.URL, "--model", os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "--language", "auto", "--output", dir, "--timeout", "90", "--update-interval-ms", interval, "--utterance-detection", "off"}
		return exec.CommandContext(ctx, "python3", append(args, extra...)...)
	}
	results := map[string]any{}
	for _, c := range []struct {
		name, input string
		samples     int
		delay       float64
		noPlayback  bool
		extra       []string
	}{
		{name: "short-default", input: shortPath, samples: 16000, delay: 2},
		{name: "long-default", input: longPath, samples: 48000, delay: 2},
		{name: "short-zero", input: shortPath, samples: 16000, extra: []string{"--playback-delay-seconds", "0"}},
		{name: "short-no-playback", input: shortPath, samples: 16000, noPlayback: true, extra: []string{"--no-playback"}},
	} {
		dir := filepath.Join(output, c.name)
		cmd := command(c.input, dir, c.extra...)
		if log, err := cmd.CombinedOutput(); err != nil {
			t.Fatal(c.name, err, string(log))
		}
		timing := readJSON(filepath.Join(dir, "playback-timing.json"))
		summary := readJSON(filepath.Join(dir, "run/summary.json"))
		if timing["status"] != "complete" || summary["status"] != "complete" || summary["cleanup"] != "acknowledged" {
			t.Fatal(c.name, timing, summary)
		}
		decoded, clientBytes := number(timing, "decoded_bytes"), number(timing, "client_bytes")
		if decoded != float64(c.samples*2) || clientBytes != decoded {
			t.Fatal("ASR byte loss", c.name, timing)
		}
		firstDecode, firstClient := number(timing, "first_decoded_seconds"), number(timing, "first_client_write_seconds")
		if firstClient < firstDecode || firstClient-firstDecode > .25 {
			t.Fatal("playback delay held ASR input", c.name, timing)
		}
		if number(timing, "max_client_queue_bytes") > number(timing, "client_queue_capacity_bytes") || number(timing, "max_player_queue_bytes") > number(timing, "playback_queue_capacity_bytes") {
			t.Fatal("fanout queue exceeded cap", c.name, timing)
		}
		if c.noPlayback {
			if number(timing, "delay_seconds") != 0 || number(timing, "player_bytes") != 0 || timing["first_player_write_seconds"] != nil || timing["player_stdin_closed_seconds"] != nil {
				t.Fatal("no-playback acquired delay/player", timing)
			}
		} else {
			if number(timing, "delay_seconds") != c.delay || number(timing, "player_bytes") != decoded {
				t.Fatal("player byte loss/delay", c.name, timing)
			}
			firstPlayer := number(timing, "first_player_write_seconds")
			if firstPlayer-firstDecode < c.delay-.05 {
				t.Fatal("player started too early", c.name, timing)
			}
			if c.delay == 0 && firstPlayer-firstDecode > .25 {
				t.Fatal("zero delay introduced startup hold", timing)
			}
			if c.delay > 0 && firstPlayer-firstClient < c.delay-.30 {
				t.Fatal("ASR did not get configured head start", c.name, timing)
			}
			clientEOF, playerEOF := number(timing, "client_stdin_closed_seconds"), number(timing, "player_stdin_closed_seconds")
			if clientEOF < firstClient || playerEOF < firstPlayer {
				t.Fatal("invalid EOF/write timing", c.name, timing)
			}
			if c.name == "short-default" && (clientEOF >= firstPlayer || clientEOF > playerEOF) {
				t.Fatal("short clip ASR EOF waited for playback", timing)
			}
		}
		results[c.name] = map[string]any{"timing": timing, "summary": summary}
	}
	// Interrupt only the managed Python process while the default player delay is
	// still pending; the previously completed cases have warmed model acquisition.
	dir := filepath.Join(output, "interrupted")
	cmd := command(longPath, dir)
	var terminal bytes.Buffer
	cmd.Stdout = &terminal
	cmd.Stderr = &terminal
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	defer func() { _ = cmd.Process.Kill() }()
	until := time.NewTimer(20 * time.Second)
	defer until.Stop()
	tick := time.NewTicker(20 * time.Millisecond)
	defer tick.Stop()
	ready := false
	for !ready {
		if data, err := os.ReadFile(filepath.Join(dir, "run/events.ndjson")); err == nil {
			for _, line := range bytes.Split(data, []byte{'\n'}) {
				var e map[string]any
				if json.Unmarshal(line, &e) == nil && e["type"] == "created" {
					ready = true
					break
				}
			}
		}
		if ready {
			break
		}
		select {
		case err := <-done:
			t.Fatal("playback ended before cancel", err, terminal.String())
		case <-until.C:
			t.Fatal("no client session before cancel")
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-tick.C:
		}
	}
	if err := cmd.Process.Signal(os.Interrupt); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		var exit *exec.ExitError
		if !errors.As(err, &exit) || exit.ExitCode() != 130 {
			t.Fatal("managed interrupt exit", err, terminal.String())
		}
	case <-ctx.Done():
		t.Fatal("playback interrupt did not drain", ctx.Err())
	}
	timing := readJSON(filepath.Join(dir, "playback-timing.json"))
	summary := readJSON(filepath.Join(dir, "run/summary.json"))
	if timing["status"] != "canceled" || number(timing, "player_bytes") != 0 || timing["first_player_write_seconds"] != nil {
		t.Fatal("cancellation was not during startup delay", timing)
	}
	if summary["status"] != "canceled" || summary["cleanup"] != "acknowledged" || summary["text_is_final"] != false {
		t.Fatal("playback cancellation cleanup", summary)
	}
	for _, bad := range []string{"Post \"http", "context canceled", "Broken pipe", "BrokenPipeError", "Traceback"} {
		if strings.Contains(terminal.String(), bad) {
			t.Fatal("noisy playback interrupt", terminal.String())
		}
	}
	if !strings.Contains(terminal.String(), "Stopped.") {
		t.Fatal("missing quiet cancellation", terminal.String())
	}
	results["interrupted"] = map[string]any{"timing": timing, "summary": summary, "only_python_parent_signaled": true, "exit_status": 130}
	s.asrSessions.mu.Lock()
	active := len(s.asrSessions.active)
	s.asrSessions.mu.Unlock()
	if active != 0 {
		t.Fatal("playback session leak", active)
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
			t.Fatal("playback unload failed")
		case <-time.After(10 * time.Millisecond):
		}
	}
	receipt := map[string]any{"passed": true, "digital_silence_only": true, "audible_voice_playback_tested": false, "acoustic_timing_qualified": false, "first_asr_write_not_delayed": true, "short_clip_asr_eof_independent": true, "bounded_byte_exact_fanout": true, "normal_unload": true, "results": results}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(output, "playback-delay.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
