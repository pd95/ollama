package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/model/audio"
)

func TestQwen3ASRPipeClientNative(t *testing.T) {
	if os.Getenv("OLLAMA_QWEN3_ASR_PIPE_CLIENT_NATIVE") != "1" {
		t.Skip("supervised PCM pipeline only")
	}
	t.Setenv("OLLAMA_EXPERIMENT", "asr-streaming")
	ctx, cancel := context.WithTimeout(t.Context(), 160*time.Second)
	defer cancel()
	output := os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_OUTPUT")
	interval := os.Getenv("OLLAMA_QWEN3_ASR_UPDATE_INTERVAL_MS")
	if interval == "" {
		interval = "2000"
	}
	if interval != "500" && interval != "1000" && interval != "2000" {
		t.Fatal("unsupported client qualification interval", interval)
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Clean(filepath.Join(cwd, "..", ".."))
	tools := filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools")
	binary := filepath.Join(output, "pcm-client")
	build := exec.CommandContext(ctx, "go", "build", "-o", binary, filepath.Join(tools, "sessionclient/main.go"), filepath.Join(tools, "sessionclient/pcm.go"), filepath.Join(tools, "sessionclient/render.go"), filepath.Join(tools, "sessionclient/utterance.go"), filepath.Join(tools, "sessionclient/utterance_detector.go"), filepath.Join(tools, "sessionclient/generation_limit.go"), filepath.Join(tools, "sessionclient/overlap.go"), filepath.Join(tools, "sessionclient/overlap_text.go"))
	build.Dir = filepath.Join(root, "ollama")
	if b, err := build.CombinedOutput(); err != nil {
		t.Fatal(err, string(b))
	}
	player := filepath.Join(output, "pcm-player")
	if b, err := exec.CommandContext(ctx, "clang", "-Wall", "-Wextra", "-Werror", filepath.Join(tools, "sessionclient/pcm_player.c"), "-framework", "AudioToolbox", "-framework", "CoreFoundation", "-o", player).CombinedOutput(); err != nil {
		t.Fatal("native player compile", err, string(b))
	}
	// Open the real macOS output device using200ms of digital silence.
	smoke := exec.CommandContext(ctx, player)
	smoke.Stdin = bytes.NewReader(make([]byte, 6400))
	if b, err := smoke.CombinedOutput(); err != nil {
		t.Fatal("CoreAudio silence smoke", err, string(b))
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
	var entries []struct {
		Name     string `json:"name"`
		Prepared string `json:"prepared"`
		SHA      string `json:"prepared_sha256"`
	}
	b, err := os.ReadFile(os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_INPUTS"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &entries); err != nil {
		t.Fatal(err)
	}
	var source []float32
	for _, e := range entries {
		if e.Name != "corpus-00" {
			continue
		}
		wav, err := os.ReadFile(e.Prepared)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(wav)
		if hex.EncodeToString(sum[:]) != e.SHA {
			t.Fatal("input hash mismatch")
		}
		var rate int
		source, rate, err = audio.Decode(wav)
		if err != nil || rate != 16000 {
			t.Fatal(err)
		}
		break
	}
	if len(source) < 64017 {
		t.Fatal("missing source")
	}
	readResult := func(dir string) map[string]any {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(dir, "summary.json"))
		if err != nil {
			t.Fatal(err)
		}
		var result map[string]any
		if err := json.Unmarshal(b, &result); err != nil {
			t.Fatal(err)
		}
		if result["status"] != "complete" || result["cleanup"] != "acknowledged" {
			t.Fatal(result)
		}
		return result
	}
	var results []map[string]any
	for _, frames := range []int{32017, 61*16000 + 17} {
		pcm := make([]float32, frames)
		for i := range pcm {
			pcm[i] = source[i%len(source)]
		}
		raw := audio.EncodeWAV(pcm, 16000)[44:]
		name := "roll-60"
		segment := "60"
		if frames == 32017 {
			name = "short-tail"
			segment = "2"
		}
		dir := filepath.Join(output, name)
		cmd := exec.CommandContext(ctx, binary, "-mode", "pcm", "-segment-seconds", segment, "-url", server.URL, "-model", os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "-language", "de", "-output", dir, "-update-interval-ms", interval)
		cmd.Stdin = bytes.NewReader(raw)
		if log, err := cmd.CombinedOutput(); err != nil {
			t.Fatal(name, err, string(log))
		}
		result := readResult(dir)
		if result["segments"] != float64(2) {
			t.Fatal("rollover not exercised", result)
		}
		log, err := os.ReadFile(filepath.Join(dir, "events.ndjson"))
		if err != nil {
			t.Fatal(err)
		}
		var end int64
		for _, line := range bytes.Split(log, []byte{'\n'}) {
			if len(line) == 0 {
				continue
			}
			var record struct {
				Type  string `json:"type"`
				Start int64  `json:"start_sample"`
				End   int64  `json:"end_sample"`
			}
			if err := json.Unmarshal(line, &record); err != nil {
				t.Fatal(err)
			}
			if record.Type == "final" {
				if record.Start != end {
					t.Fatal("segment sample gap/overlap")
				}
				end = record.End
			}
		}
		if end != int64(frames) {
			t.Fatal("lost PCM samples", end, frames)
		}
		results = append(results, result)
	}
	// Exercise actual ffmpeg and bounded fan-out without audible voice playback;
	// CoreAudio output was independently exercised with silence above.
	wav := filepath.Join(output, "input.wav")
	if err := os.WriteFile(wav, audio.EncodeWAV(source[:64017], 16000), 0o600); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(output, "pipeline")
	cmd := exec.CommandContext(ctx, "python3", filepath.Join(tools, "listen.py"), "--utterance-detection", "off", wav, "--no-playback", "--segment-seconds", "2", "--url", server.URL, "--model", os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "--language", "de", "--output", dir, "--timeout", "90", "--update-interval-ms", interval)
	if log, err := cmd.CombinedOutput(); err != nil {
		t.Fatal("ffmpeg pipeline", err, string(log))
	}
	result := readResult(filepath.Join(dir, "run"))
	if result["segments"] != float64(3) {
		t.Fatal("unexpected4s+tail rolling count", result)
	}
	// Exercise both actual fan-out branches together without audible speech.
	silence := filepath.Join(output, "silence.wav")
	if err := os.WriteFile(silence, audio.EncodeWAV(make([]float32, 32017), 16000), 0o600); err != nil {
		t.Fatal(err)
	}
	playDir := filepath.Join(output, "playback")
	play := exec.CommandContext(ctx, "python3", filepath.Join(tools, "listen.py"), "--utterance-detection", "off", silence, "--segment-seconds", "2", "--url", server.URL, "--model", os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "--language", "de", "--output", playDir, "--timeout", "90", "--update-interval-ms", interval)
	if log, err := play.CombinedOutput(); err != nil {
		t.Fatal("playback/transcription fan-out", err, string(log))
	}
	playResult := readResult(filepath.Join(playDir, "run"))
	if playResult["segments"] != float64(2) {
		t.Fatal("playback rollover", playResult)
	}
	s.asrSessions.mu.Lock()
	active := len(s.asrSessions.active)
	s.asrSessions.mu.Unlock()
	if active != 0 {
		t.Fatal("active session leaked")
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
		case <-time.After(10 * time.Millisecond):
		case <-ctx.Done():
			t.Fatal("unload failed")
		}
	}
	receipt := map[string]any{"passed": true, "raw_client": results, "ffmpeg_pipeline": result, "coreaudio_silence_smoke": true, "playback_fanout": playResult, "sample_continuity": true, "audible_voice_playback_tested": false, "normal_unload": true, "boundary_accuracy_qualified": false}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(output, "pipe-client.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
