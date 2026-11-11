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

func TestQwen3ASROperatorClientNative(t *testing.T) {
	if os.Getenv("OLLAMA_QWEN3_ASR_OPERATOR_CLIENT_NATIVE") != "1" {
		t.Skip("supervised operator-client qualification only")
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
	binary := filepath.Join(output, "operator-client")
	build := exec.CommandContext(ctx, "go", "build", "-o", binary, filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools/sessionclient/main.go"), filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools/sessionclient/pcm.go"), filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools/sessionclient/render.go"), filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools/sessionclient/utterance.go"), filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools/sessionclient/utterance_detector.go"), filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools/sessionclient/generation_limit.go"), filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools/sessionclient/overlap.go"), filepath.Join(root, "feature-streams/pending/mlx-qwen3-asr/tools/sessionclient/overlap_text.go"))
	build.Dir = filepath.Join(root, "ollama")
	if data, err := build.CombinedOutput(); err != nil {
		t.Fatal("client build", err, string(data))
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
	data, err := os.ReadFile(os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_INPUTS"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &inputs); err != nil {
		t.Fatal(err)
	}
	waves := map[string][]float32{}
	for _, entry := range inputs {
		if entry.Name != "corpus-00" && entry.Name != "corpus-02" {
			continue
		}
		wav, err := os.ReadFile(entry.Prepared)
		if err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256(wav)
		if hex.EncodeToString(hash[:]) != entry.SHA {
			t.Fatal("input hash mismatch")
		}
		pcm, rate, err := audio.Decode(wav)
		if err != nil || rate != 16000 || len(pcm) < 64017 {
			t.Fatal("invalid prepared WAV", err)
		}
		waves[entry.Name] = pcm
	}
	if len(waves) != 2 {
		t.Fatal("missing English/German inputs")
	}
	readSummary := func(dir string) map[string]any {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(dir, "summary.json"))
		if err != nil {
			t.Fatal(err)
		}
		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	run := func(name, mode, language, path string) map[string]any {
		t.Helper()
		dir := filepath.Join(output, name)
		cmd := exec.CommandContext(ctx, binary, "-mode", mode, "-wav", path, "-url", server.URL, "-model", os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "-language", language, "-output", dir, "-timeout", "2m", "-update-interval-ms", interval)
		log, err := os.Create(filepath.Join(output, name+".log"))
		if err != nil {
			t.Fatal(err)
		}
		cmd.Stdout, cmd.Stderr = log, log
		err = cmd.Run()
		log.Close()
		if err != nil {
			t.Fatal(name, err)
		}
		result := readSummary(dir)
		if result["status"] != "complete" || result["text_is_final"] != true {
			t.Fatal(result)
		}
		transcript, err := os.ReadFile(filepath.Join(dir, "transcript.txt"))
		if err != nil || len(transcript) < 3 {
			t.Fatal("empty transcript", err)
		}
		return result
	}
	var results []map[string]any
	for _, input := range []struct{ name, language string }{{"corpus-00", "de"}, {"corpus-02", "en"}} {
		// Non-16k input exercises the client's single whole-clip resampling step.
		pcm, err := audio.ResampleContext(ctx, waves[input.name][:64017], 16000, 24000)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(output, input.name+"-24k.wav")
		if err := os.WriteFile(path, audio.EncodeWAV(pcm, 24000), 0o600); err != nil {
			t.Fatal(err)
		}
		results = append(results, run("live-"+input.language, "live", input.language, path))
	}
	// A derived 75s recording crosses the existing 60s segmentation boundary.
	podcast := make([]float32, 75*16000)
	source := waves["corpus-00"]
	for i := range podcast {
		podcast[i] = source[i%len(source)]
	}
	path := filepath.Join(output, "derived-podcast.wav")
	if err := os.WriteFile(path, audio.EncodeWAV(podcast, 16000), 0o600); err != nil {
		t.Fatal(err)
	}
	fileResult := run("podcast", "file", "de", path)
	if fileResult["updates"].(float64) < 2 {
		t.Fatal("long-file segmentation was not exercised", fileResult)
	}
	results = append(results, fileResult)
	// Signal only our own client after a revision; cleanup must finish separately.
	dir := filepath.Join(output, "interrupted")
	childCtx, stopChild := context.WithCancel(ctx)
	defer stopChild()
	cmd := exec.CommandContext(childCtx, binary, "-mode", "live", "-wav", filepath.Join(output, "corpus-02-24k.wav"), "-url", server.URL, "-model", os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "-language", "en", "-output", dir, "-update-interval-ms", interval)
	log, err := os.Create(filepath.Join(output, "interrupted.log"))
	if err != nil {
		t.Fatal(err)
	}
	defer log.Close()
	cmd.Stdout, cmd.Stderr = log, log
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	for {
		data, _ := os.ReadFile(filepath.Join(dir, "events.ndjson"))
		if bytes.Contains(data, []byte(`"type":"packet"`)) && bytes.Contains(data, []byte(`"revision":1`)) {
			break
		}
		select {
		case err := <-done:
			t.Fatal("client ended before interrupt", err)
		case <-time.After(5 * time.Millisecond):
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
	}
	if err := cmd.Process.Signal(os.Interrupt); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("interrupted client reported success")
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	interrupted := readSummary(dir)
	if interrupted["status"] != "canceled" || interrupted["cleanup"] != "acknowledged" || interrupted["text_is_final"] != false {
		t.Fatal("canceled client cleanup receipt", interrupted)
	}
	if _, err := os.Stat(filepath.Join(dir, "partial.txt")); err != nil {
		t.Fatal(err)
	}
	s.asrSessions.mu.Lock()
	active := len(s.asrSessions.active)
	s.asrSessions.mu.Unlock()
	if active != 0 {
		t.Fatal("client left active session")
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
			t.Fatal("normal unload failed")
		}
	}
	receipt := map[string]any{"passed": true, "live_results": results[:2], "derived_podcast": fileResult, "interrupted": interrupted, "normal_unload": true, "public_contract_changed": false, "accuracy_qualified": false}
	data, err = json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(output, "operator-client.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
