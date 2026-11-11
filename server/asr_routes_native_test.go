package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/model/audio"
)

func TestQwen3ASRSessionHTTPNative(t *testing.T) {
	if os.Getenv("OLLAMA_QWEN3_ASR_SESSION_HTTP_NATIVE") != "1" {
		t.Skip("supervised paced native HTTP qualification only")
	}
	intervalMS := 2000
	if value := os.Getenv("OLLAMA_QWEN3_ASR_UPDATE_INTERVAL_MS"); value != "" {
		var err error
		intervalMS, err = strconv.Atoi(value)
		if err != nil || (intervalMS != 500 && intervalMS != 1000 && intervalMS != 2000) {
			t.Fatal("native HTTP update interval must be 500, 1000 or 2000 milliseconds")
		}
	}
	t.Setenv("OLLAMA_EXPERIMENT", "asr-streaming")
	ctx, cancel := context.WithTimeout(t.Context(), 175*time.Second)
	defer cancel()
	schedCtx, stopSched := context.WithCancel(context.Background())
	s := &Server{sched: InitScheduler(schedCtx), defaultNumCtx: 2048}
	s.sched.Run(schedCtx)
	defer func() { s.closeASRSessions(); s.sched.unloadAllRunners(); stopSched() }()
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(router)
	defer srv.Close()
	client := &http.Client{Timeout: 30 * time.Second}
	call := func(method, path, content string, data []byte, status int) map[string]any {
		t.Helper()
		req, err := http.NewRequestWithContext(ctx, method, srv.URL+path, bytes.NewReader(data))
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
			t.Fatalf("%s %s status=%d body=%s err=%v", method, path, response.StatusCode, body, err)
		}
		var out map[string]any
		if len(body) > 0 {
			if err := json.Unmarshal(body, &out); err != nil {
				t.Fatal(err)
			}
		}
		return out
	}
	newSession := func(language string) (string, int, int, float64) {
		t.Helper()
		request := map[string]any{"model": os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "language": language}
		if intervalMS != 2000 {
			request["update_interval_ms"] = intervalMS
		}
		body, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		created := call("POST", asrSessionPath, "application/json", body, 201)
		createSeconds := time.Since(started).Seconds()
		negotiated, ok := created["update_samples"].(float64)
		if !ok || negotiated != float64(intervalMS*16) {
			t.Fatal("unexpected negotiated update cadence", created)
		}
		packet, ok := created["max_packet_samples"].(float64)
		if !ok || packet != float64(min(16000, intervalMS*16)) {
			t.Fatal("unexpected negotiated packet cap", created)
		}
		return asrSessionPath + "/" + created["id"].(string), int(negotiated), int(packet), createSeconds
	}
	var entries []struct {
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
	waves := map[string][]float32{}
	for _, entry := range entries {
		if (entry.Language != "en" && entry.Language != "de") || waves[entry.Language] != nil {
			continue
		}
		wav, err := os.ReadFile(entry.Prepared)
		if err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256(wav)
		if hex.EncodeToString(hash[:]) != entry.SHA {
			t.Fatal("prepared hash mismatch")
		}
		pcm, rate, err := audio.Decode(wav)
		if err != nil || rate != 16000 {
			t.Fatal(err)
		}
		waves[entry.Language] = pcm
	}
	var records []map[string]any
	cases := []struct {
		language, source string
		seconds          int
	}{{"en", "en", 10}, {"de", "de", 30}, {"auto", "de", 60}}
	memoryOnly := os.Getenv("OLLAMA_QWEN3_ASR_SESSION_MEMORY_NATIVE") == "1"
	if memoryOnly {
		cases = nil
	}
	for _, test := range cases {
		if len(waves[test.source]) == 0 {
			t.Fatal("missing source WAV")
		}
		packetAt := func(offset, samples int) []byte {
			packet := make([]byte, 2*samples)
			source := waves[test.source]
			for i := range samples {
				value := int16(max(-32768, min(32767, int(source[(offset+i)%len(source)]*32768))))
				binary.LittleEndian.PutUint16(packet[2*i:], uint16(value))
			}
			return packet
		}
		var warmup map[string]any
		if intervalMS <= 1000 {
			// Warm each language/source independently with two audio seconds.
			// These explicitly recorded unpaced requests are not measured packets.
			base, _, packetSamples, createSeconds := newSession(test.language)
			var latency []float64
			for offset := 0; offset < 32000; offset += packetSamples {
				started := time.Now()
				call("POST", fmt.Sprintf("%s/audio?offset_samples=%d", base, offset), "application/octet-stream", packetAt(offset, packetSamples), 200)
				latency = append(latency, time.Since(started).Seconds())
			}
			final := call("POST", base+"/finish", "", nil, 200)
			if final["final"] != true || final["audio_end_sample"] != float64(32000) {
				t.Fatal("invalid warmup final", final)
			}
			call("DELETE", base, "", nil, 204)
			warmup = map[string]any{"audio_seconds": 2, "paced": false, "create_seconds": createSeconds, "first_request_latency_seconds": latency[0], "packet_latency_seconds": latency}
		}
		base, updateSamples, packetSamples, createSeconds := newSession(test.language)
		var concurrentDone chan struct{}
		var concurrentResponse *httptest.ResponseRecorder
		var concurrentRequest *http.Request
		if test.seconds == 10 {
			concurrentDone = make(chan struct{})
			concurrentResponse = httptest.NewRecorder()
			concurrentRequest = audioTextRequest(t, "/v1/audio/transcriptions", map[string]string{"model": os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "language": "en"}, audio.EncodeWAV(waves["en"][:32017], 16000)).WithContext(ctx)
		}
		started := time.Now()
		var latencies, backlog []float64
		updates := 0
		for offset := 0; offset < test.seconds*16000; offset += packetSamples {
			end := offset + packetSamples
			due := started.Add(time.Duration(end) * time.Second / 16000)
			if wait := time.Until(due); wait > 0 {
				timer := time.NewTimer(wait)
				select {
				case <-timer.C:
				case <-ctx.Done():
					timer.Stop()
					t.Fatal(ctx.Err())
				}
			}
			packet := packetAt(offset, packetSamples)
			if end == 32000 && concurrentDone != nil {
				go func() { router.ServeHTTP(concurrentResponse, concurrentRequest); close(concurrentDone) }()
			}
			requestStart := time.Now()
			result := call("POST", fmt.Sprintf("%s/audio?offset_samples=%d", base, offset), "application/octet-stream", packet, 200)
			latencies = append(latencies, time.Since(requestStart).Seconds())
			backlog = append(backlog, max(0, time.Since(due).Seconds()))
			if result["accepted_samples"] != float64(end) {
				t.Fatal("packet acceptance mismatch")
			}
			_, updated := result["revision"]
			if updated != (end%updateSamples == 0) {
				t.Fatal("update did not follow negotiated cadence", result)
			}
			if updated {
				updates++
				if result["audio_end_sample"] != float64(end) {
					t.Fatal("update boundary mismatch")
				}
			}
		}
		final := call("POST", base+"/finish", "", nil, 200)
		if final["final"] != true || final["audio_end_sample"] != float64(test.seconds*16000) || updates != test.seconds*16000/updateSamples {
			t.Fatal("invalid final", final)
		}
		call("DELETE", base, "", nil, 204)
		if concurrentDone != nil {
			select {
			case <-concurrentDone:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			if concurrentResponse.Code != 200 {
				t.Fatal("competing complete-file request failed", concurrentResponse.Code, concurrentResponse.Body.String())
			}
		}
		record := map[string]any{"language": test.language, "source": test.source, "audio_seconds": test.seconds, "wall_seconds": time.Since(started).Seconds(), "packet_latency_seconds": latencies, "client_backlog_seconds": backlog, "updates": updates, "final": final, "source_repeats_for_60s": test.seconds == 60, "create_seconds": createSeconds, "update_samples": updateSamples, "packet_samples": packetSamples}
		if intervalMS <= 1000 {
			timing, err := asrFrequentUpdateTimings(latencies, backlog)
			if err != nil {
				t.Fatal(err)
			}
			timingKey := "one_second_timing"
			if intervalMS == 500 {
				timingKey = "half_second_timing"
			}
			record["warmup"], record[timingKey] = warmup, timing
			if timing["due_update_p95_seconds"] >= float64(intervalMS)/1000 || timing["lateness_growth_seconds"] > 1 {
				// Preserve every sample, including the first and outliers, even if
				// a cohort fails before the final full receipt can be written.
				failed, _ := json.MarshalIndent(record, "", "  ")
				_ = os.WriteFile(filepath.Join(os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_OUTPUT"), fmt.Sprintf("session-api-%ds-timing-failure.json", test.seconds)), failed, 0o600)
				t.Fatalf("%ds %dms cadence failed predeclared timing gates: %v", test.seconds, intervalMS, timing)
			}
		}
		records = append(records, record)
	}
	var memorySamples []uint64
	if memoryOnly {
		for i := range 21 {
			base, _, packetSamples, _ := newSession("en")
			pcm := audio.EncodeWAV(waves["en"][:32017], 16000)[44:]
			for offset := 0; offset < 32017; {
				end := min(offset+packetSamples, 32017)
				call("POST", fmt.Sprintf("%s/audio?offset_samples=%d", base, offset), "application/octet-stream", pcm[offset*2:end*2], 200)
				offset = end
			}
			call("POST", base+"/finish", "", nil, 200)
			ps := call("GET", "/api/ps", "", nil, 200)
			models, ok := ps["models"].([]any)
			if !ok || len(models) != 1 {
				t.Fatal("missing model memory sample", ps)
			}
			memory := uint64(models[0].(map[string]any)["size_vram"].(float64))
			if memory == 0 {
				t.Fatal("empty memory sample")
			}
			if i > 0 {
				memorySamples = append(memorySamples, memory)
			}
		}
		lo, hi := memorySamples[0], memorySamples[0]
		for _, n := range memorySamples {
			lo = min(lo, n)
			hi = max(hi, n)
		}
		if hi-lo > 32<<20 {
			t.Fatalf("warmed public-session memory range %d exceeds 32MiB", hi-lo)
		}
	}
	// Normal scheduler expiration after completed public sessions must unload.
	loaded, err := s.getExistingModelForRunner(os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"), "")
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
			t.Fatal("model did not unload")
		}
	}
	receipt := map[string]any{"passed": true, "experimental_opt_in": true, "records": records, "normal_scheduler_unload": true, "competing_complete_file_request": !memoryOnly, "accuracy_qualified": false, "transport": "HTTP PCM16", "paced_packets": !memoryOnly, "warmed_api_memory_samples": memorySamples, "update_interval_ms": intervalMS}
	encoded, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	filename := "session-api.json"
	if memoryOnly {
		filename = "session-api-memory.json"
	}
	if err := os.WriteFile(filepath.Join(os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_OUTPUT"), filename), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
}

// Every frequent-mode packet is a due update. Keep all measured requests, including
// the first, in the nearest-rank p95; report first latency independently as well.
func asrFrequentUpdateTimings(latencies, lateness []float64) (map[string]float64, error) {
	if len(latencies) < 5 || len(latencies) != len(lateness) {
		return nil, fmt.Errorf("frequent-update timing needs matching latency/lateness windows")
	}
	for _, values := range [][]float64{latencies, lateness} {
		for _, value := range values {
			if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("frequent-update timing samples must be finite and nonnegative")
			}
		}
	}
	ordered := slices.Clone(latencies)
	slices.Sort(ordered)
	p95 := ordered[(95*len(ordered)+99)/100-1]
	mean := func(values []float64) float64 {
		var sum float64
		for _, value := range values {
			sum += value
		}
		return sum / float64(len(values))
	}
	first, last := mean(lateness[:5]), mean(lateness[len(lateness)-5:])
	return map[string]float64{"first_request_latency_seconds": latencies[0], "due_update_p95_seconds": p95, "first5_lateness_mean_seconds": first, "last5_lateness_mean_seconds": last, "lateness_growth_seconds": last - first, "measured_requests": float64(len(latencies))}, nil
}

func TestQwen3ASRSessionOneSecondTimingPolicy(t *testing.T) {
	latencies := make([]float64, 20)
	lateness := make([]float64, 20)
	for i := range latencies {
		latencies[i] = float64(i+1) / 100
		lateness[i] = 0.2
	}
	latencies[0] = 0.9
	timing, err := asrFrequentUpdateTimings(latencies, lateness)
	if err != nil || timing["due_update_p95_seconds"] != 0.2 || timing["first_request_latency_seconds"] != 0.9 || timing["lateness_growth_seconds"] != 0 {
		t.Fatal(timing, err)
	}
	// Ten requests use their maximum as nearest-rank p95, so the first slow
	// request must remain visible rather than being silently discarded.
	timing, err = asrFrequentUpdateTimings(latencies[:10], lateness[:10])
	if err != nil || timing["due_update_p95_seconds"] != 0.9 {
		t.Fatal(timing, err)
	}
	for _, pair := range []struct{ latency, late []float64 }{{nil, nil}, {latencies, lateness[:5]}, {[]float64{0, 0, 0, 0, math.NaN()}, make([]float64, 5)}} {
		if _, err := asrFrequentUpdateTimings(pair.latency, pair.late); err == nil {
			t.Fatal("invalid timing data accepted")
		}
	}
}
