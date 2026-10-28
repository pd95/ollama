package qwen3tts

import (
	"context"
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner/model/audio"
)

func TestTextChunks(t *testing.T) {
	for _, input := range []string{"Hello!", strings.Repeat("Grüße aus Köln. ", 40), strings.Repeat("語", 350), " \n "} {
		chunks := textChunks(input, 160)
		for _, chunk := range chunks {
			if chunk == "" || !utf8.ValidString(chunk) || utf8.RuneCountInString(chunk) > 160 {
				t.Fatalf("invalid chunk %q", chunk)
			}
		}
		// Only whitespace at the split may disappear.
		compact := func(s string) string { return strings.Join(strings.Fields(s), "") }
		if compact(strings.Join(chunks, "")) != compact(input) {
			t.Fatal("lost or duplicated text")
		}
	}
}

func TestSpeechOptions(t *testing.T) {
	raw, err := os.ReadFile("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := parseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{cfg: cfg}
	speed := .75
	if err := m.ValidateSpeechRequest(llm.SpeechRequest{Input: "Hello", Speed: &speed}); err == nil {
		t.Fatal("Qwen accepted a speed change")
	}
	for _, req := range []llm.SpeechRequest{{Input: "Hello"}, {Input: "Hello", Voice: "Ryan", Language: "en"}, {Input: "Hello", Voice: "ryan", Language: "German"}} {
		if err := m.ValidateSpeechRequest(req); err != nil {
			t.Fatal(err)
		}
	}
	// The original snapshots name dialect voices without their codec IDs.
	// Reject that pairing before inference; explicit English remains valid.
	cfg.Talker.Speakers["eric"] = 2875
	if cfg.Talker.Dialects == nil {
		cfg.Talker.Dialects = map[string]json.RawMessage{}
	}
	cfg.Talker.Dialects["eric"] = json.RawMessage(`"sichuan_dialect"`)
	delete(cfg.Talker.Languages, "sichuan_dialect")
	for _, language := range []string{"auto", "zh"} {
		if err := m.ValidateSpeechRequest(llm.SpeechRequest{Input: "Hello", Voice: "eric", Language: language}); err == nil {
			t.Fatal("accepted missing dialect codec ID")
		}
	}
	if err := m.ValidateSpeechRequest(llm.SpeechRequest{Input: "Hello", Voice: "eric", Language: "en"}); err != nil {
		t.Fatal(err)
	}
	for _, req := range []llm.SpeechRequest{{Input: "Hello", Voice: "missing"}, {Input: "Hello", Language: "missing"}, {Input: "Hello", Instructions: "whisper"}, {Input: "Hello", ReferenceText: "example"}, {Input: "Hello", ReferenceAudio: &llm.ReferenceAudio{Format: "wav"}}} {
		if err := m.ValidateSpeechRequest(req); err == nil {
			t.Fatal("accepted unknown option")
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := m.Speak(ctx, llm.SpeechRequest{Input: "Hello"}, func([]float32) error { t.Fatal("emitted after cancellation"); return nil }); err != context.Canceled {
		t.Fatalf("cancellation: %v", err)
	}
}

func TestBaseOptions(t *testing.T) {
	raw, err := os.ReadFile("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	var values map[string]any
	if err := json.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}
	values["tts_model_type"] = "base"
	delete(values, "quantization")
	values["talker_config"].(map[string]any)["spk_id"] = map[string]any{}
	base, err := json.Marshal(values)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := parseConfig(base)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{cfg: cfg}
	ref := &llm.ReferenceAudio{Format: "wav", Data: audio.EncodeWAV(make([]float32, 24000), 24000)}
	valid := llm.SpeechRequest{Input: "Hello", Language: "en", ReferenceAudio: ref, ReferenceText: "A reference."}
	if err := m.ValidateSpeechRequest(valid); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*llm.SpeechRequest){
		func(r *llm.SpeechRequest) { r.ReferenceAudio = nil },
		func(r *llm.SpeechRequest) { r.ReferenceText = " " },
		func(r *llm.SpeechRequest) { r.ReferenceText = strings.Repeat("é", 4097) },
		func(r *llm.SpeechRequest) { r.Voice = "Ryan" },
		func(r *llm.SpeechRequest) { r.Instructions = "Speak cheerfully." },
		func(r *llm.SpeechRequest) { r.ReferenceAudio = &llm.ReferenceAudio{Format: "mp3", Data: ref.Data} },
		func(r *llm.SpeechRequest) {
			r.ReferenceAudio = &llm.ReferenceAudio{Format: "wav", Data: []byte("invalid")}
		},
	} {
		request := valid
		change(&request)
		if err := m.ValidateSpeechRequest(request); err == nil {
			t.Fatal("invalid Base option accepted")
		}
	}
	values["tts_model_type"] = "voice_design"
	unsupported, _ := json.Marshal(values)
	if _, err := parseConfig(unsupported); err == nil {
		t.Fatal("VoiceDesign was admitted")
	}
}

func TestSampling(t *testing.T) {
	rng := func() *rand.Rand { return rand.New(rand.NewPCG(42, 43)) }
	gen := generation{Temperature: .9, TopK: 1, TopP: 1, Repetition: 2}
	for _, tc := range []struct {
		values   []float32
		previous []int32
		eos      int
		want     int32
	}{
		{[]float32{1, 2, 3}, nil, -1, 2},
		{[]float32{1, 2, 3}, []int32{2}, -1, 1},
		{[]float32{-1, -2, -3}, []int32{0}, -1, 0},
	} {
		got, err := sampleValues(tc.values, rng(), gen, tc.previous, tc.eos)
		if err != nil || got != tc.want {
			t.Fatalf("got %d %v, want %d", got, err, tc.want)
		}
	}
	values := make([]float32, 3072)
	values[3000] = 100
	values[2150] = 5
	if got, err := sampleValues(values, rng(), gen, nil, 2150); err != nil || got != 2150 {
		t.Fatalf("special token suppression: %d %v", got, err)
	}
	if _, err := sampleValues([]float32{float32(math.NaN())}, rng(), gen, nil, -1); err == nil {
		t.Fatal("accepted NaN")
	}
	gen.TopK = 3
	a, b := rng(), rng()
	for range 30 {
		one, _ := sampleValues([]float32{1, 2, 3}, a, gen, nil, -1)
		two, _ := sampleValues([]float32{1, 2, 3}, b, gen, nil, -1)
		if one != two {
			t.Fatal("seed is not deterministic")
		}
	}
}

func TestCustomVoiceVariants(t *testing.T) {
	raw, err := os.ReadFile("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	quant := cfg["quantization"].(map[string]any)
	quant["bits"] = 4
	four, _ := json.Marshal(cfg)
	if _, err := parseConfig(four); err != nil {
		t.Fatalf("supplied 0.6B 4-bit: %v", err)
	}
	cfg["tts_model_size"] = "1b7"
	delete(cfg, "quantization")
	talker := cfg["talker_config"].(map[string]any)
	talker["hidden_size"] = 2048
	talker["intermediate_size"] = 6144
	large, _ := json.Marshal(cfg)
	parsed, err := parseConfig(large)
	if err != nil {
		t.Fatalf("supplied 1.7B bf16: %v", err)
	}
	if parsed.Talker.Predictor.Hidden == parsed.Talker.Hidden {
		t.Fatal("1.7B requires the predictor projection")
	}
	m := &Model{cfg: parsed}
	if err := m.ValidateSpeechRequest(llm.SpeechRequest{Input: "Hello", Instructions: "Speak cheerfully."}); err != nil {
		t.Fatal(err)
	}
}

func TestRepetitionWindow(t *testing.T) {
	// MLX-Audio limits repetition penalties to the most recent 64 semantic codes.
	previous := append([]int32{0}, make([]int32, 64)...)
	for i := 1; i < len(previous); i++ {
		previous[i] = 2
	}
	gen := generation{Temperature: .9, TopK: 1, TopP: 1, Repetition: 2}
	token, err := sampleValues([]float32{3, 2, 0}, rand.New(rand.NewPCG(1, 2)), gen, previous, -1)
	if err != nil || token != 0 {
		t.Fatalf("old token outside the window was penalized: %d %v", token, err)
	}
}
