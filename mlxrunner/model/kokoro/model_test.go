package kokoro

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner/model/kokoro/config"
)

func TestSpeechOptions(t *testing.T) {
	m := &Model{cfg: &config.Config{Voices: []string{"af_heart", "bf_emma"}, Vocab: map[string]int{}}}
	for _, r := range "həlˈO wˈɜɹld." {
		m.cfg.Vocab[string(r)] = 1
	}
	speed := .75
	for _, req := range []llm.SpeechRequest{{Input: "Hello world."}, {Input: "Hello world.", Language: "en-US", Speed: &speed}} {
		if err := m.ValidateSpeechRequest(req); err != nil {
			t.Fatal(err)
		}
	}
	for _, req := range []llm.SpeechRequest{
		{Input: "Hello", Voice: "missing"},
		{Input: "Hello", Language: "de"},
		{Input: "Hello", Voice: "bf_emma", Language: "en-US"},
		{Input: "Hello", Instructions: "whisper"},
		{Input: "Hello", ReferenceText: "example"},
		{Input: "Hello", ReferenceAudio: &llm.ReferenceAudio{Format: "wav"}},
		{Input: ""},
		{Input: strings.Repeat("a", 4097)},
	} {
		if err := m.ValidateSpeechRequest(req); err == nil {
			t.Fatalf("accepted unsupported request: %+v", req)
		}
	}
	for _, value := range []float64{0, .49, 2.01, math.NaN(), math.Inf(1)} {
		if err := m.ValidateSpeechRequest(llm.SpeechRequest{Input: "Hello", Speed: &value}); err == nil {
			t.Fatalf("accepted speed %v", value)
		}
	}
}

func TestFitPhonemesDurationAndCancellation(t *testing.T) {
	text := "first sentence. second sentence."
	var accepted []string
	err := fitPhonemes(t.Context(), text, 510, func(part string) ([]float32, error) {
		if len(part) > 20 {
			return nil, errUtteranceTooLong
		}
		accepted = append(accepted, part)
		return []float32{1}, nil
	}, func([]float32) error { return nil })
	if err != nil || !slices.Equal(accepted, []string{"first sentence.", "second sentence."}) {
		t.Fatalf("fit: %q / %v", accepted, err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	accepted = nil
	err = fitPhonemes(ctx, text, 510, func(part string) ([]float32, error) {
		if len(part) > 20 {
			return nil, errUtteranceTooLong
		}
		accepted = append(accepted, part)
		return []float32{1}, nil
	}, func([]float32) error { cancel(); return nil })
	if !errors.Is(err, context.Canceled) || len(accepted) != 1 {
		t.Fatalf("generated after cancellation: %q / %v", accepted, err)
	}
	err = fitPhonemes(t.Context(), "ɑ", 510, func(string) ([]float32, error) { return nil, errUtteranceTooLong }, func([]float32) error { t.Fatal("emitted oversized candidate"); return nil })
	if !errors.Is(err, errUtteranceTooLong) {
		t.Fatalf("unsplittable: %v", err)
	}
}

func TestPhonemeSentenceCanExceedOldPolicy(t *testing.T) {
	text := "wi wɪl kəntˈɪnju ˈɪntu ðə nˈɛkst ʧˈʌŋk ænd fˈɪnəʃ ðə kəmplˈit pˈɛɹəɡɹˌæf wɪðˈWt lˈuzɪŋ ɪts fˈInᵊl sˈɛntᵊns."
	parts := phonemeChunks(text, 510)
	if !slices.Equal(parts, []string{text}) {
		t.Fatalf("sentence fragmented: %q", parts)
	}
}

func TestPhonemeChunksPreserveInput(t *testing.T) {
	source := strings.Repeat("həlˈO wˈɜɹld. ", 30) + strings.Repeat("ɑ", 160)
	chunks := phonemeChunks(source, 80)
	for _, s := range chunks {
		if len([]rune(s)) > 80 {
			t.Fatal("oversized chunk")
		}
	}
	compact := func(s string) string { return strings.ReplaceAll(s, " ", "") }
	if compact(strings.Join(chunks, "")) != compact(source) {
		t.Fatal("phonemes dropped or duplicated")
	}
}
