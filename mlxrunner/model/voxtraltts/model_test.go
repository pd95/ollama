package voxtraltts

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestTextNormalization(t *testing.T) {
	for _, pair := range [][2]string{{"Hello", "Hello."}, {"Grüße aus Köln!", "Grüße aus Köln!"}, {"Pay $12.50.", "Pay twelve dollars and fifty cents."}, {"1,000,123", "one million one hundred and twenty-three."}, {"Hi!!!!", "Hi!"}, {"word (note) again", "word—note—again."}} {
		got, err := normalizeText(pair[0])
		if err != nil || got != pair[1] {
			t.Errorf("%s = %q, %v; want %q", pair[0], got, err, pair[1])
		}
	}
}

func TestSpeechOptions(t *testing.T) {
	m := &Model{cfg: &config{Voices: map[string]int{"casual_male": 147, "de_male": 163}}}
	for _, req := range []llm.SpeechRequest{{Input: "Hello"}, {Input: "Guten Tag", Language: "de"}, {Input: "Hallo", Voice: "de_male"}} {
		if err := m.ValidateSpeechRequest(req); err != nil {
			t.Fatal(err)
		}
	}
	speed := .75
	for _, req := range []llm.SpeechRequest{{Input: "Hello", ReferenceAudio: &llm.ReferenceAudio{Format: "wav"}}, {Input: "Hello", ReferenceText: "example"}, {Input: "Hello", Language: "fr"}, {Input: "Hello", Voice: "missing"}, {Input: "Hello", Instructions: "whisper"}, {Input: "Hello", Speed: &speed}, {Input: "Hello", Voice: "de_male", Language: "en"}, {Input: ""}} {
		if err := m.ValidateSpeechRequest(req); err == nil {
			t.Fatalf("accepted %+v", req)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := m.Speak(ctx, llm.SpeechRequest{Input: "Hello"}, func([]float32) error { t.Fatal("emitted after cancellation"); return nil }); err != context.Canceled {
		t.Fatal(err)
	}
}

func TestTextChunks(t *testing.T) {
	input := strings.Repeat("Hallo! Grüße aus Köln. ", 40)
	chunks := textChunks(input)
	for _, s := range chunks {
		if len([]rune(s)) > 160 {
			t.Fatal("chunk exceeds bound")
		}
	}
	compact := func(s string) string { return strings.Join(strings.Fields(s), "") }
	if compact(strings.Join(chunks, "")) != compact(input) {
		t.Fatal("text omitted or duplicated")
	}
}
