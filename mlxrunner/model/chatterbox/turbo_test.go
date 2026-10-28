package chatterbox

import (
	"math/rand/v2"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestTurboSamplerAndOptions(t *testing.T) {
	m := &turboModel{}
	if err := m.ValidateSpeechRequest(llm.SpeechRequest{Input: "Hello"}); err != nil {
		t.Fatal(err)
	}
	for _, r := range []llm.SpeechRequest{{Input: "Hello", Language: "de"}, {Input: "Hello", Voice: "Ryan"}, {Input: "Hello", ReferenceText: "unused"}, {Input: "Hello", Instructions: "whisper"}, {Input: " "}} {
		if m.ValidateSpeechRequest(r) == nil {
			t.Fatal("accepted invalid options")
		}
	}
	a, b := rand.New(rand.NewPCG(42, 43)), rand.New(rand.NewPCG(42, 43))
	for range 100 {
		x, e := sampleTurbo([]float32{10, 9, -10}, []int32{0, 0}, a)
		y, f := sampleTurbo([]float32{10, 9, -10}, []int32{0}, b)
		if e != nil || f != nil || x != y || x == 2 {
			t.Fatal(x, y, e, f)
		}
	}
	for raw, want := range map[string]string{"hello": "Hello.", "Hello...": "Hello...", "First: second": "First, second.", "“Hello”": "“Hello”."} {
		if got := normalizedTurboText(raw); got != want {
			t.Fatal(got, want)
		}
	}
}
