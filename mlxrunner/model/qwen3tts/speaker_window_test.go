package qwen3tts

import "testing"

func TestSpeakerHannWindowIsPeriodic(t *testing.T) {
	window := speakerHannWindow()
	if len(window) != 1024 || window[0] != 0 || window[256] != .5 || window[512] != 1 {
		t.Fatal("unexpected speaker Hann window geometry")
	}
	// A symmetric 1024-point window ends at zero. The original periodic window
	// instead completes its cycle at the next (omitted) sample.
	if window[1023] == 0 || window[1023] != window[1] {
		t.Fatalf("speaker window repeats an endpoint: first neighbor %g, last %g", window[1], window[1023])
	}
	for i := 1; i < len(window); i++ {
		if window[i] != window[len(window)-i] {
			t.Fatalf("periodic window differs across its origin at %d", i)
		}
	}
}
