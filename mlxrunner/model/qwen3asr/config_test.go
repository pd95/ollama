package qwen3asr

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/audio"
)

func TestOriginalConfig(t *testing.T) {
	raw, err := os.ReadFile("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := parseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Text.HiddenSize != 1024 || cfg.Text.HeadDim != 128 || cfg.Text.NumAttentionHeads != 16 || cfg.Audio.Width != 896 {
		t.Fatalf("wrong dimensions: %+v", cfg)
	}
	shapes, err := TensorShapes(raw)
	if err != nil || len(shapes) != 612 {
		t.Fatalf("wrong checkpoint schema: %d / %v", len(shapes), err)
	}
	bad := bytes.Replace(raw, []byte(`"head_dim": 128`), []byte(`"head_dim": 64`), 1)
	if _, err := parseConfig(bad); err == nil {
		t.Fatal("derived hidden/head dimension was accepted")
	}
}

func TestAudioTokenLengths(t *testing.T) {
	// Provider's three convolutions run independently on each 100-frame chunk.
	for _, tc := range []struct{ frames, want int }{{1, 1}, {8, 1}, {9, 2}, {99, 13}, {100, 13}, {101, 14}, {200, 26}, {800, 104}, {801, 105}} {
		if got := audioTokens(tc.frames); got != tc.want {
			t.Errorf("%d frames: got %d want %d", tc.frames, got, tc.want)
		}
	}
}

func TestWhisperSilence(t *testing.T) {
	f, err := extractFeatures(make([]float32, 16000))
	if err != nil {
		t.Fatal(err)
	}
	if f.Frames != 100 || len(f.Data) != 12800 {
		t.Fatalf("wrong feature shape: %d/%d", f.Frames, len(f.Data))
	}
	for _, v := range f.Data {
		if v != -1.5 {
			t.Fatalf("silence feature %g, want -1.5", v)
		}
	}
}

func TestWhisperCompleteAudioBound(t *testing.T) {
	f, err := extractFeatures(make([]float32, maxAudioSeconds*16000))
	if err != nil || f.Frames != maxAudioSeconds*100 || audioTokens(f.Frames) != 780 {
		t.Fatalf("60-second frontend: frames=%d err=%v", f.Frames, err)
	}
	if _, err := extractFeatures(make([]float32, maxAudioSeconds*16000+1)); err == nil {
		t.Fatal("unbounded encoder input accepted")
	}
}

func TestPrepareMediaIsolation(t *testing.T) {
	m := &Model{}
	segments := []model.Segment{
		{Tokens: []int32{151669}},
		{Kind: "audio", Data: audio.EncodeWAV(make([]float32, 16000), 16000)},
		{Tokens: []int32{151670}},
	}
	first, err := m.PrepareMedia(segments)
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Tokens) != 15 || len(first.Items) != 1 || first.Items[0].Range != [2]int{1, 14} || first.Items[0].Source != 1 || !first.Items[0].Causal {
		t.Fatalf("wrong expansion: %+v", first)
	}
	second, err := m.PrepareMedia(segments)
	if err != nil {
		t.Fatal(err)
	}
	first.Items[0].MediaData[0] = 123
	if second.Items[0].MediaData[0] != -1.5 {
		t.Fatal("feature data is shared across requests")
	}
	if _, err := m.PrepareMedia([]model.Segment{{Data: []byte("invalid")}}); err == nil {
		t.Fatal("unknown media was silently dropped")
	}
	segments[1].Data = audio.EncodeWAV(make([]float32, 16000), 48000)
	resampled, err := m.PrepareMedia(segments)
	if err != nil || len(resampled.Items) != 1 || resampled.Items[0].Dims[1] != 33 {
		t.Fatalf("48 kHz audio was not resampled to 16 kHz: %v / %+v", err, resampled)
	}
}

func TestQwen3ASRRejectsMultipleAudioBeforeDecode(t *testing.T) {
	_, err := (&Model{}).PrepareMedia([]model.Segment{
		{Kind: "audio", Data: []byte("invalid first WAV")},
		{Kind: "audio", Data: []byte("invalid second WAV")},
	})
	if err == nil || !strings.Contains(err.Error(), "one audio clip") {
		t.Fatalf("must reject multiple occurrences before attempting to decode: %v", err)
	}
}
