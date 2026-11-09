package voxtraltts

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

// Expected IDs come directly from mistral-common, independently of MLX-Audio.
func TestOfficialTokenizerAndSpeechFraming(t *testing.T) {
	path := os.Getenv("OLLAMA_VOXTRAL_TTS_OFFICIAL_TEKKEN")
	if path == "" {
		t.Skip("set OLLAMA_VOXTRAL_TTS_OFFICIAL_TEKKEN to pinned provider tekken.json")
	}
	var fixture struct {
		Hash   string         `json:"tekken_sha256"`
		Frames map[string]int `json:"voice_frames"`
		Cases  []struct {
			Text string
			IDs  []int32
		} `json:"cases"`
		Speech []struct {
			Voice, Text string
			IDs         []int32
		} `json:"speech_cases"`
	}
	raw, err := os.ReadFile("testdata/official-tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	raw, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(raw)) != fixture.Hash {
		t.Fatal("wrong official tokenizer identity")
	}
	tok, err := tokenizer.LoadTekkenFromBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range fixture.Cases {
		if got := tok.Encode(item.Text, false); !slices.Equal(got, item.IDs) {
			t.Fatalf("official text token IDs differ: %q", item.Text)
		}
	}
	m := &Model{tok: tok, cfg: &config{Voices: fixture.Frames}}
	if len(fixture.Cases) != 9 || len(fixture.Speech) != 20 {
		t.Fatal("incomplete official catalog fixture")
	}
	for _, item := range fixture.Speech {
		if got := m.prompt(item.Text, item.Voice); !slices.Equal(got, item.IDs) {
			t.Fatalf("official speech framing differs: %s", item.Voice)
		}
	}
}

func TestReferenceTokenizer(t *testing.T) {
	source, reference := os.Getenv("OLLAMA_VOXTRAL_TTS_SOURCE"), os.Getenv("OLLAMA_SPEECH_REFERENCE_DIR")
	if source == "" || reference == "" {
		t.Skip("set source/reference for paired Voxtral checks")
	}
	raw, err := os.ReadFile(filepath.Join(source, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = parseConfig(raw); err != nil {
		t.Fatal(err)
	}
	raw, err = os.ReadFile(filepath.Join(source, "tekken.json"))
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadTekkenFromBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	var metadata struct {
		Audio struct {
			Voices map[string]int `json:"voice_num_audio_tokens"`
		} `json:"audio"`
	}
	if err = json.Unmarshal(raw, &metadata); err != nil {
		t.Fatal(err)
	}
	raw, err = os.ReadFile(filepath.Join(reference, "summary.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		TextCases []struct {
			Raw        string  `json:"raw"`
			Normalized string  `json:"normalized"`
			IDs        []int32 `json:"ids"`
			Error      bool    `json:"error"`
		} `json:"text_cases"`
		Cases []struct {
			Text  string  `json:"normalized"`
			Voice string  `json:"voice"`
			IDs   []int32 `json:"input_ids"`
		} `json:"cases"`
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	for _, item := range fixture.TextCases {
		text, err := normalizeText(item.Raw)
		if item.Error {
			if err == nil {
				t.Errorf("accepted empty text %q", item.Raw)
			}
			continue
		}
		if err != nil || text != item.Normalized {
			t.Errorf("normalization %q: got %q (%v), want %q", item.Raw, text, err, item.Normalized)
			continue
		}
		if got := tok.Encode(text, false); !slices.Equal(got, item.IDs) {
			t.Errorf("ordinary text %q tokenization differs: got %v, want %v", text, got, item.IDs)
		}
	}
	for _, item := range fixture.Cases {
		expected := []int32{1, 25}
		for range metadata.Audio.Voices[item.Voice] {
			expected = append(expected, 24)
		}
		expected = append(expected, 36)
		expected = append(expected, tok.Encode(item.Text, false)...)
		expected = append(expected, 35, 25)
		if !slices.Equal(expected, item.IDs) {
			t.Errorf("%s tokenization differs: got %v, want %v", item.Voice, expected, item.IDs)
		}
	}
}
