package cmd

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner/model/audio"
)

func TestSpeechReferenceTransport(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reference.wav")
	wav := audio.EncodeWAV([]float32{.25, -.5, 0}, 24000)
	if err := os.WriteFile(path, wav, 0o600); err != nil {
		t.Fatal(err)
	}
	reference, err := readSpeechReference(path)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req llm.SpeechRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
		}
		if req.ReferenceAudio == nil || req.ReferenceAudio.Format != "wav" || !bytes.Equal(req.ReferenceAudio.Data, wav) || req.ReferenceText != "A reference." || req.Voice != "" {
			t.Errorf("reference bytes/transcript lost: %+v", req)
		}
		w.Write(make([]byte, 46))
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	if _, err := synthesize(t.Context(), "base", "", "Hello", speechOptions{ReferenceAudio: reference, ReferenceText: "A reference."}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("not WAV"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readSpeechReference(path); err == nil {
		t.Fatal("invalid reference admitted")
	}
}

func TestSpeechCut(t *testing.T) {
	long := strings.Repeat("This sentence is long enough to say. ", 3)
	for _, tc := range []struct {
		text string
		want int
	}{
		{"Too short to say yet. ", 0},
		{"Heading\n\nBody", 7},
		{long + "And then", len(long) - 1},
		{"no sentence end yet " + strings.Repeat("words ", 20), 0},
	} {
		if got := speechCut(tc.text); got != tc.want {
			t.Errorf("speechCut(%q) = %d, want %d", tc.text, got, tc.want)
		}
	}
}

// Speech starts once a few sentences arrive, before the input ends.
func TestSpeechPiecesStream(t *testing.T) {
	r, w := io.Pipe()
	pieces := speechPieces(t.Context(), r)
	go w.Write([]byte(strings.Repeat("This sentence is long enough to say. ", 3) + "Unfinished"))
	select {
	case p := <-pieces:
		if !strings.HasSuffix(p, "to say.") {
			t.Errorf("first piece = %q", p)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("speech waited for the end of the input")
	}
	go func() {
		w.Write([]byte(" tail"))
		w.Close()
	}()
	var rest []string
	for p := range pieces {
		rest = append(rest, p)
	}
	if !slices.Equal(rest, []string{"Unfinished tail"}) {
		t.Errorf("rest = %q", rest)
	}
}

func TestRunSpeechCompleteText(t *testing.T) {
	text := "Heute testen wir die Sprachausgabe mit einem längeren Absatz. Jedes Wort soll genau einmal gesprochen werden. Achten Sie auf klare Satzgrenzen und gleichmäßige Pausen. Auch der letzte Satz muss vollständig zu hören sein."
	var inputs []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct{ Input string }
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		inputs = append(inputs, body.Input)
		w.Write(audio.EncodeWAV([]float32{.1}, 24000))
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	output, err := os.CreateTemp(t.TempDir(), "speech-*.wav")
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	previous := os.Stdout
	os.Stdout = output
	defer func() { os.Stdout = previous }()
	if err := runSpeech(t.Context(), "test", "Ryan", text, nil, false); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(inputs, []string{text}) {
		t.Fatalf("complete text fragmented into requests: %q", inputs)
	}
}

func TestCompleteSpeechPiecesBounded(t *testing.T) {
	text := strings.Repeat("Grüße aus Köln! ", 400) + strings.Repeat("語", 4300)
	var got []string
	for part := range completeSpeechPieces(t.Context(), text) {
		if utf8.RuneCountInString(part) > maxSpeechInput || !utf8.ValidString(part) {
			t.Fatal("invalid API-sized piece")
		}
		got = append(got, part)
	}
	compact := func(s string) string { return strings.Join(strings.Fields(s), "") }
	if len(got) < 2 || compact(strings.Join(got, "")) != compact(text) {
		t.Fatal("long complete text lost/duplicated")
	}
}

func TestSpeechChunks(t *testing.T) {
	for _, tc := range []struct {
		text  string
		limit int
		want  []string
	}{
		{"Hello.", 100, []string{"Hello."}},
		{"One.\n\n Two.\r\n\r\nThree.", 100, []string{"One.", "Two.", "Three."}},
		{"One two. Three four. Five six.", 21, []string{"One two. Three four.", "Five six."}},
		{"one two three four", 10, []string{"one two", "three four"}},
		{"abcdefghijkl", 5, []string{"abcde", "fghij", "kl"}},
		{"ééééé", 2, []string{"éé", "éé", "é"}},
		{" \n\n ", 100, nil},
	} {
		if got := speechChunks(tc.text, tc.limit); !slices.Equal(got, tc.want) {
			t.Errorf("speechChunks(%q, %d) = %q, want %q", tc.text, tc.limit, got, tc.want)
		}
	}
}

func TestSynthesize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct{ Input, Voice string }
		if r.URL.Path != "/v1/audio/speech" || json.NewDecoder(r.Body).Decode(&req) != nil || req.Input == "" || req.Voice != "Toby" {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":{"message":"input is required"}}`))
			return
		}
		wav := make([]byte, 48)
		binary.LittleEndian.PutUint32(wav[4:], 0xffffffff)
		binary.LittleEndian.PutUint32(wav[40:], 0xffffffff)
		copy(wav[44:], []byte{1, 0, 2, 0})
		w.Write(wav)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	wav, err := synthesize(t.Context(), "phonon", "Toby", "hello")
	if err != nil || binary.LittleEndian.Uint32(wav[4:]) != 40 || binary.LittleEndian.Uint32(wav[40:]) != 4 {
		t.Fatalf("WAV lengths not recorded: %x, %v", wav, err)
	}
	if _, err := synthesize(t.Context(), "phonon", "Toby", ""); err == nil || err.Error() != "input is required" {
		t.Fatalf("server error lost: %v", err)
	}
}

func TestSynthesizeSpeechOptions(t *testing.T) {
	seed := uint64(0)
	speed := .75
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Instructions string   `json:"instructions"`
			Language     string   `json:"language"`
			Seed         *uint64  `json:"seed"`
			Voice        string   `json:"voice"`
			Speed        *float64 `json:"speed"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
		}
		if req.Instructions != "Speak cheerfully." || req.Language != "de" || req.Seed == nil || *req.Seed != 0 || req.Voice != "Ryan" || req.Speed == nil || *req.Speed != speed {
			t.Errorf("lost options: %+v", req)
		}
		w.Write(make([]byte, 46))
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	if _, err := synthesize(t.Context(), "qwen", "Ryan", "Hallo", speechOptions{Language: "de", Seed: &seed, Instructions: "Speak cheerfully.", Speed: &speed}); err != nil {
		t.Fatal(err)
	}
}
