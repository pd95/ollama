package server

import (
	"net/http"
	"os"
	"testing"

	"github.com/ollama/ollama/model/renderers"
)

func TestTimestampFormValidation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		fields map[string]string
		task   renderers.AudioTask
	}{
		{"format", map[string]string{"timestamp_granularities[]": "word"}, renderers.AudioTranscribe},
		{"stream", map[string]string{"timestamp_granularities[]": "word", "response_format": "verbose_json", "stream": "true"}, renderers.AudioTranscribe},
		{"unknown", map[string]string{"timestamp_granularities[]": "sentence", "response_format": "verbose_json"}, renderers.AudioTranscribe},
		{"unused aligner", map[string]string{"alignment_model": "aligner"}, renderers.AudioTranscribe},
		{"translation", map[string]string{"timestamp_granularities[]": "word", "response_format": "verbose_json"}, renderers.AudioTranslate},
		{"language", map[string]string{"timestamp_granularities[]": "word", "response_format": "verbose_json", "language": "fr"}, renderers.AudioTranscribe},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.fields["model"] = "asr"
			r := audioTextRequest(t, "/v1/audio/transcriptions", tc.fields, testSpeech(1, 16000))
			if err := r.ParseMultipartForm(32 << 20); err != nil {
				t.Fatal(err)
			}
			defer r.MultipartForm.RemoveAll()
			if _, err := parseAudioTextForm(r, tc.task); err == nil {
				t.Fatal("invalid timestamp options accepted")
			}
		})
	}
}

func TestTimestampUploadIsFileBacked(t *testing.T) {
	r := audioTextRequest(t, "/v1/audio/transcriptions", map[string]string{"model": "asr", "response_format": "verbose_json", "timestamp_granularities[]": "word"}, testSpeech(1, 16000))
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		t.Fatal(err)
	}
	defer r.MultipartForm.RemoveAll()
	f, err := parseAudioTextForm(r, renderers.AudioTranscribe)
	if err != nil {
		t.Fatal(err)
	}
	defer f.removeTimestampUpload()
	if len(f.audio) != 0 || f.timestampUpload == "" || !f.wordTimestamps {
		t.Fatal("timestamp upload retained in queued RAM")
	}
	if _, err := os.Stat(f.timestampUpload); err != nil {
		t.Fatal(err)
	}
	f.removeTimestampUpload()
	if _, err := os.Stat(f.timestampUpload); !os.IsNotExist(err) {
		t.Fatal("timestamp upload not removed")
	}
}

func TestTimestampSegmentOnlyPreservesExistingPath(t *testing.T) {
	s, _ := audioTestServer(t, "lfm3-audio", true, func(int, string) string { return "hello" })
	w := serveAudioText(s, audioTextRequest(t, "/v1/audio/transcriptions", map[string]string{"model": "speech-test", "response_format": "verbose_json", "timestamp_granularities[]": "segment"}, testSpeech(1, 16000)))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
}
