//go:build integration

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/openai"
)

func init() {
	registerIntegrationCases(integrationTestCase("openai-audio-timestamps", "", runOpenAIAudioTimestamps))
}

func requireInstalledAudioPair(ctx context.Context, client *api.Client, asr, aligner string) error {
	if strings.TrimSpace(asr) == "" {
		return fmt.Errorf("OLLAMA_TEST_MODEL must select an installed ASR model")
	}
	for _, item := range []struct{ name, capability string }{{asr, "completion"}, {aligner, "alignment"}} {
		shown, err := client.Show(ctx, &api.ShowRequest{Name: item.name})
		if err != nil {
			return fmt.Errorf("selected model %s must already be installed: %w", item.name, err)
		}
		audio, task := false, false
		for _, cap := range shown.Capabilities {
			audio = audio || string(cap) == "audio"
			task = task || string(cap) == item.capability
		}
		if !audio || !task {
			return fmt.Errorf("model %s needs audio and %s capabilities", item.name, item.capability)
		}
	}
	return nil
}

func runOpenAIAudioTimestamps(t *testing.T) {
	aligner := strings.TrimSpace(os.Getenv("OLLAMA_TEST_ALIGNMENT_MODEL"))
	if aligner == "" {
		t.Skip("set OLLAMA_TEST_MODEL and OLLAMA_TEST_ALIGNMENT_MODEL to installed models")
	}
	if strings.TrimSpace(testModel) == "" {
		t.Fatal("OLLAMA_TEST_MODEL must select an installed ASR model")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	client, endpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := requireInstalledAudioPair(ctx, client, testModel, aligner); err != nil {
		t.Fatal(err)
	}
	clip := decodeTestAudio(t)
	baseline := postAudio(ctx, t, endpoint, "transcriptions", map[string]string{"model": testModel, "language": "en", "temperature": "0"}, clip)
	requireWords(t, strings.ToLower(baseline.Text), "sky", "blue")
	for _, tc := range []struct {
		name          string
		granularities []string
		language      string
		silence       bool
	}{
		{"words", []string{"word"}, "en", false},
		{"words and segments", []string{"word", "segment"}, "en", false},
		{"segments", []string{"segment"}, "en", false},
		{"automatic", []string{"word"}, "", false},
		{"silence", []string{"word"}, "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fields := url.Values{"model": {testModel}, "temperature": {"0"}, "response_format": {"verbose_json"}, "timestamp_granularities[]": tc.granularities}
			wantsWords, wantsSegments := false, false
			for _, g := range tc.granularities {
				wantsWords = wantsWords || g == "word"
				wantsSegments = wantsSegments || g == "segment"
			}
			if wantsWords {
				fields.Set("alignment_model", aligner)
			}
			if tc.language != "" {
				fields.Set("language", tc.language)
			}
			wav := clip
			if tc.silence {
				// Match the existing one-second native API silence control.
				wav = audio.EncodeWAV(make([]float32, 16000), 16000)
			}
			raw := postAudioFields(ctx, t, endpoint, "transcriptions", fields, wav)
			var members map[string]json.RawMessage
			if err := json.Unmarshal(raw, &members); err != nil {
				t.Fatal(err)
			}
			var result openai.TranscriptionTimestampResponse
			if err := json.Unmarshal(raw, &result); err != nil {
				t.Fatal(err)
			}
			_, hasWords := members["words"]
			_, hasSegments := members["segments"]
			if hasWords != wantsWords || hasSegments != wantsSegments {
				t.Fatalf("unexpected response members: %s", raw)
			}
			if wantsWords && wantsSegments {
				fields.Set("stream", "true")
				verifyTimestampStream(t, postAudioFields(ctx, t, endpoint, "transcriptions", fields, wav), result)
			}
			if tc.silence {
				if result.Text != "" || result.Words == nil || len(result.Words) != 0 {
					t.Fatalf("expected empty transcription and word array: %s", raw)
				}
			} else {
				if result.Text != baseline.Text {
					t.Fatalf("timestamp request changed text: %q versus %q", result.Text, baseline.Text)
				}
				if wantsWords && len(result.Words) == 0 {
					t.Fatal("speech has no word entries")
				}
			}
			if wantsWords {
				if err := validAudioWordIntervals(result.Words, result.Duration); err != nil {
					t.Fatal(err)
				}
			}
			if wantsSegments && (result.Segments == nil || len(*result.Segments) == 0) {
				t.Fatal("missing segments")
			}
		})
	}
	t.Run("long WAV and MP3", func(t *testing.T) {
		samples, rate, err := audio.Decode(clip)
		if err != nil {
			t.Fatal(err)
		}
		referenceRaw := postAudioFields(ctx, t, endpoint, "transcriptions", url.Values{"model": {testModel}, "alignment_model": {aligner}, "language": {"en"}, "response_format": {"verbose_json"}, "timestamp_granularities[]": {"word"}}, clip)
		var reference openai.TranscriptionTimestampResponse
		if err := json.Unmarshal(referenceRaw, &reference); err != nil {
			t.Fatal(err)
		}
		firstStart := -1.0
		for _, w := range reference.Words {
			if strings.EqualFold(w.Word, "sky") && w.End > w.Start {
				firstStart = 30 - (w.Start+w.End)/2
				break
			}
		}
		if firstStart < 25 || firstStart >= 30 {
			t.Fatalf("cannot place complete sky across boundary: %s", referenceRaw)
		}
		pcm := make([]float32, 95*rate)
		// The first sentence crosses the30-second ownership seam. Later sentences
		// prove offsets beyond the old60-second limit without repeating context.
		for _, start := range []float64{firstStart, 63, 90} {
			copy(pcm[int(start*float64(rate)):], samples)
		}
		wav := audio.EncodeWAV(pcm, rate)
		encoder, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		cmd := exec.CommandContext(encoder, "ffmpeg", "-nostdin", "-v", "error", "-f", "wav", "-i", "pipe:0", "-f", "mp3", "-codec:a", "libmp3lame", "-b:a", "64k", "pipe:1")
		cmd.Stdin = bytes.NewReader(wav)
		mp3, err := cmd.Output()
		if err != nil {
			t.Fatalf("fixed MP3 regression fixture generation requires ffmpeg: %v", err)
		}
		for _, input := range []struct {
			name string
			data []byte
		}{{"WAV", wav}, {"MP3", mp3}} {
			t.Run(input.name, func(t *testing.T) {
				raw := postAudioFields(ctx, t, endpoint, "transcriptions", url.Values{"model": {testModel}, "alignment_model": {aligner}, "language": {"en"}, "temperature": {"0"}, "response_format": {"verbose_json"}, "timestamp_granularities[]": {"word", "segment"}}, input.data)
				var result openai.TranscriptionTimestampResponse
				if err := json.Unmarshal(raw, &result); err != nil {
					t.Fatal(err)
				}
				streamRaw := postAudioFields(ctx, t, endpoint, "transcriptions", url.Values{"model": {testModel}, "alignment_model": {aligner}, "language": {"en"}, "temperature": {"0"}, "response_format": {"verbose_json"}, "timestamp_granularities[]": {"word", "segment"}, "stream": {"true"}}, input.data)
				verifyTimestampStream(t, streamRaw, result)
				if result.Duration < 95 || result.Duration > 96 {
					t.Fatalf("lost recording duration: %g", result.Duration)
				}
				if err := validAudioWordIntervals(result.Words, result.Duration); err != nil {
					t.Fatal(err)
				}
				crossing := false
				for _, w := range result.Words {
					if strings.EqualFold(w.Word, "sky") && w.Start < 30 && w.End > 30 {
						crossing = true
					}
				}
				if !crossing {
					t.Fatalf("boundary word sky was not preserved whole across30seconds: %s", raw)
				}
				t.Logf("timestamp-regression %s %s", input.name, raw)
				for _, word := range []string{"why", "sky", "blue"} {
					count := 0
					for _, w := range result.Words {
						if strings.EqualFold(w.Word, word) {
							count++
						}
					}
					if count != 3 {
						t.Fatalf("boundary omission/duplication: word %s count%d, response %s", word, count, raw)
					}
				}
				for _, start := range []float64{firstStart, 63, 90} {
					found := false
					for _, w := range result.Words {
						if w.Start >= start-1 && w.Start < start+5 && strings.EqualFold(w.Word, "blue") {
							found = true
						}
					}
					if !found {
						t.Fatalf("no globally offset blue near%gs: %s", start, raw)
					}
				}
				if result.Segments == nil || len(*result.Segments) < 3 {
					t.Fatalf("missing chunk segments %s", raw)
				}
				for i, seg := range *result.Segments {
					if seg.ID != i || seg.Start < 0 || seg.End < seg.Start || seg.End > result.Duration {
						t.Fatalf("invalid global segment %+v", seg)
					}
				}
			})
		}
	})
}

func validAudioWordIntervals(words []openai.TranscriptionWord, duration float64) error {
	if math.IsNaN(duration) || math.IsInf(duration, 0) || duration <= 0 {
		return fmt.Errorf("invalid recording duration")
	}
	previous := 0.0
	for _, w := range words {
		if strings.TrimSpace(w.Word) == "" || math.IsNaN(w.Start) || math.IsNaN(w.End) || math.IsInf(w.Start, 0) || math.IsInf(w.End, 0) || w.Start < previous || w.End < w.Start || w.End > duration {
			return fmt.Errorf("invalid word interval: %+v", w)
		}
		previous = w.End
	}
	return nil
}

func TestAudioTimestampHelpers(t *testing.T) {
	t.Run("repeated fields and raw members", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if err := r.ParseMultipartForm(1 << 20); err != nil {
				t.Error(err)
				w.WriteHeader(400)
				return
			}
			defer r.MultipartForm.RemoveAll()
			values := r.MultipartForm.Value["timestamp_granularities[]"]
			if len(values) != 2 || values[0] != "word" || values[1] != "segment" {
				t.Errorf("lost repeated granularities: %v", values)
			}
			_, _ = w.Write([]byte(`{"words":[],"segments":[]}`))
		}))
		defer server.Close()
		raw := postAudioFields(t.Context(), t, strings.TrimPrefix(server.URL, "http://"), "transcriptions", url.Values{"timestamp_granularities[]": {"word", "segment"}}, []byte("fixture"))
		if string(raw) != `{"words":[],"segments":[]}` {
			t.Fatal("raw response changed")
		}
	})
	t.Run("missing selected model never pulls", func(t *testing.T) {
		var calls atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			if r.URL.Path != "/api/show" {
				t.Errorf("unexpected request %s", r.URL.Path)
			}
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":"missing"}`))
		}))
		defer server.Close()
		endpoint, _ := url.Parse(server.URL)
		client := api.NewClient(endpoint, http.DefaultClient)
		if requireInstalledAudioPair(t.Context(), client, "asr", "aligner") == nil || calls.Load() != 1 {
			t.Fatal("missing model must fail after Show only")
		}
		if requireInstalledAudioPair(t.Context(), client, "", "aligner") == nil || calls.Load() != 1 {
			t.Fatal("missing ASR selection must fail before requests")
		}
	})
	t.Run("intervals", func(t *testing.T) {
		if err := validAudioWordIntervals([]openai.TranscriptionWord{{Word: "a", Start: 0, End: 0}, {Word: "b", Start: .1, End: .5}}, 1); err != nil {
			t.Fatal(err)
		}
		for _, words := range [][]openai.TranscriptionWord{
			{{Word: "a", Start: .2, End: .1}}, {{Word: "a", Start: 0, End: 2}}, {{Word: "a", Start: 0, End: math.NaN()}}, {{Word: "", Start: 0, End: .1}}, {{Word: "a", Start: 0, End: .5}, {Word: "b", Start: .4, End: .6}},
		} {
			if validAudioWordIntervals(words, 1) == nil {
				t.Fatalf("invalid intervals accepted: %+v", words)
			}
		}
	})
	t.Run("missing aligner and incompatible capabilities", func(t *testing.T) {
		for _, missing := range []bool{true, false} {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.URL.Path != "/api/show" {
					t.Errorf("unexpected request %s", r.URL.Path)
				}
				if calls.Load() == 2 && missing {
					w.WriteHeader(http.StatusNotFound)
					_, _ = w.Write([]byte(`{"error":"missing"}`))
					return
				}
				_, _ = w.Write([]byte(`{"capabilities":["completion","audio"]}`))
			}))
			endpoint, _ := url.Parse(server.URL)
			client := api.NewClient(endpoint, http.DefaultClient)
			err := requireInstalledAudioPair(t.Context(), client, "asr", "aligner")
			server.Close()
			if err == nil || calls.Load() != 2 {
				t.Fatalf("invalid aligner must fail after Show only: %v, %d requests", err, calls.Load())
			}
		}
	})
}

func verifyTimestampStream(t *testing.T, raw []byte, baseline openai.TranscriptionTimestampResponse) {
	t.Helper()
	var text strings.Builder
	words := make([]openai.TranscriptionWord, 0)
	segments := make([]openai.TranscriptionSegment, 0)
	completed := false
	progress := false
	totalKnown := false
	for _, line := range strings.Split(string(raw), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var e struct {
			Type     string                                `json:"type"`
			Delta    string                                `json:"delta"`
			Offset   int                                   `json:"word_offset"`
			Words    []openai.TranscriptionWord            `json:"words"`
			Segments []openai.TranscriptionSegment         `json:"segments"`
			Total    *float64                              `json:"total_seconds"`
			Result   openai.TranscriptionTimestampResponse `json:"result"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &e); err != nil {
			t.Fatal(err)
		}
		if completed {
			t.Fatal("event after completion")
		}
		switch e.Type {
		case "transcript.progress":
			progress = true
			totalKnown = totalKnown || e.Total != nil
		case "transcript.text.delta":
			text.WriteString(e.Delta)
		case "transcript.words.delta":
			if e.Offset != len(words) {
				t.Fatal("noncontiguous words")
			}
			words = append(words, e.Words...)
			segments = append(segments, e.Segments...)
		case "transcript.text.done":
			completed = true
			if text.String() != e.Result.Text || !reflect.DeepEqual(words, e.Result.Words) || e.Result.Segments == nil || !reflect.DeepEqual(segments, *e.Result.Segments) {
				t.Fatalf("stream prefix differs from final result: %s", raw)
			}
			if !reflect.DeepEqual(e.Result, baseline) {
				t.Fatalf("stream differs from JSON: %s", raw)
			}
		case "error":
			t.Fatalf("stream failed: %s", raw)
		default:
			t.Fatalf("unknown event: %s", line)
		}
	}
	if !completed || !progress || !totalKnown {
		t.Fatalf("incomplete stream: %s", raw)
	}
}
