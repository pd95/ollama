package mlxrunner

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestAudioTaskClientRequiresExactDrainAcknowledgment(t *testing.T) {
	id := strings.Repeat("a", 48)
	for _, body := range []string{
		fmt.Sprintf(`{"id":%q,"drained":false}`, id),
		`{"id":"other","drained":true}`,
		fmt.Sprintf(`{"id":%q,"drained":true,"unexpected":1}`, id),
		fmt.Sprintf(`{"id":%q,"drained":true} {}`, id),
	} {
		t.Run(body, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { fmt.Fprint(w, body) }))
			defer s.Close()
			c := &Client{port: s.Listener.Addr().(*net.TCPAddr).Port, client: s.Client()}
			if err := c.CloseAudioTask(t.Context(), id); err == nil {
				t.Fatal("invalid cleanup acknowledged")
			}
		})
	}
}

func TestAudioTaskSerializedCompletionWithoutFormat(t *testing.T) {
	request := AudioTaskRequest{Kind: "transcribe", Completion: &CompletionRequest{
		Prompt: "audio", Media: []llm.MediaData{llm.NewMediaData(0, []byte("RIFF1234WAVE"))}, Options: api.Options{NumPredict: 512},
	}}
	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	var decoded AudioTaskRequest
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if !validAudioTranscriptionTask(decoded.Completion) {
		t.Fatal("ordinary completion rejected after private IPC serialization")
	}
	decoded.Completion.Format = json.RawMessage(`"json"`)
	if validAudioTranscriptionTask(decoded.Completion) {
		t.Fatal("structured generation accepted for transcription")
	}
}

func TestAudioTaskClientRoundTrip(t *testing.T) {
	id := strings.Repeat("b", 48)
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/audio/task/reserve":
			fmt.Fprintf(w, `{"id":%q,"begin_deadline_ms":1}`, id)
		case "/v1/audio/task/run":
			fmt.Fprintf(w, `{"id":%q,"drained":true,"result":{"raw_text":"hello"}}`, id)
		case "/v1/audio/task/close":
			fmt.Fprintf(w, `{"id":%q,"drained":true}`, id)
		default:
			t.Error("unexpected task route", r.URL.Path)
		}
	}))
	defer s.Close()
	c := &Client{port: s.Listener.Addr().(*net.TCPAddr).Port, client: s.Client()}
	got, err := c.ReserveAudioTask(context.Background())
	if err != nil || got != id {
		t.Fatal(got, err)
	}
	result, err := c.RunAudioTask(t.Context(), AudioTaskRequest{ID: id, Kind: "transcribe"})
	if err != nil || result.RawText != "hello" {
		t.Fatal(result, err)
	}
	if err := c.CloseAudioTask(t.Context(), id); err != nil {
		t.Fatal(err)
	}
}
