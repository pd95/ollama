//go:build integration

package integration

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestApertus15MultipleMedia(t *testing.T) {
	if testModel == "" {
		t.Skip("Apertus 1.5 multi-media validation requires OLLAMA_TEST_MODEL")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	client, endpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	setupVisionModel(ctx, t, client, testModel)
	setupAudioModel(ctx, t, client, testModel)
	abbey, docs, _ := decodeTestImages(t)
	audio := decodeTestAudio(t)
	noThink := &api.ThinkValue{Value: false}

	assertImageAndAudio := func(t *testing.T, req api.ChatRequest) {
		t.Helper()
		message := DoChat(ctx, t, client, req, []string{"4", "four"}, 120*time.Second, 30*time.Second)
		text := strings.ToLower(message.Content)
		if !strings.Contains(text, "4") && !strings.Contains(text, "four") {
			t.Fatalf("response did not describe the four animals: %q", message.Content)
		}
	}

	t.Run("speech_then_silence", func(t *testing.T) {
		req := api.ChatRequest{
			Model: testModel, Think: noThink,
			Messages: []api.Message{{Role: "user", Content: "Transcribe the spoken clip and state that the second clip is silent.", Images: []api.ImageData{audio, silentWAV(t, audio)}}},
			Options:  map[string]any{"temperature": 0, "seed": 123, "num_predict": 80},
		}
		DoChat(ctx, t, client, req, []string{"sky", "blue"}, 120*time.Second, 30*time.Second)
	})

	t.Run("abbey_then_docs", func(t *testing.T) {
		req := api.ChatRequest{
			Model: testModel, Think: noThink,
			Messages: []api.Message{{Role: "user", Content: "Compare these two images briefly. What kind of animal appears in them?", Images: []api.ImageData{abbey, docs}}},
			Options:  map[string]any{"temperature": 0, "seed": 123, "num_predict": 80},
		}
		DoChat(ctx, t, client, req, []string{"llama", "alpaca", "animal", "cat"}, 120*time.Second, 30*time.Second)
	})

	t.Run("mixed_same_message", func(t *testing.T) {
		assertImageAndAudio(t, api.ChatRequest{
			Model: testModel, Think: noThink,
			Messages: []api.Message{{Role: "user", Content: "The first media item is audio and the second is an image. What exact words are spoken, and how many animals are shown?", Images: []api.ImageData{audio, abbey}}},
			Options:  map[string]any{"temperature": 0, "seed": 123, "num_predict": 80},
		})
	})

	t.Run("mixed_across_history", func(t *testing.T) {
		assertImageAndAudio(t, api.ChatRequest{
			Model: testModel, Think: noThink,
			Messages: []api.Message{
				{Role: "user", Content: "Remember how many animals are in this image.", Images: []api.ImageData{abbey}},
				{Role: "assistant", Content: "I will remember the image."},
				{Role: "user", Content: "Now transcribe this audio and report both the earlier count and the spoken words.", Images: []api.ImageData{audio}},
			},
			Options: map[string]any{"temperature": 0, "seed": 123, "num_predict": 80},
		})
	})

	t.Run("openai_mixed_same_message", func(t *testing.T) {
		imageB64 := base64.StdEncoding.EncodeToString(abbey)
		body := fmt.Sprintf(`{
			"model": %q,
			"messages": [{"role":"user","content":[
				{"type":"text","text":"The first media item is audio and the second is an image. What exact words are spoken, and how many animals are shown?"},
				{"type":"input_audio","input_audio":{"data":%q,"format":"wav"}},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}
			]}],
			"temperature":0,"seed":123,"max_tokens":160,"think":false
		}`, testModel, strings.TrimSpace(audioEncodingPrompt), imageB64)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://%s/v1/chat/completions", endpoint), strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		responseBody, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected 200, got %d: %s", resp.StatusCode, responseBody)
		}
		var result struct {
			Choices []struct {
				Message struct {
					Content   string `json:"content"`
					Reasoning string `json:"reasoning"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(responseBody, &result); err != nil || len(result.Choices) == 0 {
			t.Fatalf("invalid OpenAI response: %s (%v)", responseBody, err)
		}
		answer := result.Choices[0].Message.Content + " " + result.Choices[0].Message.Reasoning
		text := strings.ToLower(answer)
		if strings.TrimSpace(answer) == "" || (!strings.Contains(text, "4") && !strings.Contains(text, "four") && !strings.Contains(text, "ollama")) {
			t.Fatalf("OpenAI mixed response did not contain image evidence: %q (raw %s)", answer, responseBody)
		}
	})
}

func silentWAV(t *testing.T, source []byte) api.ImageData {
	t.Helper()
	data := append([]byte(nil), source...)
	for offset := 12; offset+8 <= len(data); {
		size := int(uint32(data[offset+4]) | uint32(data[offset+5])<<8 | uint32(data[offset+6])<<16 | uint32(data[offset+7])<<24)
		start, end := offset+8, offset+8+size
		if end > len(data) {
			t.Fatal("truncated test WAV")
		}
		if string(data[offset:offset+4]) == "data" {
			clear(data[start:end])
			return data
		}
		offset = end + size%2
	}
	t.Fatal("test WAV has no data chunk")
	return nil
}
