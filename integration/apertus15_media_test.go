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
		message := DoChat(ctx, t, client, req, []string{"sky", "blue"}, 120*time.Second, 30*time.Second)
		text := strings.ToLower(message.Content)
		if !strings.Contains(text, "4") && !strings.Contains(text, "four") {
			t.Fatalf("response did not describe the four animals: %q", message.Content)
		}
	}

	t.Run("speech_then_silence", func(t *testing.T) {
		req := api.ChatRequest{
			Model: testModel, Think: noThink,
			Messages: []api.Message{{Role: "user", Content: "Which clip contains speech, the first or the second? Include the exact spoken words in your answer.", Images: []api.ImageData{audio, silentWAV(t, audio)}}},
			Options:  map[string]any{"temperature": 0, "seed": 123, "num_predict": 80},
		}
		message := DoChat(ctx, t, client, req, []string{"sky", "blue"}, 120*time.Second, 30*time.Second)
		if !strings.Contains(strings.ToLower(message.Content), "first") {
			t.Fatalf("response did not distinguish the first speech clip from the second silent clip: %q", message.Content)
		}
	})

	t.Run("abbey_then_docs", func(t *testing.T) {
		req := api.ChatRequest{
			Model: testModel, Think: noThink,
			Messages: []api.Message{{Role: "user", Content: "Answer in order: (1) What is the first image an album cover for? (2) How many animals are in the second image?", Images: []api.ImageData{abbey, docs}}},
			Options:  map[string]any{"temperature": 0, "seed": 123, "num_predict": 80},
		}
		message := DoChat(ctx, t, client, req, []string{"ollama", "album"}, 120*time.Second, 30*time.Second)
		if text := strings.ToLower(message.Content); !strings.Contains(text, "4") && !strings.Contains(text, "four") && !strings.Contains(text, "5") && !strings.Contains(text, "five") {
			t.Fatalf("response did not count the animals in the second image: %q", message.Content)
		}
	})

	t.Run("mixed_same_message", func(t *testing.T) {
		assertImageAndAudio(t, api.ChatRequest{
			Model: testModel, Think: noThink,
			Messages: []api.Message{
				{Role: "system", Content: "Transcribe words only from the attached WAV audio, never from text visible in an image."},
				{Role: "user", Content: "The first item is audio and the second is an image. What exact words are spoken, and how many animals are shown?", Images: []api.ImageData{audio, docs}},
			},
			Options: map[string]any{"temperature": 0, "seed": 123, "num_predict": 80},
		})
	})

	t.Run("mixed_across_history", func(t *testing.T) {
		assertImageAndAudio(t, api.ChatRequest{
			Model: testModel, Think: noThink,
			Messages: []api.Message{
				{Role: "user", Content: "Remember the exact words spoken in this audio.", Images: []api.ImageData{audio}},
				{Role: "assistant", Content: "Audio received."},
				{Role: "user", Content: "Now count the animals in this image and report both the count and the earlier spoken words.", Images: []api.ImageData{docs}},
			},
			Options: map[string]any{"temperature": 0, "seed": 123, "num_predict": 80},
		})
	})

	t.Run("openai_mixed_same_message", func(t *testing.T) {
		imageB64 := base64.StdEncoding.EncodeToString(docs)
		body := fmt.Sprintf(`{
			"model": %q,
			"messages": [{"role":"user","content":[
				{"type":"text","text":"The first item is an image and the second is audio. How many animals are shown, and what exact words are spoken?"},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},
				{"type":"input_audio","input_audio":{"data":%q,"format":"wav"}}
			]}],
			"temperature":0,"seed":123,"max_tokens":400,"think":false
		}`, testModel, imageB64, strings.TrimSpace(audioEncodingPrompt))
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
			Usage struct {
				PromptTokens int `json:"prompt_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(responseBody, &result); err != nil || len(result.Choices) == 0 {
			t.Fatalf("invalid OpenAI response: %s (%v)", responseBody, err)
		}
		answer := result.Choices[0].Message.Content + " " + result.Choices[0].Message.Reasoning
		text := strings.ToLower(answer)
		if !strings.Contains(text, "4") && !strings.Contains(text, "four") {
			t.Fatalf("OpenAI mixed response did not use the image: %q (raw %s)", answer, responseBody)
		}
		// The deterministic request is 617 prompt tokens with both inputs. A
		// count above 600 proves the later audio placeholder span was expanded
		// even when this small model elects to discuss only the image.
		if result.Usage.PromptTokens < 600 {
			t.Fatalf("OpenAI mixed prompt used %d tokens, want at least 600 to include both media spans (raw %s)", result.Usage.PromptTokens, responseBody)
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
