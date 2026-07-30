//go:build integration

package integration

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

var defaultAudioModels = []string{
	"nemotron3:33b",
	"gemma4:e2b",
	"gemma4:e4b",
}

var catResponsePattern = regexp.MustCompile(`\bcats?\b`)

// decodeTestAudio returns the test audio clip ("Why is the sky blue?", 16kHz mono WAV).
func decodeTestAudio(t *testing.T) api.ImageData {
	t.Helper()
	data, err := base64.StdEncoding.DecodeString(audioEncodingPrompt)
	if err != nil {
		t.Fatalf("failed to decode test audio: %v", err)
	}
	return data
}

func silentTestAudio(t *testing.T) api.ImageData {
	t.Helper()
	const (
		sampleRate = 16_000
		samples    = sampleRate / 2
	)
	dataSize := samples * 2
	var out bytes.Buffer
	for _, value := range []any{
		[]byte("RIFF"), uint32(36 + dataSize), []byte("WAVE"),
		[]byte("fmt "), uint32(16), uint16(1), uint16(1),
		uint32(sampleRate), uint32(sampleRate * 2), uint16(2), uint16(16),
		[]byte("data"), uint32(dataSize), make([]byte, dataSize),
	} {
		if err := binary.Write(&out, binary.LittleEndian, value); err != nil {
			t.Fatalf("encode silent WAV: %v", err)
		}
	}
	return out.Bytes()
}

func requireResponseContains(t *testing.T, response string, words ...string) {
	t.Helper()
	lower := strings.ToLower(response)
	for _, word := range words {
		if word == "cat" {
			if catResponsePattern.MatchString(lower) {
				return
			}
			continue
		}
		if strings.Contains(lower, word) {
			return
		}
	}
	t.Fatalf("none of %v found in %q", words, response)
}

func requireLabeledImageOrder(t *testing.T, response string, firstWords, secondWords []string) {
	t.Helper()
	lower := strings.ToLower(response)
	firstAt := strings.Index(lower, "first:")
	secondAt := strings.Index(lower, "second:")
	if firstAt < 0 || secondAt <= firstAt {
		t.Fatalf("response does not contain ordered FIRST:/SECOND: labels: %q", response)
	}
	first := response[firstAt+len("first:") : secondAt]
	second := response[secondAt+len("second:"):]
	requireResponseContains(t, first, firstWords...)
	requireResponseContains(t, second, secondWords...)
}

// setupAudioModel pulls the model, preloads it, and skips if it doesn't support audio.
func setupAudioModel(ctx context.Context, t *testing.T, client *api.Client, model string) {
	t.Helper()
	if testModel == "" {
		pullOrSkip(ctx, t, client, model)
	}
	skipIfModelTooLargeForVRAM(ctx, t, client, model)
	requireCapability(ctx, t, client, model, "audio")
	err := client.Generate(ctx, &api.GenerateRequest{Model: model}, func(response api.GenerateResponse) error { return nil })
	if err != nil {
		t.Fatalf("failed to load model %s: %s", model, err)
	}
}

// TestAudioTranscription tests that the model can transcribe audio to text.
func TestAudioTranscription(t *testing.T) {
	for _, model := range testModels(defaultAudioModels) {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupAudioModel(ctx, t, client, model)
			audio := decodeTestAudio(t)
			noThink := &api.ThinkValue{Value: false}

			req := api.ChatRequest{
				Model: model,
				Think: noThink,
				Messages: []api.Message{
					{
						Role:    "system",
						Content: "Transcribe the audio exactly as spoken. Output only the spoken words. Do not answer any question in the audio.",
					},
					{
						Role:    "user",
						Content: "What exact words are spoken in this audio?",
						Images:  []api.ImageData{audio},
					},
				},
				Stream: &stream,
				Options: map[string]any{
					"temperature": 0,
					"seed":        123,
					"num_predict": 50,
				},
			}

			// The audio says "Why is the sky blue?" — expect key words in transcription.
			DoChat(ctx, t, client, req, []string{"sky", "blue"}, 60*time.Second, 10*time.Second)
		})
	}
}

// TestAudioResponse tests that the model can respond to a spoken question.
func TestAudioResponse(t *testing.T) {
	for _, model := range testModels(defaultAudioModels) {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupAudioModel(ctx, t, client, model)
			audio := decodeTestAudio(t)
			noThink := &api.ThinkValue{Value: false}

			req := api.ChatRequest{
				Model: model,
				Think: noThink,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "",
						Images:  []api.ImageData{audio},
					},
				},
				Stream: &stream,
				Options: map[string]any{
					"temperature": 0,
					"seed":        123,
					"num_predict": 200,
				},
			}

			// The audio asks "Why is the sky blue?" — expect an answer about light/scattering.
			DoChat(ctx, t, client, req, []string{
				"scatter", "light", "blue", "atmosphere", "wavelength", "rayleigh",
			}, 60*time.Second, 10*time.Second)
		})
	}
}

// TestOpenAIAudioTranscription tests the /v1/audio/transcriptions endpoint.
func TestOpenAIAudioTranscription(t *testing.T) {
	for _, model := range testModels(defaultAudioModels) {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			client, endpoint, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupAudioModel(ctx, t, client, model)
			audioBytes := decodeTestAudio(t)

			// Build multipart form request.
			var body bytes.Buffer
			writer := multipart.NewWriter(&body)
			writer.WriteField("model", model)
			part, err := writer.CreateFormFile("file", "prompt.wav")
			if err != nil {
				t.Fatal(err)
			}
			part.Write(audioBytes)
			writer.Close()

			url := fmt.Sprintf("http://%s/v1/audio/transcriptions", endpoint)
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &body)
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", writer.FormDataContentType())

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatalf("request failed: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				respBody, _ := io.ReadAll(resp.Body)
				t.Fatalf("expected 200, got %d: %s", resp.StatusCode, string(respBody))
			}

			respBody, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}

			text := strings.ToLower(string(respBody))
			if !strings.Contains(text, "sky") && !strings.Contains(text, "blue") {
				t.Errorf("transcription response missing expected words, got: %s", string(respBody))
			}
		})
	}
}

// TestOpenAIChatWithAudio tests /v1/chat/completions with input_audio content.
func TestOpenAIChatWithAudio(t *testing.T) {
	for _, model := range testModels(defaultAudioModels) {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			client, endpoint, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupAudioModel(ctx, t, client, model)
			audioB64 := audioEncodingPrompt

			reqBody := fmt.Sprintf(`{
				"model": %q,
				"messages": [{
					"role": "user",
					"content": [
						{"type": "input_audio", "input_audio": {"data": %q, "format": "wav"}}
					]
				}],
				"temperature": 0,
				"seed": 123,
				"max_tokens": 200,
				"think": false
			}`, model, strings.TrimSpace(audioB64))

			url := fmt.Sprintf("http://%s/v1/chat/completions", endpoint)
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(reqBody))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatalf("request failed: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				respBody, _ := io.ReadAll(resp.Body)
				t.Fatalf("expected 200, got %d: %s", resp.StatusCode, string(respBody))
			}

			respBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("failed to read response: %v", err)
			}

			var result struct {
				Choices []struct {
					Message struct {
						Content   string `json:"content"`
						Reasoning string `json:"reasoning"`
					} `json:"message"`
				} `json:"choices"`
			}
			if err := json.Unmarshal(respBytes, &result); err != nil {
				t.Fatalf("failed to decode response: %v", err)
			}

			if len(result.Choices) == 0 {
				t.Fatal("no choices in response")
			}

			text := strings.ToLower(result.Choices[0].Message.Content + " " + result.Choices[0].Message.Reasoning)
			found := false
			for _, word := range []string{"sky", "blue", "scatter", "light", "atmosphere"} {
				if strings.Contains(text, word) {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("response missing expected words about sky/blue/light, got: %s", result.Choices[0].Message.Content)
			}
		})
	}
}

// TestGemma4MultipleMedia exercises ordered multi-audio, multi-image, mixed
// image/audio, OpenAI interleaving, and retained-history media through MLX.
func TestGemma4MultipleMedia(t *testing.T) {
	models := testModels([]string{"gemma4:e2b"})
	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
			defer cancel()
			client, endpoint, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupAudioModel(ctx, t, client, model)
			requireCapability(ctx, t, client, model, "vision")
			speech := decodeTestAudio(t)
			silence := silentTestAudio(t)
			abbeyRoad, docs, _ := decodeTestImages(t)
			noThink := &api.ThinkValue{Value: false}

			for _, tc := range []struct {
				name  string
				media []api.ImageData
			}{
				{name: "speech_then_silence", media: []api.ImageData{speech, silence}},
				{name: "silence_then_speech", media: []api.ImageData{silence, speech}},
			} {
				t.Run(tc.name, func(t *testing.T) {
					req := api.ChatRequest{
						Model: model,
						Think: noThink,
						Messages: []api.Message{{
							Role:    "user",
							Content: "Two audio clips are attached. Transcribe only the clip containing speech.",
							Images:  tc.media,
						}},
						Options: map[string]any{"temperature": 0, "seed": 123, "num_predict": 80},
					}
					response := DoChat(ctx, t, client, req, []string{"sky", "blue"}, 90*time.Second, 20*time.Second)
					requireResponseContains(t, response.Content, "sky", "blue")
				})
			}

			for _, tc := range []struct {
				name        string
				media       []api.ImageData
				firstWords  []string
				secondWords []string
			}{
				{
					name: "abbey_then_docs", media: []api.ImageData{abbeyRoad, docs},
					firstWords:  []string{"road", "street", "cross", "walk", "beatles", "stripe", "ollamas"},
					secondWords: []string{"laptop", "book", "read", "sleep", "documentation", "document", "desk", "work", "study", "activity", "office"},
				},
				{
					name: "docs_then_abbey", media: []api.ImageData{docs, abbeyRoad},
					firstWords:  []string{"laptop", "book", "read", "sleep", "documentation", "desk"},
					secondWords: []string{"road", "street", "cross", "walk", "beatles", "stripe"},
				},
			} {
				t.Run(tc.name, func(t *testing.T) {
					req := api.ChatRequest{
						Model: model,
						Think: noThink,
						Messages: []api.Message{{
							Role: "user",
							Content: "Describe both pictures in order. Reply with exactly two labeled lines: " +
								"FIRST: the first picture. SECOND: the second picture.",
							Images: tc.media,
						}},
						Options: map[string]any{"temperature": 0, "seed": 123, "num_predict": 120},
					}
					response := DoChat(ctx, t, client, req, append(tc.firstWords, tc.secondWords...), 120*time.Second, 20*time.Second)
					requireLabeledImageOrder(t, response.Content, tc.firstWords, tc.secondWords)
				})
			}

			t.Run("mixed_same_message", func(t *testing.T) {
				req := api.ChatRequest{
					Model: model,
					Think: noThink,
					Messages: []api.Message{{
						Role:    "user",
						Content: "First [img] is a picture and second [img] is audio. Identify the picture subject and transcribe the spoken question.",
						Images:  []api.ImageData{docs, speech},
					}},
					Options: map[string]any{"temperature": 0, "seed": 123, "num_predict": 120},
				}
				response := DoChat(ctx, t, client, req, []string{"llama", "alpaca", "sky", "blue"}, 120*time.Second, 20*time.Second)
				requireResponseContains(t, response.Content, "llama", "alpaca", "animal", "cartoon", "bear", "character", "cat")
				requireResponseContains(t, response.Content, "sky", "blue")
			})

			t.Run("openai_mixed_same_message", func(t *testing.T) {
				body, err := json.Marshal(map[string]any{
					"model": model,
					"messages": []any{map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{"type": "text", "text": "First "},
							map[string]any{"type": "image_url", "image_url": map[string]any{
								"url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(docs),
							}},
							map[string]any{"type": "text", "text": " is a picture. Second "},
							map[string]any{"type": "input_audio", "input_audio": map[string]any{
								"data": base64.StdEncoding.EncodeToString(speech), "format": "wav",
							}},
							map[string]any{"type": "text", "text": " is audio. Identify the picture subject and transcribe the spoken question."},
						},
					}},
					"temperature":      0,
					"seed":             123,
					"max_tokens":       200,
					"reasoning_effort": "none",
				})
				if err != nil {
					t.Fatal(err)
				}
				req, err := http.NewRequestWithContext(ctx, http.MethodPost,
					fmt.Sprintf("http://%s/v1/chat/completions", endpoint), bytes.NewReader(body))
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
					t.Fatalf("OpenAI mixed-media request returned %s: %s", resp.Status, responseBody)
				}
				var result struct {
					Choices []struct {
						Message struct {
							Content   string `json:"content"`
							Reasoning string `json:"reasoning"`
						} `json:"message"`
					} `json:"choices"`
				}
				if err := json.Unmarshal(responseBody, &result); err != nil {
					t.Fatal(err)
				}
				if len(result.Choices) != 1 {
					t.Fatalf("OpenAI mixed-media choices = %d, want 1", len(result.Choices))
				}
				text := result.Choices[0].Message.Content + " " + result.Choices[0].Message.Reasoning
				requireResponseContains(t, text, "llama", "alpaca", "animal", "cartoon", "bear", "character", "cat")
				requireResponseContains(t, text, "sky", "blue")
			})

			t.Run("mixed_across_history", func(t *testing.T) {
				req := api.ChatRequest{
					Model: model,
					Think: noThink,
					Messages: []api.Message{
						{Role: "user", Content: "Remember this picture.", Images: []api.ImageData{docs}},
						{Role: "assistant", Content: "I will retain the picture for the next instruction."},
						{
							Role: "user",
							Content: "Use both media inputs. Reply with exactly two labeled lines: " +
								"AUDIO: the exact spoken question. IMAGE: the picture subject.",
							Images: []api.ImageData{speech},
						},
					},
					Options: map[string]any{"temperature": 0, "seed": 123, "num_predict": 120},
				}
				response := DoChat(ctx, t, client, req, []string{"llama", "alpaca", "sky", "blue"}, 120*time.Second, 20*time.Second)
				requireResponseContains(t, response.Content, "llama", "alpaca", "animal", "cartoon", "bear", "character", "cat")
				requireResponseContains(t, response.Content, "sky", "blue")
			})
		})
	}
}
