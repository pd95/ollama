// Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
// SPDX-License-Identifier: Apache-2.0
// Native prompt adaptation of the pinned original Qwen3-ASR processor.

package renderers

import (
	"fmt"
	"strings"
	"time"

	"golang.org/x/text/language"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

type Qwen3ASRRenderer struct{ useImgTags bool }

func (*Qwen3ASRRenderer) LeadingBOS() string            { return "" }
func (*Qwen3ASRRenderer) Thinking() *model.Thinking     { return nil }
func (*Qwen3ASRRenderer) AudioClipLimit() time.Duration { return 60 * time.Second }

func (*Qwen3ASRRenderer) AudioPrompt(req AudioRequest) (AudioPrompt, error) {
	if req.Task != AudioTranscribe {
		return AudioPrompt{}, fmt.Errorf("Qwen3-ASR supports transcription, not translation")
	}
	var name string
	if req.Source != language.Und {
		name = LanguageName(req.Source)
		if name != "English" && name != "German" {
			return AudioPrompt{}, fmt.Errorf("Qwen3-ASR currently supports English, German or automatic language detection")
		}
	}
	messages := []api.Message{{Role: "system", Content: req.Prompt}, {Role: "user", Images: []api.ImageData{req.Audio}}}
	if name != "" {
		messages = append(messages, api.Message{Role: "assistant", Content: "language " + name + "<asr_text>"})
	}
	return AudioPrompt{Messages: messages}, nil
}

func (*Qwen3ASRRenderer) AudioText(_ AudioRequest, output string) string {
	output = strings.TrimSpace(output)
	if strings.HasPrefix(output, "language ") {
		if _, body, found := strings.Cut(output, "<asr_text>"); found {
			output = body
		}
	}
	return strings.TrimSpace(output)
}

func (r *Qwen3ASRRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if len(tools) != 0 || (think != nil && think.Value != false) {
		return "", fmt.Errorf("Qwen3-ASR does not support tools or thinking")
	}
	var context, prefill string
	count := 0
	for i, message := range messages {
		switch message.Role {
		case "system":
			context += message.Content
		case "user":
			if len(message.Images) != 1 || count != 0 {
				return "", fmt.Errorf("Qwen3-ASR requires one audio clip per generation")
			}
			content := strings.TrimSpace(message.Content)
			if content != "" && content != "[img]" && content != "[img-0]" {
				return "", fmt.Errorf("Qwen3-ASR context belongs in the system message")
			}
			count++
		case "assistant":
			if i != len(messages)-1 || (message.Content != "language English<asr_text>" && message.Content != "language German<asr_text>") {
				return "", fmt.Errorf("unsupported Qwen3-ASR assistant prefix")
			}
			prefill = message.Content
		default:
			return "", fmt.Errorf("Qwen3-ASR does not support message role %q", message.Role)
		}
	}
	if count != 1 {
		return "", fmt.Errorf("Qwen3-ASR requires one audio clip")
	}
	marker := "<|audio_pad|>"
	if r.useImgTags {
		marker = "[img-0]"
	}
	// The original checkpoint's rendered template contains newline characters.
	return "<|im_start|>system\n" + context + "<|im_end|>\n<|im_start|>user\n<|audio_start|>" + marker + "<|audio_end|><|im_end|>\n<|im_start|>assistant\n" + prefill, nil
}
