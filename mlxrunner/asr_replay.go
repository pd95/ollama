package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"golang.org/x/text/language"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
	"github.com/ollama/ollama/mlxrunner/sample"
	"github.com/ollama/ollama/model/renderers"
)

// ErrASRGenerationLimit means bounded replay reached its generation limit
// before EOS. Its partial output must never be published as a completed update.
var ErrASRGenerationLimit = errors.New("ASR generation limit reached before EOS")

const asrGenerationLimitCode = "asr_generation_limit"

// newASRReplaySession keeps replay/tokenizer state beside the resident native
// model. It adds no public route. The owner must retain its model lease until
// Session.Done, with separate contexts for lease lifetime and cancelable work.
func (r *Runner) newASRReplaySession(ctx context.Context, config stream.Config) (*stream.Session, error) {
	if _, ok := r.Model.(*qwen3asr.Model); !ok {
		return nil, errors.New("native replay requires Qwen3-ASR 0.6B")
	}
	if r.Tokenizer == nil || r.Requests == nil || r.mlxThread == nil {
		return nil, errors.New("native replay requires a loaded serialized runner")
	}
	return stream.New(ctx, config, r.Tokenizer, func(ctx context.Context, input stream.Request) (string, error) {
		wav, err := audio.EncodeWAVContext(ctx, input.Samples, stream.SampleRate)
		if err != nil {
			return "", err
		}
		prompt, err := asrReplayPrompt(input.Config, input.Prefix)
		if err != nil {
			return "", err
		}
		options := api.DefaultOptions()
		options.Temperature, options.Seed, options.NumPredict = 0, 0, 256
		options.RepeatPenalty, options.RepeatLastN = 1, 0
		request := Request{
			CompletionRequest: CompletionRequest{Prompt: prompt, Media: []llm.MediaData{llm.NewMediaData(0, wav)}, Options: options},
			Ctx:               ctx, Responses: make(chan CompletionResponse), Pipeline: r.TextGenerationPipeline,
			SamplerOpts: sample.Options{Temperature: 0, Seed: 0, UseSeed: true, RepeatPenalty: 1, TopP: 1},
		}
		if err := r.Prepare(&request); err != nil {
			return "", err
		}
		select {
		case <-ctx.Done():
			request.Grammar.close()
			return "", ctx.Err()
		case r.Requests <- request:
		}
		// The existing dispatcher closes Responses only after runRequest returns,
		// including ASR cache release and stream synchronization. A Done response
		// alone is earlier than that boundary. Continue draining on cancellation.
		return collectASRReplay(ctx, request.Responses)
	})
}

func asrReplayPrompt(config stream.Config, prefix string) (string, error) {
	source := language.Und
	switch config.Language {
	case "English":
		source = language.English
	case "German":
		source = language.German
	case "":
	default:
		return "", errors.New("unsupported replay language")
	}
	// A local renderer instance avoids the server's global image-tag setting.
	renderer := &renderers.Qwen3ASRRenderer{}
	prompt, err := renderer.AudioPrompt(renderers.AudioRequest{Task: renderers.AudioTranscribe, Source: source, Prompt: config.Context, Audio: api.ImageData{1}})
	if err != nil {
		return "", err
	}
	text, err := renderer.Render(prompt.Messages, nil, nil)
	if err != nil {
		return "", err
	}
	const pad = "<|audio_pad|>"
	index := strings.LastIndex(text, pad)
	if index < 0 {
		return "", errors.New("native replay audio marker missing")
	}
	// Continuation is internal session output only. General chat assistant
	// prefixes remain rejected by the existing public renderer contract.
	return text[:index] + "[img-0]" + text[index+len(pad):] + prefix, nil
}

func collectASRReplay(ctx context.Context, responses <-chan CompletionResponse) (string, error) {
	var text strings.Builder
	var resultErr error
	done := false
	for response := range responses {
		if response.Error != nil {
			resultErr = response.Error
		}
		if text.Len()+len(response.Content) > 32768 {
			resultErr = errors.New("native replay exceeds bounded output")
		} else if resultErr == nil {
			text.WriteString(response.Content)
		}
		if response.Done {
			done = true
			if response.DoneReason == 1 && resultErr == nil {
				resultErr = fmt.Errorf("native replay did not stop at EOS (reason %d): %w", response.DoneReason, ErrASRGenerationLimit)
			} else if response.DoneReason != 0 && resultErr == nil {
				resultErr = fmt.Errorf("native replay did not stop at EOS (reason %d)", response.DoneReason)
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if resultErr != nil {
		return "", resultErr
	}
	if !done {
		return "", errors.New("native replay ended without a completion result")
	}
	return text.String(), nil
}
