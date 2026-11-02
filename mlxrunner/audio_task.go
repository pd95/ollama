package mlxrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx/mlxthread"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
	"github.com/ollama/ollama/mlxrunner/sample"
)

const maxAudioTaskBody = 48 << 20

func (r *Runner) installAudioTaskRoutes(ctx context.Context, mux *http.ServeMux) (func(), error) {
	if _, ok := r.Model.(*qwen3asr.Model); !ok {
		return func() {}, nil
	}
	registry, err := newAudioTaskRegistry(ctx, r.executeAudioTask)
	if err != nil {
		return nil, err
	}
	for _, action := range []string{"reserve", "run", "close"} {
		mux.HandleFunc("POST /v1/audio/task/"+action, func(w http.ResponseWriter, req *http.Request) {
			var input AudioTaskRequest
			limit := int64(1024)
			if action == "run" {
				limit = maxAudioTaskBody
			}
			decoder := json.NewDecoder(http.MaxBytesReader(w, req.Body, limit))
			decoder.DisallowUnknownFields()
			if err := decoder.Decode(&input); err != nil {
				http.Error(w, err.Error(), 400)
				return
			}
			if err := decoder.Decode(new(any)); err != io.EOF {
				http.Error(w, "expected one audio task request", 400)
				return
			}
			var output any
			var err error
			switch action {
			case "reserve":
				id, until, reserveErr := registry.reserve()
				err = reserveErr
				output = map[string]any{"id": id, "begin_deadline_ms": until.UnixMilli()}
			case "run":
				var result AudioTaskResult
				result, err = registry.run(req.Context(), input)
				output = audioTaskReply{ID: input.ID, Drained: true, Result: &result}
			case "close":
				err = registry.close(req.Context(), input.ID)
				output = audioTaskReply{ID: input.ID, Drained: true}
			}
			if err != nil {
				status := http.StatusUnprocessableEntity
				switch {
				case errors.Is(err, errAudioTaskBusy), errors.Is(err, errAudioTaskStopped):
					status = 503
				case errors.Is(err, errAudioTaskUnknown):
					status = 404
				case errors.Is(err, context.Canceled):
					status = 408
				case errors.Is(err, context.DeadlineExceeded):
					status = 504
				}
				var typed api.StatusError
				if errors.As(err, &typed) {
					status = typed.StatusCode
				}
				http.Error(w, err.Error(), status)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(output)
		})
	}
	return registry.shutdown, nil
}

type audioTaskReply struct {
	ID      string           `json:"id"`
	Drained bool             `json:"drained"`
	Result  *AudioTaskResult `json:"result,omitempty"`
}

func (r *Runner) executeAudioTask(ctx context.Context, input AudioTaskRequest) (AudioTaskResult, error) {
	m, ok := r.Model.(*qwen3asr.Model)
	if !ok {
		return AudioTaskResult{}, errors.New("native audio tasks require Qwen3-ASR or ForcedAligner")
	}
	switch input.Kind {
	case "align":
		if !m.IsForcedAligner() || input.Completion != nil || len(input.Audio) == 0 || len(input.Audio) > 32<<20 {
			return AudioTaskResult{}, errors.New("invalid native alignment task")
		}
		result, err := mlxthread.Call(ctx, r.mlxThread, func() (*qwen3asr.AlignmentResult, error) {
			return m.AlignContext(ctx, input.Audio, input.Text, input.Language)
		})
		return AudioTaskResult{Alignment: result}, err
	case "transcribe":
		c := input.Completion
		if m.IsForcedAligner() || !validAudioTranscriptionTask(c) {
			return AudioTaskResult{}, errors.New("invalid native transcription task")
		}
		req := Request{
			CompletionRequest: *c, Ctx: ctx, Responses: make(chan CompletionResponse), Pipeline: r.TextGenerationPipeline,
			SamplerOpts: sample.Options{
				Temperature: c.Options.Temperature, TopP: c.Options.TopP, MinP: c.Options.MinP, TopK: c.Options.TopK,
				RepeatLastN: c.Options.RepeatLastN, RepeatPenalty: c.Options.RepeatPenalty, PresencePenalty: c.Options.PresencePenalty,
				FrequencyPenalty: c.Options.FrequencyPenalty, Seed: c.Options.Seed, UseSeed: c.Options.Seed >= 0,
			},
		}
		if err := r.Prepare(&req); err != nil {
			return AudioTaskResult{}, err
		}
		select {
		case <-ctx.Done():
			req.Grammar.close()
			return AudioTaskResult{}, ctx.Err()
		case r.Requests <- req:
		}
		text, err := collectASRReplay(ctx, req.Responses)
		return AudioTaskResult{RawText: text}, err
	default:
		return AudioTaskResult{}, errors.New("unknown native audio task kind")
	}
}

func validAudioTranscriptionTask(c *CompletionRequest) bool {
	return c != nil && len(c.Media) == 1 && len(c.Media[0].Data) > 0 && len(c.Media[0].Data) <= 32<<20 && len(c.Prompt) <= 64<<10 && c.Options.NumPredict > 0 && c.Options.NumPredict <= 2048 && (len(c.Format) == 0 || bytes.Equal(bytes.TrimSpace(c.Format), []byte("null")))
}
