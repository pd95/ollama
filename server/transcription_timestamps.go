package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/text/language"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
)

func (s *Server) timestampQueue() *timestampQueue {
	s.timestampOnce.Do(func() { s.timestamps = &timestampQueue{limit: int(envconfig.MaxQueue())} })
	return s.timestamps
}

func timestampFailure(status int, message string) error {
	return api.StatusError{StatusCode: status, ErrorMessage: message}
}

func audioTextErrorStatus(err error, fallback int) int {
	var value api.StatusError
	if errors.As(err, &value) {
		return value.StatusCode
	}
	var pointer *api.StatusError
	if errors.As(err, &pointer) {
		return pointer.StatusCode
	}
	if errors.Is(err, ErrMaxQueue) {
		return http.StatusServiceUnavailable
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return http.StatusGatewayTimeout
	}
	return fallback
}

func (s *Server) timestampModel(name string, alignment bool) (*Model, error) {
	if name == "" {
		return nil, timestampFailure(400, "word timestamps require alignment_model or OLLAMA_ALIGNMENT_MODEL")
	}
	ref, err := parseAndValidateModelRef(name)
	if err != nil {
		return nil, timestampFailure(400, err.Error())
	}
	if ref.Source == modelSourceCloud {
		return nil, timestampFailure(400, "word timestamps require local native models")
	}
	m, err := s.getExistingModelForRunner(ref.Name.String(), "")
	if err != nil {
		status, message := scheduleError(name, err)
		return nil, timestampFailure(status, message)
	}
	valid := m.IsMLX() && m.Config.RemoteHost == ""
	if alignment {
		valid = valid && m.Config.ModelFamily == "qwen3_forced_aligner" && m.CheckCapabilities(model.CapabilityAlignment, model.CapabilityAudio) == nil
	} else {
		valid = valid && resolveRendererName(m) == "qwen3-asr" && m.CheckCapabilities(model.CapabilityCompletion, model.CapabilityAudio) == nil
	}
	if !valid {
		return nil, timestampFailure(400, "word timestamps require a local Qwen3-ASR model and native ForcedAligner")
	}
	return m, nil
}

func (s *Server) timestampTranscription(c *gin.Context, form audioTextForm) {
	asr, err := s.timestampModel(form.model, false)
	if err != nil {
		openAIError(c, audioTextErrorStatus(err, 400), err.Error())
		return
	}
	alignerName := form.alignmentModel
	if alignerName == "" {
		alignerName = envconfig.AlignmentModel()
	}
	aligner, err := s.timestampModel(alignerName, true)
	if err != nil {
		openAIError(c, audioTextErrorStatus(err, 400), err.Error())
		return
	}
	response, err := s.timestampQueue().run(c.Request.Context(), func(ctx context.Context) (openai.TranscriptionTimestampResponse, error) {
		return s.transcribeWithTimestamps(ctx, form, asr, aligner)
	})
	if err != nil {
		if c.Request.Context().Err() == nil {
			openAIError(c, audioTextErrorStatus(err, 500), err.Error())
		}
		return
	}
	c.JSON(http.StatusOK, response)
}

// audioTaskBackend is an optional native extension. Other runners and the
// generic llm.LlamaServer contract are intentionally unchanged.
type audioTaskBackend interface {
	ReserveAudioTask(context.Context) (string, error)
	RunAudioTask(context.Context, mlxrunner.AudioTaskRequest) (mlxrunner.AudioTaskResult, error)
	CloseAudioTask(context.Context, string) error
	AudioTaskProcessDone() <-chan struct{}
}

func drainTimestampTask(backend audioTaskBackend, id string) {
	warned := false
	for {
		select {
		case <-backend.AudioTaskProcessDone():
			return
		default:
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		err := backend.CloseAudioTask(ctx, id)
		cancel()
		if err == nil {
			return
		}
		if !warned {
			slog.Warn("native audio task awaiting positive cleanup", "error", err)
			warned = true
		}
		timer := time.NewTimer(100 * time.Millisecond)
		select {
		case <-backend.AudioTaskProcessDone():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func (s *Server) timestampStage(ctx context.Context, m *Model, build func(llm.LlamaServer) (mlxrunner.AudioTaskRequest, error)) (mlxrunner.AudioTaskResult, error) {
	lease, cancelLease := context.WithCancel(context.WithoutCancel(ctx))
	setup, cancelSetup := context.WithTimeout(ctx, envconfig.LoadTimeout())
	defer cancelSetup()
	stop := context.AfterFunc(setup, cancelLease)
	runner, err := s.queueASRRunner(lease, m)
	stopped := stop()
	if setup.Err() != nil {
		err = setup.Err()
	} else if !stopped || lease.Err() != nil {
		err = context.Canceled
	}
	if err != nil {
		cancelLease()
		return mlxrunner.AudioTaskResult{}, asrAcquisitionError(m.Name, err)
	}
	backend, ok := runner.(audioTaskBackend)
	if !ok {
		cancelLease()
		return mlxrunner.AudioTaskResult{}, timestampFailure(400, "word timestamps require the native MLX runner")
	}
	id, err := backend.ReserveAudioTask(setup)
	if err != nil {
		cancelLease()
		return mlxrunner.AudioTaskResult{}, err
	}
	defer func() { drainTimestampTask(backend, id); cancelLease() }()
	input, err := build(runner)
	if err != nil {
		return mlxrunner.AudioTaskResult{}, err
	}
	input.ID = id
	return backend.RunAudioTask(ctx, input)
}

func (s *Server) transcribeWithTimestamps(ctx context.Context, form audioTextForm, asr, aligner *Model) (openai.TranscriptionTimestampResponse, error) {
	response := openai.TranscriptionTimestampResponse{Task: "transcribe", Words: make([]openai.TranscriptionWord, 0)}
	if err := ctx.Err(); err != nil {
		return response, err
	}
	wav, err := os.ReadFile(form.timestampUpload)
	if err != nil {
		return response, err
	}
	if len(wav) > maxTimestampUploadBytes {
		return response, timestampUploadTooLarge()
	}
	if len(wav) < 12 || string(wav[:4]) != "RIFF" || string(wav[8:12]) != "WAVE" {
		return response, timestampFailure(400, "word timestamps currently require a WAV file")
	}
	duration, err := timestampWAVDuration(ctx, wav)
	if err != nil {
		return response, err
	}
	response.Duration = duration.Seconds()
	request := form.request
	request.Audio = wav
	prompter := renderers.AudioPrompterForRenderer(resolveRendererName(asr))
	prompt, err := prompter.AudioPrompt(request)
	if err != nil {
		return response, timestampFailure(400, err.Error())
	}
	opts, err := s.modelOptions(asr, map[string]any{"temperature": form.temperature, "num_predict": int64(maxAudioTextTokens(duration))})
	if err != nil {
		return response, err
	}
	transcribed, err := s.timestampStage(ctx, asr, func(r llm.LlamaServer) (mlxrunner.AudioTaskRequest, error) {
		messages := chatMessagesForRequest(asr, prompt.Messages)
		text, media, err := chatPrompt(ctx, asr, r.Tokenize, optionsForPrompt(&opts, r), messages, nil, prompt.Think, false)
		if err != nil {
			return mlxrunner.AudioTaskRequest{}, err
		}
		return mlxrunner.AudioTaskRequest{Kind: "transcribe", Completion: &mlxrunner.CompletionRequest{Prompt: text, Media: media, Options: opts}}, nil
	})
	if err != nil {
		return response, err
	}
	response.Text = prompter.AudioText(request, transcribed.RawText)
	name := renderers.LanguageName(request.Source)
	if request.Source == language.Und {
		raw := strings.TrimSpace(transcribed.RawText)
		if strings.HasPrefix(raw, "language ") {
			if header, _, found := strings.Cut(raw, "<asr_text>"); found {
				name = strings.TrimSpace(strings.TrimPrefix(header, "language "))
			}
		}
	}
	if form.segmentTimestamps {
		segments := make([]openai.TranscriptionSegment, 0)
		if response.Text != "" {
			segments = append(segments, openai.TranscriptionSegment{ID: 0, Start: 0, End: response.Duration, Text: response.Text})
		}
		response.Segments = &segments
	}
	if name == "English" || name == "German" {
		response.Language = strings.ToLower(name)
	}
	if response.Text == "" {
		return response, nil
	}
	if name != "English" && name != "German" {
		return response, timestampFailure(422, "nonempty word-timestamp transcription requires detected English or German")
	}
	units, _, err := qwen3asr.AlignmentPrompt(response.Text, name)
	if err != nil {
		return response, timestampFailure(422, err.Error())
	}
	aligned, err := s.timestampStage(ctx, aligner, func(llm.LlamaServer) (mlxrunner.AudioTaskRequest, error) {
		return mlxrunner.AudioTaskRequest{Kind: "align", Audio: wav, Text: response.Text, Language: name}, nil
	})
	if err != nil {
		return response, err
	}
	words, err := timestampWords(response.Text, units, aligned.Alignment, response.Duration)
	if err != nil {
		return response, err
	}
	response.Words = words
	return response, nil
}

func timestampWAVDuration(ctx context.Context, wav []byte) (time.Duration, error) {
	pcm, rate, err := audio.DecodeLimitContext(ctx, wav, 60)
	if err != nil {
		return 0, timestampFailure(400, err.Error())
	}
	if len(pcm) > 60*rate {
		return 0, timestampFailure(400, "word timestamps currently accept recordings up to 60 seconds")
	}
	return time.Duration(len(pcm)) * time.Second / time.Duration(rate), nil
}

func timestampWords(text string, units []qwen3asr.AlignmentUnit, result *qwen3asr.AlignmentResult, duration float64) ([]openai.TranscriptionWord, error) {
	if result == nil || result.SourceText != text || len(result.Items) != len(units) || len(result.Classes) != 2*len(units) {
		return nil, timestampFailure(502, "native alignment returned incomplete results")
	}
	words := make([]openai.TranscriptionWord, 0, len(units))
	previous := float64(0)
	for i, item := range result.Items {
		if item.AlignmentUnit != units[i] {
			return nil, timestampFailure(502, "native alignment changed normalized source spans")
		}
		for _, v := range []int32{result.Classes[2*i], result.Classes[2*i+1]} {
			if v < 0 || v >= 5000 {
				return nil, timestampFailure(502, "native alignment returned invalid classes")
			}
		}
		if math.IsNaN(item.Start) || math.IsNaN(item.End) || math.IsInf(item.Start, 0) || math.IsInf(item.End, 0) || item.Start < previous || item.End < item.Start || item.End > duration || item.OutsideAudio {
			return nil, timestampFailure(422, fmt.Sprintf("alignment produced unusable timestamps for word %d", i))
		}
		words = append(words, openai.TranscriptionWord{Word: item.Text, Start: item.Start, End: item.End})
		previous = item.End
	}
	return words, nil
}
