package kokoro

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/model/kokoro/config"
	"github.com/ollama/ollama/mlxrunner/model/kokoro/english"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() { model.Register(config.Architecture, newModel) }

// Model synthesizes English speech without Python or external phonemizers.
// Arrays keeps source and derived weights visible to the runner's collector.
type Model struct {
	Arrays   []*mlx.Array
	cfg      *config.Config
	quant    map[string]*model.TensorQuantInfo
	voices   map[string]*mlx.Array
	encoders *Encoders
	prosody  *Prosody
	decoder  *Decoder
}

func newModel(root *model.Root) (model.Model, error) {
	raw, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	cfg, err := config.Parse(raw)
	if err != nil {
		return nil, err
	}
	if len(cfg.Voices) == 0 {
		return nil, fmt.Errorf("Kokoro requires imported English voices")
	}
	return &Model{cfg: cfg, quant: root.AllTensorQuant()}, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	var err error
	m.encoders, err = LoadEncoders(m.cfg, tensors, m.quant)
	if err != nil {
		return err
	}
	m.prosody, err = LoadProsody(m.cfg, tensors, m.quant)
	if err != nil {
		return err
	}
	m.decoder, err = LoadDecoder(m.cfg, tensors, m.quant)
	if err != nil {
		return err
	}
	m.voices = map[string]*mlx.Array{}
	for _, name := range m.cfg.Voices {
		a := tensors["voices."+name]
		if a == nil || a.NumDims() != 3 || a.Dim(0) != 510 || a.Dim(1) != 1 || a.Dim(2) != 256 || a.DType() != mlx.DTypeFloat32 {
			return fmt.Errorf("invalid Kokoro voice pack %q", name)
		}
		if !(strings.HasPrefix(name, "af_") || strings.HasPrefix(name, "am_") || strings.HasPrefix(name, "bf_") || strings.HasPrefix(name, "bm_")) {
			return fmt.Errorf("unsupported Kokoro voice %q", name)
		}
		m.voices[name] = a
	}
	if m.voices["af_heart"] == nil {
		return fmt.Errorf("Kokoro requires af_heart")
	}
	for _, a := range tensors {
		m.Arrays = append(m.Arrays, a)
	}
	m.Arrays = append(m.Arrays, m.encoders.Derived...)
	m.Arrays = append(m.Arrays, m.prosody.Derived...)
	m.Arrays = append(m.Arrays, m.decoder.Derived...)
	return nil
}

func (*Model) NewCaches() []cache.Cache { return nil }
func (*Model) Forward(*batch.Batch, []cache.Cache) (*mlx.Array, *mlx.Array) {
	panic("Kokoro is speech-only")
}
func (*Model) Unembed(x *mlx.Array) *mlx.Array { return x }
func (*Model) Tokenizer() *tokenizer.Tokenizer { return nil }
func (*Model) MaxContextLength() int           { return 0 }

func badRequest(format string, args ...any) error {
	return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf(format, args...)}
}

func (m *Model) ValidateSpeechRequest(req llm.SpeechRequest) error {
	if req.ReferenceAudio != nil || req.ReferenceText != "" {
		return badRequest("Kokoro does not support reference audio")
	}
	if req.Instructions != "" {
		return badRequest("Kokoro does not support instructions")
	}
	if strings.TrimSpace(req.Input) == "" || utf8.RuneCountInString(req.Input) > 4096 {
		return badRequest("Kokoro requires 1–4096 characters of English text")
	}
	if req.Speed != nil && (math.IsNaN(*req.Speed) || math.IsInf(*req.Speed, 0) || *req.Speed < .5 || *req.Speed > 2) {
		return badRequest("Kokoro speed must be between 0.5 and 2")
	}
	voice := strings.ToLower(req.Voice)
	if voice == "" {
		voice = "af_heart"
	}
	known := false
	for _, name := range m.cfg.Voices {
		if name == voice {
			known = true
			break
		}
	}
	if !known {
		return badRequest("unknown Kokoro voice %q", req.Voice)
	}
	language := strings.ToLower(strings.ReplaceAll(req.Language, "_", "-"))
	switch language {
	case "", "en", "english", "auto":
	case "en-us":
		if voice[0] != 'a' {
			return badRequest("en-US requires an American English voice")
		}
	case "en-gb":
		if voice[0] != 'b' {
			return badRequest("en-GB requires a British English voice")
		}
	default:
		return badRequest("Kokoro currently supports English only")
	}
	phonemes, err := english.Phonemes(req.Input, voice[0] == 'b')
	if err != nil {
		return badRequest("%s", err)
	}
	if strings.Trim(phonemes, " ;:,.!?…\"()“”") == "" {
		return badRequest("Kokoro requires pronounceable text")
	}
	for _, r := range phonemes {
		if _, ok := m.cfg.Vocab[string(r)]; !ok {
			return badRequest("unsupported Kokoro phoneme %q", r)
		}
	}
	return nil
}

func phonemeChunks(text string, limit int) []string {
	chunks, _ := audio.PlanSpeech(text, limit, utf8.RuneCountInString)
	return chunks
}

var errUtteranceTooLong = errors.New("Kokoro utterance exceeds the 12-second synthesis bound")

// Fit durations before stochastic waveform synthesis. Oversized candidates
// consume no RNG, so retries cannot leak into the accepted utterances.
func fitPhonemes(ctx context.Context, phonemes string, limit int, synthesize func(string) ([]float32, error), emit func([]float32) error) error {
	pending := phonemeChunks(phonemes, limit)
	for len(pending) > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		part := pending[0]
		pending = pending[1:]
		samples, err := synthesize(part)
		if errors.Is(err, errUtteranceTooLong) {
			left, right := audio.SplitSpeech(part)
			if left == "" || right == "" {
				return err
			}
			pending = append([]string{left, right}, pending...)
			continue
		}
		if err != nil {
			return err
		}
		for offset := 0; offset < len(samples); offset += 6000 {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := emit(samples[offset:min(offset+6000, len(samples))]); err != nil {
				return err
			}
		}
	}
	return nil
}

func (m *Model) Speak(ctx context.Context, req llm.SpeechRequest, emit func([]float32) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := m.ValidateSpeechRequest(req); err != nil {
		return err
	}
	voice := strings.ToLower(req.Voice)
	if voice == "" {
		voice = "af_heart"
	}
	phonemes, err := english.Phonemes(req.Input, voice[0] == 'b')
	if err != nil {
		return err
	}
	speed := float32(1)
	if req.Speed != nil {
		speed = float32(*req.Speed)
	}
	seed := rand.Uint64()
	if req.Seed != nil {
		seed = *req.Seed
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	return fitPhonemes(ctx, phonemes, m.cfg.BERT.Context-2, func(chunk string) ([]float32, error) {
		var samples []float32
		var generationErr error
		mlx.Scoped(func() { samples, generationErr = m.synthesize(ctx, chunk, m.voices[voice], speed, rng) })
		return samples, generationErr
	}, emit)
}

func (m *Model) synthesize(ctx context.Context, phonemes string, voice *mlx.Array, speed float32, rng *rand.Rand) ([]float32, error) {
	runes := []rune(phonemes)
	if len(runes) < 1 || len(runes)+2 > m.cfg.BERT.Context || len(runes) > voice.Dim(0) {
		return nil, fmt.Errorf("Kokoro phonemes exceed the context/voice limits")
	}
	ids := make([]int32, len(runes)+2)
	for i, r := range runes {
		ids[i+1] = int32(m.cfg.Vocab[string(r)])
	}
	tokens := mlx.FromValues(ids, 1, len(ids))
	style := voice.Slice(mlx.Slice(len(runes)-1, len(runes)), mlx.Slice(), mlx.Slice()).Reshape(1, 256)
	prosodyStyle := style.Slice(mlx.Slice(), mlx.Slice(128, 256))
	e := m.encoders
	encoded := e.projection.Forward(e.bert.forward(tokens))
	duration := e.encodeDuration(encoded, prosodyStyle)
	logits := e.duration.Forward(e.durationLSTM.run(duration))
	predicted := mlx.Div(mlx.Sum(logits.Sigmoid(), -1, false), mlx.FromValue(speed))
	values := mlx.Contiguous(predicted, false).Floats()
	var indices []int32
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("invalid Kokoro duration")
		}
		count := int(max(1, min(100, math.RoundToEven(float64(value)))))
		if len(indices)+count > 480 {
			return nil, errUtteranceTooLong
		}
		for range count {
			indices = append(indices, int32(i))
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	at := mlx.FromValues(indices, len(indices))
	aligned := mlx.Transpose(mlx.Take(mlx.Transpose(duration, 0, 2, 1), at, 2), 0, 2, 1)
	shared := e.shared.run(aligned)
	pitch, noise := m.prosody.run(shared, prosodyStyle)
	asr := mlx.Transpose(mlx.Take(mlx.Transpose(e.encodeText(tokens), 0, 2, 1), at, 2), 0, 2, 1)
	excitation := randomSource(pitch, rng, m.decoder.source)
	audio := m.decoder.run(asr, pitch, noise, style.Slice(mlx.Slice(), mlx.Slice(0, 128)), excitation)
	samples := append([]float32(nil), mlx.Contiguous(audio, false).Floats()...)
	if len(samples) != len(indices)*600 {
		return nil, fmt.Errorf("Kokoro decoder returned %d samples for %d frames", len(samples), len(indices))
	}
	for _, sample := range samples {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return nil, fmt.Errorf("Kokoro produced nonfinite samples")
		}
	}
	return samples, nil
}
