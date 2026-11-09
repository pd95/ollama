package voxtraltts

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	voxtralconfig "github.com/ollama/ollama/mlxrunner/model/voxtraltts/config"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() { model.Register(Architecture, newModel) }

type Model struct {
	Arrays  []*mlx.Array
	cfg     *config
	tok     *tokenizer.Tokenizer
	quant   map[string]*model.TensorQuantInfo
	network *network
	decoder *codec
	voices  map[string]*mlx.Array
}

func newModel(root *model.Root) (model.Model, error) {
	raw, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	cfg, err := parseConfig(raw)
	if err != nil {
		return nil, err
	}
	raw, err = root.Manifest.ReadConfig("tekken.json")
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadTekkenFromBytes(raw)
	if err != nil {
		return nil, err
	}
	if len(cfg.Voices) == 0 {
		return nil, fmt.Errorf("Voxtral TTS requires imported voice packs")
	}
	return &Model{cfg: cfg, tok: tok, quant: root.AllTensorQuant()}, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	var err error
	m.network, err = loadNetwork(m.cfg, tensors, m.quant)
	if err != nil {
		return err
	}
	w := &weights{tensors: tensors, cfg: m.cfg, meta: m.quant, factory: model.NewLinearFactory(tensors, m.cfg.Quant.Group, m.cfg.Quant.Bits, m.cfg.Quant.Mode, m.quant)}
	m.decoder = loadCodec(w)
	if w.err != nil {
		return w.err
	}
	m.voices = map[string]*mlx.Array{}
	for name, count := range m.cfg.Voices {
		a := tensors["voices."+name]
		if count < 1 || count > 512 || a == nil || a.NumDims() != 2 || a.Dim(0) != count || a.Dim(1) != m.cfg.Dim {
			return fmt.Errorf("Voxtral TTS voice %q has incompatible shape", name)
		}
		m.voices[name] = a
	}
	if m.voices["casual_male"] == nil || m.voices["de_male"] == nil {
		return fmt.Errorf("Voxtral TTS requires English and German presets")
	}
	for _, a := range tensors {
		m.Arrays = append(m.Arrays, a)
	}
	m.Arrays = append(m.Arrays, m.network.derived...)
	m.Arrays = append(m.Arrays, w.derived...)
	return nil
}
func (*Model) NewCaches() []cache.Cache { return nil }
func (*Model) Forward(*batch.Batch, []cache.Cache) (*mlx.Array, *mlx.Array) {
	panic("Voxtral TTS is speech-only")
}
func (*Model) Unembed(x *mlx.Array) *mlx.Array { return x }
func (*Model) Tokenizer() *tokenizer.Tokenizer { return nil }
func (*Model) MaxContextLength() int           { return 0 }
func badRequest(format string, args ...any) error {
	return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf(format, args...)}
}

func voiceLanguage(voice string) string {
	return voxtralconfig.VoiceLanguage(voice)
}

func resolveVoice(req llm.SpeechRequest) (string, string, error) {
	voice, language, err := voxtralconfig.ResolveVoice(req.Voice, req.Language)
	if err != nil {
		return "", "", badRequest("%s", err)
	}
	return voice, language, nil
}

func (m *Model) ValidateSpeechRequest(req llm.SpeechRequest) error {
	if req.ReferenceAudio != nil || req.ReferenceText != "" {
		return badRequest("Voxtral TTS does not support reference audio")
	}
	if strings.TrimSpace(req.Input) == "" || utf8.RuneCountInString(req.Input) > 4096 {
		return badRequest("Voxtral TTS requires 1–4096 characters")
	}
	if req.Instructions != "" {
		return badRequest("Voxtral TTS does not support style instructions")
	}
	if req.Speed != nil && *req.Speed != 1 {
		return badRequest("Voxtral TTS supports only speed 1")
	}
	voice, language, err := resolveVoice(req)
	if err != nil {
		return err
	}
	if _, ok := m.cfg.Voices[voice]; !ok {
		return badRequest("unknown Voxtral TTS voice %q", req.Voice)
	}
	if _, err := normalizeTextForLanguage(req.Input, language); err != nil {
		return badRequest("%s", err)
	}
	return nil
}

func textChunks(text string) []string {
	var result []string
	runes := []rune(strings.TrimSpace(text))
	// Preserve context for short utterances instead of restarting generation
	// for each sentence. Longer requests retain the bounded chunking path.
	if len(runes) > 0 && len(runes) <= 160 {
		return []string{string(runes)}
	}
	for len(runes) > 0 {
		end := min(160, len(runes))
		split := 0
		for i := range end {
			if strings.ContainsRune(".!?؟।॥", runes[i]) && (i+1 == len(runes) || unicode.IsSpace(runes[i+1])) {
				split = i + 1
				break
			}
		}
		if split != 0 {
			end = split
		} else if end < len(runes) {
			for i := end - 1; i > 0; i-- {
				if unicode.IsSpace(runes[i]) {
					end = i
					break
				}
			}
		}
		if s := strings.TrimSpace(string(runes[:end])); s != "" {
			result = append(result, s)
		}
		runes = runes[end:]
	}
	return result
}

func (m *Model) Speak(ctx context.Context, req llm.SpeechRequest, emit func([]float32) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := m.ValidateSpeechRequest(req); err != nil {
		return err
	}
	voice, language, _ := resolveVoice(req)
	text, _ := normalizeTextForLanguage(req.Input, language)
	seed := rand.Uint64()
	if req.Seed != nil {
		seed = *req.Seed
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	for _, chunk := range textChunks(text) {
		if err := m.speak(ctx, chunk, voice, rng, emit); err != nil {
			return err
		}
	}
	return nil
}

func (m *Model) prompt(text, voice string) []int32 {
	ids := []int32{1, 25}
	for range m.cfg.Voices[voice] {
		ids = append(ids, 24)
	}
	ids = append(ids, 36)
	ids = append(ids, m.tok.Encode(text, false)...)
	return append(ids, 35, 25)
}

func (m *Model) speak(ctx context.Context, text, voice string, rng *rand.Rand, emit func([]float32) error) error {
	text, err := normalizeTextForLanguage(text, voiceLanguage(voice))
	if err != nil {
		return err
	}
	ids := m.prompt(text, voice)
	states := newState(m.cfg.Layers)
	defer states.close()
	held := mlx.NewScope()
	defer held.Close()
	var hidden *mlx.Array
	mlx.Scoped(func() {
		embeddings := m.network.condition(mlx.FromValues(ids, 1, len(ids)), m.voices[voice])
		h := m.network.backbone.forward(embeddings, states, 0)
		hidden = mlx.Squeeze(h.Slice(mlx.Slice(), mlx.Slice(h.Dim(1)-1, h.Dim(1)), mlx.Slice()), 1)
		mlx.Eval(hidden)
		held.Attach(hidden)
	})
	var frames []int32
	emitted := 0
	decode := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		count := len(frames) / 37
		start := max(0, emitted-32)
		var samples []float32
		mlx.Scoped(func() {
			wave := m.decoder.decode(mlx.FromValues(frames[start*37:], 1, count-start, 37))
			samples = append([]float32(nil), mlx.Contiguous(wave, false).Floats()...)
		})
		if len(samples) != (count-start)*1920 {
			return fmt.Errorf("Voxtral TTS decoder returned an invalid sample count")
		}
		for _, v := range samples {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("Voxtral TTS returned nonfinite audio")
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := emit(samples[(emitted-start)*1920:]); err != nil {
			return err
		}
		emitted = count
		return nil
	}
	for range 256 {
		if err := ctx.Err(); err != nil {
			return err
		}
		normal := make([]float32, 36)
		for i := range normal {
			normal[i] = float32(rng.NormFloat64())
		}
		var codes []int32
		mlx.Scoped(func() { codes, err = m.network.acoustic.codes(hidden, mlx.FromValues(normal, 1, 36)) })
		if err != nil {
			return err
		}
		if codes[0] <= 1 {
			if len(frames) == 0 {
				return fmt.Errorf("Voxtral TTS generated no audio frames")
			}
			if emitted < len(frames)/37 {
				return decode()
			}
			return nil
		}
		frames = append(frames, codes...)
		if len(frames)/37-emitted >= 25 {
			if err := decode(); err != nil {
				return err
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		mlx.Scoped(func() {
			next := m.network.backbone.forward(m.network.feedback(mlx.FromValues(codes, 1, 37)), states, 0)
			next = mlx.Squeeze(next, 1)
			mlx.Eval(next)
			held.Discard(hidden)
			held.Attach(next)
			hidden = next
		})
	}
	return fmt.Errorf("Voxtral TTS reached the 256-frame limit before end of audio")
}
