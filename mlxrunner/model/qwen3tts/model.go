package qwen3tts

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() { model.Register("Qwen3TTSForConditionalGeneration", newModel) }

type generation struct {
	Temperature float64 `json:"temperature"`
	TopK        int     `json:"top_k"`
	TopP        float64 `json:"top_p"`
	Repetition  float64 `json:"repetition_penalty"`
}

// Model is a speech-only Qwen3-TTS CustomVoice or Base model. Arrays makes all source
// and derived weights visible to the runner's weight lifetime collector.
type Model struct {
	Arrays      []*mlx.Array
	cfg         *config
	codecConfig []byte
	quant       map[string]*model.TensorQuantInfo
	tok         *tokenizer.Tokenizer
	network     *network
	speaker     *speakerEncoder
	encoder     *speechEncoder
	decoder     *decoder
	generation  generation
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
	codec, err := root.Manifest.ReadConfig("speech_tokenizer/config.json")
	if err != nil {
		return nil, err
	}
	rawTok, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadFromBytes(rawTok)
	if err != nil {
		return nil, err
	}
	gen := generation{Temperature: 0.9, TopK: 50, TopP: 1, Repetition: 1.05}
	if err := root.Manifest.ReadConfigJSON("generation_config.json", &gen); err != nil {
		return nil, err
	}
	if gen.Temperature <= 0 || gen.TopK < 1 || gen.TopP != 1 || gen.Repetition < 1 || math.IsNaN(gen.Temperature) || math.IsInf(gen.Temperature, 0) {
		return nil, fmt.Errorf("unsupported Qwen3-TTS generation configuration")
	}
	return &Model{cfg: cfg, codecConfig: codec, tok: tok, quant: root.AllTensorQuant(), generation: gen}, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	var err error
	m.network, err = loadNetwork(m.cfg, tensors, m.quant)
	if err != nil {
		return err
	}
	codec := map[string]*mlx.Array{}
	for name, a := range tensors {
		m.Arrays = append(m.Arrays, a)
		if strings.HasPrefix(name, "speech_tokenizer.") {
			codec[strings.TrimPrefix(name, "speech_tokenizer.")] = a
		}
	}
	m.decoder, err = loadDecoder(m.codecConfig, codec)
	if err != nil {
		return err
	}
	m.Arrays = append(m.Arrays, m.decoder.derived...)
	if m.cfg.Type == "base" {
		m.speaker, err = loadSpeaker(m.cfg.Speaker, tensors)
		if err != nil {
			return err
		}
		m.encoder, err = loadEncoder(m.codecConfig, codec)
		if err != nil {
			return err
		}
		m.Arrays = append(m.Arrays, m.speaker.derived...)
		m.Arrays = append(m.Arrays, m.encoder.derived...)
	}
	return nil
}

func (m *Model) NewCaches() []cache.Cache { return nil }
func (m *Model) Forward(*batch.Batch, []cache.Cache) (*mlx.Array, *mlx.Array) {
	panic("Qwen3-TTS does not implement text generation")
}
func (m *Model) Unembed(x *mlx.Array) *mlx.Array { return x }
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return nil }
func (m *Model) MaxContextLength() int           { return 0 }

func speechError(format string, args ...any) error {
	return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf(format, args...)}
}

func (m *Model) ValidateSpeechRequest(req llm.SpeechRequest) error {
	if strings.TrimSpace(req.Input) == "" || utf8.RuneCountInString(req.Input) > 4096 {
		return speechError("Qwen3-TTS requires 1–4096 characters")
	}
	if m.cfg.Type == "base" {
		if req.ReferenceAudio == nil || strings.TrimSpace(req.ReferenceText) == "" || utf8.RuneCountInString(req.ReferenceText) > 4096 {
			return speechError("Qwen3-TTS Base requires reference_audio and a transcript of 1–4096 characters")
		}
		if req.Voice != "" || req.Instructions != "" || req.ReferenceAudio.Format != "wav" {
			return speechError("Qwen3-TTS Base requires reference WAV without preset voice or style instructions")
		}
		if err := audio.ValidateReferenceWAV(req.ReferenceAudio.Data); err != nil {
			return speechError("%s", err)
		}
	} else if req.ReferenceAudio != nil || req.ReferenceText != "" {
		return speechError("Qwen3-TTS CustomVoice does not support reference audio")
	}
	if req.Speed != nil && *req.Speed != 1 {
		return speechError("Qwen3-TTS supports only speed 1")
	}
	if req.Instructions != "" && m.cfg.Size != "1b7" {
		return speechError("instructions require Qwen3-TTS 1.7B CustomVoice")
	}
	if utf8.RuneCountInString(req.Instructions) > 1024 {
		return speechError("instructions must be at most 1024 characters")
	}
	voice := strings.ToLower(req.Voice)
	if voice != "" {
		if _, ok := m.cfg.Talker.Speakers[voice]; !ok {
			return speechError("unknown Qwen3-TTS voice %q", req.Voice)
		}
	}
	language := languageName(req.Language)
	if language != "auto" {
		if _, ok := m.cfg.Talker.Languages[language]; !ok {
			return speechError("unsupported Qwen3-TTS language %q", req.Language)
		}
	}
	if m.cfg.Type == "custom_voice" {
		if voice == "" {
			voice = "ryan"
		}
		if _, err := speakerLanguage(m.cfg.Talker, req.Language, voice); err != nil {
			return speechError("%s", err)
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
	voice := req.Voice
	if voice == "" && m.cfg.Type == "custom_voice" {
		voice = "Ryan"
	}
	chunks, err := textChunks(req.Input, 160, func(text string) int {
		return len(m.tok.Encode(text, false))
	})
	if err != nil {
		return speechError("%s", err)
	}
	if len(chunks) == 0 {
		return speechError("speech requires nonempty text")
	}
	seed := rand.Uint64()
	if req.Seed != nil {
		seed = *req.Seed
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	held := mlx.NewScope()
	defer held.Close()
	var profile *cloneProfile
	if m.cfg.Type == "base" {
		var err error
		profile, err = m.referenceProfile(ctx, req, held)
		if err != nil {
			return err
		}
	}
	for _, chunk := range chunks {
		if err := m.speak(ctx, chunk, req.Language, voice, req.Instructions, profile, rng, emit); err != nil {
			return err
		}
	}
	return nil
}

// Keep complete sentences inside the conservative text budget. Speak also
// checks the tokenizer count; the autoregressive 512-frame cap remains separate.
func textChunks(text string, limit int, tokens func(string) int) ([]string, error) {
	return audio.PlanSpeech(text, limit, func(part string) int { return max(utf8.RuneCountInString(part), tokens(part)) })
}

func (m *Model) speak(ctx context.Context, text, language, voice, instructions string, profile *cloneProfile, rng *rand.Rand, emit func([]float32) error) error {
	n := m.network
	ids := m.tok.Encode("<|im_start|>assistant\n"+text+"<|im_end|>\n<|im_start|>assistant\n", false)
	held := mlx.NewScope()
	defer held.Close()
	var input, pad *mlx.Array
	var err error
	mlx.Scoped(func() {
		if profile != nil {
			input, pad, err = n.prepareICL(ids, language, profile)
		} else {
			input, pad, err = n.prepare(ids, language, voice)
		}
		if err == nil && instructions != "" {
			instructIDs := m.tok.Encode("<|im_start|>user\n"+instructions+"<|im_end|>\n", false)
			input = mlx.Concatenate([]*mlx.Array{n.project(instructIDs), input}, 1)
		}
		if err == nil {
			mlx.Eval(input, pad)
			held.Attach(input, pad)
		}
	})
	if err != nil {
		return speechError("%s", err)
	}
	gen := m.generation
	if profile != nil {
		gen.Repetition = max(gen.Repetition, 1.5)
	}
	state := newState(len(n.talker.body.layers))
	defer state.close()
	var codes, semantic []int32
	emitted := 0
	decode := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		var samples []float32
		mlx.Scoped(func() {
			allCodes := codes
			cut := 0
			if profile != nil {
				allCodes = append(append([]int32(nil), profile.codes...), codes...)
				cut = len(profile.codes) / 16 * 1920
			}
			wav := m.decoder.decode(mlx.FromValues(allCodes, 1, len(allCodes)/16, 16))
			wav = wav.Slice(mlx.Slice(), mlx.Slice(cut, wav.Dim(1)))
			samples = mlx.Contiguous(wav, false).Floats()
		})
		if emitted > len(samples) {
			return fmt.Errorf("Qwen3-TTS decoder shortened a causal prefix")
		}
		out := samples[emitted:]
		emitted = len(samples)
		if err := ctx.Err(); err != nil {
			return err
		}
		return emit(out)
	}
	for frame := range 512 {
		if err := ctx.Err(); err != nil {
			return err
		}
		var tokens []int32
		var eos bool
		mlx.Scoped(func() {
			logits, hidden := n.talker.forward(input, state)
			first, e := sampleLast(logits, rng, gen, semantic, int(n.cfg.Talker.EOS))
			if e != nil {
				err = e
				return
			}
			if first == n.cfg.Talker.EOS {
				eos = true
				return
			}
			tokens, e = n.predict(ctx, first, timeSlice(hidden, hidden.Dim(1)-1, hidden.Dim(1)), rng, gen)
			if e != nil {
				err = e
				return
			}
			next := n.codec.Forward(mlx.FromValues(tokens[:1], 1, 1))
			for i, token := range tokens[1:] {
				next = mlx.Add(next, n.codeEmb[i].Forward(mlx.FromValues([]int32{token}, 1, 1)))
			}
			next = mlx.Add(pad, next)
			mlx.Eval(next)
			held.Discard(input)
			input = next
			held.Attach(input)
		})
		if err != nil {
			return err
		}
		if eos {
			if len(codes) == 0 {
				return fmt.Errorf("Qwen3-TTS stopped without generating audio")
			}
			if emitted < len(codes)/16*1920 {
				return decode()
			}
			return nil
		}
		codes = append(codes, tokens...)
		semantic = append(semantic, tokens[0])
		// Recompute causal prefixes initially. This keeps decoder state private to
		// this request, and bounds the graph to one short utterance at a time.
		if (frame+1)%25 == 0 {
			if err := decode(); err != nil {
				return err
			}
		}
	}
	return fmt.Errorf("Qwen3-TTS reached its 512-frame utterance limit before EOS")
}

func (n *network) predict(ctx context.Context, first int32, hidden *mlx.Array, rng *rand.Rand, gen generation) ([]int32, error) {
	s := newState(len(n.predictor.layers))
	defer s.close()
	tokens := []int32{first}
	for i := range n.cfg.Talker.Groups - 1 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var x *mlx.Array
		if i == 0 {
			x = mlx.Concatenate([]*mlx.Array{hidden, n.codec.Forward(mlx.FromValues(tokens[:1], 1, 1))}, 1)
		} else {
			x = n.codeEmb[i-1].Forward(mlx.FromValues(tokens[len(tokens)-1:], 1, 1))
		}
		if n.small != nil {
			x = n.small.Forward(x)
		}
		logits := n.codeHead[i].Forward(n.predictor.forward(x, s))
		token, err := sampleLast(logits, rng, gen, nil, -1)
		if err != nil {
			return nil, err
		}
		tokens = append(tokens, token)
	}
	return tokens, nil
}

// Sampling deliberately lives on the CPU with a request-local RNG. Fixed
// network inputs have exact numerical tests; stochastic token sequences need
// not equal MLX-Audio's different categorical RNG implementation.
func sampleLast(logits *mlx.Array, rng *rand.Rand, gen generation, previous []int32, eos int) (int32, error) {
	last := timeSlice(logits, logits.Dim(1)-1, logits.Dim(1))
	values := mlx.Contiguous(last.AsType(mlx.DTypeFloat32), false).Floats()
	return sampleValues(values, rng, gen, previous, eos)
}

func sampleValues(values []float32, rng *rand.Rand, gen generation, previous []int32, eos int) (int32, error) {
	previous = previous[max(0, len(previous)-64):]
	repeated := make(map[int32]bool, len(previous))
	for _, id := range previous {
		repeated[id] = true
	}
	type candidate struct {
		id    int32
		score float64
	}
	ranked := make([]candidate, 0, len(values))
	for i, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, fmt.Errorf("Qwen3-TTS produced nonfinite logits")
		}
		if eos >= 0 && i >= len(values)-1024 && i != eos {
			continue
		}
		score := float64(v)
		if repeated[int32(i)] {
			if score < 0 {
				score *= gen.Repetition
			} else {
				score /= gen.Repetition
			}
		}
		ranked = append(ranked, candidate{int32(i), score / gen.Temperature})
	}
	slices.SortFunc(ranked, func(a, b candidate) int {
		if a.score > b.score {
			return -1
		}
		if a.score < b.score {
			return 1
		}
		return int(a.id - b.id)
	})
	ranked = ranked[:min(gen.TopK, len(ranked))]
	total := 0.0
	for _, c := range ranked {
		total += math.Exp(c.score - ranked[0].score)
	}
	pick := rng.Float64() * total
	for _, c := range ranked {
		pick -= math.Exp(c.score - ranked[0].score)
		if pick <= 0 {
			return c.id, nil
		}
	}
	return ranked[len(ranked)-1].id, nil
}
