package chatterbox

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
	"github.com/ollama/ollama/mlxrunner/model/audio"
	schema "github.com/ollama/ollama/mlxrunner/model/chatterbox/config"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() { model.Register(schema.Architecture, newModel) }

type Model struct {
	Arrays               []*mlx.Array
	tok                  *tokenizer.Tokenizer
	t3                   *t3
	s3                   *s3Tokenizer
	voice                *voiceEncoder
	camp                 *campEncoder
	conformer            *conformerEncoder
	flow                 *flowDecoder
	hift                 *hiFT
	mel                  referenceMel
	embedding, flowNoise *mlx.Array
	projection, speaker  nn.LinearLayer
}

func newModel(root *model.Root) (model.Model, error) {
	raw, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	if err = schema.Validate(raw); err != nil {
		return nil, err
	}
	raw, err = root.Manifest.ReadConfig("s3_tokenizer/config.json")
	if err != nil {
		return nil, err
	}
	if err = schema.ValidateS3(raw); err != nil {
		return nil, err
	}
	raw, err = root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadFromBytes(raw)
	if err != nil {
		return nil, err
	}
	if tok.VocabSize() != 2454 {
		return nil, fmt.Errorf("Chatterbox tokenizer vocabulary mismatch")
	}
	return &Model{tok: tok}, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	s3 := map[string]*mlx.Array{}
	for key, a := range tensors {
		if a.DType() != mlx.DTypeFloat32 {
			return fmt.Errorf("Chatterbox v3 requires float32 weights (%s)", key)
		}
		if name, ok := strings.CutPrefix(key, "s3_tokenizer."); ok {
			s3[name] = a
		}
		m.Arrays = append(m.Arrays, a)
	}
	var err error
	m.s3, err = loadS3Tokenizer(s3)
	if err != nil {
		return err
	}
	w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, 0, 0, "", nil)}
	m.t3 = loadT3(w)
	m.voice = loadVoiceEncoder(w)
	m.camp = loadCAMP(w)
	m.conformer = loadConformer(w)
	m.flow = loadFlowDecoder(w)
	m.hift = loadHiFT(w)
	m.embedding = w.get("s3gen.flow.input_embedding.weight")
	m.projection = w.linear("s3gen.flow.encoder_proj")
	m.speaker = w.linear("s3gen.flow.spk_embed_affine_layer")
	if w.err != nil {
		return w.err
	}
	m.mel = newReferenceMel()
	// S3Gen uses a fixed seed-zero, channel-major noise table. A separate PCG
	// keeps its prefix stable across target lengths without touching global RNG.
	rng := rand.New(rand.NewPCG(0, 0))
	noise := make([]float32, 80*15000)
	for i := range noise {
		noise[i] = float32(rng.NormFloat64())
	}
	m.flowNoise = mlx.FromValues(noise, 1, 80, 15000)
	m.Arrays = append(m.Arrays, m.flowNoise, m.mel.bank, m.mel.window)
	for _, derived := range [][]*mlx.Array{m.s3.derived, m.t3.derived, m.voice.derived, m.camp.derived, m.hift.derived} {
		m.Arrays = append(m.Arrays, derived...)
	}
	return nil
}
func (*Model) NewCaches() []cache.Cache { return nil }
func (*Model) Forward(*batch.Batch, []cache.Cache) (*mlx.Array, *mlx.Array) {
	panic("Chatterbox is speech-only")
}
func (*Model) Unembed(x *mlx.Array) *mlx.Array { return x }
func (*Model) Tokenizer() *tokenizer.Tokenizer { return nil }
func (*Model) MaxContextLength() int           { return 0 }
func badRequest(format string, args ...any) error {
	return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf(format, args...)}
}

func languageName(language string) (string, error) {
	switch strings.ToLower(strings.ReplaceAll(language, "_", "-")) {
	case "", "auto", "en", "english", "en-us", "en-gb":
		return "en", nil
	case "de", "german", "de-de":
		return "de", nil
	default:
		return "", badRequest("Chatterbox v3 currently supports English and German")
	}
}

func (*Model) ValidateSpeechRequest(req llm.SpeechRequest) error {
	if strings.TrimSpace(req.Input) == "" || utf8.RuneCountInString(req.Input) > 4096 {
		return badRequest("Chatterbox requires 1–4096 characters")
	}
	if req.Voice != "" || req.Instructions != "" || req.ReferenceText != "" || (req.Speed != nil && *req.Speed != 1) {
		return badRequest("Chatterbox v3 accepts reference audio, language and seed; presets, reference_text, instructions and speed changes are unsupported")
	}
	if _, err := languageName(req.Language); err != nil {
		return err
	}
	if req.ReferenceAudio == nil || req.ReferenceAudio.Format != "wav" {
		return badRequest("Chatterbox v3 requires a reference_audio WAV")
	}
	if err := audio.ValidateReferenceWAV(req.ReferenceAudio.Data); err != nil {
		return badRequest("%s", err)
	}
	return nil
}

func textChunks(text string) []string {
	var chunks []string
	runes := []rune(strings.TrimSpace(text))
	for len(runes) > 0 {
		end := min(160, len(runes))
		split := 0
		for i := range end {
			if strings.ContainsRune(".!?", runes[i]) && (i+1 == len(runes) || unicode.IsSpace(runes[i+1])) {
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
			chunks = append(chunks, s)
		}
		runes = runes[end:]
	}
	return chunks
}

func (m *Model) Speak(ctx context.Context, req llm.SpeechRequest, emit func([]float32) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := m.ValidateSpeechRequest(req); err != nil {
		return err
	}
	seed := rand.Uint64()
	if req.Seed != nil {
		seed = *req.Seed
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	held := mlx.NewScope()
	defer held.Close()
	profile, err := m.referenceProfile(ctx, req, held)
	if err != nil {
		return err
	}
	language, _ := languageName(req.Language)
	for _, chunk := range textChunks(req.Input) {
		codes, err := m.generate(ctx, normalizedV3Text(chunk, language), profile.conditioning, rng)
		if err != nil {
			return err
		}
		samples, err := m.waveform(ctx, codes, profile, rng)
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := emit(samples); err != nil {
			return err
		}
	}
	return nil
}

func (m *Model) generate(ctx context.Context, text string, condition *mlx.Array, rng *rand.Rand) ([]int32, error) {
	ids := m.tok.Encode(text, false)
	ids = append(append([]int32{255}, ids...), 0)
	if len(ids) > 2048 {
		return nil, badRequest("Chatterbox text exceeds tokenizer bound")
	}
	state := newT3State()
	defer state.close()
	held := mlx.NewScope()
	defer held.Close()
	var input *mlx.Array
	mlx.Scoped(func() {
		input = m.t3.prompt(condition, mlx.FromValues(append(append([]int32(nil), ids...), ids...), 2, len(ids)))
		held.Attach(input)
	})
	history := []int32{6561}
	var codes []int32
	for step := range 512 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var token int32
		var err error
		mlx.Scoped(func() {
			hidden := m.t3.forward(input, state)
			logits := m.t3.head.Forward(hidden.Slice(mlx.Slice(), mlx.Slice(hidden.Dim(1)-1, hidden.Dim(1)), mlx.Slice()))
			conditional := logits.Slice(mlx.Slice(0, 1), mlx.Slice(), mlx.Slice())
			unconditional := logits.Slice(mlx.Slice(1, 2), mlx.Slice(), mlx.Slice())
			logits = mlx.Add(conditional, mlx.Mul(mlx.FromValue(float32(.5)), mlx.Sub(conditional, unconditional)))
			token, err = sampleT3(mlx.Contiguous(logits, false).Floats(), history, rng)
		})
		if err != nil {
			return nil, err
		}
		if token == 6562 {
			if len(codes) == 0 {
				return nil, fmt.Errorf("Chatterbox generated no audio codes")
			}
			return codes, nil
		}
		history = append(history, token)
		if token < 6561 {
			codes = append(codes, token)
		}
		mlx.Scoped(func() {
			next := mlx.Add(mlx.Take(m.t3.speech, mlx.FromValues([]int32{token}, 1, 1), 0), mlx.Take(m.t3.speechPosition, mlx.FromValues([]int32{int32(step + 1)}, 1, 1), 0))
			next = mlx.Concatenate([]*mlx.Array{next, next}, 0)
			mlx.Eval(next)
			held.Discard(input)
			held.Attach(next)
			input = next
		})
	}
	return nil, fmt.Errorf("Chatterbox reached the 512-code limit before end of audio")
}

func (m *Model) flowInputs(codes []int32, p *cloneProfile) (mu, mask, speaker, condition *mlx.Array) {
	tokens := mlx.Concatenate([]*mlx.Array{p.tokens, mlx.FromValues(codes, 1, len(codes))}, 1)
	encoded := m.conformer.forward(mlx.Take(m.embedding, tokens, 0))
	mu = mlx.Transpose(m.projection.Forward(encoded), 0, 2, 1)
	mask = mlx.BroadcastTo(mlx.FromValue(float32(1)), 1, 1, int32(mu.Dim(2)))
	speaker = m.speaker.Forward(mlx.Div(p.speaker, mlx.Add(p.speaker.Power(mlx.FromValue(2)).SumAxis(1, true).Sqrt(), mlx.FromValue(float32(1e-8)))))
	condition = mlx.Concatenate([]*mlx.Array{p.mels, mlx.Zeros(mlx.DTypeFloat32, 1, mu.Dim(2)-p.mels.Dim(1), 80)}, 1)
	condition = mlx.Transpose(condition, 0, 2, 1)
	return
}

func (m *Model) waveform(ctx context.Context, codes []int32, p *cloneProfile, rng *rand.Rand) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var samples []float32
	var err error
	mlx.Scoped(func() {
		mu, mask, speaker, condition := m.flowInputs(codes, p)
		noise := m.flowNoise.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, mu.Dim(2)))
		mels, e := m.flow.solveContext(ctx, noise, flowTimes(10), mu, mask, speaker, condition)
		if e != nil {
			err = e
			return
		}
		mels = mels.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(p.mels.Dim(1), mels.Dim(2)))
		phase := make([]float32, 9)
		for i := range phase {
			phase[i] = float32(rng.Float64() * 2 * math.Pi)
		}
		normal := make([]float32, 9*mels.Dim(2)*480)
		for i := range normal {
			normal[i] = float32(rng.NormFloat64())
		}
		source := m.hift.excitation(m.hift.pitch(mels), mlx.FromValues(phase, 1, 9, 1), mlx.FromValues(normal, 1, 9, mels.Dim(2)*480))
		if e := ctx.Err(); e != nil {
			err = e
			return
		}
		wave := fadeHiFT(m.hift.decode(mels, source))
		samples = append([]float32(nil), mlx.Contiguous(wave, false).Floats()...)
	})
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("Chatterbox generated empty audio")
	}
	for _, v := range samples {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("Chatterbox generated nonfinite audio")
		}
	}
	return samples, nil
}
