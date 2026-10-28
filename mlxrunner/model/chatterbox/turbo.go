package chatterbox

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"unicode"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	schema "github.com/ollama/ollama/mlxrunner/model/chatterbox/config"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() { model.Register(schema.TurboArchitecture, newTurboModel) }

type turboModel struct {
	Model
	generator *turboT3
	codes     nn.EmbeddingLayer
	builtin   *cloneProfile
}

func newTurboModel(root *model.Root) (model.Model, error) {
	raw, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	if err = schema.ValidateTurbo(raw); err != nil {
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
	if tok.VocabSize() != 50276 {
		return nil, fmt.Errorf("Turbo tokenizer vocabulary mismatch")
	}
	return &turboModel{Model: Model{tok: tok}}, nil
}

func (m *turboModel) LoadWeights(tensors map[string]*mlx.Array) error {
	tensors = turboTensorAliases(tensors)
	w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, 64, 4, "affine", nil)}
	m.generator = loadTurboT3(w)
	m.voice = loadVoiceEncoder(w)
	m.voice.epsilon = 1e-8
	m.voice.turbo = true
	m.camp = loadCAMP(w)
	m.conformer = loadConformer(w)
	m.flow = loadFlowDecoder(w)
	m.hift = loadHiFT(w)
	m.flow.mixer = w.linear("s3gen.flow.decoder.estimator.time_embed_mixer")
	m.conformer.positions = w.get("s3gen.flow.encoder.embed.pos_enc.pe")
	m.conformer.upPositions = w.get("s3gen.flow.encoder.up_embed.pos_enc.pe")
	m.hift.turbo = true
	m.hift.window = w.get("s3gen.mel2wav.stft_window")
	m.codes = model.MakeEmbeddingLayer(tensors, "s3gen.flow.input_embedding", 64, 4, "affine", nil)
	m.projection = w.linear("s3gen.flow.encoder_proj")
	m.speaker = w.linear("s3gen.flow.spk_embed_affine_layer")
	if w.err != nil {
		return w.err
	}
	if m.codes == nil {
		return fmt.Errorf("missing Turbo flow embedding")
	}
	s3 := map[string]*mlx.Array{}
	for name, a := range tensors {
		if key, ok := strings.CutPrefix(name, "s3_tokenizer."); ok {
			s3[key] = a
		}
		m.Arrays = append(m.Arrays, a)
	}
	var err error
	m.s3, err = loadS3Tokenizer(s3)
	if err != nil {
		return err
	}
	m.mel = newReferenceMel()
	m.mel.turbo = true
	c := &cloneProfile{tokens: w.get("default.gen.prompt_token"), mels: w.get("default.gen.prompt_feat"), speaker: w.get("default.gen.embedding")}
	speaker, reference := w.get("default.t3.speaker_emb"), w.get("default.t3.cond_prompt_speech_tokens")
	if w.err != nil {
		return w.err
	}
	c.conditioning = m.generator.conditioning(speaker, reference)
	m.builtin = c
	for _, derived := range [][]*mlx.Array{m.s3.derived, m.voice.derived, m.camp.derived, m.hift.derived} {
		m.Arrays = append(m.Arrays, derived...)
	}
	m.Arrays = append(m.Arrays, m.mel.bank, m.mel.window, c.conditioning)
	return nil
}

func (*turboModel) ValidateSpeechRequest(req llm.SpeechRequest) error {
	if req.ReferenceAudio == nil {
		if req.Voice != "" && strings.ToLower(req.Voice) != "default" {
			return badRequest("Turbo supports only the default preset")
		}
	} else if req.Voice != "" {
		return badRequest("Turbo voice and reference_audio are mutually exclusive")
	}
	language, err := languageName(req.Language)
	if err != nil || language != "en" {
		return badRequest("Chatterbox Turbo currently supports English only")
	}
	if req.ReferenceText != "" || req.Instructions != "" || (req.Speed != nil && *req.Speed != 1) {
		return badRequest("Turbo does not support transcript, instructions or speed changes")
	}
	r := req
	r.Voice = ""
	r.Language = "en"
	if r.ReferenceAudio == nil {
		if strings.TrimSpace(r.Input) == "" || len([]rune(r.Input)) > 4096 {
			return badRequest("Turbo requires 1–4096 characters")
		}
		return nil
	}
	return (*Model)(nil).ValidateSpeechRequest(r)
}

func normalizedTurboText(text string) string {
	runes := []rune(text)
	if len(runes) > 0 {
		runes[0] = unicode.ToUpper(runes[0])
	}
	text = strings.Join(strings.Fields(string(runes)), " ")
	for _, p := range [][2]string{{"…", ", "}, {":", ","}, {"—", "-"}, {"–", "-"}, {" ,", ","}} {
		text = strings.ReplaceAll(text, p[0], p[1])
	}
	text = strings.TrimRight(text, " ")
	if !strings.ContainsAny(text[len(text)-1:], ".!?-,") {
		text += "."
	}
	return text
}

func sampleTurbo(values []float32, history []int32, rng *rand.Rand) (int32, error) {
	values = append([]float32(nil), values...)
	seen := map[int32]bool{}
	for _, id := range history {
		if id >= 0 && int(id) < len(values) && !seen[id] {
			if values[id] < 0 {
				values[id] *= 1.2
			} else {
				values[id] /= 1.2
			}
			seen[id] = true
		}
	}
	order := make([]int, len(values))
	for i, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, fmt.Errorf("nonfinite Turbo logits")
		}
		values[i] /= .8
		order[i] = i
	}
	if len(order) == 0 {
		return 0, fmt.Errorf("empty Turbo logits")
	}
	slices.SortStableFunc(order, func(a, b int) int {
		if values[a] > values[b] {
			return -1
		}
		if values[a] < values[b] {
			return 1
		}
		return 0
	})
	threshold := values[order[min(1000, len(order))-1]]
	end := 0
	for end < len(order) && values[order[end]] >= threshold {
		end++
	}
	order = order[:end]
	probs := make([]float64, len(order))
	total := float64(0)
	maximum := values[order[0]]
	for i, id := range order {
		probs[i] = math.Exp(float64(values[id] - maximum))
		total += probs[i]
	}
	keep, retained := 0, float64(0)
	for keep < len(probs) && retained <= .95*total {
		retained += probs[keep]
		keep++
	}
	pick := rng.Float64() * retained
	for i := range keep {
		pick -= probs[i]
		if pick <= 0 {
			return int32(order[i]), nil
		}
	}
	return int32(order[keep-1]), nil
}

func (m *turboModel) generateTurbo(ctx context.Context, text string, p *cloneProfile, rng *rand.Rand) ([]int32, error) {
	ids := m.tok.Encode(text, false)
	if len(ids) == 0 || len(ids) > 2048 {
		return nil, badRequest("invalid Turbo text token count")
	}
	state := newTurboState()
	defer state.close()
	held := mlx.NewScope()
	defer held.Close()
	var input *mlx.Array
	mlx.Scoped(func() {
		input = m.generator.prompt(p.conditioning, mlx.FromValues(ids, 1, len(ids)))
		held.Attach(input)
	})
	var history []int32
	for step := range 512 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var token int32
		var err error
		mlx.Scoped(func() {
			hidden := m.generator.forward(input, state)
			logits := m.generator.head.Forward(hidden.Slice(mlx.Slice(), mlx.Slice(hidden.Dim(1)-1, hidden.Dim(1)), mlx.Slice()))
			token, err = sampleTurbo(mlx.Contiguous(logits.AsType(mlx.DTypeFloat32), false).Floats(), history, rng)
		})
		if err != nil {
			return nil, err
		}
		if token == 6562 {
			if len(history) == 0 {
				return nil, fmt.Errorf("Turbo generated no speech")
			}
			codes := slices.DeleteFunc(append([]int32(nil), history...), func(v int32) bool { return v >= 6561 })
			return append(codes, 4299, 4299, 4299), nil
		}
		history = append(history, token)
		if step == 0 {
			history = append(history, token)
		} // MLX-Audio nonstreaming Turbo duplicates its first code.
		mlx.Scoped(func() {
			next := m.generator.speech.Forward(mlx.FromValues([]int32{token}, 1, 1))
			mlx.Eval(next)
			held.Discard(input)
			held.Attach(next)
			input = next
		})
	}
	return nil, fmt.Errorf("Turbo reached the 512-code limit before end of audio")
}

func (m *turboModel) flowInputsTurbo(codes []int32, p *cloneProfile) (mu, mask, speaker, condition *mlx.Array) {
	tokens := mlx.Concatenate([]*mlx.Array{p.tokens, mlx.FromValues(codes, 1, len(codes))}, 1)
	embedded := mlx.Mul(m.codes.Forward(tokens), mlx.BroadcastTo(mlx.FromValue(float32(1)), 1, int32(tokens.Dim(1)), 1))
	encoded := m.conformer.forward(embedded)
	mu = mlx.Transpose(m.projection.Forward(encoded), 0, 2, 1)
	mask = mlx.BroadcastTo(mlx.FromValue(float32(1)), 1, 1, int32(mu.Dim(2)))
	speaker = m.speaker.Forward(mlx.Div(p.speaker, mlx.Add(p.speaker.Power(mlx.FromValue(2)).SumAxis(1, true).Sqrt(), mlx.FromValue(float32(1e-8)))))
	condition = mlx.Transpose(mlx.Concatenate([]*mlx.Array{p.mels, mlx.Zeros(mlx.DTypeFloat32, 1, mu.Dim(2)-p.mels.Dim(1), 80)}, 1), 0, 2, 1)
	return
}

func (m *turboModel) waveformTurbo(ctx context.Context, codes []int32, p *cloneProfile, rng *rand.Rand) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var samples []float32
	var err error
	mlx.Scoped(func() {
		mu, mask, speaker, condition := m.flowInputsTurbo(codes, p)
		normal := make([]float32, mu.Size())
		for i := range normal {
			normal[i] = float32(rng.NormFloat64())
		}
		mels, e := m.flow.solveMean(ctx, mlx.FromValues(normal, mu.Dims()...), mlx.Linspace(0, 1, 3, mlx.DTypeFloat32), mu, mask, speaker, condition)
		if e != nil {
			err = e
			return
		}
		mels = mels.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(p.mels.Dim(1), mels.Dim(2)))
		phase := make([]float32, 8)
		for i := range phase {
			phase[i] = float32((rng.Float64()*2 - 1) * math.Pi)
		}
		noise := make([]float32, 9*mels.Dim(2)*480)
		for i := range noise {
			noise[i] = float32(rng.NormFloat64())
		}
		source := m.hift.excitation(m.hift.pitch(mels), mlx.FromValues(phase, 1, 8, 1), mlx.FromValues(noise, 1, 9, mels.Dim(2)*480))
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
		return nil, fmt.Errorf("Turbo generated empty audio")
	}
	for _, v := range samples {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("Turbo generated nonfinite audio")
		}
	}
	return samples, nil
}

func (m *turboModel) Speak(ctx context.Context, req llm.SpeechRequest, emit func([]float32) error) error {
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
	profile := m.builtin
	if req.ReferenceAudio != nil {
		var err error
		profile, err = m.referenceProfileTurbo(ctx, req, held)
		if err != nil {
			return err
		}
	}
	text := normalizedTurboText(req.Input)
	for _, chunk := range textChunks(text) {
		codes, err := m.generateTurbo(ctx, chunk, profile, rng)
		if err != nil {
			return err
		}
		samples, err := m.waveformTurbo(ctx, codes, profile, rng)
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

func (m *turboModel) referenceProfileTurbo(ctx context.Context, req llm.SpeechRequest, held *mlx.Scope) (*cloneProfile, error) {
	pcm, rate, err := audio.DecodeReferenceWAV(req.ReferenceAudio.Data)
	if err != nil {
		return nil, badRequest("%s", err)
	}
	if len(pcm) <= rate*5 {
		return nil, badRequest("Turbo reference audio must be longer than 5 seconds")
	}
	pcm24 := audio.ResampleKaiserBest(pcm, rate, 24000)
	pcm16 := audio.ResampleKaiserBest(pcm24, 24000, 16000)
	p := &cloneProfile{}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	mlx.Scoped(func() {
		tokenize := func(pcm []float32) *mlx.Array {
			return m.s3.quantize(m.s3.features(m.s3.mels(mlx.FromValues(pcm, len(pcm)))))
		}
		gen16 := pcm16[:min(160000, len(pcm16))]
		gen24 := pcm24[:min(240000, len(pcm24))]
		p.tokens = tokenize(gen16)
		p.mels = m.mel.forward(mlx.FromValues(gen24, len(gen24)))
		expected := p.mels.Dim(1) / 2
		if p.tokens.Dim(1) < expected {
			p.mels = p.mels.Slice(mlx.Slice(), mlx.Slice(0, p.tokens.Dim(1)*2), mlx.Slice())
		} else if p.tokens.Dim(1) > expected {
			p.tokens = p.tokens.Slice(mlx.Slice(), mlx.Slice(0, expected))
		}
		// This checkpoint omits CAMPPlus. Its explicitly imported, qualified v3
		// encoder supplies reference embeddings; cached default conditioning stays intact.
		fbank := m.camp.fbank(mlx.FromValues(gen16, len(gen16)))
		p.speaker = m.camp.forward(mlx.ExpandDims(mlx.Sub(fbank, mlx.Mean(fbank, 0, true)), 0))
		trimmed := trimVoiceReference(pcm16)
		speaker := m.voice.embedding(m.voice.mels(mlx.FromValues(trimmed, len(trimmed))))
		codes := tokenize(pcm16)
		codes = codes.Slice(mlx.Slice(), mlx.Slice(0, min(375, codes.Dim(1))))
		p.conditioning = m.generator.conditioning(speaker, codes)
		mlx.Eval(p.tokens, p.mels, p.speaker, p.conditioning)
		held.Attach(p.tokens, p.mels, p.speaker, p.conditioning)
	})
	return p, ctx.Err()
}
