// Package apertus provides the Apertus text model implementation for MLX.
package apertus

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("ApertusForCausalLM", newModel)
}

// RopeScaling carries the Llama 3 RoPE scaling block used by Apertus.
type RopeScaling struct {
	Factor                        float32 `json:"factor"`
	HighFreqFactor                float32 `json:"high_freq_factor"`
	LowFreqFactor                 float32 `json:"low_freq_factor"`
	OriginalMaxPositionEmbeddings int32   `json:"original_max_position_embeddings"`
	RopeType                      string  `json:"rope_type,omitempty"`
	Type                          string  `json:"type,omitempty"`
}

// Config holds Apertus model configuration.
type Config struct {
	Architecture          string      `json:"-"`
	ModelType             string      `json:"model_type"`
	DType                 string      `json:"dtype"`
	HiddenSize            int32       `json:"hidden_size"`
	IntermediateSize      int32       `json:"intermediate_size"`
	NumHiddenLayers       int32       `json:"num_hidden_layers"`
	NumAttentionHeads     int32       `json:"num_attention_heads"`
	NumKeyValueHeads      int32       `json:"num_key_value_heads"`
	VocabSize             int32       `json:"vocab_size"`
	MaxPositionEmbeddings int32       `json:"max_position_embeddings"`
	RMSNormEps            float32     `json:"rms_norm_eps"`
	RopeTheta             float32     `json:"rope_theta"`
	RopeScaling           RopeScaling `json:"rope_scaling"`
	HiddenAct             string      `json:"hidden_act"`
	QKNorm                bool        `json:"qk_norm"`
	PostNorm              bool        `json:"post_norm"`
	AttentionBias         bool        `json:"attention_bias"`
	MLPBias               bool        `json:"mlp_bias"`
	TieWordEmbeddings     bool        `json:"tie_word_embeddings"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	HeadDim   int32      `json:"-"`
	Scale     float32    `json:"-"`
	RopeFreqs *mlx.Array `json:"-"`
}

// Model is an Apertus text model.
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config
}

type Layer struct {
	AttentionNorm *nn.RMSNorm
	Attention     *Attention
	FFNNorm       *nn.RMSNorm
	MLP           *MLP
}

type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm
}

type MLP struct {
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
	Act      *XIELU
}

type XIELU struct {
	AlphaP float32
	AlphaN float32
	Beta   float32
	Eps    float32
}

func newModel(root *model.Root) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	cfg, err := parseConfig(configData)
	if err != nil {
		return nil, err
	}
	ropeFreqs, err := Llama3Freqs(
		cfg.HeadDim,
		cfg.RopeTheta,
		cfg.RopeScaling.Factor,
		cfg.RopeScaling.LowFreqFactor,
		cfg.RopeScaling.HighFreqFactor,
		cfg.RopeScaling.OriginalMaxPositionEmbeddings,
	)
	if err != nil {
		return nil, fmt.Errorf("build llama3 rope frequencies: %w", err)
	}
	cfg.RopeFreqs = mlx.FromValues(ropeFreqs, len(ropeFreqs))

	if qt := root.QuantType(); qt != "" {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams(qt)
		if gs := root.GroupSize(); gs > 0 {
			cfg.QuantGroupSize = gs
		}
	} else {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams("")
	}
	cfg.TensorQuant = root.AllTensorQuant()

	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer config: %w", err)
	}

	tokConfig := &tokenizer.TokenizerConfig{
		ConfigJSON: configData,
	}
	if genConfigData, err := root.Manifest.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = genConfigData
	}
	if tokConfigData, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = tokConfigData
	}
	if specialTokensMapData, err := root.Manifest.ReadConfig("special_tokens_map.json"); err == nil {
		tokConfig.SpecialTokensMapJSON = specialTokensMapData
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	return &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
		tok:    tok,
	}, nil
}

func parseConfig(configData []byte) (Config, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(configData, &raw); err != nil {
		return Config{}, fmt.Errorf("parse config envelope: %w", err)
	}

	active := configData
	if textRaw, ok := raw["text_config"]; ok {
		active = textRaw
	}

	var cfg Config
	if err := json.Unmarshal(active, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}

	var archConfig struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(configData, &archConfig); err != nil {
		return Config{}, fmt.Errorf("parse architecture: %w", err)
	}
	if len(archConfig.Architectures) > 0 && archConfig.Architectures[0] != "" {
		cfg.Architecture = archConfig.Architectures[0]
	} else {
		cfg.Architecture = archConfig.ModelType
	}

	if cfg.Architecture == "" {
		return Config{}, fmt.Errorf("missing architecture in config.json")
	}
	if cfg.HiddenSize <= 0 {
		return Config{}, fmt.Errorf("invalid hidden_size: %d", cfg.HiddenSize)
	}
	if cfg.IntermediateSize <= 0 {
		return Config{}, fmt.Errorf("invalid intermediate_size: %d", cfg.IntermediateSize)
	}
	if cfg.NumHiddenLayers <= 0 {
		return Config{}, fmt.Errorf("invalid num_hidden_layers: %d", cfg.NumHiddenLayers)
	}
	if cfg.NumAttentionHeads <= 0 {
		return Config{}, fmt.Errorf("invalid num_attention_heads: %d", cfg.NumAttentionHeads)
	}
	if cfg.NumKeyValueHeads <= 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return Config{}, fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
	}
	cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return Config{}, fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-5
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 12000000
	}
	if cfg.MaxPositionEmbeddings <= 0 {
		return Config{}, fmt.Errorf("invalid max_position_embeddings: %d", cfg.MaxPositionEmbeddings)
	}
	if cfg.HiddenAct != "xielu" {
		return Config{}, fmt.Errorf("unsupported hidden_act %q", cfg.HiddenAct)
	}
	if !cfg.QKNorm {
		return Config{}, fmt.Errorf("unsupported qk_norm=false")
	}
	if cfg.PostNorm {
		return Config{}, fmt.Errorf("unsupported post_norm=true")
	}
	if cfg.AttentionBias {
		return Config{}, fmt.Errorf("unsupported attention_bias=true")
	}
	if cfg.MLPBias {
		return Config{}, fmt.Errorf("unsupported mlp_bias=true")
	}
	if cfg.TieWordEmbeddings {
		return Config{}, fmt.Errorf("unsupported tie_word_embeddings=true")
	}
	if ropeType := cfg.ropeType(); ropeType != "llama3" {
		return Config{}, fmt.Errorf("unsupported rope scaling type %q", ropeType)
	}
	if cfg.RopeScaling.Factor <= 0 {
		return Config{}, fmt.Errorf("invalid rope scaling factor: %v", cfg.RopeScaling.Factor)
	}
	if cfg.RopeScaling.LowFreqFactor <= 0 || cfg.RopeScaling.HighFreqFactor <= 0 {
		return Config{}, fmt.Errorf("invalid llama3 rope frequency factors")
	}
	if cfg.RopeScaling.OriginalMaxPositionEmbeddings <= 0 {
		return Config{}, fmt.Errorf("invalid original_max_position_embeddings: %d", cfg.RopeScaling.OriginalMaxPositionEmbeddings)
	}

	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	return cfg, nil
}

func (c Config) ropeType() string {
	if c.RopeScaling.RopeType != "" {
		return strings.ToLower(c.RopeScaling.RopeType)
	}
	return strings.ToLower(c.RopeScaling.Type)
}

// Llama3InvFreqs returns inverse RoPE frequencies matching Transformers'
// _compute_llama3_parameters for full-head rotary embeddings.
func Llama3InvFreqs(headDim int32, base, factor, lowFreqFactor, highFreqFactor float32, originalContext int32) ([]float32, error) {
	if headDim <= 0 || headDim%2 != 0 {
		return nil, fmt.Errorf("head_dim must be a positive even number: %d", headDim)
	}
	if base <= 0 || factor <= 0 || lowFreqFactor <= 0 || highFreqFactor <= 0 || originalContext <= 0 {
		return nil, fmt.Errorf("invalid llama3 rope parameters")
	}

	inv := make([]float32, headDim/2)
	lowFreqWavelen := float64(originalContext) / float64(lowFreqFactor)
	highFreqWavelen := float64(originalContext) / float64(highFreqFactor)

	for i := range inv {
		v := 1.0 / math.Pow(float64(base), float64(2*i)/float64(headDim))
		wavelen := 2 * math.Pi / v
		switch {
		case wavelen > lowFreqWavelen:
			v /= float64(factor)
		case wavelen >= highFreqWavelen:
			smooth := (float64(originalContext)/wavelen - float64(lowFreqFactor)) / (float64(highFreqFactor) - float64(lowFreqFactor))
			v = (1-smooth)*v/float64(factor) + smooth*v
		}
		inv[i] = float32(v)
	}
	return inv, nil
}

// Llama3Freqs returns the frequency values expected by MLX RoPEWithFreqs.
// MLX internally reciprocates this array to obtain inverse frequencies.
func Llama3Freqs(headDim int32, base, factor, lowFreqFactor, highFreqFactor float32, originalContext int32) ([]float32, error) {
	inv, err := Llama3InvFreqs(headDim, base, factor, lowFreqFactor, highFreqFactor, originalContext)
	if err != nil {
		return nil, err
	}
	freqs := make([]float32, len(inv))
	for i, v := range inv {
		if v == 0 {
			return nil, fmt.Errorf("zero llama3 inverse frequency at index %d", i)
		}
		freqs[i] = 1 / v
	}
	return freqs, nil
}

func checkRequiredTensors(tensors map[string]*mlx.Array, cfg *Config) error {
	required := []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
		"lm_head.weight",
	}
	for i := range cfg.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("model.layers.%d", i)
		required = append(required,
			layerPrefix+".attention_layernorm.weight",
			layerPrefix+".feedforward_layernorm.weight",
			layerPrefix+".self_attn.q_proj.weight",
			layerPrefix+".self_attn.k_proj.weight",
			layerPrefix+".self_attn.v_proj.weight",
			layerPrefix+".self_attn.o_proj.weight",
			layerPrefix+".self_attn.q_norm.weight",
			layerPrefix+".self_attn.k_norm.weight",
			layerPrefix+".mlp.up_proj.weight",
			layerPrefix+".mlp.down_proj.weight",
			layerPrefix+".mlp.act_fn.alpha_p",
			layerPrefix+".mlp.act_fn.alpha_n",
			layerPrefix+".mlp.act_fn.beta",
			layerPrefix+".mlp.act_fn.eps",
		)
	}
	for _, name := range required {
		if tensors[name] == nil {
			return fmt.Errorf("missing required tensor: %s", name)
		}
	}
	return nil
}

func qkNormShape(batch, seqLen, heads, headDim int32) []int32 {
	return []int32{batch, heads, seqLen, headDim}
}

func XIELUScalar(x, alphaPParam, alphaNParam, beta, eps float64) float64 {
	alphaP := math.Log1p(math.Exp(alphaPParam))
	alphaN := beta + math.Log1p(math.Exp(alphaNParam))
	if x > 0 {
		return alphaP*x*x + beta*x
	}
	return math.Expm1(math.Min(x, eps))*alphaN - x*alphaN + beta*x
}

// LoadWeights receives all tensors loaded from the manifest and assigns them
// to model fields.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := checkRequiredTensors(tensors, m.Config); err != nil {
		return err
	}

	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	embedTokens := model.MakeEmbeddingLayer(tensors, "model.embed_tokens", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	if embedTokens == nil {
		return fmt.Errorf("missing embedding weight: model.embed_tokens.weight")
	}
	m.EmbedTokens = embedTokens

	normWeight := tensors["model.norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: model.norm.weight")
	}
	m.Norm = nn.NewRMSNorm(normWeight, m.RMSNormEps)

	lmHead := linears.Make("lm_head")
	if lmHead == nil {
		return fmt.Errorf("missing lm_head weight: lm_head.weight")
	}
	m.LMHead = lmHead

	for i := range m.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("model.layers.%d", i)
		layer := &Layer{
			Attention: &Attention{},
			MLP:       &MLP{},
		}

		if w := tensors[layerPrefix+".attention_layernorm.weight"]; w != nil {
			layer.AttentionNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".feedforward_layernorm.weight"]; w != nil {
			layer.FFNNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
			layer.Attention.QNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.k_norm.weight"]; w != nil {
			layer.Attention.KNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}

		layer.Attention.QProj = linears.Make(layerPrefix + ".self_attn.q_proj")
		layer.Attention.KProj = linears.Make(layerPrefix + ".self_attn.k_proj")
		layer.Attention.VProj = linears.Make(layerPrefix + ".self_attn.v_proj")
		layer.Attention.OProj = linears.Make(layerPrefix + ".self_attn.o_proj")
		layer.MLP.UpProj = linears.Make(layerPrefix + ".mlp.up_proj")
		layer.MLP.DownProj = linears.Make(layerPrefix + ".mlp.down_proj")
		act, err := newXIELU(
			tensors[layerPrefix+".mlp.act_fn.alpha_p"],
			tensors[layerPrefix+".mlp.act_fn.alpha_n"],
			tensors[layerPrefix+".mlp.act_fn.beta"],
			tensors[layerPrefix+".mlp.act_fn.eps"],
		)
		if err != nil {
			return fmt.Errorf("layer %d: load xielu activation parameters: %w", i, err)
		}
		layer.MLP.Act = act

		if layer.AttentionNorm == nil {
			return fmt.Errorf("layer %d: missing attention_layernorm", i)
		}
		if layer.FFNNorm == nil {
			return fmt.Errorf("layer %d: missing feedforward_layernorm", i)
		}
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			return fmt.Errorf("layer %d: missing attention projections", i)
		}
		if layer.Attention.QNorm == nil || layer.Attention.KNorm == nil {
			return fmt.Errorf("layer %d: missing q/k norm", i)
		}
		if layer.MLP.UpProj == nil || layer.MLP.DownProj == nil {
			return fmt.Errorf("layer %d: missing mlp projections", i)
		}
		m.Layers[i] = layer
	}

	return nil
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))

	h := mlx.Reshape(m.EmbedTokens.Forward(b.InputIDs), B, L, m.HiddenSize)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
	}

	return mlx.Reshape(m.Norm.Forward(h, m.RMSNormEps), B, L, m.HiddenSize)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	return mlx.Reshape(m.LMHead.Forward(x), B, L, m.VocabSize)
}

func (m *Model) NumLayers() int {
	return len(m.Layers)
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.tok
}

func (m *Model) MaxContextLength() int {
	return int(m.MaxPositionEmbeddings)
}

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = cache.NewKVCache()
	}
	return caches
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	h := mlx.Add(x, l.Attention.Forward(l.AttentionNorm.Forward(x, cfg.RMSNormEps), b, c, positions, B, L, cfg))
	h = mlx.Reshape(h, B, L, cfg.HiddenSize)
	return mlx.Reshape(mlx.Add(h, l.MLP.Forward(l.FFNNorm.Forward(h, cfg.RMSNormEps))), B, L, cfg.HiddenSize)
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	qShape := qkNormShape(B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	q = headRMSNorm(a.QNorm, q, qShape[0], qShape[1], qShape[2], qShape[3], cfg.RMSNormEps)

	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	kShape := qkNormShape(B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	k = headRMSNorm(a.KNorm, k, kShape[0], kShape[1], kShape[2], kShape[3], cfg.RMSNormEps)

	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// RoPEWithFreqs uses cfg.RopeFreqs for the Llama 3 scaling table; the
	// base and scale arguments are still required by the wrapper signature.
	q = mlx.Reshape(mlx.RoPEWithFreqs(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions, cfg.RopeFreqs), B, cfg.NumAttentionHeads, L, cfg.HeadDim)
	k = mlx.Reshape(mlx.RoPEWithFreqs(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions, cfg.RopeFreqs), B, cfg.NumKeyValueHeads, L, cfg.HeadDim)

	var kv nn.SDPAOption
	if c != nil {
		history := c.(cache.Attention).Update(b, k, v)
		kv = nn.WithKVHistory(history)
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
	}
	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(nn.CausalMask()))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func headRMSNorm(norm *nn.RMSNorm, x *mlx.Array, batch, heads, seqLen, headDim int32, eps float32) *mlx.Array {
	x = mlx.Reshape(x, batch*heads*seqLen, headDim)
	x = norm.Forward(x, eps)
	return mlx.Reshape(x, batch, heads, seqLen, headDim)
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	return m.DownProj.Forward(m.Act.Forward(m.UpProj.Forward(x)))
}

func (a *XIELU) Forward(x *mlx.Array) *mlx.Array {
	outDType := x.DType()
	x = x.AsType(mlx.DTypeFloat32)
	zero := mlx.FromValue[float32](0)
	one := mlx.FromValue[float32](1)
	alphaP := mlx.FromValue(a.AlphaP)
	alphaN := mlx.FromValue(a.AlphaN)
	beta := mlx.FromValue(a.Beta)
	eps := mlx.FromValue(a.Eps)

	positive := mlx.Add(mlx.Mul(alphaP, mlx.Mul(x, x)), mlx.Mul(beta, x))

	xMin := mlx.Minimum(x, eps)
	// MLX exposes Exp but no Expm1 binding; xMin is clamped by eps before this
	// subtraction, matching the available graph operations.
	expm1 := mlx.Sub(mlx.Exp(xMin), one)
	negative := mlx.Add(
		mlx.Mul(mlx.Sub(expm1, x), alphaN),
		mlx.Mul(beta, x),
	)

	return mlx.Where(x.Greater(zero), positive, negative).AsType(outDType)
}

func newXIELU(alphaPParam, alphaNParam, betaParam, epsParam *mlx.Array) (*XIELU, error) {
	if alphaPParam == nil || alphaNParam == nil || betaParam == nil || epsParam == nil {
		return nil, fmt.Errorf("missing xielu activation parameter tensor")
	}

	alphaP, err := scalarParam(alphaPParam)
	if err != nil {
		return nil, fmt.Errorf("alpha_p: %w", err)
	}
	alphaN, err := scalarParam(alphaNParam)
	if err != nil {
		return nil, fmt.Errorf("alpha_n: %w", err)
	}
	beta, err := scalarParam(betaParam)
	if err != nil {
		return nil, fmt.Errorf("beta: %w", err)
	}
	eps, err := scalarParam(epsParam)
	if err != nil {
		return nil, fmt.Errorf("eps: %w", err)
	}

	return &XIELU{
		AlphaP: float32(softplus64(float64(alphaP))),
		AlphaN: beta + float32(softplus64(float64(alphaN))),
		Beta:   beta,
		Eps:    eps,
	}, nil
}

func scalarParam(x *mlx.Array) (float32, error) {
	if x.Size() != 1 {
		return 0, fmt.Errorf("expected scalar or single-element tensor, got shape %v", x.Dims())
	}
	x = x.AsType(mlx.DTypeFloat32)
	mlx.Eval(x)
	values := x.Floats()
	if len(values) != 1 {
		return 0, fmt.Errorf("expected one scalar value, got %d", len(values))
	}
	return values[0], nil
}

func softplus64(x float64) float64 {
	if x > 20 {
		return x
	}
	if x < -20 {
		return math.Exp(x)
	}
	return math.Log1p(math.Exp(x))
}
