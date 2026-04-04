// Package gptoss provides an experimental GPT-OSS MLX model skeleton.
package gptoss

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("GptOssForCausalLM", NewModel)
}

// Config holds the GPT-OSS fields needed for model construction.
type Config struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	HeadDim               int32   `json:"head_dim"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	InitialContextLength  int32   `json:"initial_context_length"`
	RopeScalingFactor     float32 `json:"rope_scaling_factor"`
	RopeTheta             float32 `json:"rope_theta"`
	NumExperts            int32   `json:"num_experts"`
	LocalExperts          int32   `json:"num_local_experts"`
	ExpertsPerToken       int32   `json:"experts_per_token"`
	RopeScaling           struct {
		Factor float32 `json:"factor"`
	} `json:"rope_scaling"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`
	ExpertCount    int32                             `json:"-"`
}

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
	MLPNorm       *nn.RMSNorm
	MoE           *MoE
}

type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
	Sinks *mlx.Array
}

type MoE struct {
	Gate      nn.LinearLayer
	SwitchMLP *SwitchMLP
}

type SwitchMLP struct {
	GateWeight *mlx.Array
	UpWeight   *mlx.Array
	DownWeight *mlx.Array
}

type stackedExpertWeights struct {
	Weight *mlx.Array
}

func parseConfig(configData []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}

	if cfg.HiddenSize <= 0 {
		return Config{}, fmt.Errorf("invalid hidden_size: %d", cfg.HiddenSize)
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
	if cfg.HeadDim <= 0 {
		if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
			return Config{}, fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
		}
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return Config{}, fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-5
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	if cfg.RopeScalingFactor == 0 {
		cfg.RopeScalingFactor = cfg.RopeScaling.Factor
	}
	if cfg.RopeScalingFactor == 0 {
		cfg.RopeScalingFactor = 1
	}
	if cfg.ExpertsPerToken <= 0 {
		cfg.ExpertsPerToken = 1
	}
	cfg.ExpertCount = cfg.NumExperts
	if cfg.ExpertCount <= 0 {
		cfg.ExpertCount = cfg.LocalExperts
	}
	if cfg.ExpertCount <= 0 {
		return Config{}, fmt.Errorf("invalid expert count: num_experts=%d local_experts=%d", cfg.NumExperts, cfg.LocalExperts)
	}
	if cfg.HeadDim <= 0 {
		return Config{}, fmt.Errorf("invalid head_dim: %d", cfg.HeadDim)
	}

	return cfg, nil
}

func NewModel(root *model.Root) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	cfg, err := parseConfig(configData)
	if err != nil {
		return nil, err
	}

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

	tokConfig := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	if genConfigData, err := root.Manifest.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = genConfigData
	}
	if tokConfigData, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = tokConfigData
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	return &Model{
		Layers: make([]*Layer, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}, nil
}

func tensorAny(tensors map[string]*mlx.Array, keys ...string) (*mlx.Array, string) {
	for _, key := range keys {
		if t := tensors[key]; t != nil {
			return t, key
		}
	}
	return nil, ""
}

func stackAndClone(parts []*mlx.Array) *mlx.Array {
	if len(parts) == 0 {
		return nil
	}
	stacked := mlx.Stack(parts, 0)
	cloned := stacked.Clone()
	mlx.Eval(cloned)
	return cloned
}

func transposeExpertWeightForGatherMM(w *mlx.Array) *mlx.Array {
	if w == nil || !w.Valid() || w.NumDims() != 3 {
		return w
	}
	t := mlx.Transpose(w, 0, 2, 1)
	cloned := t.Clone()
	mlx.Eval(cloned)
	return cloned
}

func loadStackedProjection(tensors map[string]*mlx.Array, cfg *Config, bases ...string) *stackedExpertWeights {
	for _, base := range bases {
		w, key := tensorAny(tensors, base+".weight", base)
		if w == nil {
			continue
		}

		scales := tensors[key+"_scale"]
		if scales == nil {
			return &stackedExpertWeights{Weight: w}
		}

		qbiases := tensors[key+"_qbias"]
		groupSize, bits, mode := model.ResolveLinearQuantParams(
			cfg.QuantGroupSize,
			cfg.QuantBits,
			cfg.QuantMode,
			cfg.TensorQuant,
			key,
			w,
			scales,
		)

		return &stackedExpertWeights{
			Weight: mlx.Dequantize(w, scales, qbiases, groupSize, bits, mode),
		}
	}

	return nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, "token_embd", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing embedding weight: token_embd.weight")
	}

	normWeight := tensors["output_norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: output_norm.weight")
	}
	m.Norm = nn.NewRMSNorm(normWeight, m.RMSNormEps)

	m.LMHead = linears.Make("output")
	if m.LMHead == nil {
		m.LMHead = m.EmbedTokens.AsLinear()
	}

	for i := int32(0); i < m.NumHiddenLayers; i++ {
		layerPrefix := fmt.Sprintf("blk.%d", i)
		layer := &Layer{
			Attention: &Attention{},
			MoE:       &MoE{SwitchMLP: &SwitchMLP{}},
		}

		if w := tensors[layerPrefix+".attn_norm.weight"]; w != nil {
			layer.AttentionNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".ffn_norm.weight"]; w != nil {
			layer.MLPNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if layer.AttentionNorm == nil || layer.MLPNorm == nil {
			return fmt.Errorf("layer %d: missing layer norms", i)
		}

		layer.Attention.QProj = linears.Make(layerPrefix + ".attn_q")
		layer.Attention.KProj = linears.Make(layerPrefix + ".attn_k")
		layer.Attention.VProj = linears.Make(layerPrefix + ".attn_v")
		layer.Attention.OProj = linears.Make(layerPrefix + ".attn_out")
		layer.Attention.Sinks = tensors[layerPrefix+".attn_sinks"]
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			return fmt.Errorf("layer %d: missing attention projections", i)
		}

		layer.MoE.Gate = linears.Make(layerPrefix + ".ffn_gate_inp")
		if layer.MoE.Gate == nil {
			return fmt.Errorf("layer %d: missing moe gate", i)
		}

		gateW := loadStackedProjection(tensors, m.Config, layerPrefix+".ffn_gate_exps")
		upW := loadStackedProjection(tensors, m.Config, layerPrefix+".ffn_up_exps")
		downW := loadStackedProjection(tensors, m.Config, layerPrefix+".ffn_down_exps")
		if gateW == nil || upW == nil || downW == nil {
			return fmt.Errorf("layer %d: missing moe expert weights", i)
		}

		layer.MoE.SwitchMLP.GateWeight = transposeExpertWeightForGatherMM(gateW.Weight)
		layer.MoE.SwitchMLP.UpWeight = transposeExpertWeightForGatherMM(upW.Weight)
		layer.MoE.SwitchMLP.DownWeight = transposeExpertWeightForGatherMM(downW.Weight)
		if layer.MoE.SwitchMLP.GateWeight == nil || layer.MoE.SwitchMLP.UpWeight == nil || layer.MoE.SwitchMLP.DownWeight == nil {
			return fmt.Errorf("layer %d: invalid moe expert weights", i)
		}

		m.Layers[i] = layer
	}

	return nil
}

func (m *Model) Forward(inputs *mlx.Array, caches []cache.Cache) *mlx.Array {
	dims := inputs.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	h := m.EmbedTokens.Forward(inputs)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, c, B, L, m.Config)
	}

	return m.Norm.Forward(h, m.RMSNormEps)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	if m.LMHead == nil {
		panic("gptoss MLX unembed called before weights loaded")
	}
	return m.LMHead.Forward(x)
}

func (m *Model) NumLayers() int {
	return len(m.Layers)
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.tok
}

func (m *Model) MaxContextLength() int {
	if m.MaxPositionEmbeddings > 0 {
		return int(m.MaxPositionEmbeddings)
	}
	if m.InitialContextLength > 0 {
		if m.RopeScalingFactor > 0 {
			return int(math.Round(float64(m.RopeScalingFactor * float32(m.InitialContextLength))))
		}
		return int(m.InitialContextLength)
	}
	return 0
}

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = cache.NewKVCache()
	}
	return caches
}

func (l *Layer) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	attn := l.Attention.Forward(l.AttentionNorm.Forward(x, cfg.RMSNormEps), c, B, L, cfg)
	h := mlx.Add(x, attn)
	ffn := l.MoE.Forward(l.MLPNorm.Forward(h, cfg.RMSNormEps), cfg)
	return mlx.Add(h, ffn)
}

func (a *Attention) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)

	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	k = mlx.Transpose(k, 0, 2, 1, 3)

	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	ropeScale := float32(1.0 / cfg.RopeScalingFactor)
	q = mlx.RoPEWithBase(q, int(cfg.HeadDim), false, cfg.RopeTheta, ropeScale, offset)
	k = mlx.RoPEWithBase(k, int(cfg.HeadDim), false, cfg.RopeTheta, ropeScale, offset)

	if c != nil {
		k, v = c.Update(k, v)
	}

	scale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	_ = a.Sinks
	out := mlx.ScaledDotProductAttentionCausal(q, k, v, scale, L > 1)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (s *SwitchMLP) Forward(x *mlx.Array, indices *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := cfg.ExpertsPerToken

	xExpanded := mlx.ExpandDims(mlx.ExpandDims(x, -2), -2)
	xFlat := mlx.Reshape(xExpanded, B*L, 1, 1, cfg.HiddenSize)
	idxFlat := mlx.Reshape(indices, B*L, topK)

	doSort := B*L >= 64
	var invOrder *mlx.Array
	n := B * L * topK

	if doSort {
		idxAll := mlx.Flatten(idxFlat)
		order := mlx.Argsort(idxAll, 0)
		invOrder = mlx.Argsort(order, 0)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), mlx.FloorDivideScalar(order, topK), 0), 1)
		idxFlat = mlx.Reshape(mlx.Take(idxAll, order, 0), n, 1)
	}

	gate := mlx.GatherMM(xFlat, s.GateWeight, nil, idxFlat, doSort)
	up := mlx.GatherMM(xFlat, s.UpWeight, nil, idxFlat, doSort)
	hidden := mlx.Mul(mlx.SiLU(gate), up)
	down := mlx.GatherMM(hidden, s.DownWeight, nil, idxFlat, doSort)

	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

func (m *MoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	probs := mlx.SoftmaxAxis(m.Gate.Forward(x), -1, true)
	neg := mlx.Neg(probs)
	inds := mlx.Argpartition(neg, int(cfg.ExpertsPerToken)-1, -1)
	shape := inds.Dims()
	inds = mlx.SliceStartStop(inds, []int32{0, 0, 0}, []int32{int32(shape[0]), int32(shape[1]), cfg.ExpertsPerToken})

	scores := mlx.TakeAlongAxis(probs, inds, -1)
	expertOut := m.SwitchMLP.Forward(x, inds, cfg)
	y := mlx.Sum(mlx.Mul(expertOut, mlx.ExpandDims(scores, -1)), 2, false)
	return mlx.Reshape(y, B, L, cfg.HiddenSize)
}
