// Package gptoss provides an experimental GPT-OSS MLX model implementation.
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

// Config holds the GPT-OSS fields needed for minimal MLX inference.
type Config struct {
	HiddenSize            int32   `json:"hidden_size"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	HeadDim               int32   `json:"head_dim"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	InitialContextLength  int32   `json:"initial_context_length"`
	RopeScalingFactor     float32 `json:"rope_scaling_factor"`
	RopeTheta             float32 `json:"rope_theta"`
	SlidingWindow         int32   `json:"sliding_window"`
	NumExperts            int32   `json:"num_experts"`
	LocalExperts          int32   `json:"num_local_experts"`
	ExpertsPerToken       int32   `json:"experts_per_token"`

	// Quantization metadata.
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	Scale          float32 `json:"-"`
	EffectiveRopeN int32   `json:"-"`
	ExpertCount    int32   `json:"-"`
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
	Weight    *mlx.Array
	Scales    *mlx.Array
	Biases    *mlx.Array
	Bits      int
	GroupSize int
	Mode      string
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

	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	cfg.EffectiveRopeN = cfg.HeadDim
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

func sliceAxis0AndMaybeSqueeze(a *mlx.Array, idx int32) *mlx.Array {
	if a == nil || !a.Valid() || a.NumDims() == 0 {
		return nil
	}
	dims := a.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	for i, d := range dims {
		stop[i] = int32(d)
	}
	start[0] = idx
	stop[0] = idx + 1
	s := mlx.SliceStartStop(a, start, stop)
	if len(dims) > 1 {
		s = mlx.Squeeze(s, 0)
	}
	return s
}

func dequantizeStackedExperts(weight, scales, qbiases *mlx.Array, groupSize, bits int, mode string) *mlx.Array {
	if weight == nil || !weight.Valid() || scales == nil || !scales.Valid() {
		return nil
	}
	if weight.NumDims() != 3 || scales.NumDims() != 3 {
		return mlx.Dequantize(weight, scales, qbiases, groupSize, bits, mode)
	}

	experts := weight.Dim(0)
	parts := make([]*mlx.Array, 0, experts)
	for e := 0; e < experts; e++ {
		w := sliceAxis0AndMaybeSqueeze(weight, int32(e))
		s := sliceAxis0AndMaybeSqueeze(scales, int32(e))
		var qb *mlx.Array
		if qbiases != nil && qbiases.Valid() {
			qb = sliceAxis0AndMaybeSqueeze(qbiases, int32(e))
		}
		part := mlx.Dequantize(w, s, qb, groupSize, bits, mode)
		if part == nil || !part.Valid() {
			return part
		}
		parts = append(parts, part)
	}
	return stackAndClone(parts)
}

func requireTensor(name string, t *mlx.Array) error {
	if t == nil || !t.Valid() {
		return fmt.Errorf("missing or invalid tensor: %s", name)
	}
	return nil
}

func requireVector(name string, t *mlx.Array, size int32) error {
	if err := requireTensor(name, t); err != nil {
		return err
	}
	if t.NumDims() != 1 {
		return fmt.Errorf("tensor %s: expected 1D shape [%d], got %v", name, size, t.Dims())
	}
	if size > 0 && t.Dim(0) != int(size) {
		return fmt.Errorf("tensor %s: expected shape [%d], got %v", name, size, t.Dims())
	}
	return nil
}

func requireStackedExperts(name string, t *mlx.Array, experts int32) error {
	if err := requireTensor(name, t); err != nil {
		return err
	}
	if t.NumDims() != 3 {
		return fmt.Errorf("tensor %s: expected 3D expert stack, got %v", name, t.Dims())
	}
	if experts > 0 && t.Dim(0) != int(experts) {
		return fmt.Errorf("tensor %s: expected %d experts, got %v", name, experts, t.Dims())
	}
	return nil
}

func requireRuntimeShape(name string, t *mlx.Array, want ...int32) *mlx.Array {
	if t == nil || !t.Valid() {
		panic(fmt.Sprintf("gptoss %s produced an invalid MLX array", name))
	}
	dims := t.Dims()
	if len(dims) != len(want) {
		panic(fmt.Sprintf("gptoss %s expected rank %d shape %v, got %v", name, len(want), want, dims))
	}
	for i, dim := range want {
		if dim >= 0 && dims[i] != int(dim) {
			panic(fmt.Sprintf("gptoss %s expected shape %v, got %v", name, want, dims))
		}
	}
	return t
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
			Weight:    dequantizeStackedExperts(w, scales, qbiases, groupSize, bits, mode),
			Bits:      bits,
			GroupSize: groupSize,
			Mode:      mode,
		}
	}

	return nil
}

func LoadWeightsSummaryMissing(layer int32, gate, up, down *stackedExpertWeights) error {
	return fmt.Errorf("layer %d: missing moe expert weights (gate=%v up=%v down=%v)", layer, gate != nil, up != nil, down != nil)
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	cfg := m.Config
	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)

	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, "token_embd", cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing embedding weight: token_embd.weight")
	}

	normWeight := tensors["output_norm.weight"]
	if err := requireVector("output_norm.weight", normWeight, cfg.HiddenSize); err != nil {
		return err
	}
	m.Norm = nn.NewRMSNorm(normWeight, cfg.RMSNormEps)

	m.LMHead = linears.Make("output")
	if m.LMHead == nil {
		m.LMHead = m.EmbedTokens.AsLinear()
	}

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		layerPrefix := fmt.Sprintf("blk.%d", i)
		layer := &Layer{
			Attention: &Attention{},
			MoE:       &MoE{SwitchMLP: &SwitchMLP{}},
		}

		if w := tensors[layerPrefix+".attn_norm.weight"]; w != nil {
			if err := requireVector(layerPrefix+".attn_norm.weight", w, cfg.HiddenSize); err != nil {
				return err
			}
			layer.AttentionNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if w := tensors[layerPrefix+".ffn_norm.weight"]; w != nil {
			if err := requireVector(layerPrefix+".ffn_norm.weight", w, cfg.HiddenSize); err != nil {
				return err
			}
			layer.MLPNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if layer.AttentionNorm == nil || layer.MLPNorm == nil {
			return fmt.Errorf("layer %d: missing layer norms", i)
		}

		layer.Attention.QProj = linears.Make(layerPrefix + ".attn_q")
		layer.Attention.KProj = linears.Make(layerPrefix + ".attn_k")
		layer.Attention.VProj = linears.Make(layerPrefix + ".attn_v")
		layer.Attention.OProj = linears.Make(layerPrefix + ".attn_out")
		if sinks := tensors[layerPrefix+".attn_sinks"]; sinks != nil {
			if err := requireVector(layerPrefix+".attn_sinks", sinks, cfg.NumAttentionHeads); err != nil {
				return err
			}
			layer.Attention.Sinks = sinks
		}
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			return fmt.Errorf("layer %d: missing attention projections", i)
		}

		layer.MoE.Gate = linears.Make(layerPrefix + ".ffn_gate_inp")
		if layer.MoE.Gate == nil {
			return fmt.Errorf("layer %d: missing moe gate", i)
		}

		gateW := loadStackedProjection(tensors, cfg, layerPrefix+".ffn_gate_exps")
		upW := loadStackedProjection(tensors, cfg, layerPrefix+".ffn_up_exps")
		downW := loadStackedProjection(tensors, cfg, layerPrefix+".ffn_down_exps")
		if gateW == nil || upW == nil || downW == nil {
			return LoadWeightsSummaryMissing(i, gateW, upW, downW)
		}

		layer.MoE.SwitchMLP.GateWeight = gateW.Weight
		layer.MoE.SwitchMLP.UpWeight = upW.Weight
		layer.MoE.SwitchMLP.DownWeight = downW.Weight
		if layer.MoE.SwitchMLP.GateWeight == nil || layer.MoE.SwitchMLP.UpWeight == nil || layer.MoE.SwitchMLP.DownWeight == nil {
			return fmt.Errorf("layer %d: invalid stacked moe weights", i)
		}
		if err := requireStackedExperts(layerPrefix+".ffn_gate_exps.weight", layer.MoE.SwitchMLP.GateWeight, cfg.ExpertCount); err != nil {
			return err
		}
		if err := requireStackedExperts(layerPrefix+".ffn_up_exps.weight", layer.MoE.SwitchMLP.UpWeight, cfg.ExpertCount); err != nil {
			return err
		}
		if err := requireStackedExperts(layerPrefix+".ffn_down_exps.weight", layer.MoE.SwitchMLP.DownWeight, cfg.ExpertCount); err != nil {
			return err
		}
		if layer.MoE.SwitchMLP.GateWeight.Dim(2) != int(cfg.HiddenSize) {
			return fmt.Errorf("tensor %s: expected input hidden dimension %d, got %v", layerPrefix+".ffn_gate_exps.weight", cfg.HiddenSize, layer.MoE.SwitchMLP.GateWeight.Dims())
		}
		if layer.MoE.SwitchMLP.UpWeight.Dim(2) != int(cfg.HiddenSize) {
			return fmt.Errorf("tensor %s: expected input hidden dimension %d, got %v", layerPrefix+".ffn_up_exps.weight", cfg.HiddenSize, layer.MoE.SwitchMLP.UpWeight.Dims())
		}
		if layer.MoE.SwitchMLP.DownWeight.Dim(1) != int(cfg.HiddenSize) {
			return fmt.Errorf("tensor %s: expected output hidden dimension %d, got %v", layerPrefix+".ffn_down_exps.weight", cfg.HiddenSize, layer.MoE.SwitchMLP.DownWeight.Dims())
		}

		m.Layers[i] = layer
	}

	return nil
}

func (a *Attention) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	x = requireRuntimeShape("attention input", x, B, L, cfg.HiddenSize)
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)

	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	k = mlx.Transpose(k, 0, 2, 1, 3)

	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	mlxDebugTensor("q_pre_rope", q)
	mlxDebugTensor("k_pre_rope", k)
	mlxDebugTensor("v_pre_rope", v)

	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	ropeScale := float32(1.0 / cfg.RopeScalingFactor)
	q = mlx.RoPEWithBase(q, int(cfg.EffectiveRopeN), false, cfg.RopeTheta, ropeScale, offset)
	k = mlx.RoPEWithBase(k, int(cfg.EffectiveRopeN), false, cfg.RopeTheta, ropeScale, offset)
	mlxDebugTensor("q_post_rope", q)
	mlxDebugTensor("k_post_rope", k)

	if c != nil {
		k, v = c.Update(k, v)
	}

	out := mlx.ScaledDotProductAttentionCausalWithSinks(q, k, v, a.Sinks, cfg.Scale, L > 1)
	requireRuntimeShape("attention sdpa output", out, B, cfg.NumAttentionHeads, L, cfg.HeadDim)
	mlxDebugTensor("sdpa_out", out)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	out = a.OProj.Forward(out)
	return requireRuntimeShape("attention output projection", out, B, L, cfg.HiddenSize)
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

	gate := mlx.GatherMM(xFlat, mlx.Transpose(s.GateWeight, 0, 2, 1), nil, idxFlat, doSort)
	up := mlx.GatherMM(xFlat, mlx.Transpose(s.UpWeight, 0, 2, 1), nil, idxFlat, doSort)
	hidden := mlx.SwiGLUAlphaLimit(gate, up, 1.702, 7.0)
	down := mlx.GatherMM(hidden, mlx.Transpose(s.DownWeight, 0, 2, 1), nil, idxFlat, doSort)

	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

func (m *MoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	x = requireRuntimeShape("moe input", x, -1, -1, cfg.HiddenSize)
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	logits := m.Gate.Forward(x)
	mlxDebugTensor("router_logits", logits)
	neg := mlx.Neg(logits)
	inds := mlx.Argpartition(neg, int(cfg.ExpertsPerToken)-1, -1)
	shape := inds.Dims()
	inds = mlx.SliceStartStop(inds, []int32{0, 0, 0}, []int32{int32(shape[0]), int32(shape[1]), cfg.ExpertsPerToken})

	scores := mlx.TakeAlongAxis(logits, inds, -1)
	order := mlx.Argsort(mlx.Neg(scores), -1)
	inds = mlx.TakeAlongAxis(inds, order, -1)
	scores = mlx.TakeAlongAxis(scores, order, -1)
	scores = mlx.SoftmaxAxis(scores, -1, true)
	mlxDebugExperts("router_topk", inds, scores)

	expertOut := m.SwitchMLP.Forward(x, inds, cfg)
	y := mlx.Sum(mlx.Mul(expertOut, mlx.ExpandDims(scores, -1)), 2, false)
	out := mlx.Reshape(y, B, L, cfg.HiddenSize)
	mlxDebugTensor("post_mlp_residual", out)
	return out
}

func (l *Layer) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	x = requireRuntimeShape("layer input", x, B, L, cfg.HiddenSize)
	norm := l.AttentionNorm.Forward(x, cfg.RMSNormEps)
	mlxDebugTensor("attn_norm", norm)
	attn := l.Attention.Forward(norm, c, B, L, cfg)
	attn = requireRuntimeShape("layer attention residual", attn, B, L, cfg.HiddenSize)
	h := mlx.Add(x, attn)
	h = requireRuntimeShape("layer post-attention state", h, B, L, cfg.HiddenSize)
	ffn := l.MoE.Forward(l.MLPNorm.Forward(h, cfg.RMSNormEps), cfg)
	ffn = requireRuntimeShape("layer ffn residual", ffn, B, L, cfg.HiddenSize)
	return mlx.Add(h, ffn)
}

func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	dims := tokens.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	_ = mlxDebugBegin(tokens, caches, m.Config)
	h := m.EmbedTokens.Forward(tokens)
	h = requireRuntimeShape("token embedding", h, B, L, m.HiddenSize)
	mlxDebugTensor("embedding", h)
	for i, layer := range m.Layers {
		mlxDebugSetLayer(i)
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, c, B, L, m.Config)
	}

	return requireRuntimeShape("final norm", m.Norm.Forward(h, m.RMSNormEps), B, L, m.HiddenSize)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	logits := m.LMHead.Forward(x)
	mlxDebugLogits("final_logits", logits, 8)
	mlxDebugFinish()
	return logits
}

func (m *Model) NumLayers() int {
	return len(m.Layers)
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.tok
}

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i := range caches {
		if m.SlidingWindow > 0 && i%2 == 0 {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
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
