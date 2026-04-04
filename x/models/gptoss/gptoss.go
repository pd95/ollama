// Package gptoss provides an experimental GPT-OSS MLX model skeleton.
package gptoss

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"sync/atomic"

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
	SlidingWindow         int32   `json:"sliding_window"`
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

var gptossFirstDecodeNormTrace atomic.Bool
var gptossFirstDecodeLayerTrace atomic.Bool
var gptossBadDecodeLayer atomic.Int32

const gptossTraceMoELayer int32 = 5

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
			Weight: dequantizeStackedExperts(w, scales, qbiases, groupSize, bits, mode),
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
	if os.Getenv("OLLAMA_GPTOSS_F32_OUTPUT_EDGE") != "" && normWeight.DType() == mlx.DTypeBFloat16 {
		normWeight = normWeight.AsType(mlx.DTypeFloat32)
	}
	m.Norm = nn.NewRMSNorm(normWeight, m.RMSNormEps)

	if os.Getenv("OLLAMA_GPTOSS_F32_OUTPUT_EDGE") != "" {
		if w := tensors["output.weight"]; w != nil && tensors["output.weight_scale"] == nil && w.DType() == mlx.DTypeBFloat16 {
			var bias *mlx.Array
			if b := tensors["output.bias"]; b != nil {
				bias = b
				if bias.DType() == mlx.DTypeBFloat16 {
					bias = bias.AsType(mlx.DTypeFloat32)
				}
			}
			m.LMHead = nn.NewLinear(w.AsType(mlx.DTypeFloat32), bias)
		}
	}
	if m.LMHead == nil {
		m.LMHead = linears.Make("output")
	}
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

		layer.MoE.SwitchMLP.GateWeight = gateW.Weight
		layer.MoE.SwitchMLP.UpWeight = upW.Weight
		layer.MoE.SwitchMLP.DownWeight = downW.Weight
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
	traceLayers := shouldTraceFirstDecodeLayers(L, caches)
	if traceLayers {
		gptossBadDecodeLayer.Store(0)
		defer gptossFirstDecodeLayerTrace.Store(true)
	}
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, c, B, L, m.Config, int32(i), traceLayers)
	}
	traceNorm := shouldTraceFirstDecodeNorm(L, caches)
	if traceNorm {
		logTensorStats("final_norm_before", h)
	}
	out := m.Norm.Forward(h, m.RMSNormEps)
	if traceNorm {
		logTensorStats("final_norm_out", out)
	}
	return out
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	if m.LMHead == nil {
		panic("gptoss MLX unembed called before weights loaded")
	}
	return m.LMHead.Forward(x)
}

func shouldTraceFirstDecodeLayers(L int32, caches []cache.Cache) bool {
	if os.Getenv("OLLAMA_GPTOSS_STEP_DEBUG") == "" || L != 1 {
		return false
	}
	if len(caches) == 0 || caches[0] == nil || caches[0].Offset() == 0 {
		return false
	}
	return !gptossFirstDecodeLayerTrace.Load()
}

func shouldTraceFirstDecodeNorm(L int32, caches []cache.Cache) bool {
	if os.Getenv("OLLAMA_GPTOSS_STEP_DEBUG") == "" || L != 1 {
		return false
	}
	if len(caches) == 0 || caches[0] == nil || caches[0].Offset() == 0 {
		return false
	}
	return gptossFirstDecodeNormTrace.CompareAndSwap(false, true)
}

type tensorSummary struct {
	valid bool
	min   float32
	max   float32
	mean  float64
	std   float32
}

func tensorStats(t *mlx.Array) tensorSummary {
	if t == nil || !t.Valid() {
		return tensorSummary{}
	}
	tf := t.AsType(mlx.DTypeFloat32)
	mlx.Eval(tf)
	vals := tf.Floats()
	if len(vals) == 0 {
		return tensorSummary{}
	}
	minV, maxV := vals[0], vals[0]
	sum := float64(0)
	for _, v := range vals {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
		sum += float64(v)
	}
	mean := sum / float64(len(vals))
	var sq float64
	for _, v := range vals {
		d := float64(v) - mean
		sq += d * d
	}
	std := float32(math.Sqrt(sq / float64(len(vals))))
	return tensorSummary{valid: true, min: minV, max: maxV, mean: mean, std: std}
}

func logTensorStats(stage string, t *mlx.Array) {
	stats := tensorStats(t)
	if !stats.valid {
		return
	}
	tf := t.AsType(mlx.DTypeFloat32)
	mlx.Eval(tf)
	vals := tf.Floats()
	n := 8
	if len(vals) < n {
		n = len(vals)
	}
	slog.Info(stage, "shape", t.Dims(), "dtype", t.DType(), "min", stats.min, "max", stats.max, "mean", stats.mean, "std", stats.std, "sample", vals[:n])
}

func shouldTraceLayer(layer int32, traceLayers bool) bool {
	if !traceLayers {
		return false
	}
	bad := gptossBadDecodeLayer.Load()
	return bad == 0 || bad == layer+1
}

func maybeMarkBadLayer(layer int32, stage string, stats tensorSummary) {
	if !stats.valid || gptossBadDecodeLayer.Load() != 0 {
		return
	}
	maxAbs := stats.max
	if -stats.min > maxAbs {
		maxAbs = -stats.min
	}
	if maxAbs < 512 && stats.std < 128 {
		return
	}
	if gptossBadDecodeLayer.CompareAndSwap(0, layer+1) {
		slog.Info("decode_first_bad_layer", "layer", layer, "after", stage, "min", stats.min, "max", stats.max, "mean", stats.mean, "std", stats.std)
	}
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
		if m.SlidingWindow > 0 && i%2 == 0 {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

func (l *Layer) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config, layerIndex int32, traceLayers bool) *mlx.Array {
	if shouldTraceLayer(layerIndex, traceLayers) {
		stats := tensorStats(x)
		if stats.valid {
			slog.Info("decode_layer_input", "layer", layerIndex, "shape", x.Dims(), "dtype", x.DType(), "min", stats.min, "max", stats.max, "mean", stats.mean, "std", stats.std)
		}
	}
	attn := l.Attention.Forward(l.AttentionNorm.Forward(x, cfg.RMSNormEps), c, B, L, cfg)
	h := mlx.Add(x, attn)
	if shouldTraceLayer(layerIndex, traceLayers) {
		stats := tensorStats(h)
		if stats.valid {
			slog.Info("decode_post_attn", "layer", layerIndex, "shape", h.Dims(), "dtype", h.DType(), "min", stats.min, "max", stats.max, "mean", stats.mean, "std", stats.std)
			maybeMarkBadLayer(layerIndex, "attention", stats)
		}
	}
	ffn := l.MoE.Forward(l.MLPNorm.Forward(h, cfg.RMSNormEps), cfg, layerIndex, traceLayers)
	out := mlx.Add(h, ffn)
	if shouldTraceLayer(layerIndex, traceLayers) && gptossBadDecodeLayer.Load() == 0 {
		stats := tensorStats(out)
		if stats.valid {
			slog.Info("decode_post_mlp", "layer", layerIndex, "shape", out.Dims(), "dtype", out.DType(), "min", stats.min, "max", stats.max, "mean", stats.mean, "std", stats.std)
			maybeMarkBadLayer(layerIndex, "mlp", stats)
		}
	}
	return out
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
	out := mlx.ScaledDotProductAttentionCausalWithSinks(q, k, v, a.Sinks, scale, L > 1)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (s *SwitchMLP) Forward(x *mlx.Array, indices *mlx.Array, cfg *Config, traceMoE bool) *mlx.Array {
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
	if traceMoE {
		logMoEStats("moe_gate", gate)
	}
	up := mlx.GatherMM(xFlat, mlx.Transpose(s.UpWeight, 0, 2, 1), nil, idxFlat, doSort)
	if traceMoE {
		logMoEStats("moe_up", up)
	}
	hidden := mlx.SwiGLUAlphaLimit(gate, up, 1.702, 7.0)
	if traceMoE {
		logMoEStats("moe_post_swiglu", hidden)
	}
	down := mlx.GatherMM(hidden, mlx.Transpose(s.DownWeight, 0, 2, 1), nil, idxFlat, doSort)
	if traceMoE {
		logMoEStats("moe_down_raw", down)
	}

	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	out := mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
	if traceMoE {
		logMoEStats("moe_down", out)
	}
	return out
}

func (m *MoE) Forward(x *mlx.Array, cfg *Config, layerIndex int32, traceLayers bool) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	traceMoE := traceLayers && layerIndex == gptossTraceMoELayer

	logits := m.Gate.Forward(x)
	neg := mlx.Neg(logits)
	inds := mlx.Argpartition(neg, int(cfg.ExpertsPerToken)-1, -1)
	shape := inds.Dims()
	inds = mlx.SliceStartStop(inds, []int32{0, 0, 0}, []int32{int32(shape[0]), int32(shape[1]), cfg.ExpertsPerToken})

	scores := mlx.TakeAlongAxis(logits, inds, -1)
	order := mlx.Argsort(mlx.Neg(scores), -1)
	inds = mlx.TakeAlongAxis(inds, order, -1)
	scores = mlx.TakeAlongAxis(scores, order, -1)
	if traceMoE {
		mlx.Eval(inds, scores)
		slog.Info("moe_router_selected", "layer", layerIndex, "ids", inds.Ints(), "scores", scores.Floats())
	}
	scores = mlx.SoftmaxAxis(scores, -1, true)
	if traceMoE {
		mlx.Eval(scores)
		slog.Info("moe_router_probs", "layer", layerIndex, "probs", scores.Floats())
	}
	expertOut := m.SwitchMLP.Forward(x, inds, cfg, traceMoE)
	y := mlx.Sum(mlx.Mul(expertOut, mlx.ExpandDims(scores, -1)), 2, false)
	out := mlx.Reshape(y, B, L, cfg.HiddenSize)
	if traceMoE {
		logMoEStats("moe_post_residual_mlp", out)
	}
	return out
}

func logMoEStats(stage string, t *mlx.Array) {
	stats := tensorStats(t)
	if !stats.valid {
		return
	}
	slog.Info(stage, "shape", t.Dims(), "dtype", t.DType(), "min", stats.min, "max", stats.max, "mean", stats.mean, "std", stats.std)
}
