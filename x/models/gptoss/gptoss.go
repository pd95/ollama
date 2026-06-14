// Package gptoss provides the gpt-oss text model implementation for MLX.
package gptoss

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
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
	base.Register("GptOssForCausalLM", NewModel)
}

var harmonyEOSTokens = []string{"<|endoftext|>", "<|return|>", "<|call|>"}

// RopeScaling carries the gpt-oss rope scaling block.
type RopeScaling struct {
	Factor                        float32 `json:"factor"`
	OriginalMaxPositionEmbeddings int32   `json:"original_max_position_embeddings"`
	RopeType                      string  `json:"rope_type,omitempty"`
	BetaFast                      float32 `json:"beta_fast,omitempty"`
	BetaSlow                      float32 `json:"beta_slow,omitempty"`
	Truncate                      bool    `json:"truncate,omitempty"`
}

// Quantization carries optional quantization metadata from config.json.
type Quantization struct {
	Bits        int    `json:"bits"`
	GroupSize   int    `json:"group_size"`
	Mode        string `json:"mode"`
	QuantMethod string `json:"quant_method"`
}

// Config holds the gpt-oss model configuration.
type Config struct {
	Architecture          string      `json:"-"`
	ModelType             string      `json:"model_type"`
	NumHiddenLayers       int32       `json:"num_hidden_layers"`
	HiddenSize            int32       `json:"hidden_size"`
	IntermediateSize      int32       `json:"intermediate_size"`
	NumAttentionHeads     int32       `json:"num_attention_heads"`
	NumKeyValueHeads      int32       `json:"num_key_value_heads"`
	HeadDim               int32       `json:"head_dim"`
	NumLocalExperts       int32       `json:"num_local_experts"`
	NumExpertsPerTok      int32       `json:"num_experts_per_tok"`
	SlidingWindow         int32       `json:"sliding_window"`
	RopeTheta             float32     `json:"rope_theta"`
	RopeScaling           RopeScaling `json:"rope_scaling"`
	RMSNormEps            float32     `json:"rms_norm_eps"`
	VocabSize             int32       `json:"vocab_size"`
	TieWordEmbeddings     bool        `json:"tie_word_embeddings"`
	MaxPositionEmbeddings int32       `json:"max_position_embeddings"`

	Quantization       Quantization `json:"quantization"`
	QuantizationConfig Quantization `json:"quantization_config"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`
	QuantMethod    string                            `json:"-"`
}

// RopeParameters returns the runtime rope settings derived from config.
func (c *Config) RopeParameters() (base, scale float32, originalContext int) {
	if c == nil {
		return 0, 1, 0
	}
	base = c.RopeTheta
	scale = 1
	if c.RopeScaling.Factor > 0 {
		scale = 1 / c.RopeScaling.Factor
	}
	if c.RopeScaling.OriginalMaxPositionEmbeddings > 0 {
		originalContext = int(c.RopeScaling.OriginalMaxPositionEmbeddings)
	}
	return base, scale, originalContext
}

// Model is the gpt-oss text-only model.
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config
}

// Layer is a single gpt-oss decoder block.
type Layer struct {
	AttentionNorm *nn.RMSNorm
	Attention     *Attention
	FFNNorm       *nn.RMSNorm
	Router        nn.LinearLayer
	Experts       *Experts
}

// Attention implements the split gpt-oss attention path.
type Attention struct {
	QProj     nn.LinearLayer
	KProj     nn.LinearLayer
	VProj     nn.LinearLayer
	OProj     nn.LinearLayer
	Sinks     *mlx.Array
	RoPEFreqs *mlx.Array
	RoPEScale float32
}

// Experts holds the loaded gpt-oss MoE expert projections.
type Experts struct {
	GateUp *ExpertPair
	Down   *ExpertProjection
}

// ExpertPair stores the split gate and up projections from the packed expert tensor.
type ExpertPair struct {
	Gate *ExpertProjection
	Up   *ExpertProjection
}

// ExpertProjection wraps a per-expert weight matrix.
// When Scales is non-nil, the weight is quantized and Forward uses GatherQMM.
type ExpertProjection struct {
	Weight    *mlx.Array
	Bias      *mlx.Array
	Scales    *mlx.Array
	QBiases   *mlx.Array
	GroupSize int
	Bits      int
	Mode      string
	Transpose bool // true for MLX-native [experts, out, packed_in] layout
}

// NewModel creates a gpt-oss model from a manifest root.
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
		if cfg.QuantMethod == "" {
			cfg.QuantMethod = strings.ToLower(qt)
		}
	} else {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams("")
	}
	cfg.TensorQuant = root.AllTensorQuant()
	if cfg.QuantMethod == "" {
		if cfg.QuantizationConfig.QuantMethod != "" {
			cfg.QuantMethod = strings.ToLower(cfg.QuantizationConfig.QuantMethod)
		} else if cfg.Quantization.QuantMethod != "" {
			cfg.QuantMethod = strings.ToLower(cfg.Quantization.QuantMethod)
		}
	}

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
	ensureHarmonyEOSTokens(tok)

	return &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
		tok:    tok,
	}, nil
}

func ensureHarmonyEOSTokens(tok *tokenizer.Tokenizer) {
	if tok == nil {
		return
	}

	ids := append([]int32(nil), tok.EOSTokens()...)
	for _, name := range harmonyEOSTokens {
		if id, ok := tok.GetSpecialToken(name); ok {
			ids = append(ids, id)
		}
	}
	if len(ids) > 0 {
		tok.SetEOSTokens(ids...)
	}
}

func parseConfig(configData []byte) (Config, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(configData, &raw); err != nil {
		return Config{}, fmt.Errorf("parse config envelope: %w", err)
	}

	var cfg Config
	active := configData
	if textRaw, ok := raw["text_config"]; ok {
		active = textRaw
	}
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
		return Config{}, fmt.Errorf("invalid num_key_value_heads: %d", cfg.NumKeyValueHeads)
	}
	if cfg.HeadDim <= 0 {
		return Config{}, fmt.Errorf("invalid head_dim: %d", cfg.HeadDim)
	}
	if cfg.NumLocalExperts <= 0 {
		return Config{}, fmt.Errorf("invalid num_local_experts: %d", cfg.NumLocalExperts)
	}
	if cfg.NumExpertsPerTok <= 0 {
		return Config{}, fmt.Errorf("invalid num_experts_per_tok: %d", cfg.NumExpertsPerTok)
	}
	if cfg.NumExpertsPerTok > cfg.NumLocalExperts {
		return Config{}, fmt.Errorf("num_experts_per_tok (%d) must be <= num_local_experts (%d)", cfg.NumExpertsPerTok, cfg.NumLocalExperts)
	}
	if cfg.SlidingWindow <= 0 {
		return Config{}, fmt.Errorf("invalid sliding_window: %d", cfg.SlidingWindow)
	}
	if cfg.RopeTheta <= 0 {
		return Config{}, fmt.Errorf("invalid rope_theta: %f", cfg.RopeTheta)
	}
	if cfg.RopeScaling.Factor <= 0 {
		return Config{}, fmt.Errorf("invalid rope_scaling.factor: %f", cfg.RopeScaling.Factor)
	}
	if cfg.RopeScaling.OriginalMaxPositionEmbeddings <= 0 {
		return Config{}, fmt.Errorf("invalid rope_scaling.original_max_position_embeddings: %d", cfg.RopeScaling.OriginalMaxPositionEmbeddings)
	}
	if cfg.RMSNormEps <= 0 {
		return Config{}, fmt.Errorf("invalid rms_norm_eps: %f", cfg.RMSNormEps)
	}
	if cfg.VocabSize <= 0 {
		return Config{}, fmt.Errorf("invalid vocab_size: %d", cfg.VocabSize)
	}
	if cfg.MaxPositionEmbeddings <= 0 {
		cfg.MaxPositionEmbeddings = int32(math.Round(float64(cfg.RopeScaling.Factor) * float64(cfg.RopeScaling.OriginalMaxPositionEmbeddings)))
	}
	if cfg.MaxPositionEmbeddings <= 0 {
		cfg.MaxPositionEmbeddings = cfg.SlidingWindow
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return Config{}, fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}

	if cfg.QuantizationConfig.QuantMethod != "" {
		cfg.QuantMethod = strings.ToLower(cfg.QuantizationConfig.QuantMethod)
	} else if cfg.Quantization.QuantMethod != "" {
		cfg.QuantMethod = strings.ToLower(cfg.Quantization.QuantMethod)
	}

	return cfg, nil
}

// NumLayers returns the configured layer count.
func (m *Model) NumLayers() int {
	if m == nil || m.Config == nil {
		return 0
	}
	return int(m.NumHiddenLayers)
}

// Tokenizer returns the loaded tokenizer.
func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	if m == nil {
		return nil
	}
	return m.tok
}

// MaxContextLength returns the derived context length.
func (m *Model) MaxContextLength() int {
	if m == nil || m.Config == nil {
		return 0
	}
	if m.MaxPositionEmbeddings > 0 {
		return int(m.MaxPositionEmbeddings)
	}
	return 0
}

// NewCaches returns one cache per layer, matching the classic gpt-oss
// alternating sliding-window / causal parity.
func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, m.NumLayers())
	for i := range caches {
		if i%2 == 0 {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
			continue
		}
		caches[i] = cache.NewKVCache()
	}
	return caches
}

func stepwisePrefillForced() bool {
	return os.Getenv("GPTOSS_PREFILL_STEPWISE") == "1"
}

// swiGLUAlphaLimit is a compiled kernel implementing the gpt-oss MoE activation:
//
//	swish = min(gate, 7) * sigmoid(1.702 * min(gate, 7))
//	result = swish * (clamp(up, -7, 7) + 1)
//
// The Compile2 wrapper traces the elementwise chain once and fuses it into a
// single Metal kernel, eliminating ~10 per-op dispatches per MoE block.
var swiGLUAlphaLimit = mlx.Compile2(
	"gptoss_swiglu_alpha_limit",
	func(gate, up *mlx.Array) *mlx.Array {
		dt := gate.DType()
		alpha := mlx.FromValue[float32](1.702).AsType(dt)
		limit := mlx.FromValue[float32](7).AsType(dt)
		negLimit := mlx.Neg(limit)
		one := mlx.FromValue[float32](1).AsType(dt)

		clippedGate := mlx.Minimum(gate, limit)
		clippedUp := mlx.Clip(up, negLimit, limit)

		swish := clippedGate.Multiply(mlx.Mul(clippedGate, alpha).Sigmoid())
		return swish.Multiply(clippedUp.Add(one))
	},
	mlx.Shapeless(),
)

func sliceSequence(x *mlx.Array, pos int) *mlx.Array {
	// Callers pass [batch, seq, hidden] tensors only.
	return mlx.SliceStartStop(x, []int32{0, int32(pos), 0}, []int32{1, int32(pos + 1), int32(x.Dim(2))})
}

func expertSlice(t *mlx.Array, expert int32) *mlx.Array {
	if t == nil || !t.Valid() {
		return nil
	}

	dims := t.Dims()
	if len(dims) == 0 {
		return nil
	}

	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	start[0] = expert
	stop[0] = expert + 1
	for i := 1; i < len(dims); i++ {
		stop[i] = int32(dims[i])
	}

	return mlx.Squeeze(mlx.SliceStartStop(t, start, stop), 0)
}

func interleavedIndices(count int, offset int32) *mlx.Array {
	indices := make([]int32, count)
	for i := range count {
		indices[i] = int32(i*2) + offset
	}
	return mlx.FromValues(indices, count)
}

func splitGateUpInterleaved(dense, bias *mlx.Array, mid int) (gateWeight, upWeight, gateBias, upBias *mlx.Array) {
	if dense == nil || !dense.Valid() || bias == nil || !bias.Valid() || mid <= 0 {
		return nil, nil, nil, nil
	}

	even := interleavedIndices(mid, 0)
	odd := interleavedIndices(mid, 1)
	gateWeight = mlx.Take(dense, even, 0)
	upWeight = mlx.Take(dense, odd, 0)
	gateBias = mlx.Take(bias, even, 0)
	upBias = mlx.Take(bias, odd, 0)
	return gateWeight, upWeight, gateBias, upBias
}

func buildGPTOSSRoPEFreqs(cfg *Config) (*mlx.Array, float32) {
	if cfg == nil || cfg.HeadDim <= 0 || cfg.RopeTheta <= 0 {
		return nil, 1
	}
	params := &nn.RopeParameters{
		RopeTheta:                     cfg.RopeTheta,
		RopeType:                      cfg.RopeScaling.RopeType,
		Factor:                        cfg.RopeScaling.Factor,
		OriginalMaxPositionEmbeddings: cfg.RopeScaling.OriginalMaxPositionEmbeddings,
		BetaFast:                      cfg.RopeScaling.BetaFast,
		BetaSlow:                      cfg.RopeScaling.BetaSlow,
	}
	return nn.BuildYarnRopeFreqs(int(cfg.HeadDim), cfg.RopeTheta, params)
}

func requireTensor(tensors map[string]*mlx.Array, name string) (*mlx.Array, error) {
	t := tensors[name]
	if t == nil || !t.Valid() {
		return nil, fmt.Errorf("missing tensor %q", name)
	}
	return t, nil
}

func validateTensorShape(name string, t *mlx.Array, want []int, wantExpr string) error {
	if t == nil || !t.Valid() {
		return fmt.Errorf("missing tensor %q", name)
	}

	got := t.Dims()
	if len(got) != len(want) {
		return fmt.Errorf("tensor %q shape %v, want %v (%s)", name, got, want, wantExpr)
	}
	for i := range want {
		if got[i] != want[i] {
			return fmt.Errorf("tensor %q shape %v, want %v (%s)", name, got, want, wantExpr)
		}
	}
	return nil
}

func validateLayerTensorShape(layer int, name string, t *mlx.Array, want []int, wantExpr string) error {
	if err := validateTensorShape(name, t, want, wantExpr); err != nil {
		return fmt.Errorf("layer %d: %w", layer, err)
	}
	return nil
}

func validateLayerTensorDType(layer int, name string, t *mlx.Array, want mlx.DType, wantExpr string) error {
	if t == nil || !t.Valid() {
		return fmt.Errorf("layer %d: missing tensor %q", layer, name)
	}
	if got := t.DType(); got != want {
		return fmt.Errorf("layer %d: tensor %q dtype %s, want %s (%s)", layer, name, got, want, wantExpr)
	}
	return nil
}

func validateLinearLayerShape(
	layer int,
	tensors map[string]*mlx.Array,
	path string,
	wantOut, wantIn int,
	wantExpr string,
	cfg *Config,
) error {
	weightName := path + ".weight"
	weight, err := requireTensor(tensors, weightName)
	if err != nil {
		return fmt.Errorf("layer %d: %w", layer, err)
	}

	scales := tensors[weightName+"_scale"]
	if scales == nil {
		return validateLayerTensorShape(layer, weightName, weight, []int{wantOut, wantIn}, wantExpr)
	}

	if len(weight.Dims()) != 2 {
		return fmt.Errorf("layer %d: tensor %q dims %v, want quantized matrix for %s", layer, weightName, weight.Dims(), wantExpr)
	}
	if weight.Dim(0) != wantOut {
		return fmt.Errorf("layer %d: tensor %q output dim %d, want %d (%s)", layer, weightName, weight.Dim(0), wantOut, wantExpr)
	}

	_, bits, mode := model.ResolveLinearQuantParams(
		cfg.QuantGroupSize,
		cfg.QuantBits,
		cfg.QuantMode,
		cfg.TensorQuant,
		weightName,
		weight,
		scales,
	)
	if mode == "affine" {
		if _, inferredBits, ok := model.InferAffineQuantParamsFromShapes(weight, scales, bits); !ok || inferredBits != bits {
			return fmt.Errorf("layer %d: tensor %q has unsupported affine quantized shapes %v / %v for %s", layer, weightName, weight.Dims(), scales.Dims(), wantExpr)
		}
	}

	return nil
}

func loadLinearLayer(tensors map[string]*mlx.Array, linears model.LinearFactory, cfg *Config, layer int, path string, wantOut, wantIn int, wantExpr string) (nn.LinearLayer, error) {
	if err := validateLinearLayerShape(
		layer,
		tensors,
		path,
		wantOut,
		wantIn,
		wantExpr,
		cfg,
	); err != nil {
		return nil, err
	}

	biasName := path + ".bias"
	bias, err := requireTensor(tensors, biasName)
	if err != nil {
		return nil, fmt.Errorf("layer %d: %w", layer, err)
	}
	if err := validateLayerTensorShape(layer, biasName, bias, []int{wantOut}, fmt.Sprintf("%s bias length", path)); err != nil {
		return nil, err
	}

	layerLinear := linears.Make(path)
	if layerLinear == nil {
		return nil, fmt.Errorf("layer %d: failed to construct linear layer from %q", layer, path+".weight")
	}
	if got := layerLinear.OutputDim(); int(got) != wantOut {
		return nil, fmt.Errorf("layer %d: linear %q output dim = %d, want %d (%s)", layer, path, got, wantOut, wantExpr)
	}
	return layerLinear, nil
}

func loadLayerNormTensor(tensors map[string]*mlx.Array, layer int, name string, want int, wantExpr string) (*nn.RMSNorm, error) {
	weight, err := requireTensor(tensors, name)
	if err != nil {
		return nil, fmt.Errorf("layer %d: %w", layer, err)
	}
	if err := validateLayerTensorShape(layer, name, weight, []int{want}, wantExpr); err != nil {
		return nil, err
	}
	return nn.NewRMSNorm(weight, 0), nil
}

func loadExpertPair(tensors map[string]*mlx.Array, layer int, prefix string, wantOut, wantIn int, cfg *Config) (*ExpertPair, error) {
	pair, err := loadDirectExpertPair(tensors, layer, prefix, wantOut/2, wantIn, cfg)
	if err != nil {
		return nil, err
	}
	if pair != nil {
		return pair, nil
	}

	weightName := prefix + ".weight"
	biasName := prefix + ".bias"

	weight, err := requireTensor(tensors, weightName)
	if err != nil {
		return nil, fmt.Errorf("layer %d: %w", layer, err)
	}
	bias, err := requireTensor(tensors, biasName)
	if err != nil {
		return nil, fmt.Errorf("layer %d: %w", layer, err)
	}

	if err := validateLayerTensorShape(layer, weightName, weight, []int{int(cfg.NumLocalExperts), wantOut, wantIn}, fmt.Sprintf("num_local_experts x %s x hidden", prefix)); err != nil {
		return nil, err
	}
	if err := validateLayerTensorShape(layer, biasName, bias, []int{int(cfg.NumLocalExperts), wantOut}, fmt.Sprintf("num_local_experts x %s bias", prefix)); err != nil {
		return nil, err
	}
	if err := validateLayerTensorDType(layer, weightName, weight, mlx.DTypeBFloat16, "offline-dequantized expert weights"); err != nil {
		return nil, err
	}
	if err := validateLayerTensorDType(layer, biasName, bias, mlx.DTypeBFloat16, "offline-dequantized expert bias"); err != nil {
		return nil, err
	}

	if wantOut%2 != 0 {
		return nil, fmt.Errorf("layer %d: %s output dim must be even, got %d", layer, prefix, wantOut)
	}

	mid := wantOut / 2
	gateWeights := make([]*mlx.Array, 0, cfg.NumLocalExperts)
	upWeights := make([]*mlx.Array, 0, cfg.NumLocalExperts)
	gateBiases := make([]*mlx.Array, 0, cfg.NumLocalExperts)
	upBiases := make([]*mlx.Array, 0, cfg.NumLocalExperts)
	for e := range cfg.NumLocalExperts {
		expertWeight := expertSlice(weight, e)
		expertBias := expertSlice(bias, e)

		expertWeightName := fmt.Sprintf("%s.expert[%d]", weightName, e)
		if err := validateLayerTensorShape(layer, expertWeightName, expertWeight, []int{wantOut, wantIn}, fmt.Sprintf("%s expert slice", prefix)); err != nil {
			return nil, err
		}
		if err := validateLayerTensorShape(layer, expertWeightName+".bias", expertBias, []int{wantOut}, fmt.Sprintf("%s expert bias", prefix)); err != nil {
			return nil, err
		}

		gateWeight, upWeight, gateBias, upBias := splitGateUpInterleaved(expertWeight, expertBias, mid)
		if gateWeight == nil || upWeight == nil || gateBias == nil || upBias == nil {
			return nil, fmt.Errorf("layer %d: failed to split interleaved gate/up expert tensor %q", layer, expertWeightName)
		}
		gateWeight = mlx.Transpose(gateWeight, 1, 0)
		upWeight = mlx.Transpose(upWeight, 1, 0)
		gateWeight = mlx.Contiguous(gateWeight, false)
		upWeight = mlx.Contiguous(upWeight, false)
		gateBias = mlx.Contiguous(gateBias, false)
		upBias = mlx.Contiguous(upBias, false)

		gateWeights = append(gateWeights, gateWeight)
		upWeights = append(upWeights, upWeight)
		gateBiases = append(gateBiases, gateBias)
		upBiases = append(upBiases, upBias)
	}

	gateWeight := mlx.Stack(gateWeights, 0)
	upWeight := mlx.Stack(upWeights, 0)
	gateBias := mlx.Stack(gateBiases, 0)
	upBias := mlx.Stack(upBiases, 0)
	mlx.Eval(gateWeight, upWeight, gateBias, upBias)

	return &ExpertPair{
		Gate: &ExpertProjection{
			Weight: gateWeight,
			Bias:   gateBias,
		},
		Up: &ExpertProjection{
			Weight: upWeight,
			Bias:   upBias,
		},
	}, nil
}

func loadExpertProjection(tensors map[string]*mlx.Array, layer int, prefix string, wantOut, wantIn int, cfg *Config) (*ExpertProjection, error) {
	proj, err := loadDirectExpertProjection(tensors, layer, prefix, wantOut, wantIn, cfg)
	if err != nil {
		return nil, err
	}
	if proj != nil {
		return proj, nil
	}

	weightName := prefix + ".weight"
	biasName := prefix + ".bias"

	weight, err := requireTensor(tensors, weightName)
	if err != nil {
		return nil, fmt.Errorf("layer %d: %w", layer, err)
	}
	bias, err := requireTensor(tensors, biasName)
	if err != nil {
		return nil, fmt.Errorf("layer %d: %w", layer, err)
	}

	if err := validateLayerTensorShape(layer, weightName, weight, []int{int(cfg.NumLocalExperts), wantOut, wantIn}, fmt.Sprintf("num_local_experts x %s x hidden", prefix)); err != nil {
		return nil, err
	}
	if err := validateLayerTensorShape(layer, biasName, bias, []int{int(cfg.NumLocalExperts), wantOut}, fmt.Sprintf("num_local_experts x %s bias", prefix)); err != nil {
		return nil, err
	}
	if err := validateLayerTensorDType(layer, weightName, weight, mlx.DTypeBFloat16, "offline-dequantized expert weights"); err != nil {
		return nil, err
	}
	if err := validateLayerTensorDType(layer, biasName, bias, mlx.DTypeBFloat16, "offline-dequantized expert bias"); err != nil {
		return nil, err
	}

	weights := make([]*mlx.Array, 0, cfg.NumLocalExperts)
	biases := make([]*mlx.Array, 0, cfg.NumLocalExperts)
	for e := range cfg.NumLocalExperts {
		expertWeight := expertSlice(weight, e)
		expertBias := expertSlice(bias, e)

		expertWeightName := fmt.Sprintf("%s.expert[%d]", weightName, e)
		if err := validateLayerTensorShape(layer, expertWeightName, expertWeight, []int{wantOut, wantIn}, fmt.Sprintf("%s expert slice", prefix)); err != nil {
			return nil, err
		}
		if err := validateLayerTensorShape(layer, expertWeightName+".bias", expertBias, []int{wantOut}, fmt.Sprintf("%s expert bias", prefix)); err != nil {
			return nil, err
		}

		expertWeight = mlx.Transpose(expertWeight, 1, 0)
		expertWeight = mlx.Contiguous(expertWeight, false)
		weights = append(weights, expertWeight)
		biases = append(biases, expertBias)
	}

	weightStack := mlx.Stack(weights, 0)
	biasStack := mlx.Stack(biases, 0)
	mlx.Eval(weightStack, biasStack)

	return &ExpertProjection{
		Weight: weightStack,
		Bias:   biasStack,
	}, nil
}

func loadDirectExpertPair(tensors map[string]*mlx.Array, layer int, legacyPrefix string, wantOut, wantIn int, cfg *Config) (*ExpertPair, error) {
	gatePrefix := strings.Replace(legacyPrefix, "gate_up_proj", "gate_proj", 1)
	upPrefix := strings.Replace(legacyPrefix, "gate_up_proj", "up_proj", 1)
	gate, err := loadDirectExpertProjection(tensors, layer, gatePrefix, wantOut, wantIn, cfg)
	if err != nil {
		return nil, err
	}
	up, err := loadDirectExpertProjection(tensors, layer, upPrefix, wantOut, wantIn, cfg)
	if err != nil {
		return nil, err
	}
	switch {
	case gate == nil && up == nil:
		return nil, nil
	case gate == nil:
		return nil, fmt.Errorf("layer %d: missing direct gate expert tensors for %q", layer, legacyPrefix)
	case up == nil:
		return nil, fmt.Errorf("layer %d: missing direct up expert tensors for %q", layer, legacyPrefix)
	}
	return &ExpertPair{Gate: gate, Up: up}, nil
}

func loadDirectExpertProjection(tensors map[string]*mlx.Array, layer int, prefix string, wantOut, wantIn int, cfg *Config) (*ExpertProjection, error) {
	weightName := prefix + ".weight"
	biasName := prefix + ".bias"
	weight := tensors[weightName]
	bias := tensors[biasName]
	if weight == nil && bias == nil {
		return nil, nil
	}
	if weight == nil || bias == nil {
		return nil, fmt.Errorf("layer %d: missing direct expert tensor %q or %q", layer, weightName, biasName)
	}

	if err := validateLayerTensorShape(layer, biasName, bias, []int{int(cfg.NumLocalExperts), wantOut}, fmt.Sprintf("num_local_experts x out bias for %s", prefix)); err != nil {
		return nil, err
	}
	if err := validateLayerTensorDType(layer, biasName, bias, mlx.DTypeBFloat16, "runtime-ready offline expert bias"); err != nil {
		return nil, err
	}

	scales := tensors[weightName+"_scale"]
	if scales != nil {
		if len(weight.Dims()) != 3 || weight.Dim(0) != int(cfg.NumLocalExperts) {
			return nil, fmt.Errorf("layer %d: tensor %q dims %v, want quantized expert layout [num_local_experts, ?, packed]", layer, weightName, weight.Dims())
		}
		if len(scales.Dims()) != 3 || scales.Dim(0) != int(cfg.NumLocalExperts) {
			return nil, fmt.Errorf("layer %d: tensor %q dims %v, want quantized expert scale layout [num_local_experts, ?, scale_groups]", layer, weightName+"_scale", scales.Dims())
		}

		qbiases := tensors[weightName+"_qbias"]
		groupSize, bits, mode := model.ResolveLinearQuantParams(
			cfg.QuantGroupSize,
			cfg.QuantBits,
			cfg.QuantMode,
			cfg.TensorQuant,
			weightName,
			weight,
			scales,
		)

		return &ExpertProjection{
			Weight:    weight,
			Bias:      bias,
			Scales:    scales,
			QBiases:   qbiases,
			GroupSize: groupSize,
			Bits:      bits,
			Mode:      mode,
			Transpose: true, // MLX-native quantized layout: [experts, out, packed_in]
		}, nil
	}

	if err := validateLayerTensorDType(layer, weightName, weight, mlx.DTypeBFloat16, "runtime-ready offline expert weights"); err != nil {
		return nil, err
	}
	if len(weight.Dims()) != 3 || weight.Dim(0) != int(cfg.NumLocalExperts) {
		return nil, fmt.Errorf("layer %d: tensor %q dims %v, want direct expert layout [num_local_experts, out, in]", layer, weightName, weight.Dims())
	}

	switch {
	case weight.Dim(1) == wantOut && weight.Dim(2) == wantIn:
		weight = mlx.Transpose(weight, 0, 2, 1)
		weight = mlx.Contiguous(weight, false)
		mlx.Eval(weight)
		return &ExpertProjection{Weight: weight, Bias: bias}, nil
	case weight.Dim(1) == wantIn && weight.Dim(2) == wantOut:
		// Legacy GPT-OSS direct BF16 layout kept weights pre-transposed for GatherMM.
		return &ExpertProjection{Weight: weight, Bias: bias}, nil
	default:
		return nil, fmt.Errorf("layer %d: tensor %q dims %v, want direct expert layout [%d %d %d] or legacy [%d %d %d]",
			layer, weightName, weight.Dims(),
			int(cfg.NumLocalExperts), wantOut, wantIn,
			int(cfg.NumLocalExperts), wantIn, wantOut,
		)
	}
}

// LoadWeights assigns dense tensors and structural placeholders to the model.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if m == nil || m.Config == nil {
		return fmt.Errorf("missing gpt-oss config")
	}
	if len(m.Layers) == 0 {
		m.Layers = make([]*Layer, m.NumLayers())
	}

	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	embeddingWeight, err := requireTensor(tensors, "embedding.weight")
	if err != nil {
		return err
	}
	if tensors["embedding.weight_scale"] == nil {
		if err := validateTensorShape("embedding.weight", embeddingWeight, []int{int(m.VocabSize), int(m.HiddenSize)}, "vocab_size x hidden_size"); err != nil {
			return err
		}
	}
	embedTokens := model.MakeEmbeddingLayer(tensors, "embedding", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	if embedTokens == nil {
		return fmt.Errorf("failed to construct embedding layer from %q", "embedding.weight")
	}
	m.EmbedTokens = embedTokens

	outputNormWeight, err := requireTensor(tensors, "output_norm.weight")
	if err != nil {
		return err
	}
	if err := validateTensorShape("output_norm.weight", outputNormWeight, []int{int(m.HiddenSize)}, "hidden_size"); err != nil {
		return err
	}
	m.Norm = nn.NewRMSNorm(outputNormWeight, m.RMSNormEps)

	if _, err := requireTensor(tensors, "output.weight"); err != nil {
		return err
	}
	if err := validateLinearLayerShape(-1, tensors, "output", int(m.VocabSize), int(m.HiddenSize), "vocab_size x hidden_size", m.Config); err != nil {
		return err
	}
	m.LMHead = linears.Make("output")
	if m.LMHead == nil {
		return fmt.Errorf("failed to construct linear layer from %q", "output.weight")
	}

	expectedQ := int(m.NumAttentionHeads * m.HeadDim)
	expectedKV := int(m.NumKeyValueHeads * m.HeadDim)
	ropeFreqs, ropeScale := buildGPTOSSRoPEFreqs(m.Config)
	for i := range m.Layers {
		prefix := fmt.Sprintf("blocks.%d", i)

		attnNorm, err := loadLayerNormTensor(tensors, i, prefix+".attn_norm.weight", int(m.HiddenSize), "hidden_size")
		if err != nil {
			return err
		}

		qProj, err := loadLinearLayer(tensors, linears, m.Config, i, prefix+".q_proj", expectedQ, int(m.HiddenSize), "num_attention_heads * head_dim x hidden_size")
		if err != nil {
			return err
		}
		kProj, err := loadLinearLayer(tensors, linears, m.Config, i, prefix+".k_proj", expectedKV, int(m.HiddenSize), "num_key_value_heads * head_dim x hidden_size")
		if err != nil {
			return err
		}
		vProj, err := loadLinearLayer(tensors, linears, m.Config, i, prefix+".v_proj", expectedKV, int(m.HiddenSize), "num_key_value_heads * head_dim x hidden_size")
		if err != nil {
			return err
		}
		oProj, err := loadLinearLayer(tensors, linears, m.Config, i, prefix+".attn_out", int(m.HiddenSize), expectedQ, "hidden_size x num_attention_heads * head_dim")
		if err != nil {
			return err
		}

		sinks, err := requireTensor(tensors, prefix+".attn_sinks")
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
		if err := validateLayerTensorShape(i, prefix+".attn_sinks", sinks, []int{int(m.NumAttentionHeads)}, "num_attention_heads"); err != nil {
			return err
		}

		ffnNorm, err := loadLayerNormTensor(tensors, i, prefix+".ffn_norm.weight", int(m.HiddenSize), "hidden_size")
		if err != nil {
			return err
		}

		router, err := loadLinearLayer(tensors, linears, m.Config, i, prefix+".router", int(m.NumLocalExperts), int(m.HiddenSize), "num_local_experts x hidden_size")
		if err != nil {
			return err
		}

		gateUp, err := loadExpertPair(tensors, i, prefix+".experts.gate_up_proj", int(m.IntermediateSize)*2, int(m.HiddenSize), m.Config)
		if err != nil {
			return err
		}
		down, err := loadExpertProjection(tensors, i, prefix+".experts.down_proj", int(m.HiddenSize), int(m.IntermediateSize), m.Config)
		if err != nil {
			return err
		}
		m.Layers[i] = &Layer{
			AttentionNorm: attnNorm,
			Attention: &Attention{
				QProj:     qProj,
				KProj:     kProj,
				VProj:     vProj,
				OProj:     oProj,
				Sinks:     sinks,
				RoPEFreqs: ropeFreqs,
				RoPEScale: ropeScale,
			},
			FFNNorm: ffnNorm,
			Router:  router,
			Experts: &Experts{
				GateUp: gateUp,
				Down:   down,
			},
		}
	}

	return nil
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	if m == nil || m.Config == nil || m.EmbedTokens == nil || m.Norm == nil || b == nil || b.InputIDs == nil {
		return nil
	}

	dims := b.InputIDs.Dims()
	if len(dims) != 2 {
		panic(fmt.Sprintf("gpt-oss forward requires 2D token input, got %v", dims))
	}

	batchSize, seqLen := dims[0], dims[1]
	return m.forwardDense(b, caches, batchSize, seqLen)
}

func (m *Model) forwardDense(b *batch.Batch, caches []cache.Cache, batchSize, seqLen int) *mlx.Array {
	m.logCachedPrefillMode(caches, batchSize, seqLen)

	h := m.EmbedTokens.Forward(b.InputIDs)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.ForwardBatch(h, b, c, batchSize, seqLen, m.Config, i)
	}

	return m.Norm.Forward(h, m.RMSNormEps)
}

func (m *Model) logCachedPrefillMode(caches []cache.Cache, batchSize, seqLen int) {
	if m == nil || batchSize != 1 || seqLen <= 1 || len(caches) == 0 {
		return
	}

	forcedStepwise := stepwisePrefillForced()

	var stepwiseLayers, batchedLayers int
	var rotatingLayers, causalLayers int
	var minOffset, maxOffset int
	haveOffset := false
	for i, layer := range m.Layers {
		if layer == nil || layer.Attention == nil || i >= len(caches) || caches[i] == nil {
			continue
		}

		if forcedStepwise {
			stepwiseLayers++
		} else {
			batchedLayers++
		}

		switch caches[i].(type) {
		case *cache.RotatingKVCache:
			rotatingLayers++
		case *cache.KVCache:
			causalLayers++
		}

		offset := caches[i].Offset()
		if !haveOffset || offset < minOffset {
			minOffset = offset
		}
		if !haveOffset || offset > maxOffset {
			maxOffset = offset
		}
		haveOffset = true
	}
	if stepwiseLayers == 0 && batchedLayers == 0 {
		return
	}

	attrs := []any{
		"batch_size", batchSize,
		"seq_len", seqLen,
		"rotating_cache_layers", rotatingLayers,
		"causal_cache_layers", causalLayers,
		"forced", forcedStepwise,
	}
	if haveOffset {
		attrs = append(attrs, "cache_offset_min", minOffset, "cache_offset_max", maxOffset)
	}
	if stepwiseLayers > 0 {
		slog.Debug(
			"gptoss cached prefill stepwise fallback",
			append(
				attrs,
				"attention_layers", stepwiseLayers,
				"stepwise_attention_calls", stepwiseLayers*seqLen,
			)...,
		)
	}
	if batchedLayers > 0 {
		slog.Debug(
			"gptoss cached prefill batched default",
			append(
				attrs,
				"attention_layers", batchedLayers,
				"avoided_stepwise_attention_calls", batchedLayers*seqLen,
			)...,
		)
	}
}

// Unembed projects hidden states back into vocabulary space.
func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	if m == nil || m.LMHead == nil || x == nil {
		return nil
	}
	if stepwisePrefillForced() {
		dims := x.Dims()
		if len(dims) == 3 && dims[0] == 1 && dims[1] > 1 {
			steps := make([]*mlx.Array, 0, dims[1])
			for pos := range dims[1] {
				steps = append(steps, m.LMHead.Forward(sliceSequence(x, pos)))
			}
			return mlx.Concatenate(steps, 1)
		}
	}
	return m.LMHead.Forward(x)
}

func (l *Layer) Forward(x *mlx.Array, c cache.Cache, batchSize, seqLen int, cfg *Config, layerIndex int) *mlx.Array {
	return l.ForwardBatch(x, batchForForward(c, seqLen), c, batchSize, seqLen, cfg, layerIndex)
}

func (l *Layer) ForwardBatch(x *mlx.Array, b *batch.Batch, c cache.Cache, batchSize, seqLen int, cfg *Config, layerIndex int) *mlx.Array {
	if l == nil || l.Attention == nil || l.AttentionNorm == nil || l.FFNNorm == nil || l.Router == nil || l.Experts == nil || x == nil || cfg == nil {
		panic("gpt-oss layer is not fully loaded")
	}
	residual := x
	x = l.AttentionNorm.Forward(x, cfg.RMSNormEps)
	x = l.Attention.ForwardBatch(x, b, c, batchSize, seqLen, cfg, layerIndex)
	if x == nil || !x.Valid() {
		panic(fmt.Sprintf("gpt-oss layer %d attention output is invalid", layerIndex))
	}

	h := residual.Add(x)
	if h == nil || !h.Valid() {
		panic(fmt.Sprintf("gpt-oss layer %d residual add output is invalid", layerIndex))
	}

	x = l.FFNNorm.Forward(h, cfg.RMSNormEps)
	router := l.Router.Forward(x)
	if router == nil || !router.Valid() {
		panic(fmt.Sprintf("gpt-oss layer %d router output is invalid", layerIndex))
	}

	x = l.Experts.Forward(x, router, cfg, layerIndex)
	if x == nil || !x.Valid() {
		panic(fmt.Sprintf("gpt-oss layer %d expert output is invalid", layerIndex))
	}

	return h.Add(x)
}

func (a *Attention) Forward(x *mlx.Array, c cache.Cache, batchSize, seqLen int, cfg *Config, layerIndex int) *mlx.Array {
	return a.ForwardBatch(x, batchForForward(c, seqLen), c, batchSize, seqLen, cfg, layerIndex)
}

func batchForForward(c cache.Cache, seqLen int) *batch.Batch {
	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	return &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, seqLen),
		SeqOffsets:   []int32{int32(offset)},
		SeqQueryLens: []int32{int32(seqLen)},
	}
}

func (a *Attention) ForwardBatch(x *mlx.Array, b *batch.Batch, c cache.Cache, batchSize, seqLen int, cfg *Config, layerIndex int) *mlx.Array {
	if a == nil || a.QProj == nil || a.KProj == nil || a.VProj == nil || a.OProj == nil || x == nil || cfg == nil {
		return x
	}
	if c != nil && batchSize == 1 && seqLen > 1 && stepwisePrefillForced() {
		// Emergency rollback path for cached prefill. Set
		// GPTOSS_PREFILL_STEPWISE=1 to force the older per-token path for A/B
		// validation or field mitigation.
		steps := make([]*mlx.Array, 0, seqLen)
		for pos := range seqLen {
			stepBatch := &batch.Batch{
				InputIDs:     b.InputIDs.Slice(mlx.Slice(), mlx.Slice(pos, pos+1)),
				SeqOffsets:   []int32{b.SeqOffsets[0] + int32(pos)},
				SeqQueryLens: []int32{1},
			}
			steps = append(steps, a.ForwardBatch(sliceSequence(x, pos), stepBatch, c, 1, 1, cfg, layerIndex))
		}
		return mlx.Concatenate(steps, 1)
	}

	query := a.QProj.Forward(x)
	key := a.KProj.Forward(x)
	value := a.VProj.Forward(x)
	if query == nil || key == nil || value == nil || !query.Valid() || !key.Valid() || !value.Valid() {
		panic(fmt.Sprintf("gpt-oss layer %d attention projections are invalid", layerIndex))
	}

	batchDim := int32(batchSize)
	seq := int32(seqLen)
	numHeads := cfg.NumAttentionHeads
	numKVHeads := cfg.NumKeyValueHeads
	headDim := cfg.HeadDim

	query = mlx.Reshape(query, batchDim, seq, numHeads, headDim)
	key = mlx.Reshape(key, batchDim, seq, numKVHeads, headDim)
	value = mlx.Reshape(value, batchDim, seq, numKVHeads, headDim)
	if query == nil || key == nil || value == nil || !query.Valid() || !key.Valid() || !value.Valid() {
		panic(fmt.Sprintf("gpt-oss layer %d attention reshape is invalid", layerIndex))
	}

	query = mlx.Transpose(query, 0, 2, 1, 3)
	key = mlx.Transpose(key, 0, 2, 1, 3)
	value = mlx.Transpose(value, 0, 2, 1, 3)
	if query == nil || key == nil || value == nil || !query.Valid() || !key.Valid() || !value.Valid() {
		panic(fmt.Sprintf("gpt-oss layer %d attention transpose is invalid", layerIndex))
	}

	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	attentionScale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	if a.RoPEFreqs != nil && a.RoPEFreqs.Valid() {
		query = mlx.RoPEWithFreqs(query, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions, a.RoPEFreqs)
		key = mlx.RoPEWithFreqs(key, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions, a.RoPEFreqs)
		ropeScale := a.RoPEScale
		if ropeScale == 0 {
			ropeScale = 1
		}
		attentionScale *= ropeScale * ropeScale
	} else {
		ropeBase, ropeScale, _ := cfg.RopeParameters()
		query = mlx.RoPEWithBase(query, int(cfg.HeadDim), false, ropeBase, ropeScale, positions)
		key = mlx.RoPEWithBase(key, int(cfg.HeadDim), false, ropeBase, ropeScale, positions)
	}
	if query == nil || key == nil || !query.Valid() || !key.Valid() {
		panic(fmt.Sprintf("gpt-oss layer %d attention RoPE is invalid", layerIndex))
	}

	var kv nn.SDPAOption
	if c != nil {
		attnCache, ok := c.(cache.Attention)
		if !ok {
			panic(fmt.Sprintf("gpt-oss layer %d cache does not support attention", layerIndex))
		}
		history := attnCache.Update(b, key, value)
		if history == nil || history.K() == nil || history.V() == nil || !history.K().Valid() || !history.V().Valid() {
			panic(fmt.Sprintf("gpt-oss layer %d attention cache update is invalid", layerIndex))
		}
		kv = nn.WithKVHistory(history)
	} else {
		kv = nn.WithKV(key, value, b.SeqQueryLens)
	}

	attention := nn.ScaledDotProductAttention(
		b,
		query,
		attentionScale,
		kv,
		nn.WithMask(nn.CausalMask()),
		nn.WithSinks(a.Sinks),
	)
	if attention == nil || !attention.Valid() {
		panic(fmt.Sprintf("gpt-oss layer %d attention sdpa is invalid", layerIndex))
	}
	attention = mlx.Transpose(attention, 0, 2, 1, 3)
	attention = mlx.Reshape(attention, batchDim, seq, numHeads*headDim)
	if attention == nil || !attention.Valid() {
		panic(fmt.Sprintf("gpt-oss layer %d attention output reshape is invalid", layerIndex))
	}
	attention = a.OProj.Forward(attention)
	if attention == nil || !attention.Valid() {
		panic(fmt.Sprintf("gpt-oss layer %d attention output projection is invalid", layerIndex))
	}
	return attention
}

func (p *ExpertProjection) Forward(x, indices *mlx.Array, sorted bool) *mlx.Array {
	if p == nil || p.Weight == nil || x == nil || indices == nil {
		return nil
	}

	var out *mlx.Array
	if p.Scales != nil {
		out = mlx.GatherQMM(x, p.Weight, p.Scales, p.QBiases, nil, indices, p.Transpose, p.GroupSize, p.Bits, p.Mode, sorted)
	} else {
		if x.DType() != p.Weight.DType() {
			x = x.AsType(p.Weight.DType())
		}
		out = mlx.GatherMM(x, p.Weight, nil, indices, sorted)
	}

	if p.Bias == nil || !p.Bias.Valid() {
		return out
	}

	bias := p.Bias.TakeAxis(indices, 0)
	bias = mlx.ExpandDims(bias, 2)
	return mlx.Add(out, bias)
}

func (e *Experts) Forward(x, router *mlx.Array, cfg *Config, layerIndex int) *mlx.Array {
	if e == nil || e.GateUp == nil || e.GateUp.Gate == nil || e.GateUp.Up == nil || e.Down == nil || x == nil || router == nil || cfg == nil {
		panic("gpt-oss expert path is not fully loaded")
	}
	if !x.Valid() || !router.Valid() {
		panic("gpt-oss expert path received invalid tensors")
	}

	dims := x.Dims()
	if len(dims) != 3 {
		panic(fmt.Sprintf("gpt-oss expert path expects 3D hidden states, got %v", dims))
	}

	B, L := int32(dims[0]), int32(dims[1])
	topK := cfg.NumExpertsPerTok

	neg := mlx.Neg(router)
	inds := mlx.Argpartition(neg, int(topK)-1, -1)
	shape := inds.Dims()
	inds = mlx.SliceStartStop(inds, []int32{0, 0, 0}, []int32{int32(shape[0]), int32(shape[1]), topK})

	scores := mlx.TakeAlongAxis(router, inds, -1)
	scores = mlx.SoftmaxAxis(scores, -1, true)

	var xFlat *mlx.Array
	if B == 1 && L == 1 {
		xFlat = mlx.Reshape(x, 1, 1, 1, cfg.HiddenSize)
	} else {
		xExpanded := mlx.ExpandDims(mlx.ExpandDims(x, -2), -2)
		xFlat = mlx.Reshape(xExpanded, B*L, 1, 1, cfg.HiddenSize)
	}
	idxFlat := mlx.Reshape(inds, B*L, topK)

	doSort := B*L >= 24
	var invOrder *mlx.Array
	n := B * L * topK
	if doSort {
		idxAll := mlx.Flatten(idxFlat)
		order := mlx.Argsort(idxAll, 0)
		invOrder = mlx.Argsort(order, 0)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), mlx.FloorDivideScalar(order, topK), 0), 1)
		idxFlat = mlx.Reshape(mlx.Take(idxAll, order, 0), n, 1)
	}

	// Try fully fused gate+up+SwiGLU+down kernel path for single-token MXFP4 MoE (decode).
	canFuse := !doSort && B*L == 1 &&
		e.GateUp.Gate.Scales != nil && e.GateUp.Up.Scales != nil &&
		e.GateUp.Gate.Bias != nil && e.GateUp.Up.Bias != nil &&
		e.GateUp.Gate.Mode == "mxfp4" && e.GateUp.Gate.GroupSize == 32

	var down *mlx.Array
	if canFuse {
		gateW := e.GateUp.Gate
		upW := e.GateUp.Up
		wDims := gateW.Weight.Dims()
		if len(wDims) == 3 {
			numRows := wDims[1]
			numColVecs := int(cfg.HiddenSize) / 32
			inputFlat := mlx.Reshape(x, B*L, cfg.HiddenSize).AsType(mlx.DTypeFloat32)

			swiGLUOut, ok := mlx.MoEFusedGateUpSwiGLU(
				inputFlat,
				gateW.Weight, gateW.Scales, gateW.Bias,
				upW.Weight, upW.Scales, upW.Bias,
				idxFlat,
				numRows, numColVecs, int(topK),
				-7.0, 7.0, // swiglu clamp limits
			)
			if ok && swiGLUOut != nil {
				// Try fused down projection on the float32 SwiGLU output.
				if e.Down.Scales != nil && e.Down.Bias != nil &&
					e.Down.Mode == "mxfp4" && e.Down.GroupSize == 32 {
					downDims := e.Down.Weight.Dims()
					if len(downDims) == 3 {
						downNumRows := int(cfg.HiddenSize)
						downNumColVecs := numRows / 32 // intermediateSize / 32

						downOut, dok := mlx.MoEFusedDown(
							swiGLUOut,
							e.Down.Weight, e.Down.Scales, e.Down.Bias,
							idxFlat,
							downNumRows, downNumColVecs, int(topK),
						)
						if dok && downOut != nil {
							// downOut is [batch, topK, hiddenSize] in float32
							down = downOut.AsType(mlx.DTypeBFloat16)
						}
					}
				}
				// If fused down failed, fall back to GatherQMM for just the down proj
				if down == nil {
					hidden := mlx.Reshape(swiGLUOut, B*L, topK, 1, int32(numRows)).AsType(mlx.DTypeBFloat16)
					down = e.Down.Forward(hidden, idxFlat, doSort)
					if down == nil || !down.Valid() {
						panic(fmt.Sprintf("gpt-oss layer %d expert down projection is invalid", layerIndex))
					}
					down = mlx.Squeeze(down, 2)
				}
			}
		}
	}

	// Fully unfused fallback path.
	if down == nil {
		gate := e.GateUp.Gate.Forward(xFlat, idxFlat, doSort)
		up := e.GateUp.Up.Forward(xFlat, idxFlat, doSort)
		if gate == nil || !gate.Valid() {
			panic(fmt.Sprintf("gpt-oss layer %d expert gate projection is invalid", layerIndex))
		}
		if up == nil || !up.Valid() {
			panic(fmt.Sprintf("gpt-oss layer %d expert up projection is invalid", layerIndex))
		}
		hidden := swiGLUAlphaLimit(gate, up)

		downResult := e.Down.Forward(hidden, idxFlat, doSort)
		if downResult == nil || !downResult.Valid() {
			panic(fmt.Sprintf("gpt-oss layer %d expert down projection is invalid", layerIndex))
		}

		if doSort {
			down = mlx.Reshape(
				mlx.Take(mlx.Squeeze(mlx.Squeeze(downResult, 2), 1), invOrder, 0),
				B*L, topK, cfg.HiddenSize,
			)
		} else {
			down = mlx.Squeeze(downResult, 2)
		}
	}

	down = mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
	return mlx.Sum(mlx.Mul(down, mlx.ExpandDims(scores, -1)), 2, false)
}
