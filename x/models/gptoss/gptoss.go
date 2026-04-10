// Package gptoss provides the gpt-oss text model implementation for MLX.
package gptoss

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

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
}

// Attention implements the split gpt-oss attention path.
type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
	Sinks *mlx.Array
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

// Forward is intentionally unavailable until the MLP/expert path lands in Phase 4.
func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	if m == nil || m.Config == nil || m.EmbedTokens == nil || m.Norm == nil || tokens == nil {
		return nil
	}
	panic("gpt-oss forward path requires unimplemented MLP/expert execution")
}

// Unembed projects hidden states back into vocabulary space.
func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	if m == nil || m.LMHead == nil || x == nil {
		return nil
	}
	return m.LMHead.Forward(x)
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

func (l *Layer) Forward(x *mlx.Array, c cache.Cache, batchSize, seqLen int, cfg *Config, layerIndex int) *mlx.Array {
	if l == nil || l.Attention == nil || l.AttentionNorm == nil || x == nil || cfg == nil {
		return x
	}

	residual := x
	x = l.AttentionNorm.Forward(x, cfg.RMSNormEps)
	x = l.Attention.Forward(x, c, batchSize, seqLen, cfg, layerIndex)
	return residual.Add(x)
}

func (a *Attention) Forward(x *mlx.Array, c cache.Cache, batchSize, seqLen int, cfg *Config, layerIndex int) *mlx.Array {
	if a == nil || a.QProj == nil || a.KProj == nil || a.VProj == nil || a.OProj == nil || x == nil || cfg == nil {
		return x
	}

	query := a.QProj.Forward(x)
	key := a.KProj.Forward(x)
	value := a.VProj.Forward(x)

	batch := int32(batchSize)
	seq := int32(seqLen)
	numHeads := int32(cfg.NumAttentionHeads)
	numKVHeads := int32(cfg.NumKeyValueHeads)
	headDim := int32(cfg.HeadDim)

	query = mlx.Reshape(query, batch, seq, numHeads, headDim)
	key = mlx.Reshape(key, batch, seq, numKVHeads, headDim)
	value = mlx.Reshape(value, batch, seq, numKVHeads, headDim)

	query = mlx.Transpose(query, 0, 2, 1, 3)
	key = mlx.Transpose(key, 0, 2, 1, 3)
	value = mlx.Transpose(value, 0, 2, 1, 3)

	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	ropeBase, ropeScale, _ := cfg.RopeParameters()
	query = mlx.RoPEWithBase(query, int(cfg.HeadDim), false, ropeBase, ropeScale, offset)
	key = mlx.RoPEWithBase(key, int(cfg.HeadDim), false, ropeBase, ropeScale, offset)

	if c != nil {
		key, value = c.Update(key, value)
	}

	maskMode := "causal"
	if layerIndex%2 == 0 {
		maskMode = "sliding_window"
	}

	attention := mlx.ScaledDotProductAttentionWithSinks(
		query,
		key,
		value,
		float32(1.0/math.Sqrt(float64(cfg.HeadDim))),
		maskMode,
		nil,
		a.Sinks,
	)
	attention = mlx.Transpose(attention, 0, 2, 1, 3)
	attention = mlx.Reshape(attention, batch, seq, numHeads*headDim)
	return a.OProj.Forward(attention)
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

func loadLinearLayer(tensors map[string]*mlx.Array, linears model.LinearFactory, layer int, path string, wantOut, wantIn int, wantExpr string) (nn.LinearLayer, error) {
	weightName := path + ".weight"
	weight, err := requireTensor(tensors, weightName)
	if err != nil {
		return nil, fmt.Errorf("layer %d: %w", layer, err)
	}
	if err := validateLayerTensorShape(layer, weightName, weight, []int{wantOut, wantIn}, wantExpr); err != nil {
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
		return nil, fmt.Errorf("layer %d: failed to construct linear layer from %q", layer, weightName)
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
	if err := validateTensorShape("embedding.weight", embeddingWeight, []int{int(m.VocabSize), int(m.HiddenSize)}, "vocab_size x hidden_size"); err != nil {
		return err
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

	outputWeight, err := requireTensor(tensors, "output.weight")
	if err != nil {
		return err
	}
	if err := validateTensorShape("output.weight", outputWeight, []int{int(m.VocabSize), int(m.HiddenSize)}, "vocab_size x hidden_size"); err != nil {
		return err
	}
	m.LMHead = linears.Make("output")
	if m.LMHead == nil {
		return fmt.Errorf("failed to construct linear layer from %q", "output.weight")
	}

	expectedQ := int(m.NumAttentionHeads * m.HeadDim)
	expectedKV := int(m.NumKeyValueHeads * m.HeadDim)
	for i := range m.Layers {
		prefix := fmt.Sprintf("blocks.%d", i)

		attnNorm, err := loadLayerNormTensor(tensors, i, prefix+".attn_norm.weight", int(m.HiddenSize), "hidden_size")
		if err != nil {
			return err
		}

		qProj, err := loadLinearLayer(tensors, linears, i, prefix+".q_proj", expectedQ, int(m.HiddenSize), "num_attention_heads * head_dim x hidden_size")
		if err != nil {
			return err
		}
		kProj, err := loadLinearLayer(tensors, linears, i, prefix+".k_proj", expectedKV, int(m.HiddenSize), "num_key_value_heads * head_dim x hidden_size")
		if err != nil {
			return err
		}
		vProj, err := loadLinearLayer(tensors, linears, i, prefix+".v_proj", expectedKV, int(m.HiddenSize), "num_key_value_heads * head_dim x hidden_size")
		if err != nil {
			return err
		}
		oProj, err := loadLinearLayer(tensors, linears, i, prefix+".attn_out", int(m.HiddenSize), expectedQ, "hidden_size x num_attention_heads * head_dim")
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

		router, err := loadLinearLayer(tensors, linears, i, prefix+".router", int(m.NumLocalExperts), int(m.HiddenSize), "num_local_experts x hidden_size")
		if err != nil {
			return err
		}

		m.Layers[i] = &Layer{
			AttentionNorm: attnNorm,
			Attention: &Attention{
				QProj: qProj,
				KProj: kProj,
				VProj: vProj,
				OProj: oProj,
				Sinks: sinks,
			},
			FFNNorm: ffnNorm,
			Router:  router,
		}
	}

	return nil
}
