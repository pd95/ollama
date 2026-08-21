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
	apertusmetadata "github.com/ollama/ollama/x/models/apertus/metadata"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/quant"
	"github.com/ollama/ollama/x/tokenizer"
)

const (
	apertus1p0Architecture = "ApertusForCausalLM"
	apertus1p5Architecture = "Apertus1p5ForConditionalGeneration"

	maxConfigDimension = int32(1 << 24)
	maxLayers          = int32(1024)
	maxHeads           = int32(1024)
	// MLX dimensions are int32, but a tensor's element product is size_t. This
	// ceiling covers the documented 70B/1.5 vocabulary matrix while rejecting
	// implausible metadata before it reaches allocation.
	maxArrayElements = uint64(1 << 34)
)

func init() {
	base.Register(apertus1p0Architecture, newModel)
	base.Register(apertus1p5Architecture, newModel)
}

// RopeScaling carries the Llama 3 RoPE scaling block used by Apertus.
type RopeScaling struct {
	Factor                        float32 `json:"factor"`
	HighFreqFactor                float32 `json:"high_freq_factor"`
	LowFreqFactor                 float32 `json:"low_freq_factor"`
	OriginalMaxPositionEmbeddings int32   `json:"original_max_position_embeddings"`
	RopeTheta                     float32 `json:"rope_theta,omitempty"`
	RopeType                      string  `json:"rope_type,omitempty"`
	Type                          string  `json:"type,omitempty"`
}

// Config holds Apertus model configuration.
type Config struct {
	Architecture          string                `json:"-"`
	ModelType             string                `json:"model_type"`
	DType                 string                `json:"dtype"`
	HiddenSize            int32                 `json:"hidden_size"`
	IntermediateSize      int32                 `json:"intermediate_size"`
	NumHiddenLayers       int32                 `json:"num_hidden_layers"`
	NumAttentionHeads     int32                 `json:"num_attention_heads"`
	NumKeyValueHeads      int32                 `json:"num_key_value_heads"`
	VocabSize             int32                 `json:"vocab_size"`
	OutputVocabSize       int32                 `json:"output_vocab_size"`
	MaxPositionEmbeddings int32                 `json:"max_position_embeddings"`
	RMSNormEps            float32               `json:"rms_norm_eps"`
	RopeTheta             float32               `json:"rope_theta"`
	RopeScaling           RopeScaling           `json:"rope_scaling"`
	RopeParameters        RopeScaling           `json:"rope_parameters"`
	HiddenAct             string                `json:"hidden_act"`
	QKNorm                bool                  `json:"qk_norm"`
	PostNorm              bool                  `json:"post_norm"`
	AttentionBias         bool                  `json:"attention_bias"`
	MLPBias               bool                  `json:"mlp_bias"`
	TieWordEmbeddings     bool                  `json:"tie_word_embeddings"`
	ImageTokenID          int32                 `json:"-"`
	AudioTokenID          int32                 `json:"-"`
	ImageTokenOffset      int32                 `json:"-"`
	AudioTokenOffset      int32                 `json:"-"`
	VisionTokenizer       VisionTokenizerConfig `json:"-"`
	AudioTokenizer        AudioTokenizerConfig  `json:"-"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	QuantType      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	HeadDim   int32      `json:"-"`
	Scale     float32    `json:"-"`
	RopeFreqs *mlx.Array `json:"-"`
	prefix    string
}

// Model is an Apertus text model.
type Model struct {
	EmbedTokens      nn.EmbeddingLayer
	Layers           []*Layer
	Norm             *nn.RMSNorm
	LMHead           nn.LinearLayer
	Vision           *VisionTokenizer
	Audio            *AudioTokenizer
	mediaMemoryLimit uint64
	mediaResident    uint64

	tok *tokenizer.Tokenizer
	*Config
}

// ConfigureMediaMemory receives the runner's stable per-process media budget
// after model weights have been materialized.
func (m *Model) ConfigureMediaMemory(limit, resident uint64) {
	m.mediaMemoryLimit = limit
	m.mediaResident = resident
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
		cfg.QuantType = qt
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
	tokData, err = tokenizerDataForConfig(cfg, tokData)
	if err != nil {
		return nil, fmt.Errorf("prepare Apertus tokenizer: %w", err)
	}

	tokConfig := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	if data, err := root.Manifest.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = data
	}
	if data, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = data
	}
	if data, err := root.Manifest.ReadConfig("special_tokens_map.json"); err == nil {
		tokConfig.SpecialTokensMapJSON = data
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	return &Model{Config: &cfg, Layers: make([]*Layer, int(cfg.NumHiddenLayers)), tok: tok}, nil
}

func isApertus1p5Config(cfg Config) bool {
	return cfg.Architecture == apertus1p5Architecture
}

func tokenizerDataForConfig(cfg Config, data []byte) ([]byte, error) {
	if !isApertus1p5Config(cfg) {
		return data, nil
	}
	return pruneApertus1p5TokenizerAddedTokens(data, cfg.VocabSize)
}

func pruneApertus1p5TokenizerAddedTokens(data []byte, inputVocabSize int32) ([]byte, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	addedRaw, ok := raw["added_tokens"]
	if !ok {
		return data, nil
	}
	var added []json.RawMessage
	if err := json.Unmarshal(addedRaw, &added); err != nil {
		return nil, err
	}
	filtered := make([]json.RawMessage, 0, len(added))
	for _, tokenRaw := range added {
		var token struct {
			ID int32 `json:"id"`
		}
		if err := json.Unmarshal(tokenRaw, &token); err != nil {
			return nil, err
		}
		if token.ID >= 0 && token.ID < inputVocabSize {
			filtered = append(filtered, tokenRaw)
		}
	}
	filteredRaw, err := json.Marshal(filtered)
	if err != nil {
		return nil, err
	}
	raw["added_tokens"] = filteredRaw
	return json.Marshal(raw)
}

func parseConfig(configData []byte) (Config, error) {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(configData, &envelope); err != nil {
		return Config{}, fmt.Errorf("parse config envelope: %w", err)
	}
	active := configData
	if textRaw, ok := envelope["text_config"]; ok {
		active = textRaw
	}

	var cfg Config
	if err := json.Unmarshal(active, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}
	var archConfig struct {
		Architectures    []string              `json:"architectures"`
		ModelType        string                `json:"model_type"`
		ImageTokenID     int32                 `json:"image_token_id"`
		AudioTokenID     int32                 `json:"audio_token_id"`
		ImageTokenOffset int32                 `json:"image_token_offset"`
		AudioTokenOffset int32                 `json:"audio_token_offset"`
		VisionTokenizer  VisionTokenizerConfig `json:"vision_tokenizer_config"`
		AudioTokenizer   AudioTokenizerConfig  `json:"audio_tokenizer_config"`
	}
	if err := json.Unmarshal(configData, &archConfig); err != nil {
		return Config{}, fmt.Errorf("parse architecture: %w", err)
	}
	if len(archConfig.Architectures) > 0 && archConfig.Architectures[0] != "" {
		cfg.Architecture = archConfig.Architectures[0]
	} else {
		cfg.Architecture = archConfig.ModelType
	}
	cfg.ImageTokenID = archConfig.ImageTokenID
	cfg.AudioTokenID = archConfig.AudioTokenID
	cfg.ImageTokenOffset = archConfig.ImageTokenOffset
	cfg.AudioTokenOffset = archConfig.AudioTokenOffset
	cfg.VisionTokenizer = archConfig.VisionTokenizer
	cfg.AudioTokenizer = archConfig.AudioTokenizer
	if cfg.Architecture == "" {
		return Config{}, fmt.Errorf("missing architecture in config.json")
	}
	prefix, err := tensorPrefixForArchitecture(cfg.Architecture)
	if err != nil {
		return Config{}, err
	}
	cfg.prefix = prefix

	for _, field := range []struct {
		name  string
		value int32
		max   int32
	}{
		{"hidden_size", cfg.HiddenSize, maxConfigDimension},
		{"intermediate_size", cfg.IntermediateSize, maxConfigDimension},
		{"num_hidden_layers", cfg.NumHiddenLayers, maxLayers},
		{"num_attention_heads", cfg.NumAttentionHeads, maxHeads},
		{"vocab_size", cfg.VocabSize, maxConfigDimension},
		{"max_position_embeddings", cfg.MaxPositionEmbeddings, maxConfigDimension},
	} {
		if field.value <= 0 || field.value > field.max {
			return Config{}, fmt.Errorf("invalid %s: %d (must be in [1,%d])", field.name, field.value, field.max)
		}
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.NumKeyValueHeads < 0 || cfg.NumKeyValueHeads > maxHeads {
		return Config{}, fmt.Errorf("invalid num_key_value_heads: %d (must be in [1,%d])", cfg.NumKeyValueHeads, maxHeads)
	}
	if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return Config{}, fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
	}
	cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	if cfg.HeadDim%2 != 0 {
		return Config{}, fmt.Errorf("head_dim must be even: %d", cfg.HeadDim)
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return Config{}, fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if _, err := checkedProduct("attention projection", uint64(cfg.NumAttentionHeads), uint64(cfg.HeadDim)); err != nil {
		return Config{}, err
	}
	if _, err := checkedProduct("key/value projection", uint64(cfg.NumKeyValueHeads), uint64(cfg.HeadDim)); err != nil {
		return Config{}, err
	}
	if isApertus1p5Config(cfg) {
		if cfg.RopeParameters.RopeTheta == 0 {
			return Config{}, fmt.Errorf("missing rope_parameters.rope_theta")
		}
		cfg.RopeTheta = cfg.RopeParameters.RopeTheta
		cfg.RopeScaling = cfg.RopeParameters
	}
	if _, err := checkedProduct("embedding", uint64(cfg.VocabSize), uint64(cfg.HiddenSize)); err != nil {
		return Config{}, err
	}
	if _, err := checkedProduct("feed-forward projection", uint64(cfg.IntermediateSize), uint64(cfg.HiddenSize)); err != nil {
		return Config{}, err
	}
	if _, err := checkedProduct("layer tensor descriptors", uint64(cfg.NumHiddenLayers), 15); err != nil {
		return Config{}, err
	}
	if _, err := checkedProduct("RoPE frequency table", uint64(cfg.HeadDim), 1); err != nil {
		return Config{}, err
	}

	if !positiveFinite(cfg.RMSNormEps) {
		return Config{}, fmt.Errorf("invalid rms_norm_eps: %v", cfg.RMSNormEps)
	}
	if !positiveFinite(cfg.RopeTheta) {
		return Config{}, fmt.Errorf("invalid rope_theta: %v", cfg.RopeTheta)
	}
	if cfg.HiddenAct != "xielu" {
		return Config{}, fmt.Errorf("unsupported hidden_act %q", cfg.HiddenAct)
	}
	if cfg.VocabSize <= 0 {
		return Config{}, fmt.Errorf("invalid vocab_size: %d", cfg.VocabSize)
	}
	if isApertus1p5Config(cfg) {
		if cfg.OutputVocabSize <= 0 || cfg.OutputVocabSize > cfg.VocabSize {
			return Config{}, fmt.Errorf("invalid output_vocab_size: %d (vocab_size: %d)", cfg.OutputVocabSize, cfg.VocabSize)
		}
	} else if cfg.OutputVocabSize == 0 {
		cfg.OutputVocabSize = cfg.VocabSize
	} else if cfg.OutputVocabSize != cfg.VocabSize {
		return Config{}, fmt.Errorf("invalid output_vocab_size: %d (vocab_size: %d)", cfg.OutputVocabSize, cfg.VocabSize)
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
	if !positiveFinite(cfg.RopeScaling.Factor) ||
		!positiveFinite(cfg.RopeScaling.LowFreqFactor) ||
		!positiveFinite(cfg.RopeScaling.HighFreqFactor) {
		return Config{}, fmt.Errorf("invalid llama3 rope scaling factors")
	}
	if cfg.RopeScaling.HighFreqFactor <= cfg.RopeScaling.LowFreqFactor {
		return Config{}, fmt.Errorf("high_freq_factor (%v) must exceed low_freq_factor (%v)", cfg.RopeScaling.HighFreqFactor, cfg.RopeScaling.LowFreqFactor)
	}
	if cfg.RopeScaling.OriginalMaxPositionEmbeddings <= 0 || cfg.RopeScaling.OriginalMaxPositionEmbeddings > maxConfigDimension {
		return Config{}, fmt.Errorf("invalid original_max_position_embeddings: %d", cfg.RopeScaling.OriginalMaxPositionEmbeddings)
	}
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	return cfg, nil
}

func positiveFinite(v float32) bool {
	return v > 0 && !math.IsInf(float64(v), 0) && !math.IsNaN(float64(v))
}

func checkedProduct(name string, values ...uint64) (uint64, error) {
	product := uint64(1)
	for _, value := range values {
		if value == 0 || product > maxArrayElements/value {
			return 0, fmt.Errorf("%s element product exceeds %d", name, maxArrayElements)
		}
		product *= value
	}
	return product, nil
}

func (c Config) ropeType() string {
	if c.RopeScaling.RopeType != "" {
		return strings.ToLower(c.RopeScaling.RopeType)
	}
	return strings.ToLower(c.RopeScaling.Type)
}

// Llama3InvFreqs returns inverse RoPE frequencies matching Transformers.
func Llama3InvFreqs(headDim int32, base, factor, lowFreqFactor, highFreqFactor float32, originalContext int32) ([]float32, error) {
	if headDim <= 0 || headDim > maxConfigDimension || headDim%2 != 0 {
		return nil, fmt.Errorf("head_dim must be a bounded positive even number: %d", headDim)
	}
	if !positiveFinite(base) || !positiveFinite(factor) || !positiveFinite(lowFreqFactor) || !positiveFinite(highFreqFactor) ||
		highFreqFactor <= lowFreqFactor || originalContext <= 0 || originalContext > maxConfigDimension {
		return nil, fmt.Errorf("invalid llama3 rope parameters")
	}
	inv := make([]float32, int(headDim/2))
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
		if v == 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("invalid llama3 inverse frequency at index %d", i)
		}
		inv[i] = float32(v)
	}
	return inv, nil
}

// Llama3Freqs returns the frequency values expected by MLX RoPEWithFreqs.
func Llama3Freqs(headDim int32, base, factor, lowFreqFactor, highFreqFactor float32, originalContext int32) ([]float32, error) {
	inv, err := Llama3InvFreqs(headDim, base, factor, lowFreqFactor, highFreqFactor, originalContext)
	if err != nil {
		return nil, err
	}
	freqs := make([]float32, len(inv))
	for i, v := range inv {
		freqs[i] = 1 / v
		if !positiveFinite(freqs[i]) {
			return nil, fmt.Errorf("invalid llama3 frequency at index %d", i)
		}
	}
	return freqs, nil
}

func tensorPrefixForArchitecture(architecture string) (string, error) {
	switch architecture {
	case apertus1p0Architecture:
		return "model.", nil
	case apertus1p5Architecture:
		return "model.language_model.", nil
	default:
		return "", fmt.Errorf("unsupported Apertus architecture %q", architecture)
	}
}

func validateTensorNamespace(tensors map[string]*mlx.Array, architecture, expected string) error {
	for name := range tensors {
		switch architecture {
		case apertus1p0Architecture:
			if strings.HasPrefix(name, "model.language_model.") {
				return fmt.Errorf("unexpected Apertus 1.5 tensor namespace %q for %s", name, architecture)
			}
			if strings.HasPrefix(name, "model.vision_tokenizer.") || strings.HasPrefix(name, "model.audio_tokenizer.") {
				return fmt.Errorf("unexpected Apertus 1.5 media tensor %q for %s", name, architecture)
			}
		case apertus1p5Architecture:
			if strings.HasPrefix(name, "model.embed_tokens.") ||
				strings.HasPrefix(name, "model.norm.") || strings.HasPrefix(name, "model.layers.") {
				return fmt.Errorf("unexpected Apertus 1.0 tensor namespace %q for %s", name, architecture)
			}
		}
	}
	if expected == "" {
		return fmt.Errorf("missing tensor prefix for %s", architecture)
	}
	return nil
}

func apertureMediaDescriptors(tensors map[string]*mlx.Array) (map[string]apertusmetadata.TensorDescriptor, error) {
	descriptors := make(map[string]apertusmetadata.TensorDescriptor, len(tensors))
	for name, tensor := range tensors {
		if tensor == nil || !tensor.Valid() {
			continue
		}
		dims := tensor.Dims()
		shape := make([]int32, len(dims))
		for i, dim := range dims {
			if dim <= 0 || uint64(dim) > uint64(math.MaxInt32) {
				return nil, fmt.Errorf("Apertus media tensor %q has invalid dimension %d", name, dim)
			}
			shape[i] = int32(dim)
		}
		descriptors[name] = apertusmetadata.TensorDescriptor{Dtype: tensor.DType().String(), Shape: shape}
	}
	return descriptors, nil
}

func (m *Model) mediaMetadataConfig() apertusmetadata.Config {
	return apertusmetadata.Config{
		Architectures:    []string{m.Architecture},
		ModelType:        m.ModelType,
		ImageTokenID:     m.ImageTokenID,
		AudioTokenID:     m.AudioTokenID,
		ImageTokenOffset: m.ImageTokenOffset,
		AudioTokenOffset: m.AudioTokenOffset,
		Vision: apertusmetadata.VisionConfig{
			CodebookSize: m.VisionTokenizer.CodebookSize, EmbedDim: m.VisionTokenizer.EmbedDim,
			InChannels: m.VisionTokenizer.InChannels, ChannelMultiplier: m.VisionTokenizer.ChannelMultiplier,
		},
		Audio: apertusmetadata.AudioConfig{
			CodebookSize: m.AudioTokenizer.CodebookSize, CodebookDim: m.AudioTokenizer.CodebookDim,
			AudioChannels: m.AudioTokenizer.AudioChannels, SamplingRate: m.AudioTokenizer.SamplingRate,
			UpsamplingRatios: m.AudioTokenizer.UpsamplingRatios,
		},
	}
}

func hasTensorPrefix(tensors map[string]*mlx.Array, prefix string) bool {
	for name := range tensors {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func canValidateVisionTokenizer(tensors map[string]*mlx.Array, cfg VisionTokenizerConfig) bool {
	return hasTensorPrefix(tensors, "model.vision_tokenizer.") && cfg.validate() == nil
}

func canValidateAudioTokenizer(tensors map[string]*mlx.Array, cfg AudioTokenizerConfig) bool {
	return hasTensorPrefix(tensors, "model.audio_tokenizer.") && cfg.validate() == nil
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

func validateTensors(tensors map[string]*mlx.Array, cfg *Config) error {
	if cfg == nil {
		return fmt.Errorf("missing Apertus config")
	}
	prefix, err := tensorPrefixForArchitecture(cfg.Architecture)
	if err != nil {
		return err
	}
	if cfg.prefix != "" && cfg.prefix != prefix {
		return fmt.Errorf("Apertus tensor prefix %q does not match architecture %s", cfg.prefix, cfg.Architecture)
	}
	if err := validateTensorNamespace(tensors, cfg.Architecture, prefix); err != nil {
		return err
	}
	if err := validateMatrix(tensors, prefix+"embed_tokens", int(cfg.VocabSize), int(cfg.HiddenSize), cfg, true); err != nil {
		return err
	}
	if err := validateDense(tensors, prefix+"norm.weight", []int{int(cfg.HiddenSize)}); err != nil {
		return err
	}
	if err := validateMatrix(tensors, "lm_head", int(cfg.OutputVocabSize), int(cfg.HiddenSize), cfg, false); err != nil {
		return err
	}
	qOut := int(cfg.NumAttentionHeads * cfg.HeadDim)
	kvOut := int(cfg.NumKeyValueHeads * cfg.HeadDim)
	for i := range cfg.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("%slayers.%d", prefix, i)
		for _, spec := range []struct {
			name  string
			shape []int
		}{
			{layerPrefix + ".attention_layernorm.weight", []int{int(cfg.HiddenSize)}},
			{layerPrefix + ".feedforward_layernorm.weight", []int{int(cfg.HiddenSize)}},
			{layerPrefix + ".self_attn.q_norm.weight", []int{int(cfg.HeadDim)}},
			{layerPrefix + ".self_attn.k_norm.weight", []int{int(cfg.HeadDim)}},
			{layerPrefix + ".mlp.act_fn.alpha_p", []int{1}},
			{layerPrefix + ".mlp.act_fn.alpha_n", []int{1}},
			{layerPrefix + ".mlp.act_fn.beta", nil},
			{layerPrefix + ".mlp.act_fn.eps", nil},
		} {
			if err := validateDense(tensors, spec.name, spec.shape); err != nil {
				return fmt.Errorf("layer %d: %w", i, err)
			}
		}
		for _, spec := range []struct {
			path       string
			out, input int
		}{
			{layerPrefix + ".self_attn.q_proj", qOut, int(cfg.HiddenSize)},
			{layerPrefix + ".self_attn.k_proj", kvOut, int(cfg.HiddenSize)},
			{layerPrefix + ".self_attn.v_proj", kvOut, int(cfg.HiddenSize)},
			{layerPrefix + ".self_attn.o_proj", int(cfg.HiddenSize), qOut},
			{layerPrefix + ".mlp.up_proj", int(cfg.IntermediateSize), int(cfg.HiddenSize)},
			{layerPrefix + ".mlp.down_proj", int(cfg.HiddenSize), int(cfg.IntermediateSize)},
		} {
			if err := validateMatrix(tensors, spec.path, spec.out, spec.input, cfg, false); err != nil {
				return fmt.Errorf("layer %d: %w", i, err)
			}
		}
	}
	return nil
}

func requireArray(tensors map[string]*mlx.Array, name string) (*mlx.Array, error) {
	t := tensors[name]
	if t == nil || !t.Valid() {
		return nil, fmt.Errorf("missing tensor %q", name)
	}
	return t, nil
}

func validateShape(name string, t *mlx.Array, want []int) error {
	if t == nil || !t.Valid() {
		return fmt.Errorf("missing tensor %q", name)
	}
	got := t.Dims()
	if len(got) != len(want) {
		return fmt.Errorf("tensor %q shape %v, want %v", name, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			return fmt.Errorf("tensor %q shape %v, want %v", name, got, want)
		}
	}
	return nil
}

func isFloatDType(dtype mlx.DType) bool {
	return dtype == mlx.DTypeBFloat16 || dtype == mlx.DTypeFloat16 || dtype == mlx.DTypeFloat32
}

func validateDense(tensors map[string]*mlx.Array, name string, want []int) error {
	t, err := requireArray(tensors, name)
	if err != nil {
		return err
	}
	if err := validateShape(name, t, want); err != nil {
		return err
	}
	if !isFloatDType(t.DType()) {
		return fmt.Errorf("tensor %q dtype %s, want floating point", name, t.DType())
	}
	for _, suffix := range []string{"_scale", "_qbias", ".global_scale", "_scale_2"} {
		if companion := tensors[name+suffix]; companion != nil {
			return fmt.Errorf("orphan quantization companion %q", name+suffix)
		}
	}
	return nil
}

func validateMatrix(tensors map[string]*mlx.Array, path string, out, input int, cfg *Config, embedding bool) error {
	name := path + ".weight"
	if err := validateExplicitQuantization(cfg, name); err != nil {
		return err
	}
	weight, err := requireArray(tensors, name)
	if err != nil {
		for _, suffix := range []string{"_scale", "_qbias", ".global_scale", "_scale_2"} {
			if tensors[name+suffix] != nil {
				return fmt.Errorf("orphan quantization companion %q", name+suffix)
			}
		}
		return err
	}
	if tensors[path+".bias"] != nil {
		return fmt.Errorf("unexpected bias tensor %q", path+".bias")
	}
	scales := tensors[name+"_scale"]
	qbiases := tensors[name+"_qbias"]
	global := tensors[name+".global_scale"]
	legacyGlobal := tensors[name+"_scale_2"]
	if global != nil && legacyGlobal != nil {
		return fmt.Errorf("duplicate global scale companions for %q", name)
	}
	if scales == nil {
		if qbiases != nil || global != nil || legacyGlobal != nil {
			return fmt.Errorf("incomplete quantization companions for %q", name)
		}
		if tq, ok := cfg.TensorQuant[name]; ok && tq != nil && quant.Canonical(tq.QuantType) != "" {
			return fmt.Errorf("missing quantization companions for explicitly quantized tensor %q", name)
		}
		if err := validateShape(name, weight, []int{out, input}); err != nil {
			return err
		}
		if !isFloatDType(weight.DType()) {
			return fmt.Errorf("tensor %q dtype %s, want floating point", name, weight.DType())
		}
		return nil
	}

	groupSize, bits, mode := model.ResolveLinearQuantParams(
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant, name, weight, scales,
	)
	if groupSize <= 0 || input%groupSize != 0 {
		return fmt.Errorf("tensor %q input size %d is incompatible with group size %d", name, input, groupSize)
	}
	scaleShape := []int{out, input / groupSize}
	switch mode {
	case "affine":
		if bits != 4 && bits != 8 {
			return fmt.Errorf("tensor %q has unsupported affine bit width %d", name, bits)
		}
		if input%(32/bits) != 0 {
			return fmt.Errorf("tensor %q input size %d cannot be packed at %d bits", name, input, bits)
		}
		if err := validateShape(name, weight, []int{out, input / (32 / bits)}); err != nil {
			return err
		}
		if weight.DType() != mlx.DTypeUint32 {
			return fmt.Errorf("tensor %q dtype %s, want U32 packed affine", name, weight.DType())
		}
		if err := validateShape(name+"_scale", scales, scaleShape); err != nil {
			return err
		}
		if !isFloatDType(scales.DType()) {
			return fmt.Errorf("tensor %q dtype %s, want floating point", name+"_scale", scales.DType())
		}
		if err := validateShape(name+"_qbias", qbiases, scaleShape); err != nil {
			return err
		}
		if !isFloatDType(qbiases.DType()) {
			return fmt.Errorf("tensor %q dtype %s, want floating point", name+"_qbias", qbiases.DType())
		}
		if global != nil || legacyGlobal != nil {
			return fmt.Errorf("unexpected global scale for affine tensor %q", name)
		}
	case "nvfp4", "mxfp4":
		if bits != 4 || input%8 != 0 {
			return fmt.Errorf("tensor %q has invalid %s packing", name, mode)
		}
		if err := validateShape(name, weight, []int{out, input / 8}); err != nil {
			return err
		}
		if weight.DType() != mlx.DTypeUint32 {
			return fmt.Errorf("tensor %q dtype %s, want U32 packed %s", name, weight.DType(), mode)
		}
		if err := validateShape(name+"_scale", scales, scaleShape); err != nil {
			return err
		}
		if scales.DType() != mlx.DTypeUint8 {
			return fmt.Errorf("tensor %q dtype %s, want U8 %s scales", name+"_scale", scales.DType(), mode)
		}
		if qbiases != nil {
			return fmt.Errorf("unexpected qbias companion for %s tensor %q", mode, name)
		}
		gs := global
		if gs == nil {
			gs = legacyGlobal
		}
		if mode == "mxfp4" && gs != nil {
			return fmt.Errorf("unexpected global scale for mxfp4 tensor %q", name)
		}
		if gs != nil {
			if err := validateGlobalScale(name, gs, out, embedding); err != nil {
				return err
			}
		}
	case "mxfp8":
		if bits != 8 {
			return fmt.Errorf("tensor %q has invalid mxfp8 bit width %d", name, bits)
		}
		if err := validateShape(name, weight, []int{out, input}); err != nil {
			return err
		}
		if weight.DType() != mlx.DTypeUint8 {
			return fmt.Errorf("tensor %q dtype %s, want U8 mxfp8", name, weight.DType())
		}
		if err := validateShape(name+"_scale", scales, scaleShape); err != nil {
			return err
		}
		if scales.DType() != mlx.DTypeUint8 || qbiases != nil || global != nil || legacyGlobal != nil {
			return fmt.Errorf("invalid mxfp8 companions for tensor %q", name)
		}
	default:
		return fmt.Errorf("tensor %q has unsupported quantization mode %q", name, mode)
	}
	return nil
}

func validateExplicitQuantization(cfg *Config, name string) error {
	if raw := strings.TrimSpace(cfg.QuantType); raw != "" {
		canonical := quant.Canonical(raw)
		if canonical == "" {
			return fmt.Errorf("unsupported model quantization type %q", cfg.QuantType)
		}
		groupSize, bits, mode := model.QuantizationParams(canonical)
		if cfg.QuantBits != bits || cfg.QuantMode != mode {
			return fmt.Errorf("conflicting model quantization metadata for %q", cfg.QuantType)
		}
		if mode != "affine" && cfg.QuantGroupSize != groupSize {
			return fmt.Errorf("conflicting model quantization group size %d for %q", cfg.QuantGroupSize, cfg.QuantType)
		}
	}
	if cfg.TensorQuant == nil {
		return nil
	}
	tq, ok := cfg.TensorQuant[name]
	if !ok {
		return nil
	}
	if tq == nil || strings.TrimSpace(tq.QuantType) == "" {
		return fmt.Errorf("missing explicit quantization type for tensor %q", name)
	}
	canonical := quant.Canonical(tq.QuantType)
	if canonical == "" {
		return fmt.Errorf("unsupported quantization type %q for tensor %q", tq.QuantType, name)
	}
	groupSize, _, mode := model.QuantizationParams(canonical)
	if tq.GroupSize < 0 || (mode != "affine" && tq.GroupSize > 0 && tq.GroupSize != groupSize) {
		return fmt.Errorf("conflicting quantization group size %d for tensor %q type %q", tq.GroupSize, name, tq.QuantType)
	}
	return nil
}

func validateGlobalScale(name string, scale *mlx.Array, out int, embedding bool) error {
	if scale == nil || !scale.Valid() {
		return fmt.Errorf("tensor %q has invalid global scale", name)
	}
	if scale.DType() != mlx.DTypeFloat32 {
		return fmt.Errorf("tensor %q global scale dtype %s, want F32", name, scale.DType())
	}
	dims := scale.Dims()
	if len(dims) == 0 {
		return nil
	}
	if !embedding && len(dims) == 1 && dims[0] == out {
		return nil
	}
	if embedding {
		return fmt.Errorf("tensor %q embedding global scale must be scalar, got shape %v", name, dims)
	}
	return fmt.Errorf("tensor %q global scale must be scalar or shape [%d], got %v", name, out, dims)
}

// LoadWeights validates all model-owned tensor contracts before constructing layers.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := validateTensors(tensors, m.Config); err != nil {
		return err
	}
	prefix, err := tensorPrefixForArchitecture(m.Architecture)
	if err != nil {
		return err
	}
	m.prefix = prefix
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, prefix+"embed_tokens", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	m.Norm = nn.NewRMSNorm(tensors[prefix+"norm.weight"], m.RMSNormEps)
	m.LMHead = linears.Make("lm_head")

	mediaDescriptors, err := apertureMediaDescriptors(tensors)
	if err != nil {
		return err
	}
	mediaConfig := m.mediaMetadataConfig()
	if canValidateVisionTokenizer(tensors, m.VisionTokenizer) {
		if err := apertusmetadata.ValidateVisionInventory(mediaConfig, mediaDescriptors); err != nil {
			return err
		}
		m.Vision, err = loadVisionTokenizer(tensors, m.VisionTokenizer)
		if err != nil {
			return fmt.Errorf("load Apertus 1.5 vision tokenizer: %w", err)
		}
	}
	if canValidateAudioTokenizer(tensors, m.AudioTokenizer) {
		if err := apertusmetadata.ValidateAudioInventory(mediaConfig, mediaDescriptors); err != nil {
			return err
		}
		m.Audio, err = loadAudioTokenizer(tensors, m.AudioTokenizer)
		if err != nil {
			return fmt.Errorf("load Apertus 1.5 audio tokenizer: %w", err)
		}
	}

	for i := range m.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("%slayers.%d", prefix, i)
		act, err := newXIELU(
			tensors[layerPrefix+".mlp.act_fn.alpha_p"], tensors[layerPrefix+".mlp.act_fn.alpha_n"],
			tensors[layerPrefix+".mlp.act_fn.beta"], tensors[layerPrefix+".mlp.act_fn.eps"],
		)
		if err != nil {
			return fmt.Errorf("layer %d: load xielu activation parameters: %w", i, err)
		}
		m.Layers[i] = &Layer{
			AttentionNorm: nn.NewRMSNorm(tensors[layerPrefix+".attention_layernorm.weight"], m.RMSNormEps),
			FFNNorm:       nn.NewRMSNorm(tensors[layerPrefix+".feedforward_layernorm.weight"], m.RMSNormEps),
			Attention: &Attention{
				QProj: linears.Make(layerPrefix + ".self_attn.q_proj"),
				KProj: linears.Make(layerPrefix + ".self_attn.k_proj"),
				VProj: linears.Make(layerPrefix + ".self_attn.v_proj"),
				OProj: linears.Make(layerPrefix + ".self_attn.o_proj"),
				QNorm: nn.NewRMSNorm(tensors[layerPrefix+".self_attn.q_norm.weight"], m.RMSNormEps),
				KNorm: nn.NewRMSNorm(tensors[layerPrefix+".self_attn.k_norm.weight"], m.RMSNormEps),
			},
			MLP: &MLP{UpProj: linears.Make(layerPrefix + ".mlp.up_proj"), DownProj: linears.Make(layerPrefix + ".mlp.down_proj"), Act: act},
		}
	}
	return nil
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	h := mlx.Reshape(m.EmbedTokens.Forward(b.InputIDs), B, L, m.HiddenSize)
	h = m.scatterMedia(h, b)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
	}
	out := mlx.Reshape(m.Norm.Forward(h, m.RMSNormEps), B, L, m.HiddenSize)
	return out, out
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	return mlx.Reshape(m.LMHead.Forward(x), B, L, m.OutputVocabSize)
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }
func (m *Model) MaxContextLength() int           { return int(m.MaxPositionEmbeddings) }
func (m *Model) NumLayers() int                  { return len(m.Layers) }
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
	q := mlx.Transpose(mlx.Reshape(a.QProj.Forward(x), B, L, cfg.NumAttentionHeads, cfg.HeadDim), 0, 2, 1, 3)
	q = headRMSNorm(a.QNorm, q, B, cfg.NumAttentionHeads, L, cfg.HeadDim, cfg.RMSNormEps)
	k := mlx.Transpose(mlx.Reshape(a.KProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim), 0, 2, 1, 3)
	k = headRMSNorm(a.KNorm, k, B, cfg.NumKeyValueHeads, L, cfg.HeadDim, cfg.RMSNormEps)
	v := mlx.Transpose(mlx.Reshape(a.VProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim), 0, 2, 1, 3)
	q = mlx.Reshape(mlx.RoPEWithFreqs(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1, positions, cfg.RopeFreqs), B, cfg.NumAttentionHeads, L, cfg.HeadDim)
	k = mlx.Reshape(mlx.RoPEWithFreqs(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1, positions, cfg.RopeFreqs), B, cfg.NumKeyValueHeads, L, cfg.HeadDim)
	var kv nn.SDPAOption
	if c != nil {
		kv = nn.WithKVHistory(c.(cache.Attention).Update(b, k, v))
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
	}
	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(nn.CausalMask()))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.HiddenSize)
	return a.OProj.Forward(out)
}

func headRMSNorm(norm *nn.RMSNorm, x *mlx.Array, batch, heads, seqLen, headDim int32, eps float32) *mlx.Array {
	x = mlx.Reshape(x, -1, headDim)
	return mlx.Reshape(norm.Forward(x, eps), batch, heads, seqLen, headDim)
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	return m.DownProj.Forward(m.Act.Forward(m.UpProj.Forward(x)))
}

func (a *XIELU) Forward(x *mlx.Array) *mlx.Array {
	outDType := x.DType()
	x = x.AsType(mlx.DTypeFloat32)
	zero, one := mlx.FromValue[float32](0), mlx.FromValue[float32](1)
	alphaP, alphaN := mlx.FromValue(a.AlphaP), mlx.FromValue(a.AlphaN)
	beta, eps := mlx.FromValue(a.Beta), mlx.FromValue(a.Eps)
	positive := mlx.Add(mlx.Mul(alphaP, mlx.Mul(x, x)), mlx.Mul(beta, x))
	expm1 := mlx.Sub(mlx.Exp(mlx.Minimum(x, eps)), one)
	negative := mlx.Add(mlx.Mul(mlx.Sub(expm1, x), alphaN), mlx.Mul(beta, x))
	return mlx.Where(x.Greater(zero), positive, negative).AsType(outDType)
}

func newXIELU(alphaPParam, alphaNParam, betaParam, epsParam *mlx.Array) (*XIELU, error) {
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
	return &XIELU{AlphaP: float32(softplus64(float64(alphaP))), AlphaN: beta + float32(softplus64(float64(alphaN))), Beta: beta, Eps: eps}, nil
}

func scalarParam(x *mlx.Array) (float32, error) {
	if x == nil || !x.Valid() || x.Size() != 1 {
		return 0, fmt.Errorf("expected scalar or single-element tensor")
	}
	x = x.AsType(mlx.DTypeFloat32)
	mlx.Eval(x)
	values := x.Floats()
	if len(values) != 1 || math.IsNaN(float64(values[0])) || math.IsInf(float64(values[0]), 0) {
		return 0, fmt.Errorf("expected one finite scalar value")
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
