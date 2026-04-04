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
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config
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
		Config: &cfg,
		tok:    tok,
	}, nil
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

	return nil
}

func (m *Model) Forward(inputs *mlx.Array, caches []cache.Cache) *mlx.Array {
	panic("gptoss MLX forward is not implemented yet")
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	if m.LMHead == nil {
		panic("gptoss MLX unembed called before weights loaded")
	}
	return m.LMHead.Forward(x)
}

func (m *Model) NumLayers() int {
	return int(m.NumHiddenLayers)
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
