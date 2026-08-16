package create

import (
	"encoding/json"
	"fmt"
	"strings"
)

type gemma4ImportTransform struct {
	numLayers    int
	numExperts   int
	unifiedAudio bool
}

// gemma4Config is a minimal subset of the Gemma 4 config.json used for quant decisions.
type gemma4Config struct {
	NumHiddenLayers int `json:"num_hidden_layers"`
	NumExperts      int `json:"num_experts"`
	AudioConfig     *struct {
		ModelType string `json:"model_type"`
	} `json:"audio_config"`
	TextConfig struct {
		NumHiddenLayers int `json:"num_hidden_layers"`
		NumExperts      int `json:"num_experts"`
	} `json:"text_config"`
}

func newGemma4ImportTransform(rawConfig json.RawMessage) (quantizePolicy, error) {
	var cfg gemma4Config
	if err := json.Unmarshal(rawConfig, &cfg); err != nil {
		return nil, fmt.Errorf("gemma4: parse config.json: %w", err)
	}

	numLayers := cfg.NumHiddenLayers
	if numLayers == 0 {
		numLayers = cfg.TextConfig.NumHiddenLayers
	}
	numExperts := cfg.NumExperts
	if numExperts == 0 {
		numExperts = cfg.TextConfig.NumExperts
	}

	return gemma4ImportTransform{
		numLayers:    numLayers,
		numExperts:   numExperts,
		unifiedAudio: cfg.AudioConfig != nil && cfg.AudioConfig.ModelType == "gemma4_unified_audio",
	}, nil
}

func (t gemma4ImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	base := normalizeQuantType(quantize)
	switch {
	case isGemma4VisionTensor(name) || isGemma4AudioTensor(name):
		if t.unifiedAudio && strings.HasSuffix(name, "embed_audio.embedding_projection.weight") {
			// The unified architecture projects raw waveform blocks directly into
			// language embeddings. Keep this semantic boundary dense.
			return ""
		}
		// Media namespaces include positions, norms, convolutions, and learned
		// biases. Only a conventional linear weight is eligible.
		if !strings.HasSuffix(name, ".weight") || isGemma4MediaPositionTensor(name) {
			return ""
		}
		eligible := getEligibleTensorQuantization(name, shape, quantize)
		if eligible == "" {
			return ""
		}
		if isGemma4MediaBoundaryProjection(name) {
			// Four-bit media boundaries use their aligned eight-bit mate or stay
			// dense. Never fall back to the requested four-bit representation.
			eight := eightBit(base)
			if isAligned(shape, eight) {
				return eight
			}
			return ""
		}
		return eligible
	case isEmbedTokensWeight(name):
		// The embedding doubles as the lm_head projection; an 8-bit type keeps
		// quality close to bf16 (matching GGUF Q6_K) while saving bandwidth.
		// Fall back to the base type when 8-bit does not fit the vocab shape.
		if e := promoteEmbedding(shape, base); e != "" {
			return e
		}
		if isAligned(shape, base) {
			return base
		}
		return ""
	case t.isSensitiveProjection(name) && eightBit(base) != base:
		return sensitiveType(t.promoteSensitive(name), shape, base)
	default:
		// Routing gates, norms, embeddings, etc. are handled by the generic
		// policy; everything else quantizes at the requested type.
		return GetTensorQuantization(name, shape, quantize)
	}
}

func isGemma4MediaPositionTensor(name string) bool {
	return strings.Contains(name, "position") || strings.Contains(name, "pos_embed")
}

func isGemma4VisionTensor(name string) bool {
	return isVision(name)
}

func isGemma4AudioTensor(name string) bool {
	return isAudioTower(name) || strings.Contains(name, "audio")
}

func isGemma4MediaBoundaryProjection(name string) bool {
	for _, suffix := range []string{
		"embed_vision.embedding_projection.weight",
		"vision_embedder.patch_dense.weight",
		"audio_tower.output_proj.weight",
		"embed_audio.embedding_projection.weight",
	} {
		if strings.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}

// isSensitiveProjection reports the value/key/down projections whose precision
// most affects quality — attention output (v/k) and the residual stream
// (down). Non-text media tensors are handled separately.
func (t gemma4ImportTransform) isSensitiveProjection(name string) bool {
	if isGemma4VisionTensor(name) || isGemma4AudioTensor(name) {
		return false
	}
	return strings.Contains(name, ".v_proj") ||
		strings.Contains(name, ".k_proj") ||
		strings.Contains(name, "down_proj")
}

// promoteSensitive decides whether a sensitive projection uses 8-bit precision.
// 8-expert models share very few KV heads, so their k/v projections are always
// promoted; otherwise v/down projections are promoted at the input and output
// layers and periodically between (useMoreBits), where residual-stream error
// accumulates most.
func (t gemma4ImportTransform) promoteSensitive(name string) bool {
	if t.numLayers == 0 {
		return false
	}
	if t.numExperts == 8 && (strings.Contains(name, ".v_proj") || strings.Contains(name, ".k_proj")) {
		return true
	}
	if strings.Contains(name, ".k_proj") {
		return false // k_proj is promoted only via the 8-expert path
	}
	layer := layerIndex(name)
	return layer >= 0 && useMoreBits(layer, t.numLayers)
}
