package create

import (
	"encoding/json"
	"fmt"
	"strings"
)

type gemma4ImportTransform struct {
	numLayers      int
	numExperts     int
	hasAudioConfig bool
	unifiedAudio   bool
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
		numLayers:      numLayers,
		numExperts:     numExperts,
		hasAudioConfig: cfg.AudioConfig != nil,
		unifiedAudio:   cfg.AudioConfig != nil && cfg.AudioConfig.ModelType == "gemma4_unified_audio",
	}, nil
}

func (t gemma4ImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	base := normalizeQuantType(quantize)
	switch {
	case isGemma4VisionTensor(name) || isGemma4AudioTensor(name):
		if t.unifiedAudio && strings.HasSuffix(name, "embed_audio.embedding_projection.weight") {
			// The unified 12B architecture projects raw waveform blocks directly
			// into language embeddings. Even MXFP8 caused mixed-media requests to
			// lose speech, while retaining this small matrix costs about 5 MB.
			return ""
		}
		// Media namespaces also contain large two-dimensional position tables.
		// Only weights consumed as linear projections are eligible here.
		if !strings.HasSuffix(name, ".weight") {
			return ""
		}
		eligible := gemma4MediaQuantization(name, shape, quantize)
		if eligible == "" {
			return ""
		}
		if isGemma4MediaBoundaryProjection(name) {
			// The media boundary is small relative to the decoder and directly
			// determines the semantic embeddings consumed by the language model.
			// Keep it quantized, but promote four-bit imports to eight bits.
			return sensitiveType(true, shape, base)
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

func gemma4MediaQuantization(name string, shape []int32, quantize string) string {
	if len(shape) != 2 || int64(shape[0])*int64(shape[1]) < 1024 {
		return ""
	}
	base := normalizeQuantType(quantize)
	if !isAligned(shape, base) {
		return ""
	}
	if base == "int4" || base == "nvfp4" || base == "mxfp4" {
		if strings.Contains(name, ".v_proj") || strings.Contains(name, ".k_proj") || strings.Contains(name, "down_proj") {
			if promoted := eightBit(base); isAligned(shape, promoted) {
				return promoted
			}
		}
	}
	return base
}

func (t gemma4ImportTransform) includeTensor(name string) bool {
	return !isGemma4AudioTensor(name) || t.hasAudioConfig
}

func isGemma4VisionTensor(name string) bool {
	return isVision(name) || strings.Contains(name, "embed_vision") || strings.Contains(name, "vision_embedder")
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
// most affects quality: attention output (v/k) and the residual stream
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
