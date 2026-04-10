package create

import (
	"strings"

	"github.com/ollama/ollama/x/safetensors"
)

type gptossImportTransform struct{}

func newGPTOSSImportTransform(modelDir string, cfg sourceModelConfig) (tensorImportTransform, error) {
	return gptossImportTransform{}, nil
}

func (t gptossImportTransform) skipTensor(string) bool { return false }

func (t gptossImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	return GetTensorQuantization(name, shape, quantize)
}

func (t gptossImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}

	return []*safetensors.TensorData{td.WithName(t.canonicalTensorName(td.Name))}, nil
}

func (t gptossImportTransform) canonicalTensorName(name string) string {
	switch name {
	case "model.embed_tokens.weight":
		return "embedding.weight"
	case "model.norm.weight":
		return "output_norm.weight"
	case "lm_head.weight":
		return "output.weight"
	}

	const layerPrefix = "model.layers."
	if !strings.HasPrefix(name, layerPrefix) {
		return name
	}

	remainder := strings.TrimPrefix(name, layerPrefix)
	layer, suffix, ok := strings.Cut(remainder, ".")
	if !ok || layer == "" {
		return name
	}

	prefix := "blocks." + layer + "."
	switch suffix {
	case "input_layernorm.weight":
		return prefix + "attn_norm.weight"
	case "self_attn.q_proj.weight":
		return prefix + "q_proj.weight"
	case "self_attn.q_proj.bias":
		return prefix + "q_proj.bias"
	case "self_attn.k_proj.weight":
		return prefix + "k_proj.weight"
	case "self_attn.k_proj.bias":
		return prefix + "k_proj.bias"
	case "self_attn.v_proj.weight":
		return prefix + "v_proj.weight"
	case "self_attn.v_proj.bias":
		return prefix + "v_proj.bias"
	case "self_attn.o_proj.weight":
		return prefix + "attn_out.weight"
	case "self_attn.o_proj.bias":
		return prefix + "attn_out.bias"
	case "self_attn.sinks":
		return prefix + "attn_sinks"
	case "post_attention_layernorm.weight":
		return prefix + "ffn_norm.weight"
	case "mlp.router.weight":
		return prefix + "router.weight"
	case "mlp.router.bias":
		return prefix + "router.bias"
	case "mlp.experts.gate_up_proj_blocks":
		return prefix + "experts.gate_up_proj.blocks"
	case "mlp.experts.gate_up_proj_scales":
		return prefix + "experts.gate_up_proj.scales"
	case "mlp.experts.gate_up_proj_bias":
		return prefix + "experts.gate_up_proj.bias"
	case "mlp.experts.down_proj_blocks":
		return prefix + "experts.down_proj.blocks"
	case "mlp.experts.down_proj_scales":
		return prefix + "experts.down_proj.scales"
	case "mlp.experts.down_proj_bias":
		return prefix + "experts.down_proj.bias"
	default:
		return name
	}
}
