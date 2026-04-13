package create

import (
	"fmt"
	"io"
	"math"
	"strings"

	fsggml "github.com/ollama/ollama/fs/ggml"
	ggml "github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/x/safetensors"
)

type gptossImportTransform struct {
	pendingBlocks map[string]*safetensors.TensorData
	pendingScales map[string]*safetensors.TensorData
}

func newGPTOSSImportTransform(modelDir string, cfg sourceModelConfig) (tensorImportTransform, error) {
	return &gptossImportTransform{
		pendingBlocks: make(map[string]*safetensors.TensorData),
		pendingScales: make(map[string]*safetensors.TensorData),
	}, nil
}

func (t *gptossImportTransform) skipTensor(string) bool { return false }

func (t *gptossImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	if strings.Contains(name, ".experts.") && strings.HasSuffix(name, ".weight") {
		return ""
	}
	return GetTensorQuantization(name, shape, quantize)
}

func (t *gptossImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}

	name := t.canonicalTensorName(td.Name)
	switch {
	case strings.HasSuffix(td.Name, "_blocks"):
		repacked, err := repackRawGPTOSSMXFP4Tensor(td, name)
		if err != nil {
			return nil, err
		}
		t.pendingBlocks[name] = repacked
		return t.maybeEmitExpertWeight(name)
	case strings.HasSuffix(td.Name, "_scales"):
		t.pendingScales[name] = td.WithName(name)
		return t.maybeEmitExpertWeight(name)
	default:
		return []*safetensors.TensorData{td.WithName(name)}, nil
	}
}

func repackRawGPTOSSMXFP4Tensor(td *safetensors.TensorData, name string) (*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}
	if td.Dtype != "U8" {
		return nil, fmt.Errorf("gpt-oss expert tensor %q dtype = %q, want U8", td.Name, td.Dtype)
	}
	if len(td.Shape) < 1 || td.Size%16 != 0 {
		return nil, fmt.Errorf("gpt-oss expert tensor %q size = %d, want a multiple of 16", td.Name, td.Size)
	}

	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		return nil, fmt.Errorf("read gpt-oss expert tensor %q: %w", td.Name, err)
	}
	if len(raw)%16 != 0 {
		return nil, fmt.Errorf("gpt-oss expert tensor %q byte length = %d, want a multiple of 16", td.Name, len(raw))
	}

	packed := make([]byte, len(raw))
	var tmp [16]byte
	for i := 0; i < len(raw); i += 16 {
		block := raw[i : i+16]
		for j := range 8 {
			a, b := block[j], block[j+8]
			tmp[2*j+0] = (a & 0x0F) | (b << 4)
			tmp[2*j+1] = (a >> 4) | (b & 0xF0)
		}
		copy(packed[i:i+16], tmp[:])
	}

	return safetensors.NewTensorDataFromBytes(name, td.Dtype, td.Shape, packed), nil
}

func (t *gptossImportTransform) maybeEmitExpertWeight(name string) ([]*safetensors.TensorData, error) {
	blocks := t.pendingBlocks[name]
	scales := t.pendingScales[name]
	if blocks == nil || scales == nil {
		return nil, nil
	}

	delete(t.pendingBlocks, name)
	delete(t.pendingScales, name)

	weight, err := dequantizeGPTOSSMXFP4Tensor(name, blocks, scales)
	if err != nil {
		return nil, err
	}
	return []*safetensors.TensorData{weight}, nil
}

func dequantizeGPTOSSMXFP4Tensor(name string, blocks, scales *safetensors.TensorData) (*safetensors.TensorData, error) {
	if blocks == nil || scales == nil {
		return nil, fmt.Errorf("gpt-oss expert tensor %q requires blocks and scales", name)
	}
	if blocks.Dtype != "U8" {
		return nil, fmt.Errorf("gpt-oss expert blocks %q dtype = %q, want U8", blocks.Name, blocks.Dtype)
	}
	if scales.Dtype != "U8" {
		return nil, fmt.Errorf("gpt-oss expert scales %q dtype = %q, want U8", scales.Name, scales.Dtype)
	}
	if len(blocks.Shape) != 4 {
		return nil, fmt.Errorf("gpt-oss expert blocks %q shape = %v, want [experts out groups 16]", blocks.Name, blocks.Shape)
	}
	if len(scales.Shape) != 3 {
		return nil, fmt.Errorf("gpt-oss expert scales %q shape = %v, want [experts out groups]", scales.Name, scales.Shape)
	}
	if blocks.Shape[0] != scales.Shape[0] || blocks.Shape[1] != scales.Shape[1] || blocks.Shape[2] != scales.Shape[2] {
		return nil, fmt.Errorf("gpt-oss expert tensor %q shape mismatch: blocks=%v scales=%v", name, blocks.Shape, scales.Shape)
	}
	if blocks.Shape[3] != 16 {
		return nil, fmt.Errorf("gpt-oss expert blocks %q trailing shape = %v, want [... 16]", blocks.Name, blocks.Shape)
	}

	blockBytes, err := io.ReadAll(blocks.Reader())
	if err != nil {
		return nil, fmt.Errorf("read gpt-oss expert blocks %q: %w", blocks.Name, err)
	}
	scaleBytes, err := io.ReadAll(scales.Reader())
	if err != nil {
		return nil, fmt.Errorf("read gpt-oss expert scales %q: %w", scales.Name, err)
	}

	groupCount := int(blocks.Shape[0] * blocks.Shape[1] * blocks.Shape[2])
	if len(blockBytes) != groupCount*16 {
		return nil, fmt.Errorf("gpt-oss expert blocks %q byte length = %d, want %d", blocks.Name, len(blockBytes), groupCount*16)
	}
	if len(scaleBytes) != groupCount {
		return nil, fmt.Errorf("gpt-oss expert scales %q byte length = %d, want %d", scales.Name, len(scaleBytes), groupCount)
	}

	ggmlBlocks := make([]byte, groupCount*17)
	for i := 0; i < groupCount; i++ {
		dst := ggmlBlocks[i*17 : (i+1)*17]
		dst[0] = scaleBytes[i]
		copy(dst[1:], blockBytes[i*16:(i+1)*16])
	}

	decodedElems := uint64(groupCount * 32)
	values := ggml.ConvertToF32(ggmlBlocks, uint32(fsggml.TensorTypeMXFP4), decodedElems)
	for i, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("gpt-oss expert tensor %q dequantized invalid value at %d", name, i)
		}
	}

	raw, err := EncodeFloatTensor("BF16", values)
	if err != nil {
		return nil, fmt.Errorf("encode gpt-oss expert tensor %q as BF16: %w", name, err)
	}

	shape := []int32{blocks.Shape[0], blocks.Shape[1], blocks.Shape[2] * 32}
	return safetensors.NewTensorDataFromBytes(name, "BF16", shape, raw), nil
}

func (t *gptossImportTransform) canonicalTensorName(name string) string {
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
		return prefix + "experts.gate_up_proj.weight"
	case "mlp.experts.gate_up_proj_scales":
		return prefix + "experts.gate_up_proj.weight"
	case "mlp.experts.gate_up_proj_bias":
		return prefix + "experts.gate_up_proj.bias"
	case "mlp.experts.down_proj_blocks":
		return prefix + "experts.down_proj.weight"
	case "mlp.experts.down_proj_scales":
		return prefix + "experts.down_proj.weight"
	case "mlp.experts.down_proj_bias":
		return prefix + "experts.down_proj.bias"
	default:
		return name
	}
}
