package create

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/d4l3k/go-bfloat16"
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
		t.pendingBlocks[name] = td.WithName(name)
		return t.maybeEmitExpertWeight(name)
	case strings.HasSuffix(td.Name, "_scales"):
		t.pendingScales[name] = td.WithName(name)
		return t.maybeEmitExpertWeight(name)
	case strings.HasSuffix(td.Name, "gate_up_proj_bias"):
		return splitGateUpBiasTensor(td.WithName(name))
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

	switch {
	case strings.Contains(name, ".experts.gate_up_proj.weight"):
		return dequantizeAndSplitGateUpTensor(name, blocks, scales)
	case strings.Contains(name, ".experts.down_proj.weight"):
		weight, err := dequantizeAndTransposeExpertWeight(name, blocks, scales)
		if err != nil {
			return nil, err
		}
		return []*safetensors.TensorData{weight}, nil
	default:
		weight, err := dequantizeGPTOSSMXFP4Tensor(name, blocks, scales)
		if err != nil {
			return nil, err
		}
		return []*safetensors.TensorData{weight}, nil
	}
}

func decodeGPTOSSMXFP4TensorValues(name string, blocks, scales *safetensors.TensorData) ([]float32, []int32, error) {
	if blocks == nil || scales == nil {
		return nil, nil, fmt.Errorf("gpt-oss expert tensor %q requires blocks and scales", name)
	}
	if blocks.Dtype != "U8" {
		return nil, nil, fmt.Errorf("gpt-oss expert blocks %q dtype = %q, want U8", blocks.Name, blocks.Dtype)
	}
	if scales.Dtype != "U8" {
		return nil, nil, fmt.Errorf("gpt-oss expert scales %q dtype = %q, want U8", scales.Name, scales.Dtype)
	}
	if len(blocks.Shape) != 4 {
		return nil, nil, fmt.Errorf("gpt-oss expert blocks %q shape = %v, want [experts out groups 16]", blocks.Name, blocks.Shape)
	}
	if len(scales.Shape) != 3 {
		return nil, nil, fmt.Errorf("gpt-oss expert scales %q shape = %v, want [experts out groups]", scales.Name, scales.Shape)
	}
	if blocks.Shape[0] != scales.Shape[0] || blocks.Shape[1] != scales.Shape[1] || blocks.Shape[2] != scales.Shape[2] {
		return nil, nil, fmt.Errorf("gpt-oss expert tensor %q shape mismatch: blocks=%v scales=%v", name, blocks.Shape, scales.Shape)
	}
	if blocks.Shape[3] != 16 {
		return nil, nil, fmt.Errorf("gpt-oss expert blocks %q trailing shape = %v, want [... 16]", blocks.Name, blocks.Shape)
	}

	blockBytes, err := io.ReadAll(blocks.Reader())
	if err != nil {
		return nil, nil, fmt.Errorf("read gpt-oss expert blocks %q: %w", blocks.Name, err)
	}
	scaleBytes, err := io.ReadAll(scales.Reader())
	if err != nil {
		return nil, nil, fmt.Errorf("read gpt-oss expert scales %q: %w", scales.Name, err)
	}

	groupCount := int(blocks.Shape[0] * blocks.Shape[1] * blocks.Shape[2])
	if len(blockBytes) != groupCount*16 {
		return nil, nil, fmt.Errorf("gpt-oss expert blocks %q byte length = %d, want %d", blocks.Name, len(blockBytes), groupCount*16)
	}
	if len(scaleBytes) != groupCount {
		return nil, nil, fmt.Errorf("gpt-oss expert scales %q byte length = %d, want %d", scales.Name, len(scaleBytes), groupCount)
	}

	ggmlBlocks := make([]byte, groupCount*17)
	var tmp [16]byte
	for i := 0; i < groupCount; i++ {
		src := blockBytes[i*16 : (i+1)*16]
		for j := range 8 {
			a, b := src[j], src[j+8]
			tmp[2*j+0] = (a & 0x0F) | (b << 4)
			tmp[2*j+1] = (a >> 4) | (b & 0xF0)
		}

		dst := ggmlBlocks[i*17 : (i+1)*17]
		dst[0] = scaleBytes[i]
		copy(dst[1:], tmp[:])
	}

	decodedElems := uint64(groupCount * 32)
	values := ggml.ConvertToF32(ggmlBlocks, uint32(fsggml.TensorTypeMXFP4), decodedElems)
	for i, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, nil, fmt.Errorf("gpt-oss expert tensor %q dequantized invalid value at %d", name, i)
		}
	}

	shape := []int32{blocks.Shape[0], blocks.Shape[1], blocks.Shape[2] * 32}
	return values, shape, nil
}

func dequantizeGPTOSSMXFP4Tensor(name string, blocks, scales *safetensors.TensorData) (*safetensors.TensorData, error) {
	values, shape, err := decodeGPTOSSMXFP4TensorValues(name, blocks, scales)
	if err != nil {
		return nil, err
	}

	raw, err := EncodeFloatTensor("BF16", values)
	if err != nil {
		return nil, fmt.Errorf("encode gpt-oss expert tensor %q as BF16: %w", name, err)
	}

	return safetensors.NewTensorDataFromBytes(name, "BF16", shape, raw), nil
}

func dequantizeAndSplitGateUpTensor(name string, blocks, scales *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	values, shape, err := decodeGPTOSSMXFP4TensorValues(name, blocks, scales)
	if err != nil {
		return nil, err
	}
	if len(shape) != 3 {
		return nil, fmt.Errorf("gpt-oss expert tensor %q shape = %v, want [experts out in]", name, shape)
	}

	experts, outDim, inDim := int(shape[0]), int(shape[1]), int(shape[2])
	if outDim%2 != 0 {
		return nil, fmt.Errorf("gpt-oss expert tensor %q output dim = %d, want even gate/up rows", name, outDim)
	}
	mid := outDim / 2

	gateRaw := make([]byte, experts*inDim*mid*2)
	upRaw := make([]byte, experts*inDim*mid*2)
	for e := 0; e < experts; e++ {
		for row := 0; row < outDim; row++ {
			dstRow := row / 2
			for col := 0; col < inDim; col++ {
				src := (e*outDim+row)*inDim + col
				dst := (e*inDim+col)*mid + dstRow
				bits := uint16(bfloat16.FromFloat32(values[src]))
				if row%2 == 0 {
					binary.LittleEndian.PutUint16(gateRaw[dst*2:], bits)
				} else {
					binary.LittleEndian.PutUint16(upRaw[dst*2:], bits)
				}
			}
		}
	}

	gateName := strings.Replace(name, "gate_up_proj", "gate_proj", 1)
	upName := strings.Replace(name, "gate_up_proj", "up_proj", 1)
	outShape := []int32{int32(experts), int32(inDim), int32(mid)}
	return []*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes(gateName, "BF16", outShape, gateRaw),
		safetensors.NewTensorDataFromBytes(upName, "BF16", outShape, upRaw),
	}, nil
}

func dequantizeAndTransposeExpertWeight(name string, blocks, scales *safetensors.TensorData) (*safetensors.TensorData, error) {
	values, shape, err := decodeGPTOSSMXFP4TensorValues(name, blocks, scales)
	if err != nil {
		return nil, err
	}
	if len(shape) != 3 {
		return nil, fmt.Errorf("gpt-oss expert tensor %q shape = %v, want [experts out in]", name, shape)
	}

	experts, outDim, inDim := int(shape[0]), int(shape[1]), int(shape[2])
	raw := make([]byte, experts*inDim*outDim*2)
	for e := 0; e < experts; e++ {
		for out := 0; out < outDim; out++ {
			for in := 0; in < inDim; in++ {
				src := (e*outDim+out)*inDim + in
				dst := (e*inDim+in)*outDim + out
				binary.LittleEndian.PutUint16(raw[dst*2:], uint16(bfloat16.FromFloat32(values[src])))
			}
		}
	}
	return safetensors.NewTensorDataFromBytes(name, "BF16", []int32{int32(experts), int32(inDim), int32(outDim)}, raw), nil
}

func splitGateUpWeightTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}
	if td.Dtype != "BF16" {
		return nil, fmt.Errorf("gpt-oss expert tensor %q dtype = %q, want BF16", td.Name, td.Dtype)
	}
	if len(td.Shape) != 3 {
		return nil, fmt.Errorf("gpt-oss expert tensor %q shape = %v, want [experts out in]", td.Name, td.Shape)
	}
	experts, outDim, inDim := int(td.Shape[0]), int(td.Shape[1]), int(td.Shape[2])
	if outDim%2 != 0 {
		return nil, fmt.Errorf("gpt-oss expert tensor %q output dim = %d, want even gate/up rows", td.Name, outDim)
	}
	mid := outDim / 2

	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		return nil, fmt.Errorf("read gpt-oss expert tensor %q: %w", td.Name, err)
	}
	values, err := DecodeFloatTensor(td.Dtype, raw)
	if err != nil {
		return nil, fmt.Errorf("decode gpt-oss expert tensor %q: %w", td.Name, err)
	}

	gateVals := make([]float32, experts*inDim*mid)
	upVals := make([]float32, experts*inDim*mid)
	for e := 0; e < experts; e++ {
		for row := 0; row < outDim; row++ {
			dstRow := row / 2
			for col := 0; col < inDim; col++ {
				src := (e*outDim+row)*inDim + col
				dst := (e*inDim+col)*mid + dstRow
				if row%2 == 0 {
					gateVals[dst] = values[src]
				} else {
					upVals[dst] = values[src]
				}
			}
		}
	}

	gateRaw, err := EncodeFloatTensor("BF16", gateVals)
	if err != nil {
		return nil, fmt.Errorf("encode gate expert tensor %q: %w", td.Name, err)
	}
	upRaw, err := EncodeFloatTensor("BF16", upVals)
	if err != nil {
		return nil, fmt.Errorf("encode up expert tensor %q: %w", td.Name, err)
	}

	gateName := strings.Replace(td.Name, "gate_up_proj", "gate_proj", 1)
	upName := strings.Replace(td.Name, "gate_up_proj", "up_proj", 1)
	shape := []int32{int32(experts), int32(inDim), int32(mid)}
	return []*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes(gateName, "BF16", shape, gateRaw),
		safetensors.NewTensorDataFromBytes(upName, "BF16", shape, upRaw),
	}, nil
}

func splitGateUpBiasTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}
	if td.Dtype != "BF16" {
		return nil, fmt.Errorf("gpt-oss expert tensor %q dtype = %q, want BF16", td.Name, td.Dtype)
	}
	if len(td.Shape) != 2 {
		return nil, fmt.Errorf("gpt-oss expert tensor %q shape = %v, want [experts out]", td.Name, td.Shape)
	}
	experts, outDim := int(td.Shape[0]), int(td.Shape[1])
	if outDim%2 != 0 {
		return nil, fmt.Errorf("gpt-oss expert tensor %q output dim = %d, want even gate/up rows", td.Name, outDim)
	}
	mid := outDim / 2

	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		return nil, fmt.Errorf("read gpt-oss expert tensor %q: %w", td.Name, err)
	}
	values, err := DecodeFloatTensor(td.Dtype, raw)
	if err != nil {
		return nil, fmt.Errorf("decode gpt-oss expert tensor %q: %w", td.Name, err)
	}

	gateVals := make([]float32, experts*mid)
	upVals := make([]float32, experts*mid)
	for e := 0; e < experts; e++ {
		for row := 0; row < outDim; row++ {
			src := e*outDim + row
			dst := e*mid + row/2
			if row%2 == 0 {
				gateVals[dst] = values[src]
			} else {
				upVals[dst] = values[src]
			}
		}
	}

	gateRaw, err := EncodeFloatTensor("BF16", gateVals)
	if err != nil {
		return nil, fmt.Errorf("encode gate expert bias %q: %w", td.Name, err)
	}
	upRaw, err := EncodeFloatTensor("BF16", upVals)
	if err != nil {
		return nil, fmt.Errorf("encode up expert bias %q: %w", td.Name, err)
	}

	gateName := strings.Replace(td.Name, "gate_up_proj", "gate_proj", 1)
	upName := strings.Replace(td.Name, "gate_up_proj", "up_proj", 1)
	shape := []int32{int32(experts), int32(mid)}
	return []*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes(gateName, "BF16", shape, gateRaw),
		safetensors.NewTensorDataFromBytes(upName, "BF16", shape, upRaw),
	}, nil
}

func transposeExpertWeightTensor(td *safetensors.TensorData) *safetensors.TensorData {
	if td == nil || td.Dtype != "BF16" || len(td.Shape) != 3 {
		return td
	}
	experts, outDim, inDim := int(td.Shape[0]), int(td.Shape[1]), int(td.Shape[2])
	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		return td
	}
	values, err := DecodeFloatTensor(td.Dtype, raw)
	if err != nil {
		return td
	}

	transposed := make([]float32, experts*inDim*outDim)
	for e := 0; e < experts; e++ {
		for out := 0; out < outDim; out++ {
			for in := 0; in < inDim; in++ {
				src := (e*outDim+out)*inDim + in
				dst := (e*inDim+in)*outDim + out
				transposed[dst] = values[src]
			}
		}
	}
	enc, err := EncodeFloatTensor("BF16", transposed)
	if err != nil {
		return td
	}
	return safetensors.NewTensorDataFromBytes(td.Name, "BF16", []int32{int32(experts), int32(inDim), int32(outDim)}, enc)
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
