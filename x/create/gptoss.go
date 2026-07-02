package create

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ollama/ollama/x/safetensors"
)

type gptossPerTensorQuant struct {
	mode      string
	bits      int
	groupSize int
}

type gptossImportTransform struct {
	pendingBlocks  map[string]*safetensors.TensorData
	pendingScales  map[string]*safetensors.TensorData
	defaultQuant   gptossPerTensorQuant
	perTensorQuant map[string]gptossPerTensorQuant
}

func validateGPTOSSDequant() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("GPTOSS_VALIDATE_DEQUANT"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func newGPTOSSImportTransform(modelDir string, cfg sourceModelConfig) (tensorImportTransform, error) {
	t := &gptossImportTransform{
		pendingBlocks:  make(map[string]*safetensors.TensorData),
		pendingScales:  make(map[string]*safetensors.TensorData),
		perTensorQuant: make(map[string]gptossPerTensorQuant),
	}

	defaultQuant, perTensor := parseGPTOSSPerTensorQuant(modelDir)
	t.defaultQuant = defaultQuant
	t.perTensorQuant = perTensor

	return t, nil
}

func parseGPTOSSPerTensorQuant(modelDir string) (gptossPerTensorQuant, map[string]gptossPerTensorQuant) {
	defaultQ := gptossPerTensorQuant{}
	perTensor := make(map[string]gptossPerTensorQuant)

	data, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return defaultQ, perTensor
	}

	var raw struct {
		Quantization json.RawMessage `json:"quantization"`
	}
	if err := json.Unmarshal(data, &raw); err != nil || raw.Quantization == nil {
		return defaultQ, perTensor
	}

	var entries map[string]json.RawMessage
	if err := json.Unmarshal(raw.Quantization, &entries); err != nil {
		return defaultQ, perTensor
	}

	type quantEntry struct {
		Bits      int    `json:"bits"`
		GroupSize int    `json:"group_size"`
		Mode      string `json:"mode"`
	}

	if v, ok := entries["bits"]; ok {
		json.Unmarshal(v, &defaultQ.bits)
	}
	if v, ok := entries["group_size"]; ok {
		json.Unmarshal(v, &defaultQ.groupSize)
	}
	if v, ok := entries["mode"]; ok {
		json.Unmarshal(v, &defaultQ.mode)
	}

	for key, val := range entries {
		if key == "bits" || key == "group_size" || key == "mode" || key == "quant_method" {
			continue
		}
		var entry quantEntry
		if err := json.Unmarshal(val, &entry); err != nil {
			continue
		}
		if entry.Bits > 0 {
			mode := entry.Mode
			// Infer mode when not specified: if the tensor has biases
			// (affine pattern) or uses group_size=64 (affine default),
			// it's affine quantization. The MLX checkpoint omits mode
			// for some tensors like the router.
			if mode == "" && entry.GroupSize == 64 {
				mode = "affine"
			}
			q := gptossPerTensorQuant{
				mode:      mode,
				bits:      entry.Bits,
				groupSize: entry.GroupSize,
			}
			perTensor[key] = q
		}
	}

	return defaultQ, perTensor
}

func (t *gptossImportTransform) skipTensor(string) bool { return false }

func isGptossRouterWeight(name string) bool {
	return strings.HasSuffix(name, ".router.weight")
}

func (t *gptossImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	// MoE router weights choose the top-k expert set. Quantization noise can
	// flip expert selection, causing downstream activations to diverge sharply.
	// The tensor is small, so leave it in source precision.
	if isGptossRouterWeight(name) {
		return ""
	}

	quantNorm := normalizeQuantType(quantize)
	if quantNorm == "" {
		return ""
	}
	if strings.Contains(name, ".experts.") && strings.HasSuffix(name, ".weight") {
		return ""
	}
	return GetTensorQuantization(name, shape, quantize)
}

func (t *gptossImportTransform) prequantizedMetadata(sourceName string, global map[string]string) map[string]string {
	prefix := strings.TrimSuffix(sourceName, ".weight")
	if prefix == sourceName {
		return global
	}

	q, ok := t.perTensorQuant[prefix]
	if !ok {
		return global
	}

	qt := sourceQuantType(q.mode, q.bits)
	if qt == "" {
		return global
	}

	override := make(map[string]string, len(global)+2)
	for k, v := range global {
		override[k] = v
	}
	override["quant_type"] = qt
	if q.groupSize > 0 {
		override["group_size"] = strconv.Itoa(q.groupSize)
	}
	return override
}

func (t *gptossImportTransform) packedGroupComplete(groupName string, tensors []PackedTensorInput) bool {
	if !strings.HasSuffix(groupName, ".experts") {
		return false
	}

	seen := make(map[string]bool, len(tensors))
	for _, tensor := range tensors {
		seen[tensor.Name] = true
	}
	for _, suffix := range []string{
		".gate_proj.weight",
		".gate_proj.weight.scale",
		".up_proj.weight",
		".up_proj.weight.scale",
		".down_proj.weight",
		".down_proj.weight.scale",
		".gate_proj.bias",
		".up_proj.bias",
		".down_proj.bias",
	} {
		if !seen[groupName+suffix] {
			return false
		}
	}
	return true
}

func isGPTOSSPackedExpertWeight(name, dtype string) bool {
	return dtype == "U32" &&
		strings.Contains(name, ".experts.") &&
		strings.HasSuffix(name, ".weight")
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
		return preserveAndSplitGateUpTensor(name, blocks, scales)
	default:
		out, err := preservePackedExpertProjection(name, blocks, scales)
		if err != nil {
			return nil, err
		}
		return out, nil
	}
}

func validateGPTOSSPackedMXFP4Inputs(name string, blocks, scales *safetensors.TensorData) error {
	if blocks == nil || scales == nil {
		return fmt.Errorf("gpt-oss expert tensor %q requires blocks and scales", name)
	}
	if blocks.Dtype != "U8" {
		return fmt.Errorf("gpt-oss expert blocks %q dtype = %q, want U8", blocks.Name, blocks.Dtype)
	}
	if scales.Dtype != "U8" {
		return fmt.Errorf("gpt-oss expert scales %q dtype = %q, want U8", scales.Name, scales.Dtype)
	}
	if len(blocks.Shape) != 4 {
		return fmt.Errorf("gpt-oss expert blocks %q shape = %v, want [experts out groups 16]", blocks.Name, blocks.Shape)
	}
	if len(scales.Shape) != 3 {
		return fmt.Errorf("gpt-oss expert scales %q shape = %v, want [experts out groups]", scales.Name, scales.Shape)
	}
	if blocks.Shape[0] != scales.Shape[0] || blocks.Shape[1] != scales.Shape[1] || blocks.Shape[2] != scales.Shape[2] {
		return fmt.Errorf("gpt-oss expert tensor %q shape mismatch: blocks=%v scales=%v", name, blocks.Shape, scales.Shape)
	}
	if blocks.Shape[3] != 16 {
		return fmt.Errorf("gpt-oss expert blocks %q trailing shape = %v, want [... 16]", blocks.Name, blocks.Shape)
	}
	return nil
}

func preservePackedExpertProjection(name string, blocks, scales *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if err := validateGPTOSSPackedMXFP4Inputs(name, blocks, scales); err != nil {
		return nil, err
	}

	sourceBlockBytes, err := io.ReadAll(blocks.Reader())
	if err != nil {
		return nil, fmt.Errorf("read gpt-oss expert blocks %q: %w", blocks.Name, err)
	}
	sourceScaleBytes, err := io.ReadAll(scales.Reader())
	if err != nil {
		return nil, fmt.Errorf("read gpt-oss expert scales %q: %w", scales.Name, err)
	}
	blockBytes, err := preserveGPTOSSMXFP4Blocks(blocks.Name, sourceBlockBytes, int(blocks.Shape[2]))
	if err != nil {
		return nil, err
	}
	scaleBytes := convertGPTOSSMXFP4Scales(sourceScaleBytes)

	blockShape := []int32{blocks.Shape[0], blocks.Shape[1], blocks.Shape[2] * 4}
	scaleShape := []int32{scales.Shape[0], scales.Shape[1], scales.Shape[2]}
	return []*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes(name, "U32", blockShape, blockBytes),
		safetensors.NewTensorDataFromBytes(name+".scale", "U8", scaleShape, scaleBytes),
	}, nil
}

func preserveAndSplitGateUpTensor(name string, blocks, scales *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if err := validateGPTOSSPackedMXFP4Inputs(name, blocks, scales); err != nil {
		return nil, err
	}

	experts, outDim, groups := int(blocks.Shape[0]), int(blocks.Shape[1]), int(blocks.Shape[2])
	if outDim%2 != 0 {
		return nil, fmt.Errorf("gpt-oss expert tensor %q output dim = %d, want even gate/up rows", name, outDim)
	}
	mid := outDim / 2

	sourceBlockBytes, err := io.ReadAll(blocks.Reader())
	if err != nil {
		return nil, fmt.Errorf("read gpt-oss expert blocks %q: %w", blocks.Name, err)
	}
	sourceScaleBytes, err := io.ReadAll(scales.Reader())
	if err != nil {
		return nil, fmt.Errorf("read gpt-oss expert scales %q: %w", scales.Name, err)
	}
	blockBytes, err := preserveGPTOSSMXFP4Blocks(blocks.Name, sourceBlockBytes, groups)
	if err != nil {
		return nil, err
	}
	scaleBytes := convertGPTOSSMXFP4Scales(sourceScaleBytes)

	rowBlockBytes := groups * 16
	rowScaleBytes := groups
	wantBlockBytes := experts * outDim * rowBlockBytes
	wantScaleBytes := experts * outDim * rowScaleBytes
	if len(blockBytes) != wantBlockBytes {
		return nil, fmt.Errorf("gpt-oss expert blocks %q byte length = %d, want %d", blocks.Name, len(blockBytes), wantBlockBytes)
	}
	if len(scaleBytes) != wantScaleBytes {
		return nil, fmt.Errorf("gpt-oss expert scales %q byte length = %d, want %d", scales.Name, len(scaleBytes), wantScaleBytes)
	}

	gateBlocks := make([]byte, experts*mid*rowBlockBytes)
	upBlocks := make([]byte, experts*mid*rowBlockBytes)
	gateScales := make([]byte, experts*mid*rowScaleBytes)
	upScales := make([]byte, experts*mid*rowScaleBytes)
	for e := range experts {
		for row := range outDim {
			dstRow := row / 2
			srcBlock := (e*outDim + row) * rowBlockBytes
			dstBlock := (e*mid + dstRow) * rowBlockBytes
			srcScale := (e*outDim + row) * rowScaleBytes
			dstScale := (e*mid + dstRow) * rowScaleBytes
			if row%2 == 0 {
				copy(gateBlocks[dstBlock:dstBlock+rowBlockBytes], blockBytes[srcBlock:srcBlock+rowBlockBytes])
				copy(gateScales[dstScale:dstScale+rowScaleBytes], scaleBytes[srcScale:srcScale+rowScaleBytes])
			} else {
				copy(upBlocks[dstBlock:dstBlock+rowBlockBytes], blockBytes[srcBlock:srcBlock+rowBlockBytes])
				copy(upScales[dstScale:dstScale+rowScaleBytes], scaleBytes[srcScale:srcScale+rowScaleBytes])
			}
		}
	}

	gateName := strings.Replace(name, "gate_up_proj", "gate_proj", 1)
	upName := strings.Replace(name, "gate_up_proj", "up_proj", 1)
	blockShape := []int32{int32(experts), int32(mid), int32(groups * 4)}
	scaleShape := []int32{int32(experts), int32(mid), int32(groups)}
	return []*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes(gateName, "U32", blockShape, gateBlocks),
		safetensors.NewTensorDataFromBytes(gateName+".scale", "U8", scaleShape, gateScales),
		safetensors.NewTensorDataFromBytes(upName, "U32", blockShape, upBlocks),
		safetensors.NewTensorDataFromBytes(upName+".scale", "U8", scaleShape, upScales),
	}, nil
}

func preserveGPTOSSMXFP4Blocks(name string, source []byte, groups int) ([]byte, error) {
	if groups <= 0 {
		return nil, fmt.Errorf("gpt-oss expert blocks %q group count must be positive, got %d", name, groups)
	}
	if len(source)%(groups*16) != 0 {
		return nil, fmt.Errorf("gpt-oss expert blocks %q byte length = %d, want multiple of row bytes %d", name, len(source), groups*16)
	}

	out := make([]byte, len(source))
	copy(out, source)
	return out, nil
}

func convertGPTOSSMXFP4Scales(source []byte) []byte {
	out := make([]byte, len(source))
	copy(out, source)
	return out
}

var gptossMXFP4Values = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

func decodeGPTOSSMXFP4Scale(scale byte) float32 {
	return math.Float32frombits(uint32(scale) << 23)
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

	values := make([]float32, groupCount*32)
	for i := range groupCount {
		src := blockBytes[i*16 : (i+1)*16]

		scale := decodeGPTOSSMXFP4Scale(scaleBytes[i])
		base := i * 32
		for j, packed := range src {
			values[base+2*j] = gptossMXFP4Values[packed&0x0F] * scale
			values[base+2*j+1] = gptossMXFP4Values[packed>>4] * scale
		}
	}

	if validateGPTOSSDequant() {
		for i, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, nil, fmt.Errorf("gpt-oss expert tensor %q dequantized invalid value at %d", name, i)
			}
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
	for e := range experts {
		for row := range outDim {
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

func (t *gptossImportTransform) canonicalTensorName(name string) string {
	switch name {
	case "model.embed_tokens.weight":
		return "embedding.weight"
	case "model.embed_tokens.scales":
		return "embedding.weight_scale"
	case "model.embed_tokens.biases":
		return "embedding.weight_qbias"
	case "model.norm.weight":
		return "output_norm.weight"
	case "lm_head.weight":
		return "output.weight"
	case "lm_head.scales":
		return "output.weight_scale"
	case "lm_head.biases":
		return "output.weight_qbias"
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
	case "self_attn.q_proj.scales":
		return prefix + "q_proj.weight_scale"
	case "self_attn.q_proj.biases":
		return prefix + "q_proj.weight_qbias"
	case "self_attn.k_proj.weight":
		return prefix + "k_proj.weight"
	case "self_attn.k_proj.bias":
		return prefix + "k_proj.bias"
	case "self_attn.k_proj.scales":
		return prefix + "k_proj.weight_scale"
	case "self_attn.k_proj.biases":
		return prefix + "k_proj.weight_qbias"
	case "self_attn.v_proj.weight":
		return prefix + "v_proj.weight"
	case "self_attn.v_proj.bias":
		return prefix + "v_proj.bias"
	case "self_attn.v_proj.scales":
		return prefix + "v_proj.weight_scale"
	case "self_attn.v_proj.biases":
		return prefix + "v_proj.weight_qbias"
	case "self_attn.o_proj.weight":
		return prefix + "attn_out.weight"
	case "self_attn.o_proj.bias":
		return prefix + "attn_out.bias"
	case "self_attn.o_proj.scales":
		return prefix + "attn_out.weight_scale"
	case "self_attn.o_proj.biases":
		return prefix + "attn_out.weight_qbias"
	case "self_attn.sinks":
		return prefix + "attn_sinks"
	case "post_attention_layernorm.weight":
		return prefix + "ffn_norm.weight"
	case "mlp.router.weight":
		return prefix + "router.weight"
	case "mlp.router.bias":
		return prefix + "router.bias"
	case "mlp.router.scales":
		return prefix + "router.weight_scale"
	case "mlp.router.biases":
		return prefix + "router.weight_qbias"
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
	case "mlp.experts.gate_proj.weight",
		"mlp.experts.gate_proj.scales",
		"mlp.experts.gate_proj.biases":
		return prefix + "experts.gate_proj.weight"
	case "mlp.experts.gate_proj.bias":
		return prefix + "experts.gate_proj.bias"
	case "mlp.experts.up_proj.weight",
		"mlp.experts.up_proj.scales",
		"mlp.experts.up_proj.biases":
		return prefix + "experts.up_proj.weight"
	case "mlp.experts.up_proj.bias":
		return prefix + "experts.up_proj.bias"
	case "mlp.experts.down_proj.weight",
		"mlp.experts.down_proj.scales",
		"mlp.experts.down_proj.biases":
		return prefix + "experts.down_proj.weight"
	case "mlp.experts.down_proj.bias":
		return prefix + "experts.down_proj.bias"
	default:
		return name
	}
}
