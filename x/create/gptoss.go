package create

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
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

func newGPTOSSImportTransform(rawConfig json.RawMessage) (quantizePolicy, error) {
	t := &gptossImportTransform{
		perTensorQuant: make(map[string]gptossPerTensorQuant),
	}

	defaultQuant, perTensor := parseGPTOSSPerTensorQuant(rawConfig)
	t.defaultQuant = defaultQuant
	t.perTensorQuant = perTensor

	return t, nil
}

func parseGPTOSSPerTensorQuant(rawConfig json.RawMessage) (gptossPerTensorQuant, map[string]gptossPerTensorQuant) {
	defaultQ := gptossPerTensorQuant{}
	perTensor := make(map[string]gptossPerTensorQuant)

	var raw struct {
		Quantization json.RawMessage `json:"quantization"`
	}
	if err := json.Unmarshal(rawConfig, &raw); err != nil || raw.Quantization == nil {
		return defaultQ, perTensor
	}

	var entries map[string]json.RawMessage
	if err := json.Unmarshal(raw.Quantization, &entries); err != nil {
		return defaultQ, perTensor
	}

	type quantEntry struct {
		Bits        int    `json:"bits"`
		GroupSize   int    `json:"group_size"`
		Mode        string `json:"mode"`
		QuantMethod string `json:"quant_method"`
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
	if defaultQ.mode == "" {
		if v, ok := entries["quant_method"]; ok {
			json.Unmarshal(v, &defaultQ.mode)
		}
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
			if mode == "" {
				mode = entry.QuantMethod
			}
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

func planGPTOSS(inv Inventory, class Classification, policy quantizePolicy) ([]BlobSpec, error) {
	t, ok := policy.(*gptossImportTransform)
	if !ok {
		return nil, fmt.Errorf("gpt-oss planner requires gptossImportTransform, got %T", policy)
	}

	consumed := make(map[string]bool)
	expertGroups := make(map[string][]TensorSpec)
	expertMetadata := make(map[string]map[string]string)
	var specs []BlobSpec

	for _, name := range sortedTensorNames(inv) {
		if consumed[name] {
			continue
		}
		if t.isNativeCompanion(inv, name) {
			continue
		}
		if strings.HasSuffix(name, "_blocks") {
			group, tensors, metadata, sources, ok, err := t.planNativeExpertTensor(inv, name)
			if err != nil {
				return nil, err
			}
			if ok {
				expertGroups[group] = append(expertGroups[group], tensors...)
				if len(metadata) > 0 {
					if expertMetadata[group] == nil {
						expertMetadata[group] = make(map[string]string)
					}
					for k, v := range metadata {
						expertMetadata[group][k] = v
					}
				}
				for _, source := range sources {
					consumed[source] = true
				}
				continue
			}
		}

		if spec, sources, ok := t.planNativeQuantizedDense(inv, name); ok {
			specs = append(specs, spec)
			for _, source := range sources {
				consumed[source] = true
			}
			continue
		}

		outName := t.canonicalTensorName(name)
		source := inv.Tensors[name]
		q := ""
		if class.Kind == SourceFloat && class.Quantize != "" {
			q = policy.quantizationType(outName, source.Shape, class.Quantize)
		}
		specs = append(specs, BlobSpec{
			Name:    outName,
			Tensors: []TensorSpec{{Name: outName, Sources: []SourceTensor{source}, Quantize: q}},
		})
	}

	for _, group := range sortedKeys(expertGroups) {
		specs = append(specs, BlobSpec{
			Name:     group,
			Tensors:  expertGroups[group],
			Metadata: expertMetadata[group],
		})
	}

	return specs, nil
}

func (t *gptossImportTransform) isNativeCompanion(inv Inventory, name string) bool {
	switch {
	case strings.HasSuffix(name, "_scales"):
		return inv.Has(strings.TrimSuffix(name, "_scales") + "_blocks")
	case strings.HasSuffix(name, "_bias"):
		return inv.Has(strings.TrimSuffix(name, "_bias") + "_blocks")
	case strings.HasSuffix(name, ".scales"):
		return inv.Has(strings.TrimSuffix(name, ".scales") + ".weight")
	case strings.HasSuffix(name, ".biases"):
		return inv.Has(strings.TrimSuffix(name, ".biases") + ".weight")
	default:
		return false
	}
}

func (t *gptossImportTransform) planNativeQuantizedDense(inv Inventory, weightName string) (BlobSpec, []string, bool) {
	if !strings.HasSuffix(weightName, ".weight") {
		return BlobSpec{}, nil, false
	}
	scaleName := strings.TrimSuffix(weightName, ".weight") + ".scales"
	if !inv.Has(scaleName) {
		return BlobSpec{}, nil, false
	}

	outWeight := t.canonicalTensorName(weightName)
	if outWeight == weightName {
		return BlobSpec{}, nil, false
	}

	weight := inv.Tensors[weightName]
	scale := inv.Tensors[scaleName]
	tensors := []TensorSpec{
		{Name: outWeight, Sources: []SourceTensor{weight}},
		{Name: outWeight + ".scale", Sources: []SourceTensor{scale}},
	}
	sources := []string{weightName, scaleName}
	if biasName := strings.TrimSuffix(weightName, ".weight") + ".biases"; inv.Has(biasName) {
		tensors = append(tensors, TensorSpec{Name: outWeight + ".bias", Sources: []SourceTensor{inv.Tensors[biasName]}})
		sources = append(sources, biasName)
	}

	return BlobSpec{
		Name:     outWeight,
		Tensors:  tensors,
		Metadata: t.prequantizedMetadata(weightName, t.defaultMetadata()),
	}, sources, true
}

func (t *gptossImportTransform) planNativeExpertTensor(inv Inventory, blocksName string) (string, []TensorSpec, map[string]string, []string, bool, error) {
	scalesName := strings.TrimSuffix(blocksName, "_blocks") + "_scales"
	if !inv.Has(scalesName) {
		return "", nil, nil, nil, false, nil
	}

	outName := t.canonicalTensorName(blocksName)
	group, ok := strings.CutSuffix(outName, ".gate_up_proj.weight")
	if ok {
		group = group + ""
	} else if group, ok = strings.CutSuffix(outName, ".down_proj.weight"); ok {
		group = group + ""
	}
	if !ok {
		return "", nil, nil, nil, false, nil
	}
	group = strings.TrimSuffix(group, ".gate_up_proj")
	group = strings.TrimSuffix(group, ".down_proj")

	blocks := inv.Tensors[blocksName]
	scales := inv.Tensors[scalesName]
	sources := []string{blocksName, scalesName}
	// The native _blocks/_scales layout is the authoritative direct-expert
	// contract. Classification accepts it even when config.json is missing or
	// stale, so never let global or per-tensor affine metadata relabel the
	// preserved MXFP4 bytes into a runtime-incompatible format.
	metadata := map[string]string{"quant_type": "mxfp4", "group_size": "32"}
	if strings.Contains(outName, ".gate_up_proj.weight") {
		gateWeight := strings.Replace(outName, "gate_up_proj", "gate_proj", 1)
		upWeight := strings.Replace(outName, "gate_up_proj", "up_proj", 1)
		tensors := []TensorSpec{
			{Name: gateWeight, Sources: []SourceTensor{blocks, scales}, Transform: TransformGPTOSSGateUpWeight},
			{Name: gateWeight + ".scale", Sources: []SourceTensor{blocks, scales}, Transform: TransformGPTOSSGateUpScale},
			{Name: upWeight, Sources: []SourceTensor{blocks, scales}, Transform: TransformGPTOSSUpWeight},
			{Name: upWeight + ".scale", Sources: []SourceTensor{blocks, scales}, Transform: TransformGPTOSSUpScale},
		}
		if biasName := strings.TrimSuffix(blocksName, "_blocks") + "_bias"; inv.Has(biasName) {
			bias := inv.Tensors[biasName]
			gateBias := strings.Replace(outName, "gate_up_proj.weight", "gate_proj.bias", 1)
			upBias := strings.Replace(outName, "gate_up_proj.weight", "up_proj.bias", 1)
			tensors = append(tensors,
				TensorSpec{Name: gateBias, Sources: []SourceTensor{bias}, Transform: TransformGPTOSSGateUpBias},
				TensorSpec{Name: upBias, Sources: []SourceTensor{bias}, Transform: TransformGPTOSSUpBias},
			)
			sources = append(sources, biasName)
		}
		return group, tensors, metadata, sources, true, nil
	}

	tensors := []TensorSpec{
		{Name: outName, Sources: []SourceTensor{blocks, scales}, Transform: TransformGPTOSSPackedExpertWeight},
		{Name: outName + ".scale", Sources: []SourceTensor{blocks, scales}, Transform: TransformGPTOSSPackedExpertScale},
	}
	if biasName := strings.TrimSuffix(blocksName, "_blocks") + "_bias"; inv.Has(biasName) {
		tensors = append(tensors, TensorSpec{Name: strings.Replace(outName, ".weight", ".bias", 1), Sources: []SourceTensor{inv.Tensors[biasName]}})
		sources = append(sources, biasName)
	}
	return group, tensors, metadata, sources, true, nil
}

func (t *gptossImportTransform) defaultMetadata() map[string]string {
	qt := sourceQuantType(t.defaultQuant.mode, t.defaultQuant.bits)
	if qt == "" {
		qt = "mxfp4"
	}
	metadata := map[string]string{"quant_type": qt}
	if t.defaultQuant.groupSize > 0 {
		metadata["group_size"] = strconv.Itoa(t.defaultQuant.groupSize)
	} else if qt == "mxfp4" {
		metadata["group_size"] = "32"
	}
	return metadata
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
		return "embedding.weight.scale"
	case "model.embed_tokens.biases":
		return "embedding.weight.bias"
	case "model.norm.weight":
		return "output_norm.weight"
	case "lm_head.weight":
		return "output.weight"
	case "lm_head.scales":
		return "output.weight.scale"
	case "lm_head.biases":
		return "output.weight.bias"
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
		return prefix + "q_proj.weight.scale"
	case "self_attn.q_proj.biases":
		return prefix + "q_proj.weight.bias"
	case "self_attn.k_proj.weight":
		return prefix + "k_proj.weight"
	case "self_attn.k_proj.bias":
		return prefix + "k_proj.bias"
	case "self_attn.k_proj.scales":
		return prefix + "k_proj.weight.scale"
	case "self_attn.k_proj.biases":
		return prefix + "k_proj.weight.bias"
	case "self_attn.v_proj.weight":
		return prefix + "v_proj.weight"
	case "self_attn.v_proj.bias":
		return prefix + "v_proj.bias"
	case "self_attn.v_proj.scales":
		return prefix + "v_proj.weight.scale"
	case "self_attn.v_proj.biases":
		return prefix + "v_proj.weight.bias"
	case "self_attn.o_proj.weight":
		return prefix + "attn_out.weight"
	case "self_attn.o_proj.bias":
		return prefix + "attn_out.bias"
	case "self_attn.o_proj.scales":
		return prefix + "attn_out.weight.scale"
	case "self_attn.o_proj.biases":
		return prefix + "attn_out.weight.bias"
	case "self_attn.sinks":
		return prefix + "attn_sinks"
	case "post_attention_layernorm.weight":
		return prefix + "ffn_norm.weight"
	case "mlp.router.weight":
		return prefix + "router.weight"
	case "mlp.router.bias":
		return prefix + "router.bias"
	case "mlp.router.scales":
		return prefix + "router.weight.scale"
	case "mlp.router.biases":
		return prefix + "router.weight.bias"
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
