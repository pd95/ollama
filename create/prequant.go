package create

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/mlx/quant"
)

// prequantPattern describes how one producer packs an already-quantized weight
// and its scale companions into safetensors files, and how to fuse them into
// the single blob our loader reads. Producers differ only in tensor names and a
// few per-field transforms; expressing them as table rows keeps those
// differences visible and prevents the per-producer drift the old separate code
// paths suffered (for example the global scale being stored as-is by one
// producer and inverted by another).
//
// All suffixes are relative to the base — the source weight name minus its
// weight suffix. The fused blob is always named "<base>.weight", with
// companions "<base>.weight.scale", ".bias", and ".global_scale".
type prequantPattern struct {
	name string

	weightSuffix string   // source suffix identifying the weight (".weight" or ".weight_packed")
	weightDtypes []string // accepted source dtypes; empty accepts any dtype
	scaleDtypes  []string // accepted scale dtypes; empty accepts any dtype
	repackWeight bool     // reinterpret four one-byte weight values as one U32 word

	scaleSuffix    string // required per-block / affine scale companion
	scaleRelabelU8 bool   // relabel an F8_E4M3 scale as U8 for the loader

	biasSuffix string // optional bias / zero-point companion ("" if none)

	globalSuffix     string // optional global-scale companion ("" if none)
	globalReciprocal bool   // producer stores the weight global scale inverted

	ignoreSuffixes []string // activation scales unused by weight-only inference

	forceQuantType   string // override the blob's quant_type metadata
	defaultGroupSize string // set group_size metadata only when the config did not
}

// prequantPatterns is consulted in order; the first whose weight suffix matches
// and whose required scale companion is present wins. MLX and ModelOpt both use
// a ".weight" weight, but their scale companions (".scales" vs ".weight_scale")
// are mutually exclusive, so the order between them does not matter.
var prequantPatterns = []prequantPattern{
	{
		name:         "mlx",
		weightSuffix: ".weight",
		scaleSuffix:  ".scales",
		biasSuffix:   ".biases",
	},
	{
		name:             "compressed-tensors-nvfp4",
		weightSuffix:     ".weight_packed",
		repackWeight:     true,
		scaleSuffix:      ".weight_scale",
		scaleRelabelU8:   true,
		globalSuffix:     ".weight_global_scale",
		globalReciprocal: true,
		ignoreSuffixes:   []string{".input_scale", ".input_global_scale"},
		forceQuantType:   "nvfp4",
		defaultGroupSize: "16",
	},
	{
		name:             "modelopt-mxfp8",
		weightSuffix:     ".weight",
		weightDtypes:     []string{"F8_E4M3", "F8_E4M3FN"},
		repackWeight:     true,
		scaleSuffix:      ".weight_scale",
		scaleDtypes:      []string{"U8"},
		forceQuantType:   "mxfp8",
		defaultGroupSize: "32",
	},
	{
		name:           "modelopt-nvfp4",
		weightSuffix:   ".weight",
		weightDtypes:   []string{"U8", "U32"},
		repackWeight:   true,
		scaleSuffix:    ".weight_scale",
		scaleRelabelU8: true,
		globalSuffix:   ".weight_scale_2",
		ignoreSuffixes: []string{".input_scale", ".input_global_scale"},
		forceQuantType: "nvfp4",
	},
}

// planPrequantized plans an already-quantized source: each weight is fused with
// its scale companions into one blob, companions are not emitted on their own,
// and any remaining tensors (norms, embeddings) pass through at source
// precision.
func planPrequantized(inv Inventory, policy quantizePolicy) ([]BlobSpec, error) {
	if err := validateMLXPrequantCompanions(inv, policy); err != nil {
		return nil, err
	}

	fused := make(map[string]BlobSpec)
	consumed := make(map[string]bool)
	for _, name := range sortedTensorNames(inv) {
		if !tensorIncluded(policy, name) {
			continue
		}
		spec, sources, ok, err := matchPrequant(name, inv)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		fused[name] = spec
		for _, s := range sources {
			consumed[s] = true
		}
	}

	// Model Optimizer keeps Flint's publisher-native per-expert tensor names.
	// As with float imports, the Ollama artifact needs one bank per projection.
	// Stack its packed words and calibrated scales without requantizing them.
	expertGroups := make(map[string][]SourceTensor)
	if slices.Contains(inv.Config.Architectures, "FlintForCausalLM") {
		for _, name := range sortedTensorNames(inv) {
			if group, ok := perExpertGroup(name); ok && strings.HasSuffix(name, ".weight") && tensorIncluded(policy, name) {
				expertGroups[group] = append(expertGroups[group], inv.Tensors[name])
			}
		}
	}
	var specs []BlobSpec
	for _, group := range sortedKeys(expertGroups) {
		experts := expertGroups[group]
		quantized := 0
		for _, weight := range experts {
			if _, ok := fused[weight.Name]; ok {
				quantized++
			}
		}
		if quantized != 0 && quantized != len(experts) {
			return nil, fmt.Errorf("prequantized expert group %s has %d of %d projections with scales", group, quantized, len(experts))
		}
		var groupSpecs []BlobSpec
		var err error
		if quantized == 0 {
			groupSpecs, err = planExpertGroup(group, experts, "", defaultQuantPolicy{})
		} else {
			groupSpecs, err = planPrequantExpertGroup(group, experts, fused)
		}
		if err != nil {
			return nil, err
		}
		specs = append(specs, groupSpecs...)
		for _, weight := range experts {
			consumed[weight.Name] = true
		}
	}

	for _, name := range sortedTensorNames(inv) {
		if !tensorIncluded(policy, name) || consumed[name] {
			continue
		}
		if spec, ok := fused[name]; ok {
			specs = append(specs, spec)
			continue
		}
		t := inv.Tensors[name]
		specs = append(specs, BlobSpec{Name: name, Tensors: []TensorSpec{{Name: name, Sources: []SourceTensor{t}}}})
	}
	return specs, nil
}

func validateMLXPrequantCompanions(inv Inventory, policy quantizePolicy) error {
	for _, name := range sortedTensorNames(inv) {
		if !tensorIncluded(policy, name) {
			continue
		}
		if base, ok := strings.CutSuffix(name, ".scales"); ok {
			if !inv.Has(base + ".weight") {
				return fmt.Errorf("MLX quantization scale %s is missing weight companion %s", name, base+".weight")
			}
			continue
		}
		if !strings.HasSuffix(name, ".weight") || !strings.EqualFold(inv.Tensors[name].Dtype, "U32") {
			continue
		}
		metadata, explicit, err := inv.Config.TensorQuantMetadata(name)
		if err != nil {
			return err
		}
		if !explicit {
			continue
		}
		_, _, mode := quant.Params(metadata["quant_type"])
		if mode == "affine" && !inv.Has(strings.TrimSuffix(name, ".weight")+".scales") {
			return fmt.Errorf("MLX affine tensor %s is missing required scale companion %s", name, strings.TrimSuffix(name, ".weight")+".scales")
		}
	}
	return nil
}

func planPrequantExpertGroup(group string, weights []SourceTensor, fused map[string]BlobSpec) ([]BlobSpec, error) {
	// The float importer's expert planner validates contiguous indices. Stacking
	// copies the source bytes, so the output dtype and shape also perform the
	// header-only U8 weight repack and FP8 scale relabeling.
	specs, err := planExpertGroup(group, weights, "", defaultQuantPolicy{})
	if err != nil {
		return nil, err
	}
	bank := &specs[0]
	weightsByProjection := bank.Tensors
	bank.Tensors = nil
	for _, packed := range weightsByProjection {
		count := len(packed.Sources)
		scaleSources := make([]SourceTensor, count)
		globalSources := make([]SourceTensor, count)
		for index, weight := range packed.Sources {
			part, ok := fused[weight.Name]
			if !ok || part.Metadata["quant_type"] != "nvfp4" || len(part.Tensors) != 3 ||
				len(weight.Shape) != 2 {
				return nil, fmt.Errorf("prequantized expert %s has unsupported NVFP4 layout", weight.Name)
			}
			wordShape := weight.Shape
			normalized := part.Tensors[0]
			switch weight.Dtype {
			case "U32":
				if normalized.Transform != TransformNone {
					return nil, fmt.Errorf("prequantized expert %s has unsupported NVFP4 weight transform", weight.Name)
				}
			case "U8":
				if normalized.Transform != TransformRepackFP4 || normalized.OutDtype != "U32" ||
					len(normalized.OutShape) != 2 || weight.Shape[1]%4 != 0 {
					return nil, fmt.Errorf("prequantized expert %s has unsupported NVFP4 weight transform", weight.Name)
				}
				wordShape = normalized.OutShape
			default:
				return nil, fmt.Errorf("prequantized expert %s has unsupported NVFP4 weight dtype %s", weight.Name, weight.Dtype)
			}
			if index == 0 {
				packed.OutDtype = "U32"
				packed.OutShape = append([]int32{int32(count)}, wordShape...)
			} else if !slices.Equal(wordShape, packed.OutShape[1:]) {
				return nil, fmt.Errorf("prequantized expert %s differs from the NVFP4 bank layout", weight.Name)
			}
			scale, global := part.Tensors[1], part.Tensors[2]
			if scale.Name != weight.Name+".scale" || global.Name != weight.Name+".global_scale" ||
				len(scale.Sources) != 1 || len(global.Sources) != 1 ||
				global.Transform != TransformScalarF32 {
				return nil, fmt.Errorf("prequantized expert %s has unsupported NVFP4 companions", weight.Name)
			}
			s, g := scale.Sources[0], global.Sources[0]
			scaleIsU8 := s.Dtype == "U8" && scale.Transform == TransformNone
			scaleIsFP8 := isE4M3Dtype(s.Dtype) && scale.Transform == TransformRelabelU8 && scale.OutDtype == "U8"
			if (!scaleIsU8 && !scaleIsFP8) || len(s.Shape) != 2 || g.Dtype != "F32" || !slices.Equal(g.Shape, []int32{1}) ||
				(index > 0 && (s.Dtype != scaleSources[0].Dtype || !slices.Equal(s.Shape, scaleSources[0].Shape) || !slices.Equal(g.Shape, globalSources[0].Shape))) {
				return nil, fmt.Errorf("prequantized expert %s differs from the NVFP4 bank layout", weight.Name)
			}
			if bank.Metadata != nil && !maps.Equal(bank.Metadata, part.Metadata) {
				return nil, fmt.Errorf("prequantized expert %s has different quantization metadata", weight.Name)
			}
			bank.Metadata = part.Metadata
			scaleSources[index], globalSources[index] = s, g
		}
		bank.Tensors = append(bank.Tensors, packed,
			TensorSpec{Name: packed.Name + ".scale", Sources: scaleSources, Transform: TransformStackExperts, OutDtype: "U8", OutShape: append([]int32{int32(count)}, scaleSources[0].Shape...)},
			TensorSpec{Name: packed.Name + ".global_scale", Sources: globalSources, Transform: TransformStackExperts, OutDtype: "F32", OutShape: []int32{int32(count)}},
		)
	}
	return specs, nil
}

// matchPrequant returns the fused blob for a weight tensor if it matches a
// prequantized producer, along with the source names it consumes. It returns
// ok=false when name is not a prequantized weight (a companion or a plain
// tensor).
func matchPrequant(name string, inv Inventory) (BlobSpec, []string, bool, error) {
	for _, p := range prequantPatterns {
		base, ok := strings.CutSuffix(name, p.weightSuffix)
		if !ok {
			continue
		}
		scaleSrc := base + p.scaleSuffix
		if !inv.Has(scaleSrc) {
			continue
		}
		weight := inv.Tensors[name]
		scale := inv.Tensors[scaleSrc]
		if !matchesDtype(weight.Dtype, p.weightDtypes) || !matchesDtype(scale.Dtype, p.scaleDtypes) {
			continue
		}

		outWeight := base + ".weight"
		var tensors []TensorSpec
		var consumed []string

		weightTensor := TensorSpec{Name: outWeight, Sources: []SourceTensor{weight}}
		if p.repackWeight && len(weight.Shape) == 2 && (strings.EqualFold(weight.Dtype, "U8") || isE4M3Dtype(weight.Dtype)) {
			weightTensor.Transform = TransformRepackFP4
			if isE4M3Dtype(weight.Dtype) {
				weightTensor.Transform = TransformRepackFP8
			}
			weightTensor.OutDtype = "U32"
			weightTensor.OutShape = []int32{weight.Shape[0], weight.Shape[1] / 4}
		}
		tensors = append(tensors, weightTensor)

		scaleTensor := TensorSpec{Name: outWeight + ".scale", Sources: []SourceTensor{scale}}
		if p.scaleRelabelU8 && isE4M3Dtype(scale.Dtype) {
			scaleTensor.Transform = TransformRelabelU8
			scaleTensor.OutDtype = "U8"
		}
		tensors = append(tensors, scaleTensor)
		consumed = append(consumed, scaleSrc)

		if p.biasSuffix != "" {
			if biasSrc := base + p.biasSuffix; inv.Has(biasSrc) {
				tensors = append(tensors, TensorSpec{Name: outWeight + ".bias", Sources: []SourceTensor{inv.Tensors[biasSrc]}})
				consumed = append(consumed, biasSrc)
			}
		}

		if p.globalSuffix != "" {
			if gSrc := base + p.globalSuffix; inv.Has(gSrc) {
				globalSource := inv.Tensors[gSrc]
				global := TensorSpec{Name: outWeight + ".global_scale", Sources: []SourceTensor{globalSource}, Transform: f32ScaleTransform(globalSource)}
				if p.globalReciprocal {
					global.Transform = TransformReciprocalF32
				}
				tensors = append(tensors, global)
				consumed = append(consumed, gSrc)
			}
		}

		for _, suffix := range p.ignoreSuffixes {
			if source := base + suffix; inv.Has(source) {
				consumed = append(consumed, source)
			}
		}

		metadata, err := prequantMetadata(inv, p, outWeight, weight, scale)
		if err != nil {
			return BlobSpec{}, nil, false, err
		}
		if p.name == "mlx" {
			if err := validateMLXAffineLayout(inv, base, weight, scale, metadata); err != nil {
				return BlobSpec{}, nil, false, err
			}
		}

		return BlobSpec{Name: outWeight, Tensors: tensors, Metadata: metadata}, consumed, true, nil
	}
	return BlobSpec{}, nil, false, nil
}

func f32ScaleTransform(t SourceTensor) Transform {
	for _, dim := range t.Shape {
		if dim != 1 {
			return TransformF32
		}
	}
	return TransformScalarF32
}

func matchesDtype(dtype string, allowed []string) bool {
	if len(allowed) == 0 {
		return true
	}
	for _, want := range allowed {
		if strings.EqualFold(dtype, want) {
			return true
		}
	}
	return false
}

// prequantMetadata builds the fused blob's metadata: the source config's quant
// metadata, with the pattern's quant_type override and group_size default
// applied. Returns nil when there is nothing to record.
func prequantMetadata(inv Inventory, p prequantPattern, weightName string, weight, scale SourceTensor) (map[string]string, error) {
	md := make(map[string]string)
	if p.name == "mlx" {
		resolved, explicit, err := inv.Config.TensorQuantMetadata(weightName)
		if err != nil {
			return nil, err
		}
		if explicit {
			for k, v := range resolved {
				md[k] = v
			}
		} else if quantType, groupSize, ok := inferLegacyMLXAffine(weight, scale); ok {
			md["quant_type"] = quantType
			md["group_size"] = strconv.Itoa(groupSize)
		} else {
			return nil, fmt.Errorf("MLX quantized tensor %s requires explicit quantization metadata; its packed layout is not an unambiguous legacy int4/int8 layout", weightName)
		}
	} else {
		for k, v := range inv.Config.QuantMetadata() {
			md[k] = v
		}
	}
	// Flint exports an embedding width separately from its default affine
	// quantization. Resolve it before validating the packed tensor layout.
	if p.name == "mlx" && weightName == "model.embed_tokens.weight" &&
		slices.Contains(inv.Config.Architectures, "FlintForCausalLM") && md["quant_type"] == "int4" {
		var config struct {
			MLXExport struct {
				EmbeddingBits int `json:"embedding_bits"`
			} `json:"mlx_export"`
		}
		if err := json.Unmarshal(inv.RawConfig, &config); err != nil {
			return nil, fmt.Errorf("Flint embedding quantization: %w", err)
		}
		bits := config.MLXExport.EmbeddingBits
		if bits != 4 && bits != 8 {
			return nil, fmt.Errorf("Flint embedding quantization: unsupported mlx_export.embedding_bits %d", bits)
		}
		md["quant_type"] = "int" + strconv.Itoa(bits)
	}
	if p.forceQuantType != "" {
		md["quant_type"] = p.forceQuantType
	}
	if p.defaultGroupSize != "" {
		if _, ok := md["group_size"]; !ok {
			md["group_size"] = p.defaultGroupSize
		}
	}
	if len(md) == 0 {
		return nil, nil
	}
	return md, nil
}

func inferLegacyMLXAffine(weight, scale SourceTensor) (string, int, bool) {
	if !strings.EqualFold(weight.Dtype, "U32") || len(weight.Shape) < 2 || len(weight.Shape) != len(scale.Shape) {
		return "", 0, false
	}
	if !slices.Equal(weight.Shape[:len(weight.Shape)-1], scale.Shape[:len(scale.Shape)-1]) {
		return "", 0, false
	}
	packed := int64(weight.Shape[len(weight.Shape)-1])
	scaleCols := int64(scale.Shape[len(scale.Shape)-1])
	if packed <= 0 || scaleCols <= 0 {
		return "", 0, false
	}
	if packed%scaleCols == 0 {
		switch packed / scaleCols {
		case 4:
			return "int4", 32, true
		case 16:
			return "int8", 64, true
		}
	}
	return "", 0, false
}

func validateMLXAffineLayout(inv Inventory, base string, weight, scale SourceTensor, metadata map[string]string) error {
	quantType := metadata["quant_type"]
	groupSize, err := strconv.Atoi(metadata["group_size"])
	if err != nil {
		return fmt.Errorf("MLX quantized tensor %s has invalid group_size %q", weight.Name, metadata["group_size"])
	}
	_, bits, mode := quant.Params(quantType)
	if !quant.Importable(quantType) {
		return fmt.Errorf("MLX quantized tensor %s has unsupported quantization %q", weight.Name, quantType)
	}
	// This strict companion/layout validation is specific to MLX affine
	// weights. Existing MLX mxfp/nvfp checkpoints use the same .scales naming
	// pattern but intentionally have no affine bias companion.
	if mode != "affine" {
		return nil
	}
	switch groupSize {
	case 32, 64, 128:
	default:
		return fmt.Errorf("MLX quantized tensor %s has unsupported affine group_size %d", weight.Name, groupSize)
	}
	if !strings.EqualFold(weight.Dtype, "U32") {
		return fmt.Errorf("MLX quantized tensor %s dtype = %q, want U32", weight.Name, weight.Dtype)
	}
	biasName := base + ".biases"
	bias, ok := inv.Tensors[biasName]
	if !ok {
		return fmt.Errorf("MLX affine tensor %s is missing required bias companion %s", weight.Name, biasName)
	}
	if len(weight.Shape) < 2 || len(scale.Shape) != len(weight.Shape) || len(bias.Shape) != len(scale.Shape) {
		return fmt.Errorf("MLX affine tensor %s has incompatible companion ranks: weight=%v scales=%v biases=%v", weight.Name, weight.Shape, scale.Shape, bias.Shape)
	}
	last := len(weight.Shape) - 1
	if !slices.Equal(weight.Shape[:last], scale.Shape[:last]) || !slices.Equal(scale.Shape, bias.Shape) {
		return fmt.Errorf("MLX affine tensor %s has incompatible companion shapes: weight=%v scales=%v biases=%v", weight.Name, weight.Shape, scale.Shape, bias.Shape)
	}
	packedColumns := int64(weight.Shape[last])
	if packedColumns <= 0 || packedColumns > (1<<63-1)/32 {
		return fmt.Errorf("MLX affine tensor %s has invalid packed column count %d", weight.Name, packedColumns)
	}
	expanded := packedColumns * 32
	if expanded%int64(bits) != 0 {
		return fmt.Errorf("MLX affine tensor %s packed columns %d do not expand exactly at %d bits", weight.Name, packedColumns, bits)
	}
	logicalColumns := expanded / int64(bits)
	if logicalColumns%int64(groupSize) != 0 || int64(scale.Shape[last]) != logicalColumns/int64(groupSize) {
		return fmt.Errorf("MLX affine tensor %s layout conflicts with quantization metadata: packed=%d bits=%d group_size=%d scales=%v", weight.Name, packedColumns, bits, groupSize, scale.Shape)
	}
	return nil
}
