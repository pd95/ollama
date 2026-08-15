package metadata

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func completeVisionTensorNames(layers int, standardize bool) []string {
	names := []string{
		"model.vision_tower.patch_embedder.input_proj.weight",
		"model.vision_tower.patch_embedder.position_embedding_table",
		"model.embed_vision.embedding_projection.weight",
	}
	for i := range layers {
		layer := "model.vision_tower.encoder.layers." + strconv.Itoa(i)
		for _, projection := range []string{
			".self_attn.q_proj.linear.weight", ".self_attn.k_proj.linear.weight",
			".self_attn.v_proj.linear.weight", ".self_attn.o_proj.linear.weight",
			".mlp.gate_proj.linear.weight", ".mlp.up_proj.linear.weight", ".mlp.down_proj.linear.weight",
		} {
			names = append(names, layer+projection)
		}
		for _, norm := range []string{
			".self_attn.q_norm.weight", ".self_attn.k_norm.weight",
			".input_layernorm.weight", ".post_attention_layernorm.weight",
			".pre_feedforward_layernorm.weight", ".post_feedforward_layernorm.weight",
		} {
			names = append(names, layer+norm)
		}
	}
	if standardize {
		names = append(names, "model.vision_tower.std_bias", "model.vision_tower.std_scale")
	}
	return names
}

func TestValidateVisionTensors(t *testing.T) {
	cfg := ConfigFile{VisionConfig: &VisionConfig{NumHiddenLayers: 2, Standardize: true}}
	names := completeVisionTensorNames(2, true)
	if err := ValidateVisionTensors(cfg, names); err != nil {
		t.Fatalf("ValidateVisionTensors() error = %v", err)
	}

	for _, missing := range []string{
		"model.vision_tower.patch_embedder.position_embedding_table",
		"model.vision_tower.encoder.layers.1.self_attn.q_norm.weight",
		"model.vision_tower.encoder.layers.1.mlp.down_proj.linear.weight",
		"model.vision_tower.std_scale",
		"model.embed_vision.embedding_projection.weight",
	} {
		partial := append([]string(nil), names...)
		for i, name := range partial {
			if name == missing {
				partial = append(partial[:i], partial[i+1:]...)
				break
			}
		}
		if err := ValidateVisionTensors(cfg, partial); err == nil {
			t.Fatalf("missing %s: error = %v", missing, err)
		}
	}
}

func TestValidateVisionTensorsBoundsLayerCount(t *testing.T) {
	if err := ValidateVisionTensors(
		ConfigFile{VisionConfig: &VisionConfig{NumHiddenLayers: MaxVisionLayers}},
		completeVisionTensorNames(MaxVisionLayers, false),
	); err != nil {
		t.Fatalf("exact layer limit error = %v", err)
	}

	for _, layers := range []int{-1, MaxVisionLayers + 1} {
		err := ValidateVisionTensors(
			ConfigFile{VisionConfig: &VisionConfig{NumHiddenLayers: layers}},
			[]string{
				"model.vision_tower.patch_embedder.input_proj.weight",
				"model.vision_tower.patch_embedder.position_embedding_table",
			},
		)
		if err == nil || !strings.Contains(err.Error(), "num_hidden_layers") {
			t.Fatalf("layer count %d error = %v", layers, err)
		}
	}
}

func executableVisionConfig() ConfigFile {
	return ConfigFile{
		TextConfig: TextConfig{HiddenSize: 24},
		VisionConfig: &VisionConfig{
			HiddenSize: 16, IntermediateSize: 32, NumHiddenLayers: 1,
			NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: 16,
			RMSNormEps: 1e-6, DefaultOutputLength: 1, PatchSize: 4,
			PositionEmbeddingSize: 16, PoolingKernelSize: 1,
		},
	}
}

func executableVisionDescriptors() map[string]TensorDescriptor {
	return visionDescriptorsForGeometry(16, 32, 24, 4, 16, 16)
}

func visionDescriptorsForGeometry(hidden, intermediate, textHidden, patch, positions, headDim int32) map[string]TensorDescriptor {
	d := map[string]TensorDescriptor{
		"model.vision_tower.patch_embedder.input_proj.weight":        {Dtype: "F32", Shape: []int32{hidden, 3 * patch * patch}},
		"model.vision_tower.patch_embedder.position_embedding_table": {Dtype: "F32", Shape: []int32{2, positions, hidden}},
		"model.embed_vision.embedding_projection.weight":             {Dtype: "F32", Shape: []int32{textHidden, hidden}},
	}
	layer := "model.vision_tower.encoder.layers.0"
	for _, suffix := range []string{".self_attn.q_proj.linear.weight", ".self_attn.k_proj.linear.weight", ".self_attn.v_proj.linear.weight", ".self_attn.o_proj.linear.weight"} {
		d[layer+suffix] = TensorDescriptor{Dtype: "F32", Shape: []int32{hidden, hidden}}
	}
	for _, suffix := range []string{".mlp.gate_proj.linear.weight", ".mlp.up_proj.linear.weight"} {
		d[layer+suffix] = TensorDescriptor{Dtype: "F32", Shape: []int32{intermediate, hidden}}
	}
	d[layer+".mlp.down_proj.linear.weight"] = TensorDescriptor{Dtype: "F32", Shape: []int32{hidden, intermediate}}
	for _, suffix := range []string{".self_attn.q_norm.weight", ".self_attn.k_norm.weight"} {
		d[layer+suffix] = TensorDescriptor{Dtype: "F32", Shape: []int32{headDim}}
	}
	for _, suffix := range []string{".input_layernorm.weight", ".post_attention_layernorm.weight", ".pre_feedforward_layernorm.weight", ".post_feedforward_layernorm.weight"} {
		d[layer+suffix] = TensorDescriptor{Dtype: "F32", Shape: []int32{hidden}}
	}
	return d
}

func cloneDescriptors(in map[string]TensorDescriptor) map[string]TensorDescriptor {
	out := make(map[string]TensorDescriptor, len(in))
	for name, descriptor := range in {
		descriptor.Shape = slices.Clone(descriptor.Shape)
		out[name] = descriptor
	}
	return out
}

func TestVisionInventoryProducerAndNormalizationMatrix(t *testing.T) {
	const target = "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear"
	dense := executableVisionDescriptors()
	cfg := executableVisionConfig()
	if err := ValidateVisionSourceInventory(cfg, dense); err != nil {
		t.Fatalf("dense source error = %v", err)
	}
	if err := ValidateVisionInstalledInventory(cfg, dense); err != nil {
		t.Fatalf("dense installed error = %v", err)
	}

	tests := []struct {
		name     string
		config   func(*ConfigFile)
		add      map[string]TensorDescriptor
		required string
	}{
		{
			name: "compressed tensors NVFP4",
			add: map[string]TensorDescriptor{
				target + ".weight_packed":       {Dtype: "U8", Shape: []int32{16, 8}},
				target + ".weight_scale":        {Dtype: "F8_E4M3", Shape: []int32{16, 1}},
				target + ".weight_global_scale": {Dtype: "F32", Shape: nil},
			},
			required: target + ".weight_global_scale",
		},
		{
			name: "MLX packed",
			config: func(cfg *ConfigFile) {
				cfg.QuantizationConfig = Quantization{Bits: 4, GroupSize: 16, Mode: "affine"}
			},
			add: map[string]TensorDescriptor{
				target + ".weight": {Dtype: "U32", Shape: []int32{16, 2}},
				target + ".scales": {Dtype: "F16", Shape: []int32{16, 1}},
			},
			required: target + ".scales",
		},
		{
			name: "ModelOpt NVFP4",
			add: map[string]TensorDescriptor{
				target + ".weight":         {Dtype: "U8", Shape: []int32{16, 8}},
				target + ".weight_scale":   {Dtype: "F8_E4M3", Shape: []int32{16, 1}},
				target + ".weight_scale_2": {Dtype: "F32", Shape: []int32{1}},
			},
			required: target + ".weight_scale",
		},
	}

	for _, tt := range []struct {
		name   string
		weight string
		scale  string
		global string
	}{
		{name: "compressed scale", weight: "weight_packed", scale: "weight_scale", global: "weight_global_scale"},
		{name: "ModelOpt scale", weight: "weight", scale: "weight_scale", global: "weight_scale_2"},
	} {
		t.Run("wrong dtype "+tt.name, func(t *testing.T) {
			for _, field := range []string{"scale", "global"} {
				inventory := cloneDescriptors(dense)
				delete(inventory, target+".weight")
				inventory[target+"."+tt.weight] = TensorDescriptor{Dtype: "U8", Shape: []int32{16, 8}}
				inventory[target+"."+tt.scale] = TensorDescriptor{Dtype: "F8_E4M3", Shape: []int32{16, 1}}
				inventory[target+"."+tt.global] = TensorDescriptor{Dtype: "F32", Shape: nil}
				bad := inventory[target+"."+map[string]string{"scale": tt.scale, "global": tt.global}[field]]
				bad.Dtype = "F16"
				inventory[target+"."+map[string]string{"scale": tt.scale, "global": tt.global}[field]] = bad
				if err := ValidateVisionSourceInventory(cfg, inventory); err == nil {
					t.Fatalf("%s %s with F16 dtype accepted", tt.name, field)
				}
			}
		})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inventory := cloneDescriptors(dense)
			delete(inventory, target+".weight")
			for name, descriptor := range tt.add {
				inventory[name] = descriptor
			}
			localCfg := cfg
			if tt.config != nil {
				tt.config(&localCfg)
			}
			if err := ValidateVisionSourceInventory(localCfg, inventory); err != nil {
				t.Fatalf("valid packed source error = %v", err)
			}
			if err := ValidateVisionInstalledInventory(localCfg, inventory); err == nil {
				t.Fatal("source-only packed form accepted as installed")
			}
			partial := cloneDescriptors(inventory)
			delete(partial, tt.required)
			if err := ValidateVisionSourceInventory(localCfg, partial); err == nil {
				t.Fatalf("missing companion %s accepted", tt.required)
			}
		})
	}

	normalized := cloneDescriptors(dense)
	normalized[target+".weight"] = TensorDescriptor{Dtype: "U32", Shape: []int32{16, 2}}
	normalized[target+".weight.scale"] = TensorDescriptor{Dtype: "U8", Shape: []int32{16, 1}}
	normalized[target+".weight.global_scale"] = TensorDescriptor{Dtype: "F32", Shape: nil}
	if err := ValidateVisionInstalledInventory(cfg, normalized); err != nil {
		t.Fatalf("normalized installed error = %v", err)
	}
	runtime := cloneDescriptors(normalized)
	delete(runtime, target+".weight.scale")
	runtime[target+".weight_scale"] = TensorDescriptor{Dtype: "U8", Shape: []int32{16, 1}}
	if err := ValidateVisionRuntimeInventory(cfg, runtime); err != nil {
		t.Fatalf("normalized runtime error = %v", err)
	}
	for _, missing := range []string{target + ".weight.scale"} {
		partial := cloneDescriptors(normalized)
		delete(partial, missing)
		if err := ValidateVisionInstalledInventory(cfg, partial); err == nil {
			t.Fatalf("normalized installed inventory missing %s accepted", missing)
		}
	}
	cross := cloneDescriptors(normalized)
	cross[target+".scales"] = TensorDescriptor{Dtype: "F16", Shape: []int32{16, 1}}
	if err := ValidateVisionInstalledInventory(cfg, cross); err == nil {
		t.Fatal("cross-producer companion accepted")
	}

	affineCfg := cfg
	affineCfg.QuantizationConfig = Quantization{Bits: 4, GroupSize: 16, Mode: "affine"}
	affine := cloneDescriptors(dense)
	affine[target+".weight"] = TensorDescriptor{Dtype: "U32", Shape: []int32{16, 2}}
	affine[target+".weight.scale"] = TensorDescriptor{Dtype: "F16", Shape: []int32{16, 1}}
	affine[target+".weight.bias"] = TensorDescriptor{Dtype: "F16", Shape: []int32{16, 1}}
	if err := ValidateVisionInstalledInventory(affineCfg, affine); err != nil {
		t.Fatalf("normalized affine installed error = %v", err)
	}
	affineRuntime := cloneDescriptors(affine)
	delete(affineRuntime, target+".weight.scale")
	delete(affineRuntime, target+".weight.bias")
	affineRuntime[target+".weight_scale"] = TensorDescriptor{Dtype: "F16", Shape: []int32{16, 1}}
	affineRuntime[target+".weight_qbias"] = TensorDescriptor{Dtype: "F16", Shape: []int32{16, 1}}
	if err := ValidateVisionRuntimeInventory(affineCfg, affineRuntime); err != nil {
		t.Fatalf("normalized affine runtime error = %v", err)
	}
}

func TestVisionInventoryReleasedUnequalHeadGeometry(t *testing.T) {
	cfg := ConfigFile{
		TextConfig: TextConfig{HiddenSize: 2560},
		VisionConfig: &VisionConfig{
			HiddenSize: 768, IntermediateSize: 3072, NumHiddenLayers: 1,
			NumAttentionHeads: 12, NumKeyValueHeads: 12, HeadDim: 64,
			RMSNormEps: 1e-6, DefaultOutputLength: 280, PatchSize: 16,
			PositionEmbeddingSize: 10240, PoolingKernelSize: 3,
		},
	}
	descriptors := visionDescriptorsForGeometry(768, 3072, 2560, 16, 10240, 64)
	for name, validate := range map[string]func(ConfigFile, map[string]TensorDescriptor) error{
		"source":    ValidateVisionSourceInventory,
		"installed": ValidateVisionInstalledInventory,
		"runtime":   ValidateVisionRuntimeInventory,
	} {
		t.Run(name, func(t *testing.T) {
			if err := validate(cfg, descriptors); err != nil {
				t.Fatalf("released-compatible unequal hidden/head geometry: %v", err)
			}
			bad := cloneDescriptors(descriptors)
			qnorm := "model.vision_tower.encoder.layers.0.self_attn.q_norm.weight"
			bad[qnorm] = TensorDescriptor{Dtype: "F32", Shape: []int32{768}}
			if err := validate(cfg, bad); err == nil {
				t.Fatal("hidden-sized q_norm accepted")
			}
		})
	}
}

func TestVisionInventoryRejectsMissingOrZeroTextWidth(t *testing.T) {
	for _, hidden := range []int{0, -1, MaxVisionHidden + 1} {
		cfg := executableVisionConfig()
		cfg.TextConfig.HiddenSize = hidden
		for name, validate := range map[string]func(ConfigFile, map[string]TensorDescriptor) error{
			"source":    ValidateVisionSourceInventory,
			"installed": ValidateVisionInstalledInventory,
			"runtime":   ValidateVisionRuntimeInventory,
		} {
			t.Run(fmt.Sprintf("%s/%d", name, hidden), func(t *testing.T) {
				if err := validate(cfg, executableVisionDescriptors()); err == nil || !strings.Contains(err.Error(), "text hidden_size") {
					t.Fatalf("text hidden_size %d error = %v", hidden, err)
				}
			})
		}
	}
}

func TestVisionInventoryDescriptorAndConfigMatrix(t *testing.T) {
	cfg := executableVisionConfig()
	valid := executableVisionDescriptors()
	for _, mutate := range []func(map[string]TensorDescriptor){
		func(d map[string]TensorDescriptor) {
			x := d["model.vision_tower.patch_embedder.input_proj.weight"]
			x.Shape = []int32{16, 47}
			d["model.vision_tower.patch_embedder.input_proj.weight"] = x
		},
		func(d map[string]TensorDescriptor) {
			x := d["model.vision_tower.patch_embedder.position_embedding_table"]
			x.Dtype = "U8"
			d["model.vision_tower.patch_embedder.position_embedding_table"] = x
		},
		func(d map[string]TensorDescriptor) {
			x := d["model.vision_tower.encoder.layers.0.input_layernorm.weight"]
			x.Shape = []int32{1, 16}
			d["model.vision_tower.encoder.layers.0.input_layernorm.weight"] = x
		},
	} {
		malformed := cloneDescriptors(valid)
		mutate(malformed)
		if err := ValidateVisionSourceInventory(cfg, malformed); err == nil {
			t.Fatal("malformed complete-name inventory accepted")
		}
	}

	clippedCfg := cfg
	clippedVision := *cfg.VisionConfig
	clippedVision.UseClippedLinears = true
	clippedCfg.VisionConfig = &clippedVision
	partialClip := cloneDescriptors(valid)
	partialClip["model.vision_tower.encoder.layers.0.self_attn.q_proj.input_min"] = TensorDescriptor{Dtype: "F32", Shape: nil}
	if err := ValidateVisionInstalledInventory(clippedCfg, partialClip); err == nil {
		t.Fatal("partial clipping bounds accepted")
	}

	invalid := []ConfigFile{}
	for _, change := range []func(*VisionConfig){
		func(v *VisionConfig) { v.HiddenSize = -1 },
		func(v *VisionConfig) { v.HiddenSize = MaxVisionHidden + 1 },
		func(v *VisionConfig) { v.NumAttentionHeads = 3 },
		func(v *VisionConfig) { v.RMSNormEps = math.NaN() },
		func(v *VisionConfig) { v.RopeParameters.RopeTheta = math.Inf(1) },
		func(v *VisionConfig) { v.RMSNormEps = math.SmallestNonzeroFloat64 },
		func(v *VisionConfig) { v.RopeParameters.RopeTheta = math.MaxFloat64 },
		func(v *VisionConfig) { v.DefaultOutputLength = MaxVisionSoftTokens + 1 },
	} {
		bad := cfg
		vision := *cfg.VisionConfig
		change(&vision)
		bad.VisionConfig = &vision
		invalid = append(invalid, bad)
	}
	for _, bad := range invalid {
		if _, err := ProjectVisionArchitecture(bad); err == nil {
			t.Fatalf("invalid architecture accepted: %+v", bad.VisionConfig)
		}
	}
}

func unifiedVisionConfig() ConfigFile {
	return ConfigFile{
		Architectures: []string{"Gemma4UnifiedForConditionalGeneration"},
		ModelType:     "gemma4_unified", TextConfig: TextConfig{HiddenSize: 5},
		VisionConfig: &VisionConfig{ModelType: "gemma4_unified_vision", MMEmbedDim: 3, MMPosembSize: 4, ModelPatchSize: 2, NumSoftTokens: 2, PatchSize: 1, PoolingKernelSize: 2},
	}
}

func unifiedVisionDescriptors() map[string]TensorDescriptor {
	return map[string]TensorDescriptor{
		"model.vision_embedder.patch_ln1.weight":         {Dtype: "F32", Shape: []int32{12}},
		"model.vision_embedder.patch_ln1.bias":           {Dtype: "F32", Shape: []int32{12}},
		"model.vision_embedder.patch_dense.weight":       {Dtype: "F32", Shape: []int32{3, 12}},
		"model.vision_embedder.patch_dense.bias":         {Dtype: "F32", Shape: []int32{3}},
		"model.vision_embedder.patch_ln2.weight":         {Dtype: "F32", Shape: []int32{3}},
		"model.vision_embedder.patch_ln2.bias":           {Dtype: "F32", Shape: []int32{3}},
		"model.vision_embedder.pos_embedding":            {Dtype: "F32", Shape: []int32{4, 2, 3}},
		"model.vision_embedder.pos_norm.weight":          {Dtype: "F32", Shape: []int32{3}},
		"model.vision_embedder.pos_norm.bias":            {Dtype: "F32", Shape: []int32{3}},
		"model.embed_vision.embedding_projection.weight": {Dtype: "F32", Shape: []int32{5, 3}},
	}
}

func TestUnifiedVisionInventoryDispatchAndBoundaries(t *testing.T) {
	cfg, valid := unifiedVisionConfig(), unifiedVisionDescriptors()
	for name, validate := range map[string]func(ConfigFile, map[string]TensorDescriptor) error{
		"source": ValidateVisionSourceInventory, "installed": ValidateVisionInstalledInventory, "runtime": ValidateVisionRuntimeInventory,
	} {
		t.Run(name, func(t *testing.T) {
			if err := validate(cfg, valid); err != nil {
				t.Fatalf("complete unified inventory: %v", err)
			}
			for _, tensor := range []string{"model.vision_embedder.patch_ln1.bias", "model.vision_embedder.patch_dense.weight", "model.embed_vision.embedding_projection.weight"} {
				partial := cloneDescriptors(valid)
				delete(partial, tensor)
				if err := validate(cfg, partial); err == nil {
					t.Fatalf("missing %s accepted", tensor)
				}
			}
			bad := cloneDescriptors(valid)
			d := bad["model.vision_embedder.pos_embedding"]
			d.Shape = []int32{4, 3, 3}
			bad["model.vision_embedder.pos_embedding"] = d
			if err := validate(cfg, bad); err == nil {
				t.Fatal("invalid position shape accepted")
			}
		})
	}

	for _, mutate := range []func(*ConfigFile){
		func(c *ConfigFile) { c.TextConfig.HiddenSize = 0 },
		func(c *ConfigFile) { c.VisionConfig.MMEmbedDim = MaxVisionHidden + 1 },
		func(c *ConfigFile) { c.VisionConfig.MMPosembSize = 1 },
		func(c *ConfigFile) { c.VisionConfig.ModelPatchSize = 3 },
		func(c *ConfigFile) { c.VisionConfig.NumSoftTokens = MaxVisionSoftTokens + 1 },
	} {
		bad := cfg
		v := *cfg.VisionConfig
		bad.VisionConfig = &v
		mutate(&bad)
		if _, err := ProjectVisionArchitecture(bad); err == nil {
			t.Fatalf("invalid unified architecture accepted: %+v", bad.VisionConfig)
		}
	}

	boundary := cfg
	boundaryVision := *cfg.VisionConfig
	boundaryVision.ModelPatchSize = 8
	boundaryVision.PatchSize = 8
	boundaryVision.PoolingKernelSize = 1
	boundaryVision.NumSoftTokens = MaxVisionSoftTokens
	boundaryVision.MMPosembSize = MaxVisionSoftTokens
	boundary.VisionConfig = &boundaryVision
	if _, err := ProjectVisionArchitecture(boundary); err != nil {
		t.Fatalf("exact unified patch allocation boundary: %v", err)
	}
	over := boundary
	overVision := boundaryVision
	overVision.ModelPatchSize, overVision.PatchSize = 9, 9
	over.VisionConfig = &overVision
	if _, err := ProjectVisionArchitecture(over); err == nil || !strings.Contains(err.Error(), "patch allocation") {
		t.Fatalf("over unified patch allocation error = %v", err)
	}

	near := cfg
	near.ModelType = "gemma4_unified_extra"
	near.Architectures = []string{"NotGemma4UnifiedForConditionalGeneration"}
	v := *cfg.VisionConfig
	v.ModelType = "gemma4_unified_vision_extra"
	near.VisionConfig = &v
	a, err := ProjectVisionArchitecture(near)
	if err != nil {
		t.Fatal(err)
	}
	if a.Unified {
		t.Fatal("near-match unified identifiers dispatched as unified")
	}

	packedCfg := cfg
	packedVision := *cfg.VisionConfig
	packedVision.ModelPatchSize, packedVision.PatchSize = 4, 2
	packedCfg.VisionConfig = &packedVision
	packed := cloneDescriptors(valid)
	packed["model.vision_embedder.patch_ln1.weight"] = TensorDescriptor{Dtype: "F32", Shape: []int32{48}}
	packed["model.vision_embedder.patch_ln1.bias"] = TensorDescriptor{Dtype: "F32", Shape: []int32{48}}
	const dense = "model.vision_embedder.patch_dense"
	delete(packed, dense+".weight")
	packed[dense+".weight_packed"] = TensorDescriptor{Dtype: "U8", Shape: []int32{3, 24}}
	packed[dense+".weight_scale"] = TensorDescriptor{Dtype: "F8_E4M3", Shape: []int32{3, 3}}
	packed[dense+".weight_global_scale"] = TensorDescriptor{Dtype: "F32"}
	if err := ValidateVisionSourceInventory(packedCfg, packed); err != nil {
		t.Fatalf("packed unified source: %v", err)
	}
	if err := ValidateVisionInstalledInventory(packedCfg, packed); err == nil {
		t.Fatal("source-only packed unified inventory accepted as installed")
	}
	delete(packed, dense+".weight_global_scale")
	if err := ValidateVisionSourceInventory(packedCfg, packed); err == nil {
		t.Fatal("incomplete packed unified inventory accepted")
	}
}
