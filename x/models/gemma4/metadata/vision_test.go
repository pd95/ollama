package metadata

import (
	"maps"
	"strconv"
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

func unifiedVisionTensorNames() []string {
	return []string{
		"model.vision_embedder.patch_ln1.weight",
		"model.vision_embedder.patch_ln1.bias",
		"model.vision_embedder.patch_dense.weight",
		"model.vision_embedder.patch_dense.bias",
		"model.vision_embedder.patch_ln2.weight",
		"model.vision_embedder.patch_ln2.bias",
		"model.vision_embedder.pos_embedding",
		"model.vision_embedder.pos_norm.weight",
		"model.vision_embedder.pos_norm.bias",
		"model.embed_vision.embedding_projection.weight",
	}
}

func TestValidateUnifiedVisionTensors(t *testing.T) {
	cfg := ConfigFile{ModelType: "gemma4_unified", VisionConfig: &VisionConfig{ModelType: "gemma4_unified_vision"}}
	names := unifiedVisionTensorNames()
	if err := ValidateVisionTensors(cfg, names); err != nil {
		t.Fatalf("ValidateVisionTensors() error = %v", err)
	}

	partial := append([]string(nil), names[:2]...)
	partial = append(partial, names[3:]...)
	if err := ValidateVisionTensors(cfg, partial); err == nil {
		t.Fatal("missing unified patch projection: error = nil")
	}

	partial = append([]string(nil), names...)
	partial = append(partial[:3], partial[4:]...)
	if err := ValidateVisionTensors(cfg, partial); err == nil {
		t.Fatal("missing unified patch projection bias: error = nil")
	}
}

func TestValidatePackedVisionSourceRequiresCompanions(t *testing.T) {
	cfg := ConfigFile{ModelType: "gemma4_unified", VisionConfig: &VisionConfig{ModelType: "gemma4_unified_vision"}}
	names := unifiedVisionTensorNames()
	for _, path := range []string{"model.vision_embedder.patch_dense", "model.embed_vision.embedding_projection"} {
		for i, name := range names {
			if name == path+".weight" {
				names[i] = path + ".weight_packed"
				names = append(names, path+".weight_scale", path+".weight_global_scale")
				break
			}
		}
	}
	if err := ValidateVisionSourceTensors(cfg, names); err != nil {
		t.Fatalf("ValidateVisionSourceTensors() error = %v", err)
	}
	if err := ValidateVisionTensors(cfg, names); err == nil {
		t.Fatal("ValidateVisionTensors() accepted unnormalized packed weights")
	}

	withoutGlobal := append([]string(nil), names...)
	withoutGlobal = withoutGlobal[:len(withoutGlobal)-1]
	if err := ValidateVisionSourceTensors(cfg, withoutGlobal); err == nil {
		t.Fatal("packed source without global scale: error = nil")
	}
}

func TestValidateUnifiedVisionSourceDescriptors(t *testing.T) {
	cfg := ConfigFile{
		ModelType:  "gemma4_unified",
		TextConfig: TextConfig{HiddenSize: 2},
		VisionConfig: &VisionConfig{
			ModelType: "gemma4_unified_vision", MMEmbedDim: 2, MMPosembSize: 4, ModelPatchSize: 1,
		},
	}
	shapes := map[string][]int32{
		"model.vision_embedder.patch_ln1.weight":         {3},
		"model.vision_embedder.patch_ln1.bias":           {3},
		"model.vision_embedder.patch_dense.weight":       {2, 3},
		"model.vision_embedder.patch_dense.bias":         {2},
		"model.vision_embedder.patch_ln2.weight":         {2},
		"model.vision_embedder.patch_ln2.bias":           {2},
		"model.vision_embedder.pos_embedding":            {4, 2, 2},
		"model.vision_embedder.pos_norm.weight":          {2},
		"model.vision_embedder.pos_norm.bias":            {2},
		"model.embed_vision.embedding_projection.weight": {2, 2},
	}
	tensors := make(map[string]TensorDescriptor, len(shapes))
	for name, shape := range shapes {
		tensors[name] = TensorDescriptor{Dtype: "BF16", Shape: shape}
	}
	if err := ValidateVisionSourceInventory(cfg, tensors); err != nil {
		t.Fatalf("ValidateVisionSourceInventory() error = %v", err)
	}

	badShape := maps.Clone(tensors)
	badShape["model.vision_embedder.pos_embedding"] = TensorDescriptor{Dtype: "BF16", Shape: []int32{5, 2, 2}}
	if err := ValidateVisionSourceInventory(cfg, badShape); err == nil {
		t.Fatal("invalid position embedding shape: error = nil")
	}
	badDtype := maps.Clone(tensors)
	badDtype["model.vision_embedder.patch_dense.bias"] = TensorDescriptor{Dtype: "U8", Shape: []int32{2}}
	if err := ValidateVisionSourceInventory(cfg, badDtype); err == nil {
		t.Fatal("invalid patch projection bias dtype: error = nil")
	}

	linearNamespace := maps.Clone(tensors)
	delete(linearNamespace, "model.vision_embedder.patch_dense.weight")
	linearNamespace["model.vision_embedder.patch_dense.linear.weight"] = TensorDescriptor{Dtype: "BF16", Shape: []int32{2, 3}}
	if err := ValidateVisionSourceInventory(cfg, linearNamespace); err == nil {
		t.Fatal("unsupported linear-namespace patch projection: error = nil")
	}
}
