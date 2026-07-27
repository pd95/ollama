package metadata

import (
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
