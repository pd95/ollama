// Package metadata contains Gemma 4 model metadata checks that do not depend
// on MLX, allowing import and server capability paths to share one definition.
package metadata

import (
	"fmt"
	"slices"
)

const defaultVisionLayers = 16

type ConfigFile struct {
	VisionConfig *VisionConfig `json:"vision_config"`
}

type VisionConfig struct {
	NumHiddenLayers int  `json:"num_hidden_layers"`
	Standardize     bool `json:"standardize"`
}

// ValidateVisionTensors verifies the weights required by the native Gemma 4
// MLX vision loader. It returns a concrete missing tensor path on failure.
func ValidateVisionTensors(cfg ConfigFile, names []string) error {
	if cfg.VisionConfig == nil {
		return fmt.Errorf("missing vision_config")
	}

	prefix, ok := visionPrefix(names)
	if !ok {
		return fmt.Errorf("missing vision_tower.patch_embedder.input_proj weight")
	}
	has := func(name string) bool { return slices.Contains(names, name) }
	require := func(name string) error {
		if !has(name) {
			return fmt.Errorf("missing %s", name)
		}
		return nil
	}
	requireLinear := func(path string) error {
		for _, suffix := range []string{".weight", ".weight_packed", ".linear.weight", ".linear.weight_packed"} {
			if has(path + suffix) {
				return nil
			}
		}
		return fmt.Errorf("missing %s weight", path)
	}

	if err := require(prefix + "vision_tower.patch_embedder.position_embedding_table"); err != nil {
		return err
	}
	layers := cfg.VisionConfig.NumHiddenLayers
	if layers == 0 {
		layers = defaultVisionLayers
	}
	if layers < 0 {
		return fmt.Errorf("invalid vision layer count %d", layers)
	}
	for i := range layers {
		layer := fmt.Sprintf("%svision_tower.encoder.layers.%d", prefix, i)
		for _, projection := range []string{
			".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj",
			".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj",
		} {
			if err := requireLinear(layer + projection); err != nil {
				return err
			}
		}
		for _, norm := range []string{
			".self_attn.q_norm.weight", ".self_attn.k_norm.weight",
			".input_layernorm.weight", ".post_attention_layernorm.weight",
			".pre_feedforward_layernorm.weight", ".post_feedforward_layernorm.weight",
		} {
			if err := require(layer + norm); err != nil {
				return err
			}
		}
	}
	if cfg.VisionConfig.Standardize {
		if err := require(prefix + "vision_tower.std_bias"); err != nil {
			return err
		}
		if err := require(prefix + "vision_tower.std_scale"); err != nil {
			return err
		}
	}

	for _, projector := range []string{"embed_vision.embedding_projection", "model.embed_vision.embedding_projection"} {
		if err := requireLinear(projector); err == nil {
			return nil
		}
	}
	return fmt.Errorf("missing embed_vision.embedding_projection weight")
}

func visionPrefix(names []string) (string, bool) {
	for _, prefix := range []string{"", "model."} {
		path := prefix + "vision_tower.patch_embedder.input_proj"
		for _, suffix := range []string{".weight", ".weight_packed", ".linear.weight", ".linear.weight_packed"} {
			if slices.Contains(names, path+suffix) {
				return prefix, true
			}
		}
	}
	return "", false
}
