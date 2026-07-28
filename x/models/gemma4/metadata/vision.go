// Package metadata contains Gemma 4 model metadata checks that do not depend
// on MLX, allowing import and server capability paths to share one definition.
package metadata

import (
	"fmt"
	"slices"
	"strings"
)

const defaultVisionLayers = 16

const maxInt32Value = int64(1<<31 - 1)

type ConfigFile struct {
	Architectures []string      `json:"architectures"`
	ModelType     string        `json:"model_type"`
	TextConfig    TextConfig    `json:"text_config"`
	VisionConfig  *VisionConfig `json:"vision_config"`
	AudioConfig   *AudioConfig  `json:"audio_config"`
}

type TextConfig struct {
	HiddenSize int `json:"hidden_size"`
}

type VisionConfig struct {
	ModelType       string `json:"model_type"`
	NumHiddenLayers int    `json:"num_hidden_layers"`
	Standardize     bool   `json:"standardize"`
	MMEmbedDim      int    `json:"mm_embed_dim"`
	MMPosembSize    int    `json:"mm_posemb_size"`
	ModelPatchSize  int    `json:"model_patch_size"`
}

type TensorDescriptor struct {
	Dtype string
	Shape []int32
}

// ValidateVisionTensors verifies normalized manifest tensor names accepted by
// the native Gemma 4 MLX loaders.
func ValidateVisionTensors(cfg ConfigFile, names []string) error {
	return validateVisionTensors(cfg, names, false)
}

// ValidateVisionSourceTensors verifies source safetensors names. Packed source
// weights are accepted only with the companions required to normalize them.
func ValidateVisionSourceTensors(cfg ConfigFile, names []string) error {
	return validateVisionTensors(cfg, names, true)
}

// ValidateVisionSourceInventory additionally checks source tensor descriptors
// when the released config provides enough dimensions to do so.
func ValidateVisionSourceInventory(cfg ConfigFile, tensors map[string]TensorDescriptor) error {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	if err := ValidateVisionSourceTensors(cfg, names); err != nil {
		return err
	}
	if isUnifiedConfig(cfg) {
		return validateUnifiedSourceDescriptors(cfg, tensors)
	}
	return nil
}

func validateVisionTensors(cfg ConfigFile, names []string, source bool) error {
	if cfg.VisionConfig == nil {
		return fmt.Errorf("missing vision_config")
	}
	if isUnifiedConfig(cfg) {
		return validateUnifiedVisionTensors(names, source)
	}
	return validateTowerVisionTensors(*cfg.VisionConfig, names, source)
}

func isUnifiedConfig(cfg ConfigFile) bool {
	if strings.Contains(strings.ToLower(cfg.ModelType), "unified") ||
		strings.Contains(strings.ToLower(cfg.VisionConfig.ModelType), "unified") {
		return true
	}
	for _, architecture := range cfg.Architectures {
		if strings.Contains(strings.ToLower(architecture), "unified") {
			return true
		}
	}
	return false
}

func validateUnifiedVisionTensors(names []string, source bool) error {
	for _, name := range []string{
		"model.vision_embedder.patch_ln1.weight",
		"model.vision_embedder.patch_ln1.bias",
		"model.vision_embedder.patch_ln2.weight",
		"model.vision_embedder.patch_ln2.bias",
		"model.vision_embedder.pos_embedding",
		"model.vision_embedder.pos_norm.weight",
		"model.vision_embedder.pos_norm.bias",
	} {
		if !slices.Contains(names, name) {
			return fmt.Errorf("missing %s", name)
		}
	}
	for _, path := range []string{
		"model.vision_embedder.patch_dense",
		"model.embed_vision.embedding_projection",
	} {
		if err := requireUnifiedLinear(names, path, source); err != nil {
			return err
		}
	}
	if !slices.Contains(names, "model.vision_embedder.patch_dense.bias") {
		return fmt.Errorf("missing model.vision_embedder.patch_dense.bias")
	}
	return nil
}

func validateUnifiedSourceDescriptors(cfg ConfigFile, tensors map[string]TensorDescriptor) error {
	vc := cfg.VisionConfig
	if vc.ModelPatchSize <= 0 || vc.MMEmbedDim <= 0 || vc.MMPosembSize <= 0 ||
		int64(vc.ModelPatchSize) > maxInt32Value || int64(vc.MMEmbedDim) > maxInt32Value || int64(vc.MMPosembSize) > maxInt32Value ||
		cfg.TextConfig.HiddenSize < 0 || int64(cfg.TextConfig.HiddenSize) > maxInt32Value {
		return fmt.Errorf("invalid unified vision dimensions")
	}
	patchDim := int64(vc.ModelPatchSize) * int64(vc.ModelPatchSize) * 3
	if patchDim <= 0 || patchDim > maxInt32Value {
		return fmt.Errorf("invalid unified patch dimension %d", patchDim)
	}
	expected := map[string][]int32{
		"model.vision_embedder.patch_ln1.weight":   {int32(patchDim)},
		"model.vision_embedder.patch_ln1.bias":     {int32(patchDim)},
		"model.vision_embedder.patch_dense.weight": {int32(vc.MMEmbedDim), int32(patchDim)},
		"model.vision_embedder.patch_dense.bias":   {int32(vc.MMEmbedDim)},
		"model.vision_embedder.patch_ln2.weight":   {int32(vc.MMEmbedDim)},
		"model.vision_embedder.patch_ln2.bias":     {int32(vc.MMEmbedDim)},
		"model.vision_embedder.pos_embedding":      {int32(vc.MMPosembSize), 2, int32(vc.MMEmbedDim)},
		"model.vision_embedder.pos_norm.weight":    {int32(vc.MMEmbedDim)},
		"model.vision_embedder.pos_norm.bias":      {int32(vc.MMEmbedDim)},
	}
	if cfg.TextConfig.HiddenSize > 0 {
		expected["model.embed_vision.embedding_projection.weight"] = []int32{int32(cfg.TextConfig.HiddenSize), int32(vc.MMEmbedDim)}
	}
	for name, shape := range expected {
		desc, ok := tensors[name]
		if !ok && strings.HasSuffix(name, ".weight") {
			base := strings.TrimSuffix(name, ".weight")
			if _, packed := tensors[base+".weight_packed"]; packed {
				// Producer-specific packed shapes are normalized by the import
				// plan; companion presence was checked above.
				continue
			}
		}
		if !ok {
			return fmt.Errorf("missing descriptor for %s", name)
		}
		if !slices.Equal(desc.Shape, shape) {
			return fmt.Errorf("%s shape %v, want %v", name, desc.Shape, shape)
		}
		if !isFloatingDtype(desc.Dtype) {
			return fmt.Errorf("%s dtype %s is not floating point", name, desc.Dtype)
		}
	}
	return nil
}

func requireUnifiedLinear(names []string, path string, source bool) error {
	if slices.Contains(names, path+".weight") {
		return nil
	}
	if source && slices.Contains(names, path+".weight_packed") {
		if slices.Contains(names, path+".weight_scale") &&
			slices.Contains(names, path+".weight_global_scale") {
			return nil
		}
		return fmt.Errorf("incomplete packed weight %s.weight_packed", path)
	}
	return fmt.Errorf("missing %s weight", path)
}

func isFloatingDtype(dtype string) bool {
	switch strings.ToUpper(dtype) {
	case "BF16", "F16", "F32":
		return true
	default:
		return false
	}
}

func validateTowerVisionTensors(cfg VisionConfig, names []string, source bool) error {
	prefix, ok := visionPrefix(names, source)
	if !ok {
		return fmt.Errorf("missing vision_tower.patch_embedder.input_proj weight")
	}
	require := func(name string) error {
		if !slices.Contains(names, name) {
			return fmt.Errorf("missing %s", name)
		}
		return nil
	}
	if err := require(prefix + "vision_tower.patch_embedder.position_embedding_table"); err != nil {
		return err
	}
	layers := cfg.NumHiddenLayers
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
			if err := requireLinear(names, layer+projection, source); err != nil {
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
	if cfg.Standardize {
		if err := require(prefix + "vision_tower.std_bias"); err != nil {
			return err
		}
		if err := require(prefix + "vision_tower.std_scale"); err != nil {
			return err
		}
	}

	for _, projector := range []string{"embed_vision.embedding_projection", "model.embed_vision.embedding_projection"} {
		if requireLinear(names, projector, source) == nil {
			return nil
		}
	}
	return fmt.Errorf("missing embed_vision.embedding_projection weight")
}

func requireLinear(names []string, path string, source bool) error {
	for _, suffix := range []string{".weight", ".linear.weight"} {
		if slices.Contains(names, path+suffix) {
			return nil
		}
	}
	if source {
		for _, suffix := range []string{".weight_packed", ".linear.weight_packed"} {
			weight := path + suffix
			if !slices.Contains(names, weight) {
				continue
			}
			base := strings.TrimSuffix(weight, ".weight_packed")
			if slices.Contains(names, base+".weight_scale") &&
				slices.Contains(names, base+".weight_global_scale") {
				return nil
			}
			return fmt.Errorf("incomplete packed weight %s", weight)
		}
	}
	return fmt.Errorf("missing %s weight", path)
}

func visionPrefix(names []string, source bool) (string, bool) {
	for _, prefix := range []string{"", "model."} {
		if requireLinear(names, prefix+"vision_tower.patch_embedder.input_proj", source) == nil {
			return prefix, true
		}
	}
	return "", false
}
