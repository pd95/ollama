package metadata

import (
	"fmt"
	"slices"
)

const gemma4AudioFeatureSize = 128

type AudioConfig struct {
	AttentionChunkSize      int   `json:"attention_chunk_size"`
	AttentionContextLeft    int   `json:"attention_context_left"`
	AttentionContextRight   int   `json:"attention_context_right"`
	ConvKernelSize          int   `json:"conv_kernel_size"`
	HiddenSize              int   `json:"hidden_size"`
	NumAttentionHeads       int   `json:"num_attention_heads"`
	NumHiddenLayers         int   `json:"num_hidden_layers"`
	OutputProjDims          int   `json:"output_proj_dims"`
	SubsamplingConvChannels []int `json:"subsampling_conv_channels"`
	UseClippedLinears       bool  `json:"use_clipped_linears"`
}

// ValidateAudioTensors verifies the normalized tensor names required by the
// native Gemma 4 MLX audio loader.
func ValidateAudioTensors(cfg ConfigFile, names []string) error {
	if err := validateAudioConfig(cfg); err != nil {
		return err
	}
	for name := range requiredAudioShapes(cfg) {
		if !slices.Contains(names, name) {
			return fmt.Errorf("missing %s", name)
		}
	}
	return nil
}

// ValidateAudioSourceInventory additionally validates released tensor shapes
// and floating-point dtypes.
func ValidateAudioSourceInventory(cfg ConfigFile, tensors map[string]TensorDescriptor) error {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	if err := ValidateAudioTensors(cfg, names); err != nil {
		return err
	}
	for name, shape := range requiredAudioShapes(cfg) {
		desc := tensors[name]
		if !slices.Equal(desc.Shape, shape) {
			return fmt.Errorf("%s shape %v, want %v", name, desc.Shape, shape)
		}
		if !isFloatingDtype(desc.Dtype) {
			return fmt.Errorf("%s dtype %s is not floating point", name, desc.Dtype)
		}
	}
	return nil
}

// RequiredAudioTensorShapes returns a copy of the normalized audio tensor
// contract derived from config.
func RequiredAudioTensorShapes(cfg ConfigFile) (map[string][]int32, error) {
	if err := validateAudioConfig(cfg); err != nil {
		return nil, err
	}
	shapes := requiredAudioShapes(cfg)
	out := make(map[string][]int32, len(shapes))
	for name, shape := range shapes {
		out[name] = append([]int32(nil), shape...)
	}
	return out, nil
}

func validateAudioConfig(cfg ConfigFile) error {
	ac := cfg.AudioConfig
	if ac == nil {
		return fmt.Errorf("missing audio_config")
	}
	if ac.HiddenSize <= 0 || ac.HiddenSize > int(maxInt32Value) ||
		ac.NumHiddenLayers <= 0 || ac.NumHiddenLayers > int(maxInt32Value) ||
		ac.NumAttentionHeads <= 0 || ac.HiddenSize%ac.NumAttentionHeads != 0 ||
		ac.OutputProjDims <= 0 || ac.OutputProjDims > int(maxInt32Value) ||
		ac.ConvKernelSize <= 0 || ac.ConvKernelSize%2 == 0 ||
		ac.AttentionChunkSize <= 0 || ac.AttentionContextLeft <= 0 || ac.AttentionContextRight < 0 ||
		len(ac.SubsamplingConvChannels) != 2 || ac.SubsamplingConvChannels[0] <= 0 || ac.SubsamplingConvChannels[1] <= 0 ||
		cfg.TextConfig.HiddenSize <= 0 || cfg.TextConfig.HiddenSize > int(maxInt32Value) {
		return fmt.Errorf("invalid Gemma 4 audio dimensions")
	}
	return nil
}

func requiredAudioShapes(cfg ConfigFile) map[string][]int32 {
	ac := cfg.AudioConfig
	hidden := int32(ac.HiddenSize)
	output := int32(ac.OutputProjDims)
	headDim := hidden / int32(ac.NumAttentionHeads)
	c0 := int32(ac.SubsamplingConvChannels[0])
	c1 := int32(ac.SubsamplingConvChannels[1])
	freq0 := int32((gemma4AudioFeatureSize + 1) / 2)
	freq1 := (freq0 + 1) / 2

	required := map[string][]int32{
		"model.audio_tower.subsample_conv_projection.layer0.conv.weight":       {c0, 1, 3, 3},
		"model.audio_tower.subsample_conv_projection.layer0.norm.weight":       {c0},
		"model.audio_tower.subsample_conv_projection.layer1.conv.weight":       {c1, c0, 3, 3},
		"model.audio_tower.subsample_conv_projection.layer1.norm.weight":       {c1},
		"model.audio_tower.subsample_conv_projection.input_proj_linear.weight": {hidden, freq1 * c1},
		"model.audio_tower.output_proj.weight":                                 {output, hidden},
		"model.audio_tower.output_proj.bias":                                   {output},
		"model.embed_audio.embedding_projection.weight":                        {int32(cfg.TextConfig.HiddenSize), output},
	}

	addLinear := func(path string, shape []int32) {
		required[path+".linear.weight"] = shape
		if ac.UseClippedLinears {
			for _, suffix := range []string{".input_min", ".input_max", ".output_min", ".output_max"} {
				required[path+suffix] = []int32{}
			}
		}
	}

	for i := range ac.NumHiddenLayers {
		layer := fmt.Sprintf("model.audio_tower.layers.%d", i)
		for _, ff := range []string{"feed_forward1", "feed_forward2"} {
			required[layer+"."+ff+".pre_layer_norm.weight"] = []int32{hidden}
			required[layer+"."+ff+".post_layer_norm.weight"] = []int32{hidden}
			addLinear(layer+"."+ff+".ffw_layer_1", []int32{4 * hidden, hidden})
			addLinear(layer+"."+ff+".ffw_layer_2", []int32{hidden, 4 * hidden})
		}

		required[layer+".norm_pre_attn.weight"] = []int32{hidden}
		required[layer+".norm_post_attn.weight"] = []int32{hidden}
		required[layer+".norm_out.weight"] = []int32{hidden}
		for _, projection := range []string{"q_proj", "k_proj", "v_proj", "post"} {
			addLinear(layer+".self_attn."+projection, []int32{hidden, hidden})
		}
		required[layer+".self_attn.per_dim_scale"] = []int32{headDim}
		required[layer+".self_attn.relative_k_proj.weight"] = []int32{hidden, hidden}

		required[layer+".lconv1d.pre_layer_norm.weight"] = []int32{hidden}
		required[layer+".lconv1d.conv_norm.weight"] = []int32{hidden}
		required[layer+".lconv1d.depthwise_conv1d.weight"] = []int32{hidden, 1, int32(ac.ConvKernelSize)}
		addLinear(layer+".lconv1d.linear_start", []int32{2 * hidden, hidden})
		addLinear(layer+".lconv1d.linear_end", []int32{hidden, hidden})
	}

	return required
}
