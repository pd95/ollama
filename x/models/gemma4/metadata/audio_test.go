package metadata

import (
	"maps"
	"slices"
	"strings"
	"testing"
)

func releasedAudioConfig(layers int) ConfigFile {
	return ConfigFile{
		TextConfig: TextConfig{HiddenSize: 2560},
		AudioConfig: &AudioConfig{
			AttentionChunkSize: 12, AttentionContextLeft: 13,
			ConvKernelSize: 5, HiddenSize: 1024, NumAttentionHeads: 8,
			NumHiddenLayers: layers, OutputProjDims: 1536,
			SubsamplingConvChannels: []int{128, 32}, UseClippedLinears: true,
		},
	}
}

func completeAudioInventory(cfg ConfigFile) map[string]TensorDescriptor {
	tensors := make(map[string]TensorDescriptor)
	shapes, err := requiredAudioShapes(cfg)
	if err != nil {
		panic(err)
	}
	for name, shape := range shapes {
		tensors[name] = TensorDescriptor{Dtype: "BF16", Shape: shape}
	}
	return tensors
}

func TestValidateReleasedAudioInventory(t *testing.T) {
	cfg := releasedAudioConfig(12)
	tensors := completeAudioInventory(cfg)
	if got := len(tensors); got != 752 {
		t.Fatalf("audio tensor count = %d, want 752", got)
	}
	if err := ValidateAudioSourceInventory(cfg, tensors); err != nil {
		t.Fatalf("ValidateAudioSourceInventory() error = %v", err)
	}

	partial := maps.Clone(tensors)
	delete(partial, "model.audio_tower.layers.11.self_attn.q_proj.input_max")
	if err := ValidateAudioSourceInventory(cfg, partial); err == nil {
		t.Fatal("partial clipping inventory: error = nil")
	}

	nearMatch := maps.Clone(partial)
	nearMatch["model.audio_tower.layers.11.self_attn.q_proj.input_max.extra"] = TensorDescriptor{Dtype: "BF16", Shape: []int32{}}
	if err := ValidateAudioSourceInventory(cfg, nearMatch); err == nil {
		t.Fatal("near-match clipping scalar substituted for exact name: error = nil")
	}

	badShape := maps.Clone(tensors)
	badShape["model.audio_tower.output_proj.weight"] = TensorDescriptor{Dtype: "BF16", Shape: []int32{1024, 1536}}
	if err := ValidateAudioSourceInventory(cfg, badShape); err == nil {
		t.Fatal("transposed output projection: error = nil")
	}

	badDtype := maps.Clone(tensors)
	badDtype["model.embed_audio.embedding_projection.weight"] = TensorDescriptor{Dtype: "U8", Shape: []int32{2560, 1536}}
	if err := ValidateAudioSourceInventory(cfg, badDtype); err == nil {
		t.Fatal("non-floating projector: error = nil")
	}
}

func TestValidateAudioTensorsExactNormalizedNames(t *testing.T) {
	cfg := releasedAudioConfig(1)
	shapes, err := RequiredAudioTensorShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(shapes))
	for name := range shapes {
		names = append(names, name)
	}
	if err := ValidateAudioTensors(cfg, names); err != nil {
		t.Fatalf("complete normalized inventory: %v", err)
	}

	missing := "model.audio_tower.subsample_conv_projection.layer0.conv.weight"
	for i, name := range names {
		if name == missing {
			names[i] = "prefix." + name
			break
		}
	}
	if err := ValidateAudioTensors(cfg, names); err == nil || !strings.Contains(err.Error(), missing) {
		t.Fatalf("near-match inventory error = %v, want exact missing name", err)
	}
}

func TestValidateAudioConfig(t *testing.T) {
	tests := []struct {
		name string
		edit func(*ConfigFile)
	}{
		{"missing", func(cfg *ConfigFile) { cfg.AudioConfig = nil }},
		{"heads", func(cfg *ConfigFile) { cfg.AudioConfig.NumAttentionHeads = 7 }},
		{"conv channels", func(cfg *ConfigFile) { cfg.AudioConfig.SubsamplingConvChannels = []int{128} }},
		{"even kernel", func(cfg *ConfigFile) { cfg.AudioConfig.ConvKernelSize = 4 }},
		{"text hidden", func(cfg *ConfigFile) { cfg.TextConfig.HiddenSize = 0 }},
		{"hidden overflow", func(cfg *ConfigFile) { cfg.AudioConfig.HiddenSize = 1 << 30; cfg.AudioConfig.NumAttentionHeads = 1 }},
		{"impractical layers", func(cfg *ConfigFile) { cfg.AudioConfig.NumHiddenLayers = maxAudioLayers + 1 }},
		{"heads bound", func(cfg *ConfigFile) { cfg.AudioConfig.NumAttentionHeads = maxAudioHeads + 1 }},
		{"output bound", func(cfg *ConfigFile) { cfg.AudioConfig.OutputProjDims = maxAudioOutputDims + 1 }},
		{"kernel bound", func(cfg *ConfigFile) { cfg.AudioConfig.ConvKernelSize = maxAudioKernelSize + 2 }},
		{"chunk bound", func(cfg *ConfigFile) { cfg.AudioConfig.AttentionChunkSize = maxAudioContextSize + 1 }},
		{"left context bound", func(cfg *ConfigFile) { cfg.AudioConfig.AttentionContextLeft = maxAudioContextSize + 1 }},
		{"right context bound", func(cfg *ConfigFile) { cfg.AudioConfig.AttentionContextRight = maxAudioContextSize + 1 }},
		{"conv channel bound", func(cfg *ConfigFile) { cfg.AudioConfig.SubsamplingConvChannels[1] = maxAudioConvChannels + 1 }},
		{"text hidden bound", func(cfg *ConfigFile) { cfg.TextConfig.HiddenSize = maxTextHiddenSize + 1 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := releasedAudioConfig(12)
			tt.edit(&cfg)
			if err := ValidateAudioTensors(cfg, nil); err == nil {
				t.Fatal("ValidateAudioTensors() error = nil")
			}
		})
	}
}

func TestAudioConfigSupportedBoundaries(t *testing.T) {
	cfg := releasedAudioConfig(maxAudioLayers)
	cfg.TextConfig.HiddenSize = maxTextHiddenSize
	cfg.AudioConfig.HiddenSize = maxAudioHiddenSize
	cfg.AudioConfig.NumAttentionHeads = maxAudioHeads
	cfg.AudioConfig.OutputProjDims = maxAudioOutputDims
	cfg.AudioConfig.ConvKernelSize = maxAudioKernelSize
	cfg.AudioConfig.AttentionChunkSize = maxAudioContextSize
	cfg.AudioConfig.AttentionContextLeft = maxAudioContextSize
	cfg.AudioConfig.AttentionContextRight = maxAudioContextSize
	cfg.AudioConfig.SubsamplingConvChannels = []int{maxAudioConvChannels, maxAudioConvChannels}
	shapes, err := RequiredAudioTensorShapes(cfg)
	if err != nil {
		t.Fatalf("supported boundary config: %v", err)
	}
	if got, want := len(shapes), 8+maxAudioLayers*62; got != want {
		t.Fatalf("boundary inventory count = %d, want %d", got, want)
	}
	names := make([]string, 0, len(shapes))
	for name := range shapes {
		names = append(names, name)
	}
	if err := ValidateAudioTensors(cfg, names); err != nil {
		t.Fatalf("supported boundary inventory: %v", err)
	}
	if got, want := shapes["model.audio_tower.layers.127.feed_forward1.ffw_layer_1.linear.weight"], []int32{4 * maxAudioHiddenSize, maxAudioHiddenSize}; !slices.Equal(got, want) {
		t.Fatalf("boundary feed-forward shape = %v, want %v", got, want)
	}
}

func TestAudioWithoutClippingScalars(t *testing.T) {
	cfg := releasedAudioConfig(1)
	cfg.AudioConfig.UseClippedLinears = false
	tensors := completeAudioInventory(cfg)
	if got := len(tensors); got != 30 {
		t.Fatalf("audio tensor count = %d, want 30", got)
	}
	if err := ValidateAudioSourceInventory(cfg, tensors); err != nil {
		t.Fatalf("ValidateAudioSourceInventory() error = %v", err)
	}
}
