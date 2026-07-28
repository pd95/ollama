package metadata

import (
	"maps"
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
	for name, shape := range requiredAudioShapes(cfg) {
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

	missing := maps.Clone(tensors)
	delete(missing, "model.audio_tower.layers.11.self_attn.q_proj.input_max")
	if err := ValidateAudioSourceInventory(cfg, missing); err == nil {
		t.Fatal("missing clipping scalar: error = nil")
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
