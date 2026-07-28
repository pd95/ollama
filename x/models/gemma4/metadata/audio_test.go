package metadata

import (
	"encoding/json"
	"maps"
	"testing"
)

func releasedAudioConfig(layers int) ConfigFile {
	return ConfigFile{
		TextConfig:   TextConfig{HiddenSize: 2560, VocabSize: 262144},
		AudioTokenID: 258881,
		AudioConfig: &AudioConfig{
			AttentionChunkSize: 12, AttentionContextLeft: 13,
			AttentionInvalidLogit: -1e9, AttentionLogitCap: 50,
			ConvKernelSize: 5, HiddenSize: 1024, NumAttentionHeads: 8,
			NumHiddenLayers: layers, OutputProjDims: 1536,
			GradientClipping: 1e10, ResidualWeight: 0.5, RMSNormEps: 1e-6,
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
		{"invalid logit cap", func(cfg *ConfigFile) { cfg.AudioConfig.AttentionLogitCap = 0 }},
		{"invalid residual", func(cfg *ConfigFile) { cfg.AudioConfig.ResidualWeight = 0 }},
		{"text hidden", func(cfg *ConfigFile) { cfg.TextConfig.HiddenSize = 0 }},
		{"excessive layers", func(cfg *ConfigFile) { cfg.AudioConfig.NumHiddenLayers = maxAudioLayers + 1 }},
		{"excessive hidden", func(cfg *ConfigFile) { cfg.AudioConfig.HiddenSize = maxAudioHiddenSize + 1 }},
		{"excessive channels", func(cfg *ConfigFile) { cfg.AudioConfig.SubsamplingConvChannels[0] = maxAudioConvChannels + 1 }},
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

	var overflow ConfigFile
	if err := json.Unmarshal([]byte(`{"audio_config":{"gradient_clipping":1e39}}`), &overflow); err == nil {
		t.Fatal("float32 overflow: json.Unmarshal() error = nil")
	}
}

func TestValidateAudioRuntimeMetadata(t *testing.T) {
	processor := []byte(`{
		"audio_seq_length":750,
		"feature_extractor":{
			"feature_size":128,"fft_length":512,"frame_length":320,"hop_length":160,
			"input_scale_factor":1,"max_frequency":8000,"mel_floor":0.001,
			"padding_side":"right","sampling_rate":16000
		}
	}`)
	tokens := []byte(`{"boa_token":"<|audio>","audio_token":"<|audio|>","eoa_token":"<audio|>"}`)
	tokenizerData := []byte(`{"model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[{"id":5,"content":"<|audio>","special":true},{"id":258881,"content":"<|audio|>","special":true},{"id":6,"content":"<audio|>","special":true}]}`)
	cfg := releasedAudioConfig(12)
	if err := ValidateAudioRuntimeMetadata(cfg, processor, tokens, tokenizerData); err != nil {
		t.Fatalf("ValidateAudioRuntimeMetadata() error = %v", err)
	}

	for _, tt := range []struct {
		name      string
		processor []byte
		tokens    []byte
		tokenizer []byte
		edit      func(*ConfigFile)
	}{
		{"missing processor", nil, tokens, tokenizerData, nil},
		{"unsupported processor", []byte(`{"audio_seq_length":749}`), tokens, tokenizerData, nil},
		{"missing tokens", processor, nil, tokenizerData, nil},
		{"incomplete tokens", processor, []byte(`{"audio_token":"<|audio|>"}`), tokenizerData, nil},
		{"missing tokenizer", processor, tokens, nil, nil},
		{"wrong tokenizer id", processor, tokens, []byte(`{"model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[{"id":5,"content":"<|audio>","special":true},{"id":7,"content":"<|audio|>","special":true},{"id":6,"content":"<audio|>","special":true}]}`), nil},
		{"vocab membership is not singleton encoding", processor, tokens, []byte(`{"model":{"type":"BPE","vocab":{"<|audio>":0,"<|audio|>":1,"<audio|>":2},"merges":[]},"added_tokens":[]}`), func(cfg *ConfigFile) { cfg.AudioTokenID = 1; cfg.TextConfig.VocabSize = 3 }},
		{"invalid token id", processor, tokens, tokenizerData, func(cfg *ConfigFile) { cfg.AudioTokenID = cfg.TextConfig.VocabSize }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			candidate := releasedAudioConfig(12)
			if tt.edit != nil {
				tt.edit(&candidate)
			}
			if err := ValidateAudioRuntimeMetadata(candidate, tt.processor, tt.tokens, tt.tokenizer); err == nil {
				t.Fatal("ValidateAudioRuntimeMetadata() error = nil")
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
