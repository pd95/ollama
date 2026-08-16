package metadata

import (
	"encoding/json"
	"maps"
	"slices"
	"strings"
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

func TestParseAudioConfigEquivalence(t *testing.T) {
	valid, err := json.Marshal(releasedAudioConfig(12))
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		data []byte
		ok   bool
	}{
		{"released", valid, true},
		{"missing", []byte(`{"model_type":"gemma4"}`), true},
		{"malformed", []byte(`{"audio_config":`), false},
		{"partial", []byte(`{"audio_config":{"hidden_size":1024}}`), false},
		{"near match", []byte(`{"audio_config_extra":{"hidden_size":1024}}`), true},
		{"bounded width", []byte(strings.Replace(string(valid), `"hidden_size":1024`, `"hidden_size":1073741824`, 1)), false},
		{"float32 overflow", []byte(strings.Replace(string(valid), `"gradient_clipping":10000000000`, `"gradient_clipping":1e39`, 1)), false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := ParseAudioConfig(tt.data)
			if (err == nil) != tt.ok {
				t.Fatalf("ParseAudioConfig() = (%+v, %v), ok = %v", cfg, err, tt.ok)
			}
			if tt.name == "missing" || tt.name == "near match" {
				if cfg != nil {
					t.Fatalf("ParseAudioConfig() = %+v, want nil", cfg)
				}
			}
		})
	}
}

func TestValidateAudioRuntimeMetadata(t *testing.T) {
	processor := []byte(`{"audio_seq_length":750,"feature_extractor":{"feature_size":128,"fft_length":512,"frame_length":320,"hop_length":160,"input_scale_factor":1,"max_frequency":8000,"mel_floor":0.001,"padding_side":"right","sampling_rate":16000}}`)
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
		{"invalid logit cap", func(cfg *ConfigFile) { cfg.AudioConfig.AttentionLogitCap = 0 }},
		{"invalid residual", func(cfg *ConfigFile) { cfg.AudioConfig.ResidualWeight = 0 }},
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

	var overflow ConfigFile
	if err := json.Unmarshal([]byte(`{"audio_config":{"gradient_clipping":1e39}}`), &overflow); err == nil {
		t.Fatal("float32 overflow: json.Unmarshal() error = nil")
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
