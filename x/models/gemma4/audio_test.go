package gemma4

import (
	"context"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	mlxmodel "github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	gemma4metadata "github.com/ollama/ollama/x/models/gemma4/metadata"
)

const releasedGemma4AudioConfig = `{
  "audio_config": {
    "attention_chunk_size": 12,
    "attention_context_left": 13,
    "attention_context_right": 0,
    "attention_invalid_logits_value": -1000000000.0,
    "attention_logit_cap": 50.0,
    "conv_kernel_size": 5,
    "gradient_clipping": 10000000000.0,
    "hidden_size": 1024,
    "num_attention_heads": 8,
    "num_hidden_layers": 12,
    "output_proj_dims": 1536,
    "residual_weight": 0.5,
    "rms_norm_eps": 0.000001,
    "subsampling_conv_channels": [128, 32],
    "use_clipped_linears": true
  }
}`

func TestParseReleasedAudioConfig(t *testing.T) {
	cfg, err := parseAudioConfig([]byte(releasedGemma4AudioConfig))
	if err != nil {
		t.Fatal(err)
	}
	if cfg == nil || cfg.HiddenSize != 1024 || cfg.NumHiddenLayers != 12 || cfg.OutputProjDims != 1536 {
		t.Fatalf("audio config = %+v", cfg)
	}
	if cfg.AttentionContextLeft != 13 || cfg.AttentionChunkSize != 12 || cfg.ConvKernelSize != 5 {
		t.Fatalf("audio context config = %+v", cfg)
	}

	if cfg, err := parseAudioConfig([]byte(`{"model_type":"gemma4"}`)); err != nil || cfg != nil {
		t.Fatalf("missing audio config = %+v, %v; want nil, nil", cfg, err)
	}
	bad := strings.Replace(releasedGemma4AudioConfig, `"num_attention_heads": 8`, `"num_attention_heads": 7`, 1)
	if _, err := parseAudioConfig([]byte(bad)); err == nil {
		t.Fatal("non-divisible head count: error = nil")
	}
}

func TestParseAudioConfigUsesMetadataPredicate(t *testing.T) {
	for _, edit := range []struct {
		name string
		old  string
		new  string
	}{
		{"hidden bound", `"hidden_size": 1024`, `"hidden_size": 8193`},
		{"layer bound", `"num_hidden_layers": 12`, `"num_hidden_layers": 129`},
		{"head bound", `"num_attention_heads": 8`, `"num_attention_heads": 257`},
		{"output bound", `"output_proj_dims": 1536`, `"output_proj_dims": 16385`},
		{"context bound", `"attention_context_left": 13`, `"attention_context_left": 4097`},
		{"float32 overflow", `"gradient_clipping": 10000000000.0`, `"gradient_clipping": 1e39`},
	} {
		t.Run(edit.name, func(t *testing.T) {
			data := strings.Replace(releasedGemma4AudioConfig, edit.old, edit.new, 1)
			if data == releasedGemma4AudioConfig {
				t.Fatal("test edit did not apply")
			}
			if _, err := parseAudioConfig([]byte(data)); err == nil {
				t.Fatal("parseAudioConfig() error = nil")
			}
		})
	}
}

func TestParseTextConfigRejectsInvalidAudioMarkers(t *testing.T) {
	valid := `"boi_token_id":1,"image_token_id":2,"eoi_token_id":3,"boa_token_id":4,"audio_token_id":5,"eoa_token_index":6,"text_config":{"vocab_size":7}`
	if _, err := parseTextConfig([]byte(`{` + valid + `}`)); err != nil {
		t.Fatalf("exact marker boundary: %v", err)
	}
	tests := []struct {
		name string
		json string
	}{
		{"negative begin", `{"boa_token_id":-1}`},
		{"negative audio", `{"audio_token_id":-1}`},
		{"negative end", `{"eoa_token_index":-1}`},
		{"begin outside vocab", `{"boa_token_id":262144}`},
		{"audio outside vocab", `{"audio_token_id":262144}`},
		{"end outside vocab", `{"eoa_token_index":262144}`},
		{"duplicate begin and audio", `{"boa_token_id":258881,"audio_token_id":258881}`},
		{"duplicate audio and end", `{"audio_token_id":258883,"eoa_token_index":258883}`},
		{"duplicate begin and end", `{"boa_token_id":258883,"eoa_token_index":258883}`},
		{"audio begin duplicates image begin", `{"boa_token_id":255999}`},
		{"audio begin duplicates image", `{"boa_token_id":258880}`},
		{"audio begin duplicates image end", `{"boa_token_id":258882}`},
		{"audio token duplicates image begin", `{"audio_token_id":255999}`},
		{"audio token duplicates image", `{"audio_token_id":258880}`},
		{"audio token duplicates image end", `{"audio_token_id":258882}`},
		{"audio end duplicates image begin", `{"eoa_token_index":255999}`},
		{"audio end duplicates image", `{"eoa_token_index":258880}`},
		{"audio end duplicates image end", `{"eoa_token_index":258882}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := parseTextConfig([]byte(tt.json)); err == nil {
				t.Fatal("parseTextConfig() error = nil")
			}
		})
	}
	cfg, err := parseTextConfig([]byte(`{` + valid + `}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name   string
		mutate func(*TextConfig)
	}{
		{"zero begin", func(cfg *TextConfig) { cfg.BOATokenIDValue = 0 }},
		{"zero audio", func(cfg *TextConfig) { cfg.AudioTokenIDValue = 0 }},
		{"zero end", func(cfg *TextConfig) { cfg.EOATokenIDValue = 0 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			candidate := cfg
			tt.mutate(&candidate)
			if err := validateGemma4MediaTokenConfig(&candidate); err == nil {
				t.Fatal("marker validator error = nil")
			}
		})
	}
}

func TestAudioAttentionMaskValues(t *testing.T) {
	valid := []bool{true, true, true, false}
	got := audioAttentionMaskValues(valid, 2, 2, 4, 2, 0)
	allowed := func(block, query, context int) bool {
		return got[(block*2+query)*4+context]
	}
	for _, tt := range []struct {
		block, query, context int
		want                  bool
	}{
		{0, 0, 0, false},
		{0, 0, 2, true},
		{0, 0, 3, false},
		{0, 1, 1, false},
		{0, 1, 3, true},
		{1, 0, 0, true},
		{1, 0, 2, true},
		{1, 0, 3, false},
		{1, 1, 3, false},
	} {
		if value := allowed(tt.block, tt.query, tt.context); value != tt.want {
			t.Errorf("mask[%d,%d,%d] = %v, want %v", tt.block, tt.query, tt.context, value, tt.want)
		}
	}
}

func TestParseGemma4AudioTokens(t *testing.T) {
	tokens := parseGemma4MediaTokens([]byte(`{
      "boa_token":"audio-start","audio_token":"audio-soft","eoa_token":"audio-end"
    }`), defaultGemma4MediaTokens())
	if tokens.BOA != "audio-start" || tokens.Audio != "audio-soft" || tokens.EOA != "audio-end" {
		t.Fatalf("audio tokens = %+v", tokens)
	}
	if tokens.Image != defaultGemma4ImageToken {
		t.Fatalf("image token = %q, want unchanged default", tokens.Image)
	}
}

func TestMediaTokenSpanAudio(t *testing.T) {
	start, end, err := mediaTokenSpan([]int32{1, 258881, 258881, 2}, 258881)
	if err != nil || start != 1 || end != 3 {
		t.Fatalf("mediaTokenSpan() = %d, %d, %v; want 1, 3, nil", start, end, err)
	}
}

func TestSupportedGemma4AudioDType(t *testing.T) {
	for _, dtype := range []mlx.DType{mlx.DTypeBFloat16, mlx.DTypeFloat16, mlx.DTypeFloat32} {
		if !supportedGemma4AudioDType(dtype) {
			t.Errorf("supportedGemma4AudioDType(%s) = false", dtype)
		}
	}
	for _, dtype := range []mlx.DType{mlx.DTypeInt32, mlx.DTypeUint8, mlx.DTypeFloat64} {
		if supportedGemma4AudioDType(dtype) {
			t.Errorf("supportedGemma4AudioDType(%s) = true", dtype)
		}
	}
}

func TestPrepareAudioMedia(t *testing.T) {
	cfg := defaultAudioProcessorConfig()
	frames := make([][]float64, 16000)
	for i := range frames {
		frames[i] = []float64{0.25}
	}
	m := &Model{
		TextConfig:  &TextConfig{AudioTokenIDValue: 1, BOATokenIDValue: 2, EOATokenIDValue: 3},
		AudioConfig: &AudioConfig{}, AudioProcessorConfig: &cfg,
		Audio: &AudioModel{}, EmbedAudio: &MultimodalEmbedder{},
	}
	wav := makeTestWAV(t, 1, 16, 16000, frames)
	prepared, err := m.PrepareMedia(context.Background(), []base.Segment{{Tokens: []int32{9}}, {Kind: "audio", Data: wav}})
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared.Items) != 1 {
		t.Fatalf("items = %d, want 1", len(prepared.Items))
	}
	payload := prepared.Items[0].Opaque.(gemma4MediaPayload)
	if got := payload.AudioEnd - payload.AudioStart; got != payload.Audio.SoftTokens {
		t.Fatalf("audio span = %d, want %d", got, payload.Audio.SoftTokens)
	}
	if payload.Audio.Features != nil || len(prepared.Items[0].MediaData) == 0 {
		t.Fatal("audio features must be owned by PreparedItem.MediaData")
	}
	ordered, err := m.PrepareMedia(context.Background(), []base.Segment{{Kind: "audio", Data: wav}, {Tokens: []int32{8}}, {Kind: "audio", Data: wav}})
	if err != nil || len(ordered.Items) != 2 || ordered.Items[0].Range[1] >= ordered.Items[1].Range[0] {
		t.Fatalf("ordered audio = items %d error %v", len(ordered.Items), err)
	}
}

func TestPrepareAudioMediaRejectsMissingWeights(t *testing.T) {
	cfg := defaultAudioProcessorConfig()
	m := &Model{TextConfig: &TextConfig{}, AudioConfig: &AudioConfig{}, AudioProcessorConfig: &cfg}
	if _, err := m.PrepareMedia(context.Background(), []base.Segment{{Kind: "audio", Data: []byte("wav")}}); err == nil || !strings.Contains(err.Error(), "does not support audio") {
		t.Fatalf("PrepareMedia() error = %v", err)
	}
}

func TestValidateAndLoadGemma4AudioWeights(t *testing.T) {
	useMLXTestThread(t)

	cfg := tinyAudioConfig()
	const textHidden = int32(6)
	for _, dtype := range []mlx.DType{mlx.DTypeBFloat16, mlx.DTypeFloat16, mlx.DTypeFloat32} {
		t.Run(dtype.String(), func(t *testing.T) {
			tensors := actualAudioTensors(t, cfg, textHidden, dtype)
			if err := validateGemma4AudioWeights(tensors, cfg, textHidden); err != nil {
				t.Fatalf("validateGemma4AudioWeights(%s): %v", dtype, err)
			}
			loaded, err := loadAudioModel(tensors, cfg, textHidden, 0, 0, "", nil)
			if err != nil || loaded == nil {
				t.Fatalf("loadAudioModel(%s) = (%v, %v)", dtype, loaded, err)
			}
			embed, err := loadMultimodalEmbedder(tensors, "embed_audio", cfg.RMSNormEps, 0, 0, "", nil)
			if err != nil || embed == nil {
				t.Fatalf("loadMultimodalEmbedder(%s) = (%v, %v)", dtype, embed, err)
			}
		})
	}

	valid := actualAudioTensors(t, cfg, textHidden, mlx.DTypeFloat32)
	const target = "model.audio_tower.output_proj.weight"
	tests := []struct {
		name   string
		mutate func(map[string]*mlx.Array)
	}{
		{"missing", func(tensors map[string]*mlx.Array) { delete(tensors, target) }},
		{"wrong shape", func(tensors map[string]*mlx.Array) { tensors[target] = mlx.Zeros(mlx.DTypeFloat32, 1, 1) }},
		{"integer dtype", func(tensors map[string]*mlx.Array) {
			tensors[target] = mlx.Zeros(mlx.DTypeInt32, int(cfg.OutputProjDims), int(cfg.HiddenSize))
		}},
		{"float64 dtype", func(tensors map[string]*mlx.Array) {
			tensors[target] = mlx.Zeros(mlx.DTypeFloat64, int(cfg.OutputProjDims), int(cfg.HiddenSize))
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensors := maps.Clone(valid)
			tt.mutate(tensors)
			if err := validateGemma4AudioWeights(tensors, cfg, textHidden); err == nil {
				t.Fatal("validateGemma4AudioWeights() error = nil")
			}
			if loaded, err := loadAudioModel(tensors, cfg, textHidden, 0, 0, "", nil); err == nil || loaded != nil {
				t.Fatalf("loadAudioModel() = (%v, %v), want (nil, error)", loaded, err)
			}
		})
	}
}

func TestNewModelDisablesMalformedAudioMetadata(t *testing.T) {
	config := `{
		"architectures":["Gemma4ForConditionalGeneration"],
		"boi_token_id":1,
		"image_token_id":2,
		"eoi_token_id":3,
		"boa_token_id":4,
		"audio_token_id":5,
		"eoa_token_index":6,
		"text_config":{
			"hidden_size":8,
			"num_hidden_layers":1,
			"intermediate_size":16,
			"num_attention_heads":1,
			"num_key_value_heads":1,
			"head_dim":8,
			"global_head_dim":8,
			"vocab_size":7,
			"rms_norm_eps":0.000001
		},
		"audio_config":{
			"attention_chunk_size":12,
			"attention_context_left":13,
			"attention_context_right":0,
			"attention_invalid_logits_value":-1000000000.0,
			"attention_logit_cap":50.0,
			"conv_kernel_size":5,
			"gradient_clipping":10000000000.0,
			"hidden_size":1024,
			"num_attention_heads":8,
			"num_hidden_layers":12,
			"output_proj_dims":1536,
			"residual_weight":0.5,
			"rms_norm_eps":0.000001,
			"subsampling_conv_channels":[128,32],
			"use_clipped_linears":true
		}
	}`
	tokenizerData := []byte(`{
		"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"<|audio>":4,"<|audio|>":5,"<audio|>":6},"merges":[]},
		"added_tokens":[
			{"id":4,"content":"<|audio>","special":true},
			{"id":5,"content":"<|audio|>","special":true},
			{"id":6,"content":"<audio|>","special":true}
		]
	}`)
	processorData := []byte(`{
		"feature_size":128,"sampling_rate":16000,"padding_value":0,
		"return_attention_mask":true,"num_mel_bins":128,"n_fft":512,
		"hop_length":160,"win_length":400,"max_length_seconds":30
	}`)
	tokenizerConfigData := []byte(`{
		"boa_token":"<|audio>","audio_token":"<|audio|>","eoa_token":"<audio|>"
	}`)
	tests := []struct {
		name      string
		config    string
		tokenizer []byte
		extra     map[string][]byte
	}{
		{
			name:      "malformed audio config",
			config:    strings.Replace(config, `"residual_weight":0.5`, `"residual_weight":0`, 1),
			tokenizer: tokenizerData,
			extra: map[string][]byte{
				"processor_config.json": processorData,
				"tokenizer_config.json": tokenizerConfigData,
			},
		},
		{
			name:      "missing processor config",
			config:    config,
			tokenizer: tokenizerData,
			extra: map[string][]byte{
				"tokenizer_config.json": tokenizerConfigData,
			},
		},
		{
			name:      "malformed processor config",
			config:    config,
			tokenizer: tokenizerData,
			extra: map[string][]byte{
				"processor_config.json": []byte(`{"feature_extractor":{"sampling_rate":0}}`),
				"tokenizer_config.json": tokenizerConfigData,
			},
		},
		{
			name:   "incomplete tokenizer markers",
			config: config,
			tokenizer: []byte(`{
				"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"<|audio>":4,"<|audio|>":5,"<audio|>":6},"merges":[]},
				"added_tokens":[]
			}`),
			extra: map[string][]byte{
				"processor_config.json": processorData,
				"tokenizer_config.json": tokenizerConfigData,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loaded, err := newModel(testGemma4Root(t, []byte(tt.config), tt.tokenizer, tt.extra))
			if err != nil {
				t.Fatalf("newModel() error = %v", err)
			}
			m := loaded.(*Model)
			if m.AudioConfig != nil && m.AudioProcessorConfig != nil || m.Audio != nil || m.EmbedAudio != nil {
				t.Fatalf("audio runtime retained state = (%+v, %+v, %v, %v)", m.AudioConfig, m.AudioProcessorConfig, m.Audio, m.EmbedAudio)
			}
			if m.TextConfig == nil || m.Tokenizer() == nil {
				t.Fatal("text runtime was not preserved")
			}

			var wantError string
			for attempt := range 2 {
				prepared, err := m.PrepareMedia(context.Background(), []base.Segment{{Kind: "audio", Data: []byte("wav")}})
				if err == nil || !strings.Contains(err.Error(), "does not support audio") {
					t.Fatalf("PrepareMedia() attempt %d = (%+v, %v), want deterministic unsupported-audio error", attempt, prepared, err)
				}
				if attempt == 0 {
					wantError = err.Error()
				} else if err.Error() != wantError {
					t.Fatalf("PrepareMedia() error = %q, want %q", err, wantError)
				}
				if prepared != nil || m.AudioConfig != nil && m.AudioProcessorConfig != nil || m.Audio != nil || m.EmbedAudio != nil {
					t.Fatalf("PrepareMedia() attempt %d retained state: prepared=%+v model=%+v", attempt, prepared, m)
				}
			}
		})
	}
}

func testGemma4Root(t *testing.T, configData, tokenizerData []byte, extra map[string][]byte) *mlxmodel.Root {
	t.Helper()

	blobDir := filepath.Join(t.TempDir(), "blobs")
	if err := os.MkdirAll(blobDir, 0o755); err != nil {
		t.Fatal(err)
	}
	layers := make([]manifest.ManifestLayer, 0, len(extra)+2)
	writeConfig := func(name string, data []byte) {
		digest := fmt.Sprintf("sha256:config-%d", len(layers))
		path := filepath.Join(blobDir, strings.Replace(digest, ":", "-", 1))
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		layers = append(layers, manifest.ManifestLayer{
			MediaType: "application/vnd.ollama.image.json",
			Digest:    digest,
			Name:      name,
		})
	}
	writeConfig("config.json", configData)
	writeConfig("tokenizer.json", tokenizerData)
	for name, data := range extra {
		writeConfig(name, data)
	}

	return &mlxmodel.Root{Manifest: &manifest.ModelManifest{
		Manifest: &manifest.Manifest{
			SchemaVersion: 2,
			MediaType:     "application/vnd.ollama.image.model",
			Layers:        layers,
		},
		BlobDir: blobDir,
	}}
}

func tinyAudioConfig() *AudioConfig {
	return &AudioConfig{
		AttentionChunkSize: 2, AttentionContextLeft: 2, AttentionContextRight: 0,
		AttentionInvalidLogit: -1e9, AttentionLogitCap: 50, ConvKernelSize: 3,
		GradientClipping: 1e4, HiddenSize: 4, NumAttentionHeads: 2,
		NumHiddenLayers: 1, OutputProjDims: 4, ResidualWeight: 0.5,
		RMSNormEps: 1e-6, SubsamplingConvChannels: []int32{2, 2},
	}
}

func actualAudioTensors(t *testing.T, cfg *AudioConfig, textHidden int32, dtype mlx.DType) map[string]*mlx.Array {
	t.Helper()
	shapes, err := gemma4metadata.RequiredAudioTensorShapes(audioMetadataConfig(cfg, textHidden))
	if err != nil {
		t.Fatal(err)
	}
	tensors := make(map[string]*mlx.Array, len(shapes))
	for name, shape := range shapes {
		dims := make([]int, len(shape))
		for i, dim := range shape {
			dims[i] = int(dim)
		}
		tensors[name] = mlx.Zeros(dtype, dims...)
	}
	return tensors
}
