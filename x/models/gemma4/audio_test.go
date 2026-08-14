package gemma4

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	mlxmodel "github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
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

func TestParseReleasedUnifiedAudioConfig(t *testing.T) {
	cfg, err := parseAudioConfig([]byte(`{
		"audio_config":{
			"model_type":"gemma4_unified_audio",
			"audio_embed_dim":640,"audio_samples_per_token":640,
			"hidden_size":640,"output_proj_dims":640,"rms_norm_eps":0.000001
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.unified() || cfg.AudioSamplesPerToken != 640 || cfg.OutputProjDims != 640 {
		t.Fatalf("unified audio config = %+v", cfg)
	}

	bad := *cfg
	bad.AudioSamplesPerToken = 320
	data, err := json.Marshal(map[string]any{"audio_config": bad})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := parseAudioConfig(data); err == nil {
		t.Fatal("invalid unified audio config: error = nil")
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

func TestPrepareAudioMedia(t *testing.T) {
	cfg := defaultAudioProcessorConfig()
	frames := make([][]float64, 16000)
	for i := range frames {
		frames[i] = []float64{0.25}
	}
	m := &Model{
		TextConfig:           &TextConfig{AudioTokenIDValue: 1, BOATokenIDValue: 2, EOATokenIDValue: 3},
		AudioConfig:          &AudioConfig{},
		AudioProcessorConfig: &cfg,
		Audio:                &AudioModel{},
		EmbedAudio:           &MultimodalEmbedder{},
	}
	wav := makeTestWAV(t, 1, 16, 16000, frames)
	prepared, err := m.PrepareMedia([]base.Segment{{Tokens: []int32{9}}, {Kind: "audio", Data: wav}})
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
	ordered, err := m.PrepareMedia([]base.Segment{{Kind: "audio", Data: wav}, {Tokens: []int32{8}}, {Kind: "audio", Data: wav}})
	if err != nil || len(ordered.Items) != 2 || ordered.Items[0].Range[1] >= ordered.Items[1].Range[0] {
		t.Fatalf("ordered audio = items %d error %v", len(ordered.Items), err)
	}
}

func TestPrepareAudioMediaRejectsMissingWeights(t *testing.T) {
	m := &Model{
		TextConfig:           &TextConfig{},
		AudioConfig:          &AudioConfig{},
		AudioProcessorConfig: ptr(defaultAudioProcessorConfig()),
	}
	if _, err := m.PrepareMedia([]base.Segment{{Kind: "audio", Data: []byte("wav")}}); err == nil || !strings.Contains(err.Error(), "does not support audio") {
		t.Fatalf("PrepareMedia() error = %v", err)
	}
}

func ptr[T any](v T) *T {
	return &v
}

func TestEncodeUnifiedAudioMedia(t *testing.T) {
	skipIfNoMLX(t)
	m := &Model{
		AudioConfig: &AudioConfig{ModelType: "gemma4_unified_audio"},
		EmbedAudio: &MultimodalEmbedder{
			Projection: nn.NewLinear(mlx.FromValues([]float32{1, 0, 0, 1}, 2, 2), nil),
			Eps:        1e-6,
		},
	}
	item := &base.PreparedItem{
		Opaque: gemma4MediaPayload{Audio: &gemma4AudioInput{FeatureSize: 2, Frames: 1, SoftTokens: 1}},
	}
	features := m.EncodeMedia(item, mlx.FromValues([]float32{3, 4}, 1, 1, 2))
	if features.NumDims() != 2 || features.Dim(0) != 1 || features.Dim(1) != 2 {
		t.Fatalf("unified audio features = %v", features.Dims())
	}
}

func TestNewModelDisablesMalformedAudioMetadata(t *testing.T) {
	config := `{
		"architectures":["Gemma4ForConditionalGeneration"],
		"audio_token_id":1,
		"text_config":{
			"hidden_size":8,
			"num_hidden_layers":1,
			"intermediate_size":16,
			"num_attention_heads":1,
			"num_key_value_heads":1,
			"head_dim":8,
			"global_head_dim":8,
			"vocab_size":3,
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
		"model":{"type":"BPE","vocab":{"<|audio>":0,"<|audio|>":1,"<audio|>":2},"merges":[]},
		"added_tokens":[
			{"id":0,"content":"<|audio>","special":true},
			{"id":1,"content":"<|audio|>","special":true},
			{"id":2,"content":"<audio|>","special":true}
		]
	}`)
	for _, tt := range []struct {
		name      string
		config    string
		tokenizer []byte
	}{
		{"invalid residual", strings.Replace(config, `"residual_weight":0.5`, `"residual_weight":0`, 1), tokenizerData},
		{"float32 overflow", strings.Replace(config, `"gradient_clipping":10000000000.0`, `"gradient_clipping":1e39`, 1), tokenizerData},
		{"vocab token is not singleton encoding", config, []byte(`{
			"model":{"type":"BPE","vocab":{"<|audio>":0,"<|audio|>":1,"<audio|>":2},"merges":[]},
			"added_tokens":[]
		}`)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			root := testGemma4Root(t, []byte(tt.config), tt.tokenizer, map[string][]byte{
				"processor_config.json": []byte(`{
					"audio_seq_length":750,
					"feature_extractor":{
						"feature_size":128,"fft_length":512,"frame_length":320,
						"hop_length":160,"input_scale_factor":1,"max_frequency":8000,
						"mel_floor":0.001,"padding_side":"right","sampling_rate":16000
					}
				}`),
				"tokenizer_config.json": []byte(`{
					"boa_token":"<|audio>","audio_token":"<|audio|>","eoa_token":"<audio|>"
				}`),
			})

			loaded, err := newModel(root)
			if err != nil {
				t.Fatalf("newModel() error = %v", err)
			}
			m := loaded.(*Model)
			if m.AudioConfig != nil || m.AudioProcessorConfig != nil {
				t.Fatalf("audio runtime = (%+v, %+v), want disabled", m.AudioConfig, m.AudioProcessorConfig)
			}
			if m.TextConfig == nil || m.Tokenizer() == nil {
				t.Fatal("text runtime was not preserved")
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
