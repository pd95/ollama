package gemma4

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	mlxmodel "github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
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

func TestPrepareAudioMediaPrompt(t *testing.T) {
	tok := testGemma4AudioTokenizer(t)
	cfg := defaultAudioProcessorConfig()
	frames := make([][]float64, 16000)
	for i := range frames {
		frames[i] = []float64{0.25}
	}
	m := &Model{
		TextConfig:           &TextConfig{AudioTokenIDValue: 1},
		AudioConfig:          &AudioConfig{},
		AudioProcessorConfig: &cfg,
		tok:                  tok,
		mediaTokens:          defaultGemma4MediaTokens(),
	}
	media := []llm.MediaData{{ID: 7, Kind: llm.MediaKindAudio, Data: makeTestWAV(t, 1, 16, 16000, frames)}}
	prepared, err := m.PrepareMediaPrompt(context.Background(), "[img-7]", media)
	if err != nil {
		t.Fatal(err)
	}
	payload, ok := prepared.Payload.(*gemma4MediaPayload)
	if !ok || payload.Audio == nil {
		t.Fatalf("payload = %#v, want audio", prepared.Payload)
	}
	if got := payload.AudioEnd - payload.AudioStart; got != payload.Audio.SoftTokens {
		t.Fatalf("audio span = %d, want %d", got, payload.Audio.SoftTokens)
	}
	for i, id := range prepared.PLEInputIDs {
		if i >= payload.AudioStart && i < payload.AudioEnd {
			if id != 0 {
				t.Fatalf("PLEInputIDs[%d] = %d, want 0", i, id)
			}
		}
	}
	if len(prepared.BidirectionalSpans) != 0 {
		t.Fatalf("audio bidirectional spans = %v, want none", prepared.BidirectionalSpans)
	}
	if _, err := m.PrepareMediaPrompt(context.Background(), "missing", media); err == nil || !strings.Contains(err.Error(), "marker") {
		t.Fatalf("missing marker error = %v", err)
	}
	if _, err := m.PrepareMediaPrompt(context.Background(), "[img-7]", append(media, media[0])); err == nil || !strings.Contains(err.Error(), "exactly one") {
		t.Fatalf("multiple media error = %v", err)
	}
}

func TestPrepareAudioMediaEmbeddingsRejectsMissingWeights(t *testing.T) {
	skipIfNoMLX(t)
	m := &Model{
		TextConfig:  &TextConfig{EmbedScale: 1},
		EmbedTokens: nn.NewEmbedding(mlx.FromValues([]float32{0, 0, 1, 1, 2, 2}, 3, 2)),
	}
	prepared := &batch.PreparedInput{
		Tokens: []int32{0, 1, 2},
		Payload: &gemma4MediaPayload{
			Audio:      &gemma4AudioInput{SoftTokens: 1},
			AudioStart: 1,
			AudioEnd:   2,
		},
	}
	if err := m.PrepareMediaEmbeddings(prepared); err == nil || !strings.Contains(err.Error(), "audio weights are not loaded") {
		t.Fatalf("PrepareMediaEmbeddings() error = %v", err)
	}
}

func TestNewModelDisablesMalformedAudioMetadata(t *testing.T) {
	config := []byte(`{
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
			"residual_weight":0,
			"rms_norm_eps":0.000001,
			"subsampling_conv_channels":[128,32],
			"use_clipped_linears":true
		}
	}`)
	tokenizerData := []byte(`{
		"model":{"type":"BPE","vocab":{"<|audio>":0,"<|audio|>":1,"<audio|>":2},"merges":[]},
		"added_tokens":[
			{"id":0,"content":"<|audio>","special":true},
			{"id":1,"content":"<|audio|>","special":true},
			{"id":2,"content":"<audio|>","special":true}
		]
	}`)
	root := testGemma4Root(t, config, tokenizerData, map[string][]byte{
		"processor_config.json": []byte(`{
			"feature_size":128,"sampling_rate":16000,"padding_value":0,
			"return_attention_mask":true,"num_mel_bins":128,"n_fft":512,
			"hop_length":160,"win_length":400,"max_length_seconds":30
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
}

func testGemma4AudioTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	data := []byte(`{
		"model":{"type":"BPE","vocab":{"<|audio>":0,"<|audio|>":1,"<audio|>":2},"merges":[]},
		"added_tokens":[
			{"id":0,"content":"<|audio>","special":true},
			{"id":1,"content":"<|audio|>","special":true},
			{"id":2,"content":"<audio|>","special":true}
		]
	}`)
	tok, err := tokenizer.LoadFromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	return tok
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
