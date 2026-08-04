package apertus

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	imagemanifest "github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func TestRegistration(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	root := minimalManifestRoot(t, "ApertusForCausalLM")
	got, err := base.New(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := got.(*Model); !ok {
		t.Fatalf("base.New() returned %T, want *apertus.Model", got)
	}
}

func TestApertus1p5Registration(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	root := minimalManifestRoot(t, "Apertus1p5ForConditionalGeneration")
	got, err := base.New(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := got.(*Model); !ok {
		t.Fatalf("base.New() returned %T, want *apertus.Model", got)
	}
}

func TestParseConfigApertus8B(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"architectures": ["ApertusForCausalLM"],
		"model_type": "apertus",
		"dtype": "bfloat16",
		"hidden_size": 4096,
		"intermediate_size": 21504,
		"num_hidden_layers": 32,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"max_position_embeddings": 65536,
		"rope_theta": 12000000,
		"rope_scaling": {
			"factor": 8,
			"high_freq_factor": 4,
			"low_freq_factor": 1,
			"original_max_position_embeddings": 8192,
			"rope_type": "llama3",
			"type": "llama3"
		},
		"hidden_act": "xielu",
		"qk_norm": true,
		"post_norm": false,
		"attention_bias": false,
		"mlp_bias": false,
		"tie_word_embeddings": false,
		"rms_norm_eps": 1e-5,
		"vocab_size": 131072
	}`))
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Architecture != "ApertusForCausalLM" {
		t.Fatalf("Architecture = %q", cfg.Architecture)
	}
	if cfg.HeadDim != 128 {
		t.Fatalf("HeadDim = %d, want 128", cfg.HeadDim)
	}
	if cfg.Scale != float32(1/math.Sqrt(128)) {
		t.Fatalf("Scale = %v, want %v", cfg.Scale, float32(1/math.Sqrt(128)))
	}
	if cfg.MaxPositionEmbeddings != 65536 {
		t.Fatalf("MaxPositionEmbeddings = %d, want 65536", cfg.MaxPositionEmbeddings)
	}
	if cfg.RopeScaling.OriginalMaxPositionEmbeddings != 8192 {
		t.Fatalf("OriginalMaxPositionEmbeddings = %d, want 8192", cfg.RopeScaling.OriginalMaxPositionEmbeddings)
	}
	if cfg.OutputVocabSize != cfg.VocabSize {
		t.Fatalf("OutputVocabSize = %d, want VocabSize %d", cfg.OutputVocabSize, cfg.VocabSize)
	}
}

func TestParseConfigApertus1p5Exact8B(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"architectures": ["Apertus1p5ForConditionalGeneration"],
		"model_type": "apertus1p5",
		"text_config": {
			"attention_bias": false,
			"dtype": "bfloat16",
			"hidden_act": "xielu",
			"hidden_size": 4096,
			"intermediate_size": 21504,
			"max_position_embeddings": 262144,
			"mlp_bias": false,
			"model_type": "apertus1p5_text",
			"num_attention_heads": 32,
			"num_hidden_layers": 32,
			"num_key_value_heads": 8,
			"output_vocab_size": 131072,
			"post_norm": false,
			"qk_norm": true,
			"rms_norm_eps": 0.00001,
			"rope_parameters": {
				"factor": 32.0,
				"high_freq_factor": 4.0,
				"low_freq_factor": 1.0,
				"original_max_position_embeddings": 8192,
				"rope_theta": 4000000,
				"rope_type": "llama3"
			},
			"tie_word_embeddings": false,
			"vocab_size": 266752
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Architecture != "Apertus1p5ForConditionalGeneration" {
		t.Fatalf("Architecture = %q", cfg.Architecture)
	}
	if cfg.ModelType != "apertus1p5_text" {
		t.Fatalf("ModelType = %q", cfg.ModelType)
	}
	if cfg.VocabSize != 266752 || cfg.OutputVocabSize != 131072 {
		t.Fatalf("vocab/output vocab = %d/%d, want 266752/131072", cfg.VocabSize, cfg.OutputVocabSize)
	}
	if cfg.MaxPositionEmbeddings != 262144 {
		t.Fatalf("MaxPositionEmbeddings = %d, want 262144", cfg.MaxPositionEmbeddings)
	}
	if cfg.RopeTheta != 4000000 {
		t.Fatalf("RopeTheta = %v, want 4000000", cfg.RopeTheta)
	}
	if cfg.RopeScaling.Factor != 32 {
		t.Fatalf("RopeScaling.Factor = %v, want 32", cfg.RopeScaling.Factor)
	}
}

func TestParseConfigApertus1p5Synthetic70B(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"architectures": ["Apertus1p5ForConditionalGeneration"],
		"model_type": "apertus1p5",
		"text_config": {
			"dtype": "bfloat16",
			"hidden_act": "xielu",
			"hidden_size": 8192,
			"intermediate_size": 43008,
			"max_position_embeddings": 262144,
			"model_type": "apertus1p5_text",
			"num_attention_heads": 64,
			"num_hidden_layers": 80,
			"num_key_value_heads": 8,
			"output_vocab_size": 131072,
			"qk_norm": true,
			"rms_norm_eps": 0.00001,
			"rope_parameters": {
				"factor": 32.0,
				"high_freq_factor": 4.0,
				"low_freq_factor": 1.0,
				"original_max_position_embeddings": 8192,
				"rope_theta": 4000000,
				"rope_type": "llama3"
			},
			"vocab_size": 266752
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.HiddenSize != 8192 || cfg.IntermediateSize != 43008 || cfg.NumHiddenLayers != 80 ||
		cfg.NumAttentionHeads != 64 || cfg.NumKeyValueHeads != 8 || cfg.HeadDim != 128 {
		t.Fatalf("70B dimensions were not preserved: %+v", cfg)
	}
	if cfg.VocabSize != 266752 || cfg.OutputVocabSize != 131072 || cfg.MaxPositionEmbeddings != 262144 {
		t.Fatalf("70B composite vocabulary/context = %d/%d/%d", cfg.VocabSize, cfg.OutputVocabSize, cfg.MaxPositionEmbeddings)
	}
}

func TestParseConfigApertusMiniRopeAndTiedEmbeddings(t *testing.T) {
	tests := []struct {
		name      string
		rope      string
		tied      bool
		wantType  string
		wantScale float32
	}{
		{
			name:      "default untied",
			rope:      `{"rope_type":"default"}`,
			wantType:  "default",
			wantScale: 1,
		},
		{
			name:      "missing scaling defaults to standard rope",
			rope:      `{}`,
			wantType:  "default",
			wantScale: 1,
		},
		{
			name:      "linear tied",
			rope:      `{"rope_type":"linear","factor":2}`,
			tied:      true,
			wantType:  "linear",
			wantScale: 0.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := parseConfig([]byte(`{
				"architectures":["ApertusForCausalLM"],
				"model_type":"apertus",
				"hidden_size":16,
				"intermediate_size":32,
				"num_hidden_layers":1,
				"num_attention_heads":4,
				"num_key_value_heads":2,
				"max_position_embeddings":4096,
				"rope_theta":500000,
				"rope_scaling":` + tt.rope + `,
				"hidden_act":"xielu",
				"qk_norm":true,
				"tie_word_embeddings":` + strconv.FormatBool(tt.tied) + `,
				"vocab_size":131072
			}`))
			if err != nil {
				t.Fatal(err)
			}
			if got := cfg.ropeType(); got != tt.wantType {
				t.Fatalf("ropeType() = %q, want %q", got, tt.wantType)
			}
			if cfg.RopeScale != tt.wantScale {
				t.Fatalf("RopeScale = %v, want %v", cfg.RopeScale, tt.wantScale)
			}
			if cfg.TieWordEmbeddings != tt.tied {
				t.Fatalf("TieWordEmbeddings = %v, want %v", cfg.TieWordEmbeddings, tt.tied)
			}
		})
	}
}

func TestParseConfigRejectsInvalidMiniRope(t *testing.T) {
	base := `{
		"architectures":["ApertusForCausalLM"],
		"hidden_size":16,
		"intermediate_size":32,
		"num_hidden_layers":1,
		"num_attention_heads":4,
		"num_key_value_heads":2,
		"max_position_embeddings":4096,
		"rope_theta":500000,
		"rope_scaling":%s,
		"hidden_act":"xielu",
		"qk_norm":true,
		"vocab_size":128
	}`
	for _, tt := range []struct {
		name string
		rope string
		want string
	}{
		{name: "linear factor", rope: `{"rope_type":"linear","factor":0}`, want: "invalid linear rope scaling factor"},
		{name: "unsupported type", rope: `{"rope_type":"yarn","factor":1}`, want: `unsupported rope scaling type "yarn"`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseConfig([]byte(fmt.Sprintf(base, tt.rope)))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("parseConfig error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestParseConfigApertus1p5RequiresRopeParameters(t *testing.T) {
	_, err := parseConfig([]byte(`{
		"architectures": ["Apertus1p5ForConditionalGeneration"],
		"text_config": {
			"hidden_size": 16,
			"intermediate_size": 32,
			"num_hidden_layers": 1,
			"num_attention_heads": 4,
			"num_key_value_heads": 2,
			"max_position_embeddings": 64,
			"hidden_act": "xielu",
			"qk_norm": true,
			"vocab_size": 128
		}
	}`))
	if err == nil || !strings.Contains(err.Error(), "missing rope_parameters.rope_theta") {
		t.Fatalf("parseConfig error = %v, want missing rope_parameters.rope_theta", err)
	}
}

func TestApertus1p5MediaLoaderRequiresValidIndependentConfig(t *testing.T) {
	tensors := map[string]*mlx.Array{}
	completeApertusVisionNames(func(name string) bool {
		tensors[name] = mlx.New(name)
		return true
	})
	completeApertusAudioNames(func(name string) bool {
		tensors[name] = mlx.New(name)
		return true
	})

	vision := VisionTokenizerConfig{
		AttnResolutions: []int32{16}, BaseChannels: 256, ChannelMultiplier: []int32{1, 1, 2, 2, 4},
		CodebookSize: 131072, EmbedDim: 256, InChannels: 3, LatentChannels: 256,
		NumResBlocks: 4, Resolution: 256,
	}
	audio := AudioTokenizerConfig{
		AudioChannels: 1, CodebookDim: 512, CodebookSize: 4096, Compress: 2,
		DilationGrowthRate: 2, HiddenSize: 512, KernelSize: 7, LastKernelSize: 7,
		NormType: "weight_norm", NumFilters: 32, NumLSTMLayers: 2, NumResidualLayers: 1,
		PadMode: "reflect", ResidualKernelSize: 3, SamplingRate: 24000,
		UpsamplingRatios: []int32{6, 5, 5, 4}, UseConvShortcut: true,
	}
	if !canLoadVisionTokenizer(tensors, vision) || !canLoadAudioTokenizer(tensors, audio) {
		t.Fatal("complete tensors with supported configs must enable both media tokenizers")
	}
	if canLoadVisionTokenizer(tensors, VisionTokenizerConfig{}) {
		t.Fatal("complete vision tensors with missing vision config enabled vision")
	}
	invalidVision := vision
	invalidVision.Resolution = 512
	if canLoadVisionTokenizer(tensors, invalidVision) || !canLoadAudioTokenizer(tensors, audio) {
		t.Fatal("invalid vision config did not suppress only vision")
	}
	if canLoadAudioTokenizer(tensors, AudioTokenizerConfig{}) {
		t.Fatal("complete audio tensors with missing audio config enabled audio")
	}
	invalidAudio := audio
	invalidAudio.SamplingRate = 16000
	if canLoadAudioTokenizer(tensors, invalidAudio) || !canLoadVisionTokenizer(tensors, vision) {
		t.Fatal("invalid audio config did not suppress only audio")
	}
}

func TestParseConfigRejectsInvalidOutputVocab(t *testing.T) {
	_, err := parseConfig([]byte(`{
		"architectures": ["ApertusForCausalLM"],
		"hidden_size": 16,
		"intermediate_size": 32,
		"num_hidden_layers": 1,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"max_position_embeddings": 64,
		"rope_theta": 12000000,
		"rope_scaling": {
			"factor": 8,
			"high_freq_factor": 4,
			"low_freq_factor": 1,
			"original_max_position_embeddings": 8192,
			"rope_type": "llama3"
		},
		"hidden_act": "xielu",
		"qk_norm": true,
		"vocab_size": 128,
		"output_vocab_size": 129
	}`))
	if err == nil || !strings.Contains(err.Error(), "invalid output_vocab_size") {
		t.Fatalf("parseConfig error = %v, want invalid output_vocab_size", err)
	}
}

func TestParseConfigRejectsUnsupportedVariants(t *testing.T) {
	baseConfig := `{
		"architectures": ["ApertusForCausalLM"],
		"hidden_size": 16,
		"intermediate_size": 32,
		"num_hidden_layers": 1,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"max_position_embeddings": 64,
		"rope_theta": 12000000,
		"rope_scaling": {
			"factor": 8,
			"high_freq_factor": 4,
			"low_freq_factor": 1,
			"original_max_position_embeddings": 8192,
			"rope_type": "llama3"
		},
		"hidden_act": "silu",
		"qk_norm": true,
		"vocab_size": 128
	}`

	_, err := parseConfig([]byte(baseConfig))
	if err == nil || !strings.Contains(err.Error(), `unsupported hidden_act "silu"`) {
		t.Fatalf("parseConfig error = %v, want unsupported hidden_act", err)
	}
}

func TestLlama3RoPEReferenceValues(t *testing.T) {
	got, err := Llama3InvFreqs(8, 12000000, 8, 1, 4, 8192)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{
		1,
		0.016990442,
		0.000036084392,
		0.00000061308978,
	}
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-9 {
			t.Fatalf("inv_freq[%d] = %.12g, want %.12g", i, got[i], want[i])
		}
	}

	freqs, err := Llama3Freqs(8, 12000000, 8, 1, 4, 8192)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		wantFreq := 1 / want[i]
		if diff := math.Abs(float64(freqs[i] - wantFreq)); diff/float64(wantFreq) > 1e-6 {
			t.Fatalf("freq[%d] = %.12g, want reciprocal %.12g", i, freqs[i], wantFreq)
		}
	}
}

func TestQKNormShape(t *testing.T) {
	got := qkNormShape(2, 3, 4, 5)
	want := []int32{2, 4, 3, 5}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("qkNormShape() = %v, want %v", got, want)
		}
	}
}

func TestXIELUScalarParity(t *testing.T) {
	tests := []struct {
		x    float64
		want float64
	}{
		{x: 2.0, want: 3.772588722239781},
		{x: -2.0, want: 0.35462209218399154},
	}
	for _, tt := range tests {
		got := XIELUScalar(tt.x, 0.0, 0.0, 0.5, -1e-6)
		if diff := math.Abs(got - tt.want); diff > 1e-12 {
			t.Fatalf("XIELUScalar(%v) = %.15g, want %.15g", tt.x, got, tt.want)
		}
	}
}

func TestRequiredTensorErrors(t *testing.T) {
	cfg := &Config{NumHiddenLayers: 1}
	err := checkRequiredTensors(nil, cfg)
	if err == nil || !strings.Contains(err.Error(), "model.embed_tokens.weight") {
		t.Fatalf("checkRequiredTensors error = %v, want missing embedding", err)
	}

	tensors := map[string]*mlx.Array{}
	for _, name := range []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
		"lm_head.weight",
		"model.layers.0.attention_layernorm.weight",
		"model.layers.0.feedforward_layernorm.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.self_attn.q_norm.weight",
		"model.layers.0.self_attn.k_norm.weight",
		"model.layers.0.mlp.up_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
		"model.layers.0.mlp.act_fn.alpha_p",
		"model.layers.0.mlp.act_fn.alpha_n",
		"model.layers.0.mlp.act_fn.beta",
	} {
		tensors[name] = mlx.New(name)
	}
	err = checkRequiredTensors(tensors, cfg)
	if err == nil || !strings.Contains(err.Error(), "model.layers.0.mlp.act_fn.eps") {
		t.Fatalf("checkRequiredTensors error = %v, want missing xielu eps", err)
	}
}

func TestRequiredTensorsAllowTiedHeadToBeAbsent(t *testing.T) {
	cfg := &Config{NumHiddenLayers: 0, TieWordEmbeddings: true}
	tensors := map[string]*mlx.Array{
		"model.embed_tokens.weight": mlx.New("model.embed_tokens.weight"),
		"model.norm.weight":         mlx.New("model.norm.weight"),
	}
	if err := checkRequiredTensors(tensors, cfg); err != nil {
		t.Fatalf("checkRequiredTensors() error = %v", err)
	}

	cfg.TieWordEmbeddings = false
	err := checkRequiredTensors(tensors, cfg)
	if err == nil || !strings.Contains(err.Error(), "lm_head.weight") {
		t.Fatalf("checkRequiredTensors() error = %v, want missing lm_head.weight", err)
	}
}

func TestTensorPrefixVariants(t *testing.T) {
	for _, prefix := range []string{"model.", "model.language_model."} {
		t.Run(prefix, func(t *testing.T) {
			tensors := map[string]*mlx.Array{
				prefix + "embed_tokens.weight": mlx.New(prefix + "embed_tokens.weight"),
				prefix + "norm.weight":         mlx.New(prefix + "norm.weight"),
			}
			got, err := tensorPrefix(tensors)
			if err != nil {
				t.Fatal(err)
			}
			if got != prefix {
				t.Fatalf("tensorPrefix() = %q, want %q", got, prefix)
			}
		})
	}
}

func minimalManifestRoot(t *testing.T, architecture string) *model.Root {
	t.Helper()

	dir := t.TempDir()
	configJSON := `{
		"architectures": [` + strconv.Quote(architecture) + `],
		"model_type": "apertus",
		"dtype": "bfloat16",
		"hidden_size": 16,
		"intermediate_size": 32,
		"num_hidden_layers": 1,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"max_position_embeddings": 64,
		"rope_theta": 12000000,
		"rope_scaling": {
			"factor": 8,
			"high_freq_factor": 4,
			"low_freq_factor": 1,
			"original_max_position_embeddings": 8192,
			"rope_type": "llama3"
		},
		"hidden_act": "xielu",
		"qk_norm": true,
		"post_norm": false,
		"attention_bias": false,
		"mlp_bias": false,
		"tie_word_embeddings": false,
		"rms_norm_eps": 1e-5,
		"vocab_size": 3
	}`
	if architecture == "Apertus1p5ForConditionalGeneration" {
		configJSON = `{
			"architectures": ["Apertus1p5ForConditionalGeneration"],
			"model_type": "apertus1p5",
			"text_config": {
				"dtype": "bfloat16",
				"hidden_size": 16,
				"intermediate_size": 32,
				"num_hidden_layers": 1,
				"num_attention_heads": 4,
				"num_key_value_heads": 2,
				"max_position_embeddings": 64,
				"rope_parameters": {
					"rope_theta": 4000000,
					"factor": 32,
					"high_freq_factor": 4,
					"low_freq_factor": 1,
					"original_max_position_embeddings": 8192,
					"rope_type": "llama3"
				},
				"hidden_act": "xielu",
				"qk_norm": true,
				"post_norm": false,
				"attention_bias": false,
				"mlp_bias": false,
				"tie_word_embeddings": false,
				"rms_norm_eps": 1e-5,
				"vocab_size": 6,
				"output_vocab_size": 3
			}
		}`
	}
	configDigest := writeManifestBlob(t, dir, "config", []byte(configJSON))
	tokenizerDigest := writeManifestBlob(t, dir, "tokenizer", []byte(`{
		"model": {
			"type": "BPE",
			"vocab": {"</s>": 0, "hello": 1, "world": 2},
			"merges": []
		},
		"added_tokens": [
			{"id": 0, "content": "</s>", "special": true}
		]
	}`))

	return &model.Root{
		Manifest: &imagemanifest.ModelManifest{
			BlobDir: dir,
			Manifest: &imagemanifest.Manifest{
				SchemaVersion: 2,
				MediaType:     "application/vnd.ollama.image.model",
				Layers: []imagemanifest.ManifestLayer{
					{
						MediaType: "application/vnd.ollama.image.json",
						Digest:    configDigest,
						Size:      1,
						Name:      "config.json",
					},
					{
						MediaType: "application/vnd.ollama.image.json",
						Digest:    tokenizerDigest,
						Size:      1,
						Name:      "tokenizer.json",
					},
				},
			},
		},
	}
}

func writeManifestBlob(t *testing.T, dir, name string, data []byte) string {
	t.Helper()

	digest := "sha256:" + name
	path := filepath.Join(dir, "sha256-"+name)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return digest
}
