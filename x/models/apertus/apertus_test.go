package apertus

import (
	"math"
	"os"
	"path/filepath"
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

	root := minimalManifestRoot(t)
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

func minimalManifestRoot(t *testing.T) *model.Root {
	t.Helper()

	dir := t.TempDir()
	configDigest := writeManifestBlob(t, dir, "config", []byte(`{
		"architectures": ["ApertusForCausalLM"],
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
	}`))
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
