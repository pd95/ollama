package gptoss

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func TestParseConfig(t *testing.T) {
	data := []byte(`{
		"architectures": ["GptOssForCausalLM"],
		"model_type": "gpt_oss",
		"num_hidden_layers": 24,
		"hidden_size": 2880,
		"intermediate_size": 2880,
		"num_attention_heads": 64,
		"num_key_value_heads": 8,
		"head_dim": 64,
		"num_local_experts": 32,
		"num_experts_per_tok": 4,
		"sliding_window": 128,
		"rope_theta": 150000,
		"rope_scaling": {
			"factor": 32.0,
			"original_max_position_embeddings": 4096,
			"rope_type": "yarn",
			"beta_fast": 32.0,
			"beta_slow": 1.0,
			"truncate": false
		},
		"rms_norm_eps": 0.00001,
		"vocab_size": 201088,
		"tie_word_embeddings": false,
		"quantization_config": {
			"quant_method": "mxfp4"
		}
	}`)

	cfg, err := parseConfig(data)
	if err != nil {
		t.Fatalf("parseConfig() error = %v", err)
	}

	if cfg.Architecture != "GptOssForCausalLM" {
		t.Fatalf("Architecture = %q, want %q", cfg.Architecture, "GptOssForCausalLM")
	}
	if cfg.ModelType != "gpt_oss" {
		t.Fatalf("ModelType = %q, want %q", cfg.ModelType, "gpt_oss")
	}
	if cfg.NumHiddenLayers != 24 || cfg.HiddenSize != 2880 || cfg.IntermediateSize != 2880 {
		t.Fatalf("unexpected core dims: %+v", cfg)
	}
	if cfg.NumAttentionHeads != 64 || cfg.NumKeyValueHeads != 8 || cfg.HeadDim != 64 {
		t.Fatalf("unexpected attention dims: %+v", cfg)
	}
	if cfg.NumLocalExperts != 32 || cfg.NumExpertsPerTok != 4 {
		t.Fatalf("unexpected expert dims: %+v", cfg)
	}
	if cfg.MaxPositionEmbeddings != 131072 {
		t.Fatalf("MaxPositionEmbeddings = %d, want 131072", cfg.MaxPositionEmbeddings)
	}
	if cfg.QuantMethod != "mxfp4" {
		t.Fatalf("QuantMethod = %q, want %q", cfg.QuantMethod, "mxfp4")
	}
}

func TestNewModelRegistersGptOss(t *testing.T) {
	root := testRoot(t, []byte(`{
		"architectures": ["GptOssForCausalLM"],
		"model_type": "gpt_oss",
		"num_hidden_layers": 24,
		"hidden_size": 2880,
		"intermediate_size": 2880,
		"num_attention_heads": 64,
		"num_key_value_heads": 8,
		"head_dim": 64,
		"num_local_experts": 32,
		"num_experts_per_tok": 4,
		"sliding_window": 128,
		"rope_theta": 150000,
		"rope_scaling": {
			"factor": 32.0,
			"original_max_position_embeddings": 4096
		},
		"rms_norm_eps": 0.00001,
		"vocab_size": 201088,
		"tie_word_embeddings": false,
		"quantization_config": {
			"quant_method": "mxfp4"
		}
	}`), []byte(`{
		"model": {
			"type": "BPE",
			"vocab": {"a": 0, "b": 1},
			"merges": []
		},
		"added_tokens": []
	}`))

	m, err := base.New(root)
	if err != nil {
		t.Fatalf("base.New() error = %v", err)
	}

	got, ok := m.(*Model)
	if !ok {
		t.Fatalf("base.New() type = %T, want *Model", m)
	}

	if got.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want loaded tokenizer")
	}
	if got.NumLayers() != 24 {
		t.Fatalf("NumLayers() = %d, want 24", got.NumLayers())
	}
	if got.MaxContextLength() != 131072 {
		t.Fatalf("MaxContextLength() = %d, want 131072", got.MaxContextLength())
	}
	if got.Architecture != "GptOssForCausalLM" {
		t.Fatalf("Architecture = %q, want %q", got.Architecture, "GptOssForCausalLM")
	}
	if got.QuantMethod != "mxfp4" {
		t.Fatalf("QuantMethod = %q, want %q", got.QuantMethod, "mxfp4")
	}
}

func TestLoadWeightsIsSkeletal(t *testing.T) {
	m := &Model{}
	if err := m.LoadWeights(nil); err == nil {
		t.Fatal("LoadWeights() error = nil, want skeletal implementation error")
	}
}

func testRoot(t *testing.T, configJSON, tokenizerJSON []byte) *model.Root {
	t.Helper()

	dir := t.TempDir()
	blobDir := filepath.Join(dir, "blobs")
	if err := os.MkdirAll(blobDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}

	writeBlob := func(digest string, content []byte) {
		t.Helper()
		path := filepath.Join(blobDir, strings.Replace(digest, ":", "-", 1))
		if err := os.WriteFile(path, content, 0o644); err != nil {
			t.Fatalf("WriteFile(%s) error = %v", path, err)
		}
	}

	writeBlob("sha256:config", configJSON)
	writeBlob("sha256:tokenizer", tokenizerJSON)

	mf := &manifest.Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.ollama.image.model",
		Layers: []manifest.ManifestLayer{
			{MediaType: "application/vnd.ollama.image.json", Digest: "sha256:config", Name: "config.json"},
			{MediaType: "application/vnd.ollama.image.json", Digest: "sha256:tokenizer", Name: "tokenizer.json"},
		},
	}

	return &model.Root{
		Manifest: &manifest.ModelManifest{
			Manifest: mf,
			BlobDir:  blobDir,
		},
	}
}
