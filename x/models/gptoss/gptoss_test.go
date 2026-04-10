package gptoss

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
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

func TestLoadWeightsDensePath(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}

	tensors := denseTestTensors(t, cfg)
	if err := m.LoadWeights(tensors); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}

	if m.EmbedTokens == nil {
		t.Fatal("EmbedTokens = nil")
	}
	if m.Norm == nil {
		t.Fatal("Norm = nil")
	}
	if m.LMHead == nil {
		t.Fatal("LMHead = nil")
	}

	if len(m.Layers) != int(cfg.NumHiddenLayers) {
		t.Fatalf("len(Layers) = %d, want %d", len(m.Layers), cfg.NumHiddenLayers)
	}
	for i, layer := range m.Layers {
		if layer == nil {
			t.Fatalf("layer %d = nil", i)
		}
		if layer.AttentionNorm == nil {
			t.Fatalf("layer %d AttentionNorm = nil", i)
		}
		if layer.FFNNorm == nil {
			t.Fatalf("layer %d FFNNorm = nil", i)
		}
		if layer.Router == nil {
			t.Fatalf("layer %d Router = nil", i)
		}
		if layer.Attention == nil {
			t.Fatalf("layer %d Attention = nil", i)
		}
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			t.Fatalf("layer %d attention projections not fully loaded", i)
		}
		if layer.Attention.Sinks == nil {
			t.Fatalf("layer %d Attention.Sinks = nil", i)
		}
	}

	caches := m.NewCaches()
	if len(caches) != int(cfg.NumHiddenLayers) {
		t.Fatalf("len(NewCaches()) = %d, want %d", len(caches), cfg.NumHiddenLayers)
	}
	if _, ok := caches[0].(*cache.RotatingKVCache); !ok {
		t.Fatalf("cache[0] = %T, want *cache.RotatingKVCache", caches[0])
	}
	if _, ok := caches[1].(*cache.KVCache); !ok {
		t.Fatalf("cache[1] = %T, want *cache.KVCache", caches[1])
	}
}

func TestLoadWeightsMissingTensorFails(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}

	tensors := denseTestTensors(t, cfg)
	delete(tensors, "blocks.0.q_proj.weight")

	err := m.LoadWeights(tensors)
	if err == nil {
		t.Fatal("LoadWeights() error = nil, want missing tensor failure")
	}
	if !strings.Contains(err.Error(), "layer 0") || !strings.Contains(err.Error(), "blocks.0.q_proj.weight") {
		t.Fatalf("LoadWeights() error = %q, want layer and tensor name", err)
	}
}

func TestLoadWeightsShapeValidationFails(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}

	tensors := denseTestTensors(t, cfg)
	tensors["blocks.0.q_proj.weight"] = mlx.FromValues([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}, 3, 4)

	err := m.LoadWeights(tensors)
	if err == nil {
		t.Fatal("LoadWeights() error = nil, want shape validation failure")
	}
	if !strings.Contains(err.Error(), "blocks.0.q_proj.weight") || !strings.Contains(err.Error(), "shape [3 4]") {
		t.Fatalf("LoadWeights() error = %q, want q_proj shape mismatch", err)
	}
}

func TestNewCachesLayerParity(t *testing.T) {
	cfg := denseTestConfig(t)
	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}

	caches := m.NewCaches()
	if len(caches) != 2 {
		t.Fatalf("len(NewCaches()) = %d, want 2", len(caches))
	}
	if _, ok := caches[0].(*cache.RotatingKVCache); !ok {
		t.Fatalf("cache[0] = %T, want *cache.RotatingKVCache", caches[0])
	}
	if _, ok := caches[1].(*cache.KVCache); !ok {
		t.Fatalf("cache[1] = %T, want *cache.KVCache", caches[1])
	}
}

func TestRopeParametersDerivedFromConfig(t *testing.T) {
	cfg := denseTestConfig(t)

	base, scale, originalContext := cfg.RopeParameters()
	if base != cfg.RopeTheta {
		t.Fatalf("rope base = %v, want %v", base, cfg.RopeTheta)
	}
	if scale != 0.5 {
		t.Fatalf("rope scale = %v, want 0.5", scale)
	}
	if originalContext != 4 {
		t.Fatalf("original context = %d, want 4", originalContext)
	}
}

func TestForwardFailsBeforePartialExecution(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	m := &Model{
		Config:      &cfg,
		Layers:      make([]*Layer, cfg.NumHiddenLayers),
		EmbedTokens: nn.NewEmbedding(denseMatrix(int(cfg.VocabSize), int(cfg.HiddenSize), 1)),
		Norm:        nn.NewRMSNorm(denseVector(int(cfg.HiddenSize), 2), cfg.RMSNormEps),
	}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("Forward() did not panic, want explicit failure")
		}
		if !strings.Contains(fmt.Sprint(r), "unimplemented MLP/expert execution") {
			t.Fatalf("panic = %v, want explicit gpt-oss forward failure", r)
		}
	}()

	tokens := mlx.FromValues([]int32{1, 2}, 1, 2)
	m.Forward(tokens, nil)
}

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

func denseTestConfig(t *testing.T) Config {
	t.Helper()

	cfg, err := parseConfig([]byte(`{
		"architectures": ["GptOssForCausalLM"],
		"model_type": "gpt_oss",
		"num_hidden_layers": 2,
		"hidden_size": 4,
		"intermediate_size": 8,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 2,
		"num_local_experts": 2,
		"num_experts_per_tok": 1,
		"sliding_window": 4,
		"rope_theta": 150000,
		"rope_scaling": {
			"factor": 2.0,
			"original_max_position_embeddings": 4
		},
		"rms_norm_eps": 0.00001,
		"vocab_size": 8,
		"tie_word_embeddings": false,
		"quantization_config": {
			"quant_method": "mxfp4"
		}
	}`))
	if err != nil {
		t.Fatalf("parseConfig() error = %v", err)
	}

	return cfg
}

func denseTestTensors(t *testing.T, cfg Config) map[string]*mlx.Array {
	t.Helper()

	tensors := map[string]*mlx.Array{
		"embedding.weight":   denseMatrix(8, 4, 1),
		"output_norm.weight": denseVector(4, 1),
		"output.weight":      denseMatrix(8, 4, 2),
	}

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("blocks.%d", i)
		tensors[prefix+".attn_norm.weight"] = denseVector(int(cfg.HiddenSize), 3+float32(i))
		tensors[prefix+".q_proj.weight"] = denseMatrix(int(cfg.NumAttentionHeads*cfg.HeadDim), int(cfg.HiddenSize), 4+float32(i))
		tensors[prefix+".q_proj.bias"] = denseVector(int(cfg.NumAttentionHeads*cfg.HeadDim), 5+float32(i))
		tensors[prefix+".k_proj.weight"] = denseMatrix(int(cfg.NumKeyValueHeads*cfg.HeadDim), int(cfg.HiddenSize), 6+float32(i))
		tensors[prefix+".k_proj.bias"] = denseVector(int(cfg.NumKeyValueHeads*cfg.HeadDim), 7+float32(i))
		tensors[prefix+".v_proj.weight"] = denseMatrix(int(cfg.NumKeyValueHeads*cfg.HeadDim), int(cfg.HiddenSize), 8+float32(i))
		tensors[prefix+".v_proj.bias"] = denseVector(int(cfg.NumKeyValueHeads*cfg.HeadDim), 9+float32(i))
		tensors[prefix+".attn_out.weight"] = denseMatrix(int(cfg.HiddenSize), int(cfg.NumAttentionHeads*cfg.HeadDim), 10+float32(i))
		tensors[prefix+".attn_out.bias"] = denseVector(int(cfg.HiddenSize), 11+float32(i))
		tensors[prefix+".attn_sinks"] = denseVector(int(cfg.NumAttentionHeads), 12+float32(i))
		tensors[prefix+".ffn_norm.weight"] = denseVector(int(cfg.HiddenSize), 13+float32(i))
		tensors[prefix+".router.weight"] = denseMatrix(int(cfg.NumLocalExperts), int(cfg.HiddenSize), 14+float32(i))
		tensors[prefix+".router.bias"] = denseVector(int(cfg.NumLocalExperts), 15+float32(i))
	}

	return tensors
}

func denseMatrix(rows, cols int, start float32) *mlx.Array {
	values := make([]float32, rows*cols)
	for i := range values {
		values[i] = start + float32(i)
	}
	return mlx.FromValues(values, rows, cols)
}

func denseVector(length int, start float32) *mlx.Array {
	values := make([]float32, length)
	for i := range values {
		values[i] = start + float32(i)
	}
	return mlx.FromValues(values, length)
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
