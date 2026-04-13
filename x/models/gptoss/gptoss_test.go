package gptoss

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

func forwardModel(m *Model, tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	return m.Forward(&batch.Batch{
		InputIDs:   tokens,
		SeqOffsets: []int32{0},
	}, caches)
}

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
		if layer.Experts == nil || layer.Experts.GateUp == nil || layer.Experts.GateUp.Gate == nil || layer.Experts.GateUp.Up == nil || layer.Experts.Down == nil {
			t.Fatalf("layer %d Experts = %+v, want loaded expert projections", i, layer.Experts)
		}
		if got := layer.Experts.GateUp.Gate.Weight.DType(); got != mlx.DTypeBFloat16 {
			t.Fatalf("layer %d GateUp.Gate dtype = %v, want %v", i, got, mlx.DTypeBFloat16)
		}
		if got := layer.Experts.GateUp.Gate.Bias.DType(); got != mlx.DTypeBFloat16 {
			t.Fatalf("layer %d GateUp.Gate bias dtype = %v, want %v", i, got, mlx.DTypeBFloat16)
		}
		if got := layer.Experts.GateUp.Up.Weight.DType(); got != mlx.DTypeBFloat16 {
			t.Fatalf("layer %d GateUp.Up dtype = %v, want %v", i, got, mlx.DTypeBFloat16)
		}
		if got := layer.Experts.GateUp.Up.Bias.DType(); got != mlx.DTypeBFloat16 {
			t.Fatalf("layer %d GateUp.Up bias dtype = %v, want %v", i, got, mlx.DTypeBFloat16)
		}
		if got := layer.Experts.Down.Weight.DType(); got != mlx.DTypeBFloat16 {
			t.Fatalf("layer %d Down dtype = %v, want %v", i, got, mlx.DTypeBFloat16)
		}
		if got := layer.Experts.Down.Bias.DType(); got != mlx.DTypeBFloat16 {
			t.Fatalf("layer %d Down bias dtype = %v, want %v", i, got, mlx.DTypeBFloat16)
		}
		if dims := layer.Experts.GateUp.Gate.Weight.Dims(); len(dims) != 3 || dims[0] != int(cfg.NumLocalExperts) || dims[1] != int(cfg.HiddenSize) || dims[2] != int(cfg.IntermediateSize) {
			t.Fatalf("layer %d GateUp.Gate dims = %v, want [%d %d %d]", i, dims, cfg.NumLocalExperts, cfg.HiddenSize, cfg.IntermediateSize)
		}
		if dims := layer.Experts.GateUp.Gate.Bias.Dims(); len(dims) != 2 || dims[0] != int(cfg.NumLocalExperts) || dims[1] != int(cfg.IntermediateSize) {
			t.Fatalf("layer %d GateUp.Gate bias dims = %v, want [%d %d]", i, dims, cfg.NumLocalExperts, cfg.IntermediateSize)
		}
		if dims := layer.Experts.GateUp.Up.Weight.Dims(); len(dims) != 3 || dims[0] != int(cfg.NumLocalExperts) || dims[1] != int(cfg.HiddenSize) || dims[2] != int(cfg.IntermediateSize) {
			t.Fatalf("layer %d GateUp.Up dims = %v, want [%d %d %d]", i, dims, cfg.NumLocalExperts, cfg.HiddenSize, cfg.IntermediateSize)
		}
		if dims := layer.Experts.GateUp.Up.Bias.Dims(); len(dims) != 2 || dims[0] != int(cfg.NumLocalExperts) || dims[1] != int(cfg.IntermediateSize) {
			t.Fatalf("layer %d GateUp.Up bias dims = %v, want [%d %d]", i, dims, cfg.NumLocalExperts, cfg.IntermediateSize)
		}
		if dims := layer.Experts.Down.Weight.Dims(); len(dims) != 3 || dims[0] != int(cfg.NumLocalExperts) || dims[1] != int(cfg.IntermediateSize) || dims[2] != int(cfg.HiddenSize) {
			t.Fatalf("layer %d Down dims = %v, want [%d %d %d]", i, dims, cfg.NumLocalExperts, cfg.IntermediateSize, cfg.HiddenSize)
		}
		if dims := layer.Experts.Down.Bias.Dims(); len(dims) != 2 || dims[0] != int(cfg.NumLocalExperts) || dims[1] != int(cfg.HiddenSize) {
			t.Fatalf("layer %d Down bias dims = %v, want [%d %d]", i, dims, cfg.NumLocalExperts, cfg.HiddenSize)
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

func TestLoadWeightsExpertDTypeValidationFails(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}

	tensors := denseTestTensors(t, cfg)
	tensors["blocks.0.experts.gate_proj.weight"] = tensors["blocks.0.experts.gate_proj.weight"].AsType(mlx.DTypeFloat32)

	err := m.LoadWeights(tensors)
	if err == nil {
		t.Fatal("LoadWeights() error = nil, want expert dtype validation failure")
	}
	if !strings.Contains(err.Error(), "blocks.0.experts.gate_proj.weight") || !strings.Contains(err.Error(), "dtype F32, want BF16") {
		t.Fatalf("LoadWeights() error = %q, want expert dtype mismatch", err)
	}
}

func TestLoadWeightsExpertMissingTensorFails(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}

	tensors := denseTestTensors(t, cfg)
	delete(tensors, "blocks.0.experts.down_proj.bias")

	err := m.LoadWeights(tensors)
	if err == nil {
		t.Fatal("LoadWeights() error = nil, want expert missing tensor failure")
	}
	if !strings.Contains(err.Error(), "blocks.0.experts.down_proj.bias") || !strings.Contains(err.Error(), "missing direct expert tensor") {
		t.Fatalf("LoadWeights() error = %q, want missing expert tensor", err)
	}
}

func TestLoadWeightsExpertShapeValidationFails(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}

	tensors := denseTestTensors(t, cfg)
	tensors["blocks.0.experts.down_proj.weight"] = denseExpertWeight(int(cfg.NumLocalExperts), int(cfg.HiddenSize), int(cfg.IntermediateSize)-1, 99)

	err := m.LoadWeights(tensors)
	if err == nil {
		t.Fatal("LoadWeights() error = nil, want expert shape validation failure")
	}
	if !strings.Contains(err.Error(), "blocks.0.experts.down_proj.weight") || !strings.Contains(err.Error(), "shape [2 4 7]") {
		t.Fatalf("LoadWeights() error = %q, want expert shape mismatch", err)
	}
}

func TestSplitGateUpInterleavedUsesEvenOddOrdering(t *testing.T) {
	skipIfNoMLX(t)

	dense := mlx.FromValues([]float32{
		0, 1,
		10, 11,
		20, 21,
		30, 31,
		40, 41,
		50, 51,
	}, 6, 2).AsType(mlx.DTypeBFloat16)
	bias := mlx.FromValues([]float32{0, 10, 20, 30, 40, 50}, 6).AsType(mlx.DTypeBFloat16)

	gateWeight, upWeight, gateBias, upBias := splitGateUpInterleaved(dense, bias, 3)
	if gateWeight == nil || upWeight == nil || gateBias == nil || upBias == nil {
		t.Fatal("splitGateUpInterleaved() returned nil tensors")
	}

	if dims := gateWeight.Dims(); len(dims) != 2 || dims[0] != 3 || dims[1] != 2 {
		t.Fatalf("gateWeight dims = %v, want [3 2]", dims)
	}
	if dims := upWeight.Dims(); len(dims) != 2 || dims[0] != 3 || dims[1] != 2 {
		t.Fatalf("upWeight dims = %v, want [3 2]", dims)
	}

	gateWeightVals := materializedFloats(gateWeight.AsType(mlx.DTypeFloat32))
	upWeightVals := materializedFloats(upWeight.AsType(mlx.DTypeFloat32))
	gateBiasVals := materializedFloats(gateBias.AsType(mlx.DTypeFloat32))
	upBiasVals := materializedFloats(upBias.AsType(mlx.DTypeFloat32))
	if got := []float32{gateWeightVals[0], gateWeightVals[2], gateWeightVals[4]}; !slices.Equal(got, []float32{0, 20, 40}) {
		t.Fatalf("gateWeight first-column values = %v, want [0 20 40]", got)
	}
	if got := []float32{upWeightVals[0], upWeightVals[2], upWeightVals[4]}; !slices.Equal(got, []float32{10, 30, 50}) {
		t.Fatalf("upWeight first-column values = %v, want [10 30 50]", got)
	}
	if !slices.Equal(gateBiasVals, []float32{0, 20, 40}) {
		t.Fatalf("gateBias values = %v, want [0 20 40]", gateBiasVals)
	}
	if !slices.Equal(upBiasVals, []float32{10, 30, 50}) {
		t.Fatalf("upBias values = %v, want [10 30 50]", upBiasVals)
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

func TestBuildGPTOSSRoPEFreqsMatchesReference(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HeadDim = 64
	cfg.RopeScaling.Factor = 32
	cfg.RopeScaling.OriginalMaxPositionEmbeddings = 4096
	cfg.RopeScaling.BetaFast = 32
	cfg.RopeScaling.BetaSlow = 1
	gotTensor := buildGPTOSSRoPEFreqs(&cfg)
	if gotTensor == nil || !gotTensor.Valid() {
		t.Fatal("buildGPTOSSRoPEFreqs() returned invalid tensor")
	}

	got := materializedFloats(gotTensor.AsType(mlx.DTypeFloat32))
	want := referenceGPTOSSRoPEDenominators(&cfg)
	if len(got) != len(want) {
		t.Fatalf("rope frequency length = %d, want %d", len(got), len(want))
	}

	for i := range want {
		tol := 1e-5 * math.Max(1, math.Abs(float64(want[i])))
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Fatalf("rope frequency[%d] = %v, want %v (diff %v)", i, got[i], want[i], diff)
		}
	}
}

func TestForwardRunsCompletePath(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}
	caches := m.NewCaches()
	tokens := mlx.FromValues([]int32{1, 2}, 1, 2)
	out := forwardModel(m, tokens, caches)

	if out == nil || !out.Valid() {
		t.Fatal("Forward() returned invalid output")
	}
	if dims := out.Dims(); len(dims) != 3 || dims[0] != 1 || dims[1] != 2 || dims[2] != int(cfg.HiddenSize) {
		t.Fatalf("Forward() dims = %v, want [1 2 %d]", dims, cfg.HiddenSize)
	}
}

func TestForwardLastTokenMatchesPrefillStepPath(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 64
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2
	cfg.SlidingWindow = 16
	cfg.RopeTheta = 150000
	cfg.RopeScaling.Factor = 32
	cfg.RopeScaling.OriginalMaxPositionEmbeddings = 4096
	cfg.RopeScaling.BetaFast = 32
	cfg.RopeScaling.BetaSlow = 1

	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}

	tokens := mlx.FromValues([]int32{1, 2, 3, 4, 5, 6}, 1, 6)

	fullCaches := m.NewCaches()
	fullHidden := forwardModel(m, tokens, fullCaches)
	fullLogits := m.Unembed(fullHidden)
	fullLast := materializedFloats(fullLogits.Slice(mlx.Slice(), mlx.Slice(fullLogits.Dim(1)-1), mlx.Slice()).Squeeze(1).AsType(mlx.DTypeFloat32))

	stepCaches := m.NewCaches()
	forwardModel(m, mlx.FromValues([]int32{1, 2, 3, 4, 5}, 1, 5), stepCaches)
	stepHidden := forwardModel(m, mlx.FromValues([]int32{6}, 1, 1), stepCaches)
	stepLogits := m.Unembed(stepHidden)
	stepLast := materializedFloats(stepLogits.Squeeze(1).AsType(mlx.DTypeFloat32))

	if len(fullLast) != len(stepLast) {
		t.Fatalf("logit length mismatch: full=%d step=%d", len(fullLast), len(stepLast))
	}
	for i := range fullLast {
		if diff := math.Abs(float64(fullLast[i] - stepLast[i])); diff > 2e-1 {
			t.Fatalf("last-token logit[%d] = %v, want %v (diff %v)", i, stepLast[i], fullLast[i], diff)
		}
	}
}

func TestLayerLastTokenMatchesPrefillStepPath(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 64
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2
	cfg.SlidingWindow = 16
	cfg.RopeTheta = 150000
	cfg.RopeScaling.Factor = 32
	cfg.RopeScaling.OriginalMaxPositionEmbeddings = 4096
	cfg.RopeScaling.BetaFast = 32
	cfg.RopeScaling.BetaSlow = 1

	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}
	layer := m.Layers[0]

	xVals := make([]float32, 6*64)
	for i := range xVals {
		xVals[i] = float32((i%17)-8) / 8
	}

	fullCache := cache.NewRotatingKVCache(int(cfg.SlidingWindow))
	full := layer.Forward(mlx.FromValues(xVals, 1, 6, 64).AsType(mlx.DTypeBFloat16), fullCache, 1, 6, &cfg, 0)
	fullLast := materializedFloats(full.Slice(mlx.Slice(), mlx.Slice(full.Dim(1)-1), mlx.Slice()).Squeeze(1).AsType(mlx.DTypeFloat32))

	stepCache := cache.NewRotatingKVCache(int(cfg.SlidingWindow))
	layer.Forward(mlx.FromValues(xVals[:5*64], 1, 5, 64).AsType(mlx.DTypeBFloat16), stepCache, 1, 5, &cfg, 0)
	step := layer.Forward(mlx.FromValues(xVals[5*64:], 1, 1, 64).AsType(mlx.DTypeBFloat16), stepCache, 1, 1, &cfg, 0)
	stepLast := materializedFloats(step.Squeeze(1).AsType(mlx.DTypeFloat32))

	if len(fullLast) != len(stepLast) {
		t.Fatalf("layer output length mismatch: full=%d step=%d", len(fullLast), len(stepLast))
	}
	for i := range fullLast {
		if diff := math.Abs(float64(fullLast[i] - stepLast[i])); diff > 1e-2 {
			t.Fatalf("layer last-token output[%d] = %v, want %v (diff %v)", i, stepLast[i], fullLast[i], diff)
		}
	}
}

func TestUnembedLastTokenMatchesPrefillStepPath(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 64
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2
	cfg.SlidingWindow = 16

	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}

	hVals := make([]float32, 6*64)
	for i := range hVals {
		hVals[i] = float32((i%23)-11) / 11
	}

	full := m.Unembed(mlx.FromValues(hVals, 1, 6, 64).AsType(mlx.DTypeBFloat16))
	fullLast := materializedFloats(full.Slice(mlx.Slice(), mlx.Slice(full.Dim(1)-1), mlx.Slice()).Squeeze(1).AsType(mlx.DTypeFloat32))

	step := m.Unembed(mlx.FromValues(hVals[5*64:], 1, 1, 64).AsType(mlx.DTypeBFloat16))
	stepLast := materializedFloats(step.Squeeze(1).AsType(mlx.DTypeFloat32))

	if len(fullLast) != len(stepLast) {
		t.Fatalf("unembed output length mismatch: full=%d step=%d", len(fullLast), len(stepLast))
	}
	for i := range fullLast {
		if diff := math.Abs(float64(fullLast[i] - stepLast[i])); diff > 1e-2 {
			t.Fatalf("unembed last-token output[%d] = %v, want %v (diff %v)", i, stepLast[i], fullLast[i], diff)
		}
	}
}

func TestAttentionLastTokenMatchesPrefillStepPathLoadedLayer(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 64
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2
	cfg.SlidingWindow = 16
	cfg.RopeTheta = 150000
	cfg.RopeScaling.Factor = 32
	cfg.RopeScaling.OriginalMaxPositionEmbeddings = 4096
	cfg.RopeScaling.BetaFast = 32
	cfg.RopeScaling.BetaSlow = 1

	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}
	attn := m.Layers[0].Attention

	xVals := make([]float32, 6*64)
	for i := range xVals {
		xVals[i] = float32((i%17)-8) / 8
	}

	fullCache := cache.NewRotatingKVCache(int(cfg.SlidingWindow))
	full := attn.Forward(mlx.FromValues(xVals, 1, 6, 64).AsType(mlx.DTypeBFloat16), fullCache, 1, 6, &cfg, 0)
	fullLast := materializedFloats(full.Slice(mlx.Slice(), mlx.Slice(full.Dim(1)-1), mlx.Slice()).Squeeze(1).AsType(mlx.DTypeFloat32))

	stepCache := cache.NewRotatingKVCache(int(cfg.SlidingWindow))
	attn.Forward(mlx.FromValues(xVals[:5*64], 1, 5, 64).AsType(mlx.DTypeBFloat16), stepCache, 1, 5, &cfg, 0)
	step := attn.Forward(mlx.FromValues(xVals[5*64:], 1, 1, 64).AsType(mlx.DTypeBFloat16), stepCache, 1, 1, &cfg, 0)
	stepLast := materializedFloats(step.Squeeze(1).AsType(mlx.DTypeFloat32))

	if len(fullLast) != len(stepLast) {
		t.Fatalf("attention output length mismatch: full=%d step=%d", len(fullLast), len(stepLast))
	}
	for i := range fullLast {
		if diff := math.Abs(float64(fullLast[i] - stepLast[i])); diff > 1e-2 {
			t.Fatalf("loaded attention last-token output[%d] = %v, want %v (diff %v)", i, stepLast[i], fullLast[i], diff)
		}
	}
}

func TestAttentionLastTokenMatchesPrefillStepPathLoadedLayerCausalCache(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 64
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2
	cfg.SlidingWindow = 16
	cfg.RopeTheta = 150000
	cfg.RopeScaling.Factor = 32
	cfg.RopeScaling.OriginalMaxPositionEmbeddings = 4096
	cfg.RopeScaling.BetaFast = 32
	cfg.RopeScaling.BetaSlow = 1

	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}
	attn := m.Layers[0].Attention

	xVals := make([]float32, 6*64)
	for i := range xVals {
		xVals[i] = float32((i%17)-8) / 8
	}

	fullCache := cache.NewKVCache()
	full := attn.Forward(mlx.FromValues(xVals, 1, 6, 64).AsType(mlx.DTypeBFloat16), fullCache, 1, 6, &cfg, 0)
	fullLast := materializedFloats(full.Slice(mlx.Slice(), mlx.Slice(full.Dim(1)-1), mlx.Slice()).Squeeze(1).AsType(mlx.DTypeFloat32))

	stepCache := cache.NewKVCache()
	attn.Forward(mlx.FromValues(xVals[:5*64], 1, 5, 64).AsType(mlx.DTypeBFloat16), stepCache, 1, 5, &cfg, 0)
	step := attn.Forward(mlx.FromValues(xVals[5*64:], 1, 1, 64).AsType(mlx.DTypeBFloat16), stepCache, 1, 1, &cfg, 0)
	stepLast := materializedFloats(step.Squeeze(1).AsType(mlx.DTypeFloat32))

	if len(fullLast) != len(stepLast) {
		t.Fatalf("causal attention output length mismatch: full=%d step=%d", len(fullLast), len(stepLast))
	}
	for i := range fullLast {
		if diff := math.Abs(float64(fullLast[i] - stepLast[i])); diff > 1e-2 {
			t.Fatalf("loaded causal attention last-token output[%d] = %v, want %v (diff %v)", i, stepLast[i], fullLast[i], diff)
		}
	}
}

func TestQProjLastTokenMatchesBatchPathLoadedLayer(t *testing.T) {
	skipIfNoMLX(t)
	t.Skip("diagnostic only: MLX batched affine path diverges on macOS; gpt-oss runtime avoids this path")

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 64
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2

	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}
	qproj := m.Layers[0].Attention.QProj

	xVals := make([]float32, 6*64)
	for i := range xVals {
		xVals[i] = float32((i%17)-8) / 8
	}

	full := qproj.Forward(mlx.FromValues(xVals, 1, 6, 64).AsType(mlx.DTypeBFloat16))
	fullLast := materializedFloats(full.Slice(mlx.Slice(), mlx.Slice(full.Dim(1)-1), mlx.Slice()).Squeeze(1).AsType(mlx.DTypeFloat32))

	step := qproj.Forward(mlx.FromValues(xVals[5*64:], 1, 1, 64).AsType(mlx.DTypeBFloat16))
	stepLast := materializedFloats(step.Squeeze(1).AsType(mlx.DTypeFloat32))

	if len(fullLast) != len(stepLast) {
		t.Fatalf("q_proj output length mismatch: full=%d step=%d", len(fullLast), len(stepLast))
	}
	for i := range fullLast {
		if diff := math.Abs(float64(fullLast[i] - stepLast[i])); diff > 1e-3 {
			t.Fatalf("q_proj last-token output[%d] = %v, want %v (diff %v)", i, stepLast[i], fullLast[i], diff)
		}
	}
}

func TestQProjLoadedLayerMatchesExplicitAffine(t *testing.T) {
	skipIfNoMLX(t)
	t.Skip("diagnostic only: MLX batched affine path diverges on macOS; gpt-oss runtime avoids this path")

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 64
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2

	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}
	qproj := m.Layers[0].Attention.QProj.(*nn.Linear)

	xVals := make([]float32, 6*64)
	for i := range xVals {
		xVals[i] = float32((i%17)-8) / 8
	}
	x := mlx.FromValues(xVals, 1, 6, 64).AsType(mlx.DTypeBFloat16)

	got := qproj.Forward(x)
	want := x.Matmul(qproj.Weight.Transpose(1, 0)).Add(qproj.Bias)

	gotVals := materializedFloats(got.AsType(mlx.DTypeFloat32))
	wantVals := materializedFloats(want.AsType(mlx.DTypeFloat32))
	if len(gotVals) != len(wantVals) {
		t.Fatalf("q_proj explicit affine length mismatch: got=%d want=%d", len(gotVals), len(wantVals))
	}
	for i := range gotVals {
		if diff := math.Abs(float64(gotVals[i] - wantVals[i])); diff > 1e-3 {
			t.Fatalf("q_proj explicit affine output[%d] = %v, want %v (diff %v)", i, gotVals[i], wantVals[i], diff)
		}
	}
}

func TestQProjExplicitAffineMatchesCPUReference(t *testing.T) {
	skipIfNoMLX(t)
	t.Skip("diagnostic only: MLX batched affine path diverges on macOS; gpt-oss runtime avoids this path")

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 64
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2

	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}
	qproj := m.Layers[0].Attention.QProj.(*nn.Linear)

	xVals := make([]float32, 6*64)
	for i := range xVals {
		xVals[i] = float32((i%17)-8) / 8
	}
	lastX := xVals[5*64:]

	weightVals := materializedFloats(qproj.Weight.AsType(mlx.DTypeFloat32))
	biasVals := materializedFloats(qproj.Bias.AsType(mlx.DTypeFloat32))
	w := make([][]float32, 64)
	for in := 0; in < 64; in++ {
		w[in] = make([]float32, 64)
		for out := 0; out < 64; out++ {
			w[in][out] = weightVals[out*64+in]
		}
	}
	want := affineRef(lastX, w, biasVals)

	fullX := mlx.FromValues(xVals, 1, 6, 64).AsType(mlx.DTypeBFloat16)
	full := fullX.Matmul(qproj.Weight.Transpose(1, 0)).Add(qproj.Bias)
	fullLast := materializedFloats(full.Slice(mlx.Slice(), mlx.Slice(full.Dim(1)-1), mlx.Slice()).Squeeze(1).AsType(mlx.DTypeFloat32))

	stepX := mlx.FromValues(lastX, 1, 1, 64).AsType(mlx.DTypeBFloat16)
	step := stepX.Matmul(qproj.Weight.Transpose(1, 0)).Add(qproj.Bias)
	stepLast := materializedFloats(step.Squeeze(1).AsType(mlx.DTypeFloat32))

	for i := range want {
		if diff := math.Abs(float64(fullLast[i] - want[i])); diff > 1e-2 {
			t.Fatalf("explicit affine batch output[%d] = %v, want %v (diff %v)", i, fullLast[i], want[i], diff)
		}
		if diff := math.Abs(float64(stepLast[i] - want[i])); diff > 1e-2 {
			t.Fatalf("explicit affine step output[%d] = %v, want %v (diff %v)", i, stepLast[i], want[i], diff)
		}
	}
}

func TestQProjExplicitAffineFloat32MatchesCPUReference(t *testing.T) {
	skipIfNoMLX(t)
	t.Skip("diagnostic only: MLX batched affine path diverges on macOS; gpt-oss runtime avoids this path")

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 64
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2

	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}
	qproj := m.Layers[0].Attention.QProj.(*nn.Linear)

	xVals := make([]float32, 6*64)
	for i := range xVals {
		xVals[i] = float32((i%17)-8) / 8
	}
	lastX := xVals[5*64:]

	weightVals := materializedFloats(qproj.Weight.AsType(mlx.DTypeFloat32))
	biasVals := materializedFloats(qproj.Bias.AsType(mlx.DTypeFloat32))
	w := make([][]float32, 64)
	for in := 0; in < 64; in++ {
		w[in] = make([]float32, 64)
		for out := 0; out < 64; out++ {
			w[in][out] = weightVals[out*64+in]
		}
	}
	want := affineRef(lastX, w, biasVals)

	fullX := mlx.FromValues(xVals, 1, 6, 64).AsType(mlx.DTypeFloat32)
	fullW := qproj.Weight.Transpose(1, 0).AsType(mlx.DTypeFloat32)
	fullB := qproj.Bias.AsType(mlx.DTypeFloat32)
	full := fullX.Matmul(fullW).Add(fullB)
	fullLast := materializedFloats(full.Slice(mlx.Slice(), mlx.Slice(full.Dim(1)-1), mlx.Slice()).Squeeze(1))

	stepX := mlx.FromValues(lastX, 1, 1, 64).AsType(mlx.DTypeFloat32)
	step := stepX.Matmul(fullW).Add(fullB)
	stepLast := materializedFloats(step.Squeeze(1))

	for i := range want {
		if diff := math.Abs(float64(fullLast[i] - want[i])); diff > 1e-3 {
			t.Fatalf("float32 affine batch output[%d] = %v, want %v (diff %v)", i, fullLast[i], want[i], diff)
		}
		if diff := math.Abs(float64(stepLast[i] - want[i])); diff > 1e-3 {
			t.Fatalf("float32 affine step output[%d] = %v, want %v (diff %v)", i, stepLast[i], want[i], diff)
		}
	}
}

func TestQProjExplicitAffine2DAndContiguousMatchCPUReference(t *testing.T) {
	skipIfNoMLX(t)
	t.Skip("diagnostic only: MLX batched affine path diverges on macOS; gpt-oss runtime avoids this path")

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 64
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2

	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}
	qproj := m.Layers[0].Attention.QProj.(*nn.Linear)

	xVals := make([]float32, 6*64)
	for i := range xVals {
		xVals[i] = float32((i%17)-8) / 8
	}
	lastX := xVals[5*64:]

	weightVals := materializedFloats(qproj.Weight.AsType(mlx.DTypeFloat32))
	biasVals := materializedFloats(qproj.Bias.AsType(mlx.DTypeFloat32))
	w := make([][]float32, 64)
	for in := 0; in < 64; in++ {
		w[in] = make([]float32, 64)
		for out := 0; out < 64; out++ {
			w[in][out] = weightVals[out*64+in]
		}
	}
	want := affineRef(lastX, w, biasVals)

	wT := qproj.Weight.Transpose(1, 0).AsType(mlx.DTypeFloat32)
	bias := qproj.Bias.AsType(mlx.DTypeFloat32)

	full3D := mlx.FromValues(xVals, 1, 6, 64).AsType(mlx.DTypeFloat32)
	out3D := full3D.Matmul(wT).Add(bias)
	last3D := materializedFloats(out3D.Slice(mlx.Slice(), mlx.Slice(out3D.Dim(1)-1), mlx.Slice()).Squeeze(1))

	full2D := mlx.FromValues(xVals, 6, 64).AsType(mlx.DTypeFloat32)
	out2D := full2D.Matmul(wT).Add(bias)
	last2D := materializedFloats(out2D.Slice(mlx.Slice(out2D.Dim(0)-1), mlx.Slice()).Squeeze(0))

	full2DContig := mlx.Contiguous(full2D, false)
	wTContig := mlx.Contiguous(wT, false)
	out2DContig := full2DContig.Matmul(wTContig).Add(bias)
	last2DContig := materializedFloats(out2DContig.Slice(mlx.Slice(out2DContig.Dim(0)-1), mlx.Slice()).Squeeze(0))

	step2D := mlx.FromValues(lastX, 1, 64).AsType(mlx.DTypeFloat32)
	outStep2D := step2D.Matmul(wT).Add(bias)
	lastStep2D := materializedFloats(outStep2D.Squeeze(0))

	check := func(label string, got []float32) {
		t.Helper()
		for i := range want {
			if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-3 {
				t.Errorf("%s output[%d] = %v, want %v (diff %v)", label, i, got[i], want[i], diff)
				return
			}
		}
	}

	check("3D affine", last3D)
	check("2D affine", last2D)
	check("2D contiguous affine", last2DContig)
	check("2D step affine", lastStep2D)
	if t.Failed() {
		t.FailNow()
	}
}

func TestQProjExplicitAffineBatch2DDiffersFromRowWise2D(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 64
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2

	m := &Model{
		Config: &cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}
	if err := m.LoadWeights(denseTestTensors(t, cfg)); err != nil {
		t.Fatalf("LoadWeights() error = %v", err)
	}
	qproj := m.Layers[0].Attention.QProj.(*nn.Linear)

	xVals := make([]float32, 6*64)
	for i := range xVals {
		xVals[i] = float32((i%17)-8) / 8
	}

	wT := qproj.Weight.Transpose(1, 0).AsType(mlx.DTypeFloat32)
	bias := qproj.Bias.AsType(mlx.DTypeFloat32)

	full2D := mlx.FromValues(xVals, 6, 64).AsType(mlx.DTypeFloat32)
	batched := full2D.Matmul(wT).Add(bias)
	batchedVals := materializedFloats(batched)

	rowWiseParts := make([]*mlx.Array, 0, 6)
	for row := 0; row < 6; row++ {
		xRow := mlx.FromValues(xVals[row*64:(row+1)*64], 1, 64).AsType(mlx.DTypeFloat32)
		rowWiseParts = append(rowWiseParts, xRow.Matmul(wT).Add(bias))
	}
	rowWise := mlx.Concatenate(rowWiseParts, 0)
	rowWiseVals := materializedFloats(rowWise)

	if len(batchedVals) != len(rowWiseVals) {
		t.Fatalf("batched vs row-wise length mismatch: got=%d want=%d", len(batchedVals), len(rowWiseVals))
	}

	foundDiff := false
	for i := range batchedVals {
		if diff := math.Abs(float64(batchedVals[i] - rowWiseVals[i])); diff > 1e-3 {
			foundDiff = true
			break
		}
	}
	if !foundDiff {
		t.Fatalf("expected batched 2D affine to diverge from row-wise 2D affine on macOS MLX, but it matched")
	}
}

func TestAttentionLastTokenMatchesPrefillStepPathScaledRoPE(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.HeadDim = 64
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.SlidingWindow = 16
	cfg.RopeTheta = 150000
	cfg.RopeScaling.Factor = 32
	cfg.RopeScaling.OriginalMaxPositionEmbeddings = 4096
	cfg.RopeScaling.BetaFast = 32
	cfg.RopeScaling.BetaSlow = 1

	xVals := make([]float32, 6*64)
	for i := range xVals {
		xVals[i] = float32((i%17)-8) / 8
	}
	x := mlx.FromValues(xVals, 1, 6, 64)

	attn := &Attention{
		QProj:     nnLinearFromValues(identityFlat(64), nil, 64, 64),
		KProj:     nnLinearFromValues(identityFlat(64), nil, 64, 64),
		VProj:     nnLinearFromValues(identityFlat(64), nil, 64, 64),
		OProj:     nnLinearFromValues(identityFlat(64), nil, 64, 64),
		Sinks:     mlx.FromValues([]float32{0.3}, 1).AsType(mlx.DTypeBFloat16),
		RoPEFreqs: buildGPTOSSRoPEFreqs(&cfg),
	}

	fullCache := cache.NewRotatingKVCache(int(cfg.SlidingWindow))
	full := attn.Forward(x, fullCache, 1, 6, &cfg, 0)
	fullLast := materializedFloats(full.Slice(mlx.Slice(), mlx.Slice(full.Dim(1)-1), mlx.Slice()).Squeeze(1).AsType(mlx.DTypeFloat32))

	stepCache := cache.NewRotatingKVCache(int(cfg.SlidingWindow))
	attn.Forward(mlx.FromValues(xVals[:5*64], 1, 5, 64), stepCache, 1, 5, &cfg, 0)
	step := attn.Forward(mlx.FromValues(xVals[5*64:], 1, 1, 64), stepCache, 1, 1, &cfg, 0)
	stepLast := materializedFloats(step.Squeeze(1).AsType(mlx.DTypeFloat32))

	if len(fullLast) != len(stepLast) {
		t.Fatalf("attention output length mismatch: full=%d step=%d", len(fullLast), len(stepLast))
	}
	for i := range fullLast {
		if diff := math.Abs(float64(fullLast[i] - stepLast[i])); diff > 3e-2 {
			t.Fatalf("attention last-token output[%d] = %v, want %v (diff %v)", i, stepLast[i], fullLast[i], diff)
		}
	}
}

func TestExpertsLastTokenMatchesBatchPath(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.IntermediateSize = 128
	cfg.NumLocalExperts = 4
	cfg.NumExpertsPerTok = 2

	xVals := make([]float32, 6*64)
	routerVals := make([]float32, 6*int(cfg.NumLocalExperts))
	for i := range xVals {
		xVals[i] = float32((i%19)-9) / 9
	}
	for i := range routerVals {
		routerVals[i] = float32((i%11)-5) / 4
	}

	experts := &Experts{
		GateUp: &ExpertPair{
			Gate: &ExpertProjection{
				Weight: denseExpertWeight(int(cfg.NumLocalExperts), int(cfg.HiddenSize), int(cfg.IntermediateSize), 1).AsType(mlx.DTypeBFloat16),
				Bias:   expertBias(int(cfg.NumLocalExperts), int(cfg.IntermediateSize), 0).AsType(mlx.DTypeBFloat16),
			},
			Up: &ExpertProjection{
				Weight: denseExpertWeight(int(cfg.NumLocalExperts), int(cfg.HiddenSize), int(cfg.IntermediateSize), 10).AsType(mlx.DTypeBFloat16),
				Bias:   expertBias(int(cfg.NumLocalExperts), int(cfg.IntermediateSize), 0.5).AsType(mlx.DTypeBFloat16),
			},
		},
		Down: &ExpertProjection{
			Weight: denseExpertWeight(int(cfg.NumLocalExperts), int(cfg.IntermediateSize), int(cfg.HiddenSize), 20).AsType(mlx.DTypeBFloat16),
			Bias:   expertBias(int(cfg.NumLocalExperts), int(cfg.HiddenSize), 1).AsType(mlx.DTypeBFloat16),
		},
	}

	full := experts.Forward(
		mlx.FromValues(xVals, 1, 6, int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16),
		mlx.FromValues(routerVals, 1, 6, int(cfg.NumLocalExperts)).AsType(mlx.DTypeBFloat16),
		&cfg,
		0,
	)
	fullLast := materializedFloats(full.Slice(mlx.Slice(), mlx.Slice(full.Dim(1)-1), mlx.Slice()).Squeeze(1).AsType(mlx.DTypeFloat32))

	step := experts.Forward(
		mlx.FromValues(xVals[5*64:], 1, 1, int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16),
		mlx.FromValues(routerVals[5*int(cfg.NumLocalExperts):], 1, 1, int(cfg.NumLocalExperts)).AsType(mlx.DTypeBFloat16),
		&cfg,
		0,
	)
	stepLast := materializedFloats(step.Squeeze(1).AsType(mlx.DTypeFloat32))

	if len(fullLast) != len(stepLast) {
		t.Fatalf("expert output length mismatch: full=%d step=%d", len(fullLast), len(stepLast))
	}
	for i := range fullLast {
		if diff := math.Abs(float64(fullLast[i] - stepLast[i])); diff > 5e-2 {
			t.Fatalf("expert last-token output[%d] = %v, want %v (diff %v)", i, stepLast[i], fullLast[i], diff)
		}
	}
}

func TestExpertsForwardMatchesReference(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 2
	cfg.IntermediateSize = 2
	cfg.NumLocalExperts = 2
	cfg.NumExpertsPerTok = 2

	xVals := []float32{0.5, -1.0}
	routerVals := []float32{2.0, 1.0}
	x := mlx.FromValues(xVals, 1, 1, int(cfg.HiddenSize))
	router := mlx.FromValues(routerVals, 1, 1, int(cfg.NumLocalExperts))

	experts := &Experts{
		GateUp: &ExpertPair{
			Gate: &ExpertProjection{
				Weight: mlx.FromValues([]float32{
					1.0, 0.0,
					0.0, 1.0,

					-0.5, 1.0,
					1.5, -1.0,
				}, int(cfg.NumLocalExperts), int(cfg.HiddenSize), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
				Bias: mlx.FromValues([]float32{
					0.1, -0.2,
					0.3, 0.4,
				}, int(cfg.NumLocalExperts), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
			},
			Up: &ExpertProjection{
				Weight: mlx.FromValues([]float32{
					2.0, 0.0,
					0.0, 2.0,

					1.0, -1.0,
					0.5, 1.5,
				}, int(cfg.NumLocalExperts), int(cfg.HiddenSize), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
				Bias: mlx.FromValues([]float32{
					0.3, 0.4,
					-0.2, 0.1,
				}, int(cfg.NumLocalExperts), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
			},
		},
		Down: &ExpertProjection{
			Weight: mlx.FromValues([]float32{
				1.0, 0.0,
				0.0, 1.0,

				0.5, -1.0,
				1.0, 0.5,
			}, int(cfg.NumLocalExperts), int(cfg.IntermediateSize), int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16),
			Bias: mlx.FromValues([]float32{
				0.05, -0.05,
				-0.1, 0.2,
			}, int(cfg.NumLocalExperts), int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16),
		},
	}

	got := experts.Forward(x, router, &cfg, 0)
	if got == nil || !got.Valid() {
		t.Fatal("Experts.Forward() returned invalid tensor")
	}
	gotVals := materializedFloats(got.AsType(mlx.DTypeFloat32))

	wantVals := referenceExpertsForward(
		xVals,
		routerVals,
		int(cfg.NumExpertsPerTok),
		[][][]float32{
			{{1.0, 0.0}, {0.0, 1.0}},
			{{-0.5, 1.0}, {1.5, -1.0}},
		},
		[][]float32{
			{0.1, -0.2},
			{0.3, 0.4},
		},
		[][][]float32{
			{{2.0, 0.0}, {0.0, 2.0}},
			{{1.0, -1.0}, {0.5, 1.5}},
		},
		[][]float32{
			{0.3, 0.4},
			{-0.2, 0.1},
		},
		[][][]float32{
			{{1.0, 0.0}, {0.0, 1.0}},
			{{0.5, -1.0}, {1.0, 0.5}},
		},
		[][]float32{
			{0.05, -0.05},
			{-0.1, 0.2},
		},
	)

	if len(gotVals) != len(wantVals) {
		t.Fatalf("Experts.Forward() output length = %d, want %d", len(gotVals), len(wantVals))
	}
	for i := range wantVals {
		if diff := math.Abs(float64(gotVals[i] - wantVals[i])); diff > 1e-2 {
			t.Fatalf("Experts.Forward() output[%d] = %v, want %v (diff %v)", i, gotVals[i], wantVals[i], diff)
		}
	}
}

func TestExpertsForwardMatchesReferenceSortedPath(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 2
	cfg.IntermediateSize = 2
	cfg.NumLocalExperts = 3
	cfg.NumExpertsPerTok = 2

	const seqLen = 64
	xVals := make([]float32, seqLen*int(cfg.HiddenSize))
	routerVals := make([]float32, seqLen*int(cfg.NumLocalExperts))
	for i := 0; i < seqLen; i++ {
		xVals[i*2+0] = float32(i%7)/7 + 0.1
		xVals[i*2+1] = float32((i*3)%11)/11 - 0.2

		routerVals[i*3+0] = float32((i%5)-2) * 0.4
		routerVals[i*3+1] = float32(((i+2)%7)-3) * 0.3
		routerVals[i*3+2] = float32(((i*2)%9)-4) * 0.2
	}

	x := mlx.FromValues(xVals, 1, seqLen, int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16)
	router := mlx.FromValues(routerVals, 1, seqLen, int(cfg.NumLocalExperts)).AsType(mlx.DTypeBFloat16)

	experts := &Experts{
		GateUp: &ExpertPair{
			Gate: &ExpertProjection{
				Weight: mlx.FromValues([]float32{
					1.0, 0.0,
					0.2, 0.8,

					-0.4, 0.9,
					1.1, -0.3,

					0.7, -0.2,
					0.5, 1.2,
				}, int(cfg.NumLocalExperts), int(cfg.HiddenSize), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
				Bias: mlx.FromValues([]float32{
					0.1, -0.2,
					0.0, 0.05,
					-0.1, 0.2,
				}, int(cfg.NumLocalExperts), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
			},
			Up: &ExpertProjection{
				Weight: mlx.FromValues([]float32{
					0.9, 0.1,
					-0.3, 1.1,

					1.4, -0.6,
					0.2, 0.7,

					-0.5, 0.8,
					1.0, 0.4,
				}, int(cfg.NumLocalExperts), int(cfg.HiddenSize), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
				Bias: mlx.FromValues([]float32{
					0.0, 0.15,
					-0.05, 0.1,
					0.2, -0.1,
				}, int(cfg.NumLocalExperts), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
			},
		},
		Down: &ExpertProjection{
			Weight: mlx.FromValues([]float32{
				0.8, -0.2,
				0.1, 1.0,

				1.2, 0.3,
				-0.6, 0.9,

				0.4, 1.1,
				0.7, -0.5,
			}, int(cfg.NumLocalExperts), int(cfg.IntermediateSize), int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16),
			Bias: mlx.FromValues([]float32{
				0.05, -0.05,
				0.1, 0.0,
				-0.15, 0.2,
			}, int(cfg.NumLocalExperts), int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16),
		},
	}

	got := experts.Forward(x, router, &cfg, 0)
	if got == nil || !got.Valid() {
		t.Fatal("Experts.Forward() returned invalid tensor")
	}
	gotVals := materializedFloats(got.AsType(mlx.DTypeFloat32))

	wantVals := make([]float32, 0, len(gotVals))
	for i := 0; i < seqLen; i++ {
		wantVals = append(wantVals, referenceExpertsForward(
			xVals[i*2:(i+1)*2],
			routerVals[i*3:(i+1)*3],
			int(cfg.NumExpertsPerTok),
			[][][]float32{
				{{1.0, 0.0}, {0.2, 0.8}},
				{{-0.4, 0.9}, {1.1, -0.3}},
				{{0.7, -0.2}, {0.5, 1.2}},
			},
			[][]float32{
				{0.1, -0.2},
				{0.0, 0.05},
				{-0.1, 0.2},
			},
			[][][]float32{
				{{0.9, 0.1}, {-0.3, 1.1}},
				{{1.4, -0.6}, {0.2, 0.7}},
				{{-0.5, 0.8}, {1.0, 0.4}},
			},
			[][]float32{
				{0.0, 0.15},
				{-0.05, 0.1},
				{0.2, -0.1},
			},
			[][][]float32{
				{{0.8, -0.2}, {0.1, 1.0}},
				{{1.2, 0.3}, {-0.6, 0.9}},
				{{0.4, 1.1}, {0.7, -0.5}},
			},
			[][]float32{
				{0.05, -0.05},
				{0.1, 0.0},
				{-0.15, 0.2},
			},
		)...)
	}

	if len(gotVals) != len(wantVals) {
		t.Fatalf("Experts.Forward() output length = %d, want %d", len(gotVals), len(wantVals))
	}
	for i := range wantVals {
		if diff := math.Abs(float64(gotVals[i] - wantVals[i])); diff > 2e-2 {
			t.Fatalf("Experts.Forward() sorted output[%d] = %v, want %v (diff %v)", i, gotVals[i], wantVals[i], diff)
		}
	}
}

func TestExpertsForwardMatchesReferencePromptLength63(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 2
	cfg.IntermediateSize = 2
	cfg.NumLocalExperts = 3
	cfg.NumExpertsPerTok = 2

	const seqLen = 63
	xVals := make([]float32, seqLen*int(cfg.HiddenSize))
	routerVals := make([]float32, seqLen*int(cfg.NumLocalExperts))
	for i := 0; i < seqLen; i++ {
		xVals[i*2+0] = float32(i%7)/7 + 0.1
		xVals[i*2+1] = float32((i*3)%11)/11 - 0.2

		routerVals[i*3+0] = float32((i%5)-2) * 0.4
		routerVals[i*3+1] = float32(((i+2)%7)-3) * 0.3
		routerVals[i*3+2] = float32(((i*2)%9)-4) * 0.2
	}

	x := mlx.FromValues(xVals, 1, seqLen, int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16)
	router := mlx.FromValues(routerVals, 1, seqLen, int(cfg.NumLocalExperts)).AsType(mlx.DTypeBFloat16)

	experts := &Experts{
		GateUp: &ExpertPair{
			Gate: &ExpertProjection{
				Weight: mlx.FromValues([]float32{
					1.0, 0.0,
					0.2, 0.8,

					-0.4, 0.9,
					1.1, -0.3,

					0.7, -0.2,
					0.5, 1.2,
				}, int(cfg.NumLocalExperts), int(cfg.HiddenSize), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
				Bias: mlx.FromValues([]float32{
					0.1, -0.2,
					0.0, 0.05,
					-0.1, 0.2,
				}, int(cfg.NumLocalExperts), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
			},
			Up: &ExpertProjection{
				Weight: mlx.FromValues([]float32{
					0.9, 0.1,
					-0.3, 1.1,

					1.4, -0.6,
					0.2, 0.7,

					-0.5, 0.8,
					1.0, 0.4,
				}, int(cfg.NumLocalExperts), int(cfg.HiddenSize), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
				Bias: mlx.FromValues([]float32{
					0.0, 0.15,
					-0.05, 0.1,
					0.2, -0.1,
				}, int(cfg.NumLocalExperts), int(cfg.IntermediateSize)).AsType(mlx.DTypeBFloat16),
			},
		},
		Down: &ExpertProjection{
			Weight: mlx.FromValues([]float32{
				0.8, -0.2,
				0.1, 1.0,

				1.2, 0.3,
				-0.6, 0.9,

				0.4, 1.1,
				0.7, -0.5,
			}, int(cfg.NumLocalExperts), int(cfg.IntermediateSize), int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16),
			Bias: mlx.FromValues([]float32{
				0.05, -0.05,
				0.1, 0.0,
				-0.15, 0.2,
			}, int(cfg.NumLocalExperts), int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16),
		},
	}

	got := experts.Forward(x, router, &cfg, 0)
	if got == nil || !got.Valid() {
		t.Fatal("Experts.Forward() returned invalid tensor")
	}
	gotVals := materializedFloats(got.AsType(mlx.DTypeFloat32))

	wantVals := make([]float32, 0, len(gotVals))
	for i := 0; i < seqLen; i++ {
		wantVals = append(wantVals, referenceExpertsForward(
			xVals[i*2:(i+1)*2],
			routerVals[i*3:(i+1)*3],
			int(cfg.NumExpertsPerTok),
			[][][]float32{
				{{1.0, 0.0}, {0.2, 0.8}},
				{{-0.4, 0.9}, {1.1, -0.3}},
				{{0.7, -0.2}, {0.5, 1.2}},
			},
			[][]float32{
				{0.1, -0.2},
				{0.0, 0.05},
				{-0.1, 0.2},
			},
			[][][]float32{
				{{0.9, 0.1}, {-0.3, 1.1}},
				{{1.4, -0.6}, {0.2, 0.7}},
				{{-0.5, 0.8}, {1.0, 0.4}},
			},
			[][]float32{
				{0.0, 0.15},
				{-0.05, 0.1},
				{0.2, -0.1},
			},
			[][][]float32{
				{{0.8, -0.2}, {0.1, 1.0}},
				{{1.2, 0.3}, {-0.6, 0.9}},
				{{0.4, 1.1}, {0.7, -0.5}},
			},
			[][]float32{
				{0.05, -0.05},
				{0.1, 0.0},
				{-0.15, 0.2},
			},
		)...)
	}

	if len(gotVals) != len(wantVals) {
		t.Fatalf("Experts.Forward() output length = %d, want %d", len(gotVals), len(wantVals))
	}
	for i := range wantVals {
		if diff := math.Abs(float64(gotVals[i] - wantVals[i])); diff > 2e-2 {
			t.Fatalf("Experts.Forward() prompt-length-63 output[%d] = %v, want %v (diff %v)", i, gotVals[i], wantVals[i], diff)
		}
	}
}

func TestAttentionForwardMatchesReference(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 2
	cfg.HeadDim = 2
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.RopeTheta = 10000
	cfg.RopeScaling.Factor = 1
	cfg.RopeScaling.OriginalMaxPositionEmbeddings = 4

	xVals := []float32{
		1.0, 0.5,
		-0.25, 0.75,
	}
	x := mlx.FromValues(xVals, 1, 2, 2)

	attn := &Attention{
		QProj: nnLinearFromValues([]float32{
			1, 0,
			0, 1,
		}, nil, 2, 2),
		KProj: nnLinearFromValues([]float32{
			0.5, 0,
			0, 1.5,
		}, []float32{0.1, -0.2}, 2, 2),
		VProj: nnLinearFromValues([]float32{
			1.2, 0,
			0, 0.8,
		}, []float32{-0.05, 0.2}, 2, 2),
		OProj: nnLinearFromValues([]float32{
			1, 0,
			0, 1,
		}, []float32{0.01, -0.02}, 2, 2),
	}

	got := attn.Forward(x, nil, 1, 2, &cfg, 0)
	if got == nil || !got.Valid() {
		t.Fatal("Attention.Forward() returned invalid tensor")
	}

	want := referenceAttentionForward(
		xVals,
		[][]float32{{1, 0}, {0, 1}},
		nil,
		[][]float32{{0.5, 0}, {0, 1.5}},
		[]float32{0.1, -0.2},
		[][]float32{{1.2, 0}, {0, 0.8}},
		[]float32{-0.05, 0.2},
		[][]float32{{1, 0}, {0, 1}},
		[]float32{0.01, -0.02},
		cfg.RopeTheta,
	)

	gotVals := materializedFloats(got.AsType(mlx.DTypeFloat32))
	if len(gotVals) != len(want) {
		t.Fatalf("Attention.Forward() output length = %d, want %d", len(gotVals), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(gotVals[i] - want[i])); diff > 2e-1 {
			t.Fatalf("Attention.Forward() output[%d] = %v, want %v (diff %v)", i, gotVals[i], want[i], diff)
		}
	}
}

func TestAttentionForwardMatchesReferenceWithSinksAndScaledRoPE(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 64
	cfg.HeadDim = 64
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.RopeTheta = 150000
	cfg.RopeScaling.Factor = 32
	cfg.RopeScaling.OriginalMaxPositionEmbeddings = 4096
	cfg.RopeScaling.BetaFast = 32
	cfg.RopeScaling.BetaSlow = 1

	xVals := make([]float32, 2*64)
	for i := range xVals {
		xVals[i] = float32((i%17)-8) / 8
	}
	x := mlx.FromValues(xVals, 1, 2, 64)

	attn := &Attention{
		QProj:     nnLinearFromValues(identityFlat(64), nil, 64, 64),
		KProj:     nnLinearFromValues(identityFlat(64), nil, 64, 64),
		VProj:     nnLinearFromValues(identityFlat(64), nil, 64, 64),
		OProj:     nnLinearFromValues(identityFlat(64), nil, 64, 64),
		Sinks:     mlx.FromValues([]float32{0.3}, 1).AsType(mlx.DTypeBFloat16),
		RoPEFreqs: buildGPTOSSRoPEFreqs(&cfg),
	}

	got := attn.Forward(x, nil, 1, 2, &cfg, 0)
	if got == nil || !got.Valid() {
		t.Fatal("Attention.Forward() returned invalid tensor")
	}
	gotVals := materializedFloats(got.AsType(mlx.DTypeFloat32))

	wantVals := referenceAttentionForwardWithSinksScaledRoPE(xVals, &cfg, 0.3)
	if len(gotVals) != len(wantVals) {
		t.Fatalf("Attention.Forward() output length = %d, want %d", len(gotVals), len(wantVals))
	}
	for i := range wantVals {
		if diff := math.Abs(float64(gotVals[i] - wantVals[i])); diff > 3e-2 {
			t.Fatalf("Attention.Forward() scaled-rope+sinks output[%d] = %v, want %v (diff %v)", i, gotVals[i], wantVals[i], diff)
		}
	}
}

func TestExpertsForwardMatchesExplicitRealImportedModel(t *testing.T) {
	skipIfNoMLX(t)

	if os.Getenv("OLLAMA_MODELS") == "" {
		t.Skip("OLLAMA_MODELS not set")
	}

	root, err := model.Open("gptoss-mlx-runtime")
	if err != nil {
		t.Skipf("imported model not available: %v", err)
	}
	defer root.Close()

	baseModel, err := base.New(root)
	if err != nil {
		t.Fatalf("base.New() error = %v", err)
	}
	tensors, err := loadRuntimeTensorsForTest(root)
	if err != nil {
		t.Fatalf("loadRuntimeTensorsForTest() error = %v", err)
	}
	if err := base.Weights(baseModel)(tensors); err != nil {
		t.Fatalf("load weights error = %v", err)
	}

	m := baseModel.(*Model)
	layer := m.Layers[0]
	cfg := m.Config

	xVals := make([]float32, int(cfg.HiddenSize))
	for i := range xVals {
		xVals[i] = float32((i%11)-5) / 5
	}
	x := mlx.FromValues(xVals, 1, 1, int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16)

	routerVals := make([]float32, int(cfg.NumLocalExperts))
	for i := range routerVals {
		routerVals[i] = float32(((i*7)%17)-8) / 3
	}
	router := mlx.FromValues(routerVals, 1, 1, int(cfg.NumLocalExperts)).AsType(mlx.DTypeBFloat16)

	got := layer.Experts.Forward(x, router, cfg, 0)
	gotVals := materializedFloats(got.AsType(mlx.DTypeFloat32))
	wantVals := explicitExpertsForwardFromLoadedWeights(t, layer.Experts, cfg, x, routerVals)

	if len(gotVals) != len(wantVals) {
		t.Fatalf("Experts.Forward() output length = %d, want %d", len(gotVals), len(wantVals))
	}
	for i := range wantVals {
		if diff := math.Abs(float64(gotVals[i] - wantVals[i])); diff > 5e-2 {
			t.Fatalf("Experts.Forward() real imported output[%d] = %v, want %v (diff %v)", i, gotVals[i], wantVals[i], diff)
		}
	}
}

func TestAttentionForwardMatchesExplicitRealImportedModel(t *testing.T) {
	skipIfNoMLX(t)

	if os.Getenv("OLLAMA_MODELS") == "" {
		t.Skip("OLLAMA_MODELS not set")
	}

	root, err := model.Open("gptoss-mlx-runtime")
	if err != nil {
		t.Skipf("imported model not available: %v", err)
	}
	defer root.Close()

	baseModel, err := base.New(root)
	if err != nil {
		t.Fatalf("base.New() error = %v", err)
	}
	tensors, err := loadRuntimeTensorsForTest(root)
	if err != nil {
		t.Fatalf("loadRuntimeTensorsForTest() error = %v", err)
	}
	if err := base.Weights(baseModel)(tensors); err != nil {
		t.Fatalf("load weights error = %v", err)
	}

	m := baseModel.(*Model)
	layer := m.Layers[0]
	cfg := m.Config

	xVals := make([]float32, 2*int(cfg.HiddenSize))
	for i := range xVals {
		xVals[i] = float32((i%13)-6) / 6
	}
	x := mlx.FromValues(xVals, 1, 2, int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16)

	got := layer.Attention.Forward(x, nil, 1, 2, cfg, 0)
	gotVals := materializedFloats(got.AsType(mlx.DTypeFloat32))
	wantVals := explicitAttentionForwardFromLoadedWeights(t, layer.Attention, cfg, xVals)

	if len(gotVals) != len(wantVals) {
		t.Fatalf("Attention.Forward() output length = %d, want %d", len(gotVals), len(wantVals))
	}
	for i := range wantVals {
		if diff := math.Abs(float64(gotVals[i] - wantVals[i])); diff > 1e-1 {
			t.Fatalf("Attention.Forward() real imported output[%d] = %v, want %v (diff %v)", i, gotVals[i], wantVals[i], diff)
		}
	}
}

func TestLayerForwardMatchesExplicitRealImportedModel(t *testing.T) {
	skipIfNoMLX(t)

	if os.Getenv("OLLAMA_MODELS") == "" {
		t.Skip("OLLAMA_MODELS not set")
	}

	root, err := model.Open("gptoss-mlx-runtime")
	if err != nil {
		t.Skipf("imported model not available: %v", err)
	}
	defer root.Close()

	baseModel, err := base.New(root)
	if err != nil {
		t.Fatalf("base.New() error = %v", err)
	}
	tensors, err := loadRuntimeTensorsForTest(root)
	if err != nil {
		t.Fatalf("loadRuntimeTensorsForTest() error = %v", err)
	}
	if err := base.Weights(baseModel)(tensors); err != nil {
		t.Fatalf("load weights error = %v", err)
	}

	m := baseModel.(*Model)
	layer := m.Layers[0]
	cfg := m.Config

	xVals := make([]float32, 2*int(cfg.HiddenSize))
	for i := range xVals {
		xVals[i] = float32((i%13)-6) / 6
	}
	x := mlx.FromValues(xVals, 1, 2, int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16)

	got := layer.Forward(x, nil, 1, 2, cfg, 0)
	gotVals := materializedFloats(got.AsType(mlx.DTypeFloat32))
	wantVals := explicitLayerForwardFromLoadedWeights(t, layer, cfg, xVals)

	if len(gotVals) != len(wantVals) {
		t.Fatalf("Layer.Forward() output length = %d, want %d", len(gotVals), len(wantVals))
	}
	for i := range wantVals {
		if diff := math.Abs(float64(gotVals[i] - wantVals[i])); diff > 1.5e-1 {
			t.Fatalf("Layer.Forward() real imported output[%d] = %v, want %v (diff %v)", i, gotVals[i], wantVals[i], diff)
		}
	}
}

func TestAttentionForwardMatchesReferenceWithKVCache(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 2
	cfg.HeadDim = 2
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.RopeTheta = 10000
	cfg.RopeScaling.Factor = 1
	cfg.RopeScaling.OriginalMaxPositionEmbeddings = 4

	xVals := []float32{
		1.0, 0.5,
		-0.25, 0.75,
		0.4, -0.6,
		0.2, 0.3,
	}

	attn := &Attention{
		QProj: nnLinearFromValues([]float32{
			1, 0,
			0, 1,
		}, nil, 2, 2),
		KProj: nnLinearFromValues([]float32{
			0.5, 0,
			0, 1.5,
		}, []float32{0.1, -0.2}, 2, 2),
		VProj: nnLinearFromValues([]float32{
			1.2, 0,
			0, 0.8,
		}, []float32{-0.05, 0.2}, 2, 2),
		OProj: nnLinearFromValues([]float32{
			1, 0,
			0, 1,
		}, []float32{0.01, -0.02}, 2, 2),
	}

	c := cache.NewKVCache()
	gotVals := make([]float32, 0, len(xVals))
	for pos := 0; pos < len(xVals)/2; pos++ {
		x := mlx.FromValues(xVals[pos*2:(pos+1)*2], 1, 1, 2)
		got := attn.Forward(x, c, 1, 1, &cfg, 0)
		if got == nil || !got.Valid() {
			t.Fatalf("Attention.Forward() returned invalid tensor at token %d", pos)
		}
		gotVals = append(gotVals, materializedFloats(got.AsType(mlx.DTypeFloat32))...)
	}

	want := referenceAttentionForwardWindowed(
		xVals,
		[][]float32{{1, 0}, {0, 1}},
		nil,
		[][]float32{{0.5, 0}, {0, 1.5}},
		[]float32{0.1, -0.2},
		[][]float32{{1.2, 0}, {0, 0.8}},
		[]float32{-0.05, 0.2},
		[][]float32{{1, 0}, {0, 1}},
		[]float32{0.01, -0.02},
		cfg.RopeTheta,
		0,
	)

	if len(gotVals) != len(want) {
		t.Fatalf("Attention.Forward() output length = %d, want %d", len(gotVals), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(gotVals[i] - want[i])); diff > 2e-1 {
			t.Fatalf("cached Attention.Forward() output[%d] = %v, want %v (diff %v)", i, gotVals[i], want[i], diff)
		}
	}
}

func TestAttentionForwardMatchesReferenceWithSlidingWindowCache(t *testing.T) {
	skipIfNoMLX(t)

	cfg := denseTestConfig(t)
	cfg.HiddenSize = 2
	cfg.HeadDim = 2
	cfg.NumAttentionHeads = 1
	cfg.NumKeyValueHeads = 1
	cfg.RopeTheta = 10000
	cfg.RopeScaling.Factor = 1
	cfg.RopeScaling.OriginalMaxPositionEmbeddings = 4
	cfg.SlidingWindow = 2

	xVals := []float32{
		1.0, 0.5,
		-0.25, 0.75,
		0.4, -0.6,
		0.2, 0.3,
	}

	attn := &Attention{
		QProj: nnLinearFromValues([]float32{
			1, 0,
			0, 1,
		}, nil, 2, 2),
		KProj: nnLinearFromValues([]float32{
			0.5, 0,
			0, 1.5,
		}, []float32{0.1, -0.2}, 2, 2),
		VProj: nnLinearFromValues([]float32{
			1.2, 0,
			0, 0.8,
		}, []float32{-0.05, 0.2}, 2, 2),
		OProj: nnLinearFromValues([]float32{
			1, 0,
			0, 1,
		}, []float32{0.01, -0.02}, 2, 2),
	}

	c := cache.NewRotatingKVCache(int(cfg.SlidingWindow))
	gotVals := make([]float32, 0, len(xVals))
	for pos := 0; pos < len(xVals)/2; pos++ {
		x := mlx.FromValues(xVals[pos*2:(pos+1)*2], 1, 1, 2)
		got := attn.Forward(x, c, 1, 1, &cfg, 0)
		if got == nil || !got.Valid() {
			t.Fatalf("Attention.Forward() returned invalid tensor at token %d", pos)
		}
		gotVals = append(gotVals, materializedFloats(got.AsType(mlx.DTypeFloat32))...)
	}

	want := referenceAttentionForwardWindowed(
		xVals,
		[][]float32{{1, 0}, {0, 1}},
		nil,
		[][]float32{{0.5, 0}, {0, 1.5}},
		[]float32{0.1, -0.2},
		[][]float32{{1.2, 0}, {0, 0.8}},
		[]float32{-0.05, 0.2},
		[][]float32{{1, 0}, {0, 1}},
		[]float32{0.01, -0.02},
		cfg.RopeTheta,
		int(cfg.SlidingWindow),
	)

	if len(gotVals) != len(want) {
		t.Fatalf("Attention.Forward() output length = %d, want %d", len(gotVals), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(gotVals[i] - want[i])); diff > 2e-1 {
			t.Fatalf("sliding Attention.Forward() output[%d] = %v, want %v (diff %v)", i, gotVals[i], want[i], diff)
		}
	}
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
		"embedding.weight":   denseMatrix(int(cfg.VocabSize), int(cfg.HiddenSize), 1),
		"output_norm.weight": denseVector(int(cfg.HiddenSize), 1),
		"output.weight":      denseMatrix(int(cfg.VocabSize), int(cfg.HiddenSize), 2),
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
		tensors[prefix+".experts.gate_proj.weight"] = denseExpertWeight(int(cfg.NumLocalExperts), int(cfg.HiddenSize), int(cfg.IntermediateSize), 16+float32(i)).AsType(mlx.DTypeBFloat16)
		tensors[prefix+".experts.gate_proj.bias"] = expertBias(int(cfg.NumLocalExperts), int(cfg.IntermediateSize), 0)
		tensors[prefix+".experts.up_proj.weight"] = denseExpertWeight(int(cfg.NumLocalExperts), int(cfg.HiddenSize), int(cfg.IntermediateSize), 20+float32(i)).AsType(mlx.DTypeBFloat16)
		tensors[prefix+".experts.up_proj.bias"] = expertBias(int(cfg.NumLocalExperts), int(cfg.IntermediateSize), 0.5)
		tensors[prefix+".experts.down_proj.weight"] = denseExpertWeight(int(cfg.NumLocalExperts), int(cfg.IntermediateSize), int(cfg.HiddenSize), 24+float32(i)).AsType(mlx.DTypeBFloat16)
		tensors[prefix+".experts.down_proj.bias"] = expertBias(int(cfg.NumLocalExperts), int(cfg.HiddenSize), 0)
	}

	return tensors
}

func loadRuntimeTensorsForTest(root *model.Root) (map[string]*mlx.Array, error) {
	rawTensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	for _, layer := range root.Manifest.GetTensorLayers("") {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		blobPath := root.Manifest.BlobPath(layer.Digest)
		for name, arr := range mlx.Load(blobPath) {
			rawTensors[name] = arr
		}
	}

	scaleBaseNames := make(map[string]bool)
	allTensors := make(map[string]*mlx.Array, len(rawTensors))
	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") {
			baseName := strings.TrimSuffix(name, ".scale")
			allTensors[baseName+"_scale"] = arr
			scaleBaseNames[baseName] = true
		}
	}

	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") {
			continue
		}
		if strings.HasSuffix(name, ".bias") && !strings.HasSuffix(name, ".weight_qbias") {
			baseName := strings.TrimSuffix(name, ".bias")
			if scaleBaseNames[baseName] {
				allTensors[baseName+"_qbias"] = arr
			} else {
				allTensors[name] = arr
			}
		} else {
			allTensors[name] = arr
		}
	}
	return allTensors, nil
}

func explicitExpertsForwardFromLoadedWeights(t *testing.T, experts *Experts, cfg *Config, x *mlx.Array, routerVals []float32) []float32 {
	t.Helper()

	type routedExpert struct {
		index int
		logit float32
	}

	dims := x.Dims()
	if len(dims) != 3 {
		t.Fatalf("explicitExpertsForwardFromLoadedWeights() x dims = %v, want [batch seq hidden]", dims)
	}
	if dims[0] != 1 {
		t.Fatalf("explicitExpertsForwardFromLoadedWeights() batch = %d, want 1", dims[0])
	}

	seqLen := dims[1]
	hidden := dims[2]
	if len(routerVals) != seqLen*int(cfg.NumLocalExperts) {
		t.Fatalf("explicitExpertsForwardFromLoadedWeights() router length = %d, want %d", len(routerVals), seqLen*int(cfg.NumLocalExperts))
	}

	xVals := materializedFloats(x.AsType(mlx.DTypeFloat32))
	out := make([]float32, seqLen*hidden)

	gateW := materializedFloats(experts.GateUp.Gate.Weight.AsType(mlx.DTypeFloat32))
	gateB := materializedFloats(experts.GateUp.Gate.Bias.AsType(mlx.DTypeFloat32))
	upW := materializedFloats(experts.GateUp.Up.Weight.AsType(mlx.DTypeFloat32))
	upB := materializedFloats(experts.GateUp.Up.Bias.AsType(mlx.DTypeFloat32))
	downW := materializedFloats(experts.Down.Weight.AsType(mlx.DTypeFloat32))
	downB := materializedFloats(experts.Down.Bias.AsType(mlx.DTypeFloat32))

	gateWMats := make([][][]float32, int(cfg.NumLocalExperts))
	gateBMats := make([][]float32, int(cfg.NumLocalExperts))
	upWMats := make([][][]float32, int(cfg.NumLocalExperts))
	upBMats := make([][]float32, int(cfg.NumLocalExperts))
	downWMats := make([][][]float32, int(cfg.NumLocalExperts))
	downBMats := make([][]float32, int(cfg.NumLocalExperts))

	for expert := range int(cfg.NumLocalExperts) {
		gateWMats[expert] = make([][]float32, int(cfg.IntermediateSize))
		upWMats[expert] = make([][]float32, int(cfg.IntermediateSize))
		downWMats[expert] = make([][]float32, int(cfg.HiddenSize))
		gateBMats[expert] = append([]float32(nil), gateB[expert*int(cfg.IntermediateSize):(expert+1)*int(cfg.IntermediateSize)]...)
		upBMats[expert] = append([]float32(nil), upB[expert*int(cfg.IntermediateSize):(expert+1)*int(cfg.IntermediateSize)]...)
		downBMats[expert] = append([]float32(nil), downB[expert*int(cfg.HiddenSize):(expert+1)*int(cfg.HiddenSize)]...)

		for row := range int(cfg.IntermediateSize) {
			start := (expert*int(cfg.IntermediateSize) + row) * hidden
			gateWMats[expert][row] = append([]float32(nil), gateW[start:start+hidden]...)
			upWMats[expert][row] = append([]float32(nil), upW[start:start+hidden]...)
		}
		for row := range int(cfg.HiddenSize) {
			start := (expert*int(cfg.HiddenSize) + row) * int(cfg.IntermediateSize)
			downWMats[expert][row] = append([]float32(nil), downW[start:start+int(cfg.IntermediateSize)]...)
		}
	}

	return out

	for pos := range seqLen {
		selected := make([]routedExpert, int(cfg.NumLocalExperts))
		for expert := range int(cfg.NumLocalExperts) {
			selected[expert] = routedExpert{
				index: expert,
				logit: routerVals[pos*int(cfg.NumLocalExperts)+expert],
			}
		}
		slices.SortFunc(selected, func(a, b routedExpert) int {
			switch {
			case a.logit > b.logit:
				return -1
			case a.logit < b.logit:
				return 1
			default:
				return 0
			}
		})
		topK := int(cfg.NumExpertsPerTok)
		if topK > len(selected) {
			topK = len(selected)
		}
		selected = selected[:topK]

		xRow := xVals[pos*hidden : (pos+1)*hidden]
		routerRow := make([]float32, len(selected))
		selectedIndices := make([]int, len(selected))
		for i, s := range selected {
			routerRow[i] = s.logit
			selectedIndices[i] = s.index
		}
		expertOut := referenceExpertsForward(
			xRow,
			routerRow,
			topK,
			pickExpertMatrices(gateWMats, selectedIndices),
			pickExpertBiases(gateBMats, selectedIndices),
			pickExpertMatrices(upWMats, selectedIndices),
			pickExpertBiases(upBMats, selectedIndices),
			pickExpertMatrices(downWMats, selectedIndices),
			pickExpertBiases(downBMats, selectedIndices),
		)
		copy(out[pos*hidden:(pos+1)*hidden], expertOut)
	}

	return out
}

func pickExpertMatrices(src [][][]float32, selected []int) [][][]float32 {
	out := make([][][]float32, len(selected))
	for i, index := range selected {
		out[i] = src[index]
	}
	return out
}

func pickExpertBiases(src [][]float32, selected []int) [][]float32 {
	out := make([][]float32, len(selected))
	for i, index := range selected {
		out[i] = src[index]
	}
	return out
}

func explicitAttentionForwardFromLoadedWeights(t *testing.T, attn *Attention, cfg *Config, xVals []float32) []float32 {
	t.Helper()

	hidden := int(cfg.HiddenSize)
	seqLen := len(xVals) / hidden
	qOut := int(cfg.NumAttentionHeads * cfg.HeadDim)
	kvOut := int(cfg.NumKeyValueHeads * cfg.HeadDim)
	headDim := int(cfg.HeadDim)
	numHeads := int(cfg.NumAttentionHeads)
	numKVHeads := int(cfg.NumKeyValueHeads)
	qMul := numHeads / numKVHeads

	qW := materializedFloats(attn.QProj.(*nn.Linear).Weight.AsType(mlx.DTypeFloat32))
	qB := materializedFloats(attn.QProj.(*nn.Linear).Bias.AsType(mlx.DTypeFloat32))
	kW := materializedFloats(attn.KProj.(*nn.Linear).Weight.AsType(mlx.DTypeFloat32))
	kB := materializedFloats(attn.KProj.(*nn.Linear).Bias.AsType(mlx.DTypeFloat32))
	vW := materializedFloats(attn.VProj.(*nn.Linear).Weight.AsType(mlx.DTypeFloat32))
	vB := materializedFloats(attn.VProj.(*nn.Linear).Bias.AsType(mlx.DTypeFloat32))
	oW := materializedFloats(attn.OProj.(*nn.Linear).Weight.AsType(mlx.DTypeFloat32))
	oB := materializedFloats(attn.OProj.(*nn.Linear).Bias.AsType(mlx.DTypeFloat32))
	sinks := materializedFloats(attn.Sinks.AsType(mlx.DTypeFloat32))

	denoms := referenceGPTOSSRoPEDenominators(cfg)
	concentration := yarnConcentration(cfg)
	scale := float32(1 / math.Sqrt(float64(headDim)))

	query := make([][][]float32, seqLen)
	key := make([][][]float32, seqLen)
	value := make([][][]float32, seqLen)
	for pos := 0; pos < seqLen; pos++ {
		x := xVals[pos*hidden : (pos+1)*hidden]
		qVec := affineFlatRef(x, qW, qB, qOut, hidden)
		kVec := affineFlatRef(x, kW, kB, kvOut, hidden)
		vVec := affineFlatRef(x, vW, vB, kvOut, hidden)

		query[pos] = make([][]float32, numHeads)
		for h := 0; h < numHeads; h++ {
			query[pos][h] = applyRoPEGeneric(qVec[h*headDim:(h+1)*headDim], pos, denoms, concentration)
		}
		key[pos] = make([][]float32, numKVHeads)
		value[pos] = make([][]float32, numKVHeads)
		for h := 0; h < numKVHeads; h++ {
			key[pos][h] = applyRoPEGeneric(kVec[h*headDim:(h+1)*headDim], pos, denoms, concentration)
			value[pos][h] = append([]float32(nil), vVec[h*headDim:(h+1)*headDim]...)
		}
	}

	out := make([]float32, seqLen*hidden)
	for pos := 0; pos < seqLen; pos++ {
		attnHidden := make([]float32, numHeads*headDim)
		for h := 0; h < numHeads; h++ {
			kvHead := h / qMul
			sink := sinks[h]
			scores := make([]float32, pos+2)
			scores[0] = sink
			maxScore := sink
			for j := 0; j <= pos; j++ {
				score := dotRef(query[pos][h], key[j][kvHead]) * scale
				scores[j+1] = score
				if score > maxScore {
					maxScore = score
				}
			}
			sum := float32(0)
			for i := range scores {
				scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
				sum += scores[i]
			}
			for i := range scores {
				scores[i] /= sum
			}
			for j := 0; j <= pos; j++ {
				w := scores[j+1]
				for d := 0; d < headDim; d++ {
					attnHidden[h*headDim+d] += w * value[j][kvHead][d]
				}
			}
		}
		projected := affineFlatRef(attnHidden, oW, oB, hidden, numHeads*headDim)
		copy(out[pos*hidden:(pos+1)*hidden], projected)
	}
	return out
}

func explicitLayerForwardFromLoadedWeights(t *testing.T, layer *Layer, cfg *Config, xVals []float32) []float32 {
	t.Helper()

	hidden := int(cfg.HiddenSize)
	seqLen := len(xVals) / hidden
	attnNormWeight := materializedFloats(layer.AttentionNorm.Weight.AsType(mlx.DTypeFloat32))
	ffnNormWeight := materializedFloats(layer.FFNNorm.Weight.AsType(mlx.DTypeFloat32))
	routerLinear := layer.Router.(*nn.Linear)
	routerW := materializedFloats(routerLinear.Weight.AsType(mlx.DTypeFloat32))
	routerB := materializedFloats(routerLinear.Bias.AsType(mlx.DTypeFloat32))

	attnIn := make([]float32, len(xVals))
	for pos := 0; pos < seqLen; pos++ {
		copy(attnIn[pos*hidden:(pos+1)*hidden], referenceRMSNorm(xVals[pos*hidden:(pos+1)*hidden], attnNormWeight, cfg.RMSNormEps))
	}
	attnOut := explicitAttentionForwardFromLoadedWeights(t, layer.Attention, cfg, attnIn)

	postAttn := make([]float32, len(xVals))
	for i := range xVals {
		postAttn[i] = xVals[i] + attnOut[i]
	}

	ffnIn := make([]float32, len(postAttn))
	for pos := 0; pos < seqLen; pos++ {
		copy(ffnIn[pos*hidden:(pos+1)*hidden], referenceRMSNorm(postAttn[pos*hidden:(pos+1)*hidden], ffnNormWeight, cfg.RMSNormEps))
	}

	routerVals := make([]float32, seqLen*int(cfg.NumLocalExperts))
	for pos := 0; pos < seqLen; pos++ {
		row := affineFlatRef(
			ffnIn[pos*hidden:(pos+1)*hidden],
			routerW,
			routerB,
			int(cfg.NumLocalExperts),
			hidden,
		)
		copy(routerVals[pos*int(cfg.NumLocalExperts):(pos+1)*int(cfg.NumLocalExperts)], row)
	}

	xArr := mlx.FromValues(ffnIn, 1, seqLen, hidden).AsType(mlx.DTypeBFloat16)
	expertOut := explicitExpertsForwardFromLoadedWeights(t, layer.Experts, cfg, xArr, routerVals)

	out := make([]float32, len(postAttn))
	for i := range out {
		out[i] = postAttn[i] + expertOut[i]
	}
	return out
}

func referenceRMSNorm(x, weight []float32, eps float32) []float32 {
	ss := float32(0)
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i := range x {
		out[i] = x[i] * inv * weight[i]
	}
	return out
}

func affineFlatRef(x, w, b []float32, outDim, inDim int) []float32 {
	out := make([]float32, outDim)
	for o := 0; o < outDim; o++ {
		sum := float32(0)
		row := w[o*inDim : (o+1)*inDim]
		for i := 0; i < inDim; i++ {
			sum += row[i] * x[i]
		}
		if len(b) > 0 {
			sum += b[o]
		}
		out[o] = sum
	}
	return out
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

func denseExpertWeight(experts, out, in int, start float32) *mlx.Array {
	values := make([]float32, experts*out*in)
	for i := range values {
		values[i] = start + float32(i)/100
	}
	return mlx.FromValues(values, experts, out, in)
}

func referenceExpertsForward(
	x []float32,
	router []float32,
	topK int,
	gateW [][][]float32,
	gateB [][]float32,
	upW [][][]float32,
	upB [][]float32,
	downW [][][]float32,
	downB [][]float32,
) []float32 {
	type routedExpert struct {
		index int
		logit float32
	}

	selected := make([]routedExpert, len(router))
	for i, v := range router {
		selected[i] = routedExpert{index: i, logit: v}
	}
	slices.SortFunc(selected, func(a, b routedExpert) int {
		switch {
		case a.logit > b.logit:
			return -1
		case a.logit < b.logit:
			return 1
		default:
			return 0
		}
	})
	if topK > 0 && topK < len(selected) {
		selected = selected[:topK]
	}

	maxLogit := float32(math.Inf(-1))
	for _, s := range selected {
		if s.logit > maxLogit {
			maxLogit = s.logit
		}
	}

	scores := make([]float32, len(selected))
	sum := float32(0)
	for i, s := range selected {
		scores[i] = float32(math.Exp(float64(s.logit - maxLogit)))
		sum += scores[i]
	}
	for i := range scores {
		scores[i] /= sum
	}

	out := make([]float32, len(x))
	for i, score := range scores {
		e := selected[i].index
		gate := affineRef(x, gateW[e], gateB[e])
		up := affineRef(x, upW[e], upB[e])

		hidden := make([]float32, len(gate))
		for i := range gate {
			clippedGate := gate[i]
			if clippedGate > 7 {
				clippedGate = 7
			}

			clippedUp := up[i]
			if clippedUp < -7 {
				clippedUp = -7
			}
			if clippedUp > 7 {
				clippedUp = 7
			}

			gated := clippedGate / (1 + float32(math.Exp(float64(-1.702*clippedGate))))
			hidden[i] = gated * (clippedUp + 1)
		}

		down := affineRef(hidden, downW[e], downB[e])
		for i := range out {
			out[i] += score * down[i]
		}
	}

	return out
}

func TestSwiGLUAlphaLimitMatchesReference(t *testing.T) {
	skipIfNoMLX(t)

	gate := mlx.FromValues([]float32{-8, -1, 0.5, 9}, 1, 1, 1, 4)
	up := mlx.FromValues([]float32{-9, -0.5, 2, 8}, 1, 1, 1, 4)

	got := materializedFloats(swiGLUAlphaLimit(gate, up).AsType(mlx.DTypeFloat32))
	want := []float32{
		referenceSwiGLUAlphaLimit(-8, -9),
		referenceSwiGLUAlphaLimit(-1, -0.5),
		referenceSwiGLUAlphaLimit(0.5, 2),
		referenceSwiGLUAlphaLimit(9, 8),
	}

	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 5e-4 {
			t.Fatalf("value %d = %v, want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

func referenceSwiGLUAlphaLimit(gate, up float32) float32 {
	clippedGate := gate
	if clippedGate > 7 {
		clippedGate = 7
	}

	clippedUp := up
	if clippedUp > 7 {
		clippedUp = 7
	}
	if clippedUp < -7 {
		clippedUp = -7
	}

	gated := clippedGate / (1 + float32(math.Exp(float64(-1.702*clippedGate))))
	return gated * (clippedUp + 1)
}

func materializedFloats(a *mlx.Array) []float32 {
	if a == nil {
		return nil
	}
	cloned := a.Clone()
	mlx.Eval(cloned)
	return cloned.Floats()
}

func identityFlat(n int) []float32 {
	out := make([]float32, n*n)
	for i := 0; i < n; i++ {
		out[i*n+i] = 1
	}
	return out
}

func referenceGPTOSSRoPEDenominators(cfg *Config) []float32 {
	dims := int(cfg.HeadDim)
	dHalf := float64(dims) / 2
	base := float64(cfg.RopeTheta)
	factor := float64(cfg.RopeScaling.Factor)
	origCtx := float64(cfg.RopeScaling.OriginalMaxPositionEmbeddings)
	betaFast := float64(cfg.RopeScaling.BetaFast)
	betaSlow := float64(cfg.RopeScaling.BetaSlow)
	if betaFast == 0 {
		betaFast = 32
	}
	if betaSlow == 0 {
		betaSlow = 1
	}

	low := math.Floor(dHalf * math.Log(origCtx/(betaFast*2*math.Pi)) / math.Log(base))
	high := math.Ceil(dHalf * math.Log(origCtx/(betaSlow*2*math.Pi)) / math.Log(base))
	out := make([]float32, 0, dims/2)
	for j := 0; j < dims/2; j++ {
		divisor := math.Pow(base, float64(2*j)/float64(dims))
		ramp := (float64(j) - low) / (high - low)
		if ramp < 0 {
			ramp = 0
		}
		if ramp > 1 {
			ramp = 1
		}
		mask := 1 - ramp
		invFreq := (1/(factor*divisor))*(1-mask) + (1/divisor)*mask
		out = append(out, float32(1/invFreq))
	}
	return out
}

func referenceAttentionForward(
	x []float32,
	qW [][]float32, qB []float32,
	kW [][]float32, kB []float32,
	vW [][]float32, vB []float32,
	oW [][]float32, oB []float32,
	ropeBase float32,
) []float32 {
	seqLen := len(x) / 2
	query := make([][]float32, seqLen)
	key := make([][]float32, seqLen)
	value := make([][]float32, seqLen)
	for pos := 0; pos < seqLen; pos++ {
		query[pos] = applyRoPE2D(affineRef(x[pos*2:(pos+1)*2], qW, qB), pos, ropeBase)
		key[pos] = applyRoPE2D(affineRef(x[pos*2:(pos+1)*2], kW, kB), pos, ropeBase)
		value[pos] = affineRef(x[pos*2:(pos+1)*2], vW, vB)
	}

	scale := float32(1 / math.Sqrt(2))
	out := make([]float32, len(x))
	for pos := 0; pos < seqLen; pos++ {
		scores := make([]float32, pos+1)
		maxScore := float32(math.Inf(-1))
		for j := 0; j <= pos; j++ {
			score := dotRef(query[pos], key[j]) * scale
			scores[j] = score
			if score > maxScore {
				maxScore = score
			}
		}
		sum := float32(0)
		for j := range scores {
			scores[j] = float32(math.Exp(float64(scores[j] - maxScore)))
			sum += scores[j]
		}
		for j := range scores {
			scores[j] /= sum
		}

		hidden := make([]float32, 2)
		for j, score := range scores {
			for d := range hidden {
				hidden[d] += score * value[j][d]
			}
		}
		projected := affineRef(hidden, oW, oB)
		copy(out[pos*2:(pos+1)*2], projected)
	}
	return out
}

func referenceAttentionForwardWithSinksScaledRoPE(x []float32, cfg *Config, sink float32) []float32 {
	seqLen := len(x) / int(cfg.HiddenSize)
	headDim := int(cfg.HeadDim)
	denoms := referenceGPTOSSRoPEDenominators(cfg)
	concentration := yarnConcentration(cfg)

	query := make([][]float32, seqLen)
	key := make([][]float32, seqLen)
	value := make([][]float32, seqLen)
	for pos := 0; pos < seqLen; pos++ {
		base := append([]float32(nil), x[pos*headDim:(pos+1)*headDim]...)
		query[pos] = applyRoPEGeneric(base, pos, denoms, concentration)
		key[pos] = applyRoPEGeneric(base, pos, denoms, concentration)
		value[pos] = append([]float32(nil), base...)
	}

	scale := float32(1 / math.Sqrt(float64(headDim)))
	out := make([]float32, len(x))
	for pos := 0; pos < seqLen; pos++ {
		scores := make([]float32, pos+2)
		maxScore := sink
		scores[0] = sink
		for j := 0; j <= pos; j++ {
			score := dotRef(query[pos], key[j]) * scale
			scores[j+1] = score
			if score > maxScore {
				maxScore = score
			}
		}
		sum := float32(0)
		for j := range scores {
			scores[j] = float32(math.Exp(float64(scores[j] - maxScore)))
			sum += scores[j]
		}
		for j := range scores {
			scores[j] /= sum
		}

		hidden := make([]float32, headDim)
		for j := 0; j <= pos; j++ {
			score := scores[j+1]
			for d := 0; d < headDim; d++ {
				hidden[d] += score * value[j][d]
			}
		}
		copy(out[pos*headDim:(pos+1)*headDim], hidden)
	}
	return out
}

func referenceAttentionForwardWindowed(
	x []float32,
	qW [][]float32, qB []float32,
	kW [][]float32, kB []float32,
	vW [][]float32, vB []float32,
	oW [][]float32, oB []float32,
	ropeBase float32,
	window int,
) []float32 {
	seqLen := len(x) / 2
	query := make([][]float32, seqLen)
	key := make([][]float32, seqLen)
	value := make([][]float32, seqLen)
	for pos := 0; pos < seqLen; pos++ {
		query[pos] = applyRoPE2D(affineRef(x[pos*2:(pos+1)*2], qW, qB), pos, ropeBase)
		key[pos] = applyRoPE2D(affineRef(x[pos*2:(pos+1)*2], kW, kB), pos, ropeBase)
		value[pos] = affineRef(x[pos*2:(pos+1)*2], vW, vB)
	}

	scale := float32(1 / math.Sqrt(2))
	out := make([]float32, len(x))
	for pos := 0; pos < seqLen; pos++ {
		start := 0
		if window > 0 && pos+1 > window {
			start = pos + 1 - window
		}
		scores := make([]float32, pos-start+1)
		maxScore := float32(math.Inf(-1))
		for j := start; j <= pos; j++ {
			score := dotRef(query[pos], key[j]) * scale
			scores[j-start] = score
			if score > maxScore {
				maxScore = score
			}
		}
		sum := float32(0)
		for j := range scores {
			scores[j] = float32(math.Exp(float64(scores[j] - maxScore)))
			sum += scores[j]
		}
		for j := range scores {
			scores[j] /= sum
		}

		hidden := make([]float32, 2)
		for j, score := range scores {
			val := value[start+j]
			for d := range hidden {
				hidden[d] += score * val[d]
			}
		}
		projected := affineRef(hidden, oW, oB)
		copy(out[pos*2:(pos+1)*2], projected)
	}
	return out
}

func applyRoPE2D(v []float32, position int, base float32) []float32 {
	if len(v) != 2 {
		panic("applyRoPE2D expects length-2 vector")
	}
	theta := float64(position)
	if base <= 0 {
		base = 10000
	}
	c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
	return []float32{
		v[0]*c - v[1]*s,
		v[0]*s + v[1]*c,
	}
}

func applyRoPEGeneric(v []float32, position int, denoms []float32, concentration float32) []float32 {
	if len(v)%2 != 0 {
		panic("applyRoPEGeneric expects even-length vector")
	}
	out := make([]float32, len(v))
	for i := 0; i < len(v); i += 2 {
		theta := float64(position) / float64(denoms[i/2])
		c := concentration * float32(math.Cos(theta))
		s := concentration * float32(math.Sin(theta))
		out[i] = v[i]*c - v[i+1]*s
		out[i+1] = v[i]*s + v[i+1]*c
	}
	return out
}

func dotRef(a, b []float32) float32 {
	sum := float32(0)
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func nnLinearFromValues(weightVals, biasVals []float32, out, in int) *nn.Linear {
	weight := mlx.FromValues(weightVals, out, in).AsType(mlx.DTypeBFloat16)
	var bias *mlx.Array
	if biasVals != nil {
		bias = mlx.FromValues(biasVals, out).AsType(mlx.DTypeBFloat16)
	}
	return nn.NewLinear(weight, bias)
}

func affineRef(x []float32, w [][]float32, b []float32) []float32 {
	width := 0
	if len(b) > 0 {
		width = len(b)
	} else if len(w) > 0 {
		width = len(w[0])
	}
	out := make([]float32, width)
	copy(out, b)
	for i := range x {
		for j := range out {
			out[j] += x[i] * w[i][j]
		}
	}
	return out
}

func expertBias(experts, out int, start float32) *mlx.Array {
	values := make([]float32, experts*out)
	for i := range values {
		values[i] = start + float32(i)/100
	}
	return mlx.FromValues(values, experts, out).AsType(mlx.DTypeBFloat16)
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
