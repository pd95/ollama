package apertus

import (
	"encoding/json"
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

func TestPruneApertus1p5TokenizerAddedTokens(t *testing.T) {
	data := []byte(`{
		"model": {
			"type": "BPE",
			"vocab": {"a": 0},
			"merges": []
		},
		"added_tokens": [
			{"id": 61, "content": "<|system_start|>", "special": true},
			{"id": 73, "content": "<|tool_output_start|>", "special": true},
			{"id": 131073, "content": "<|img_start|>", "special": true},
			{"id": 262344, "content": "<|audio token 0|>", "special": true}
		]
	}`)

	pruned, err := pruneApertus1p5TokenizerAddedTokens(data, 131072)
	if err != nil {
		t.Fatalf("pruneApertus1p5TokenizerAddedTokens error: %v", err)
	}

	var got struct {
		AddedTokens []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(pruned, &got); err != nil {
		t.Fatalf("parse pruned tokenizer: %v", err)
	}

	want := []struct {
		id      int32
		content string
	}{
		{id: 61, content: "<|system_start|>"},
		{id: 73, content: "<|tool_output_start|>"},
	}
	if len(got.AddedTokens) != len(want) {
		t.Fatalf("kept added tokens = %v, want %d", got.AddedTokens, len(want))
	}
	for i, wantToken := range want {
		if got.AddedTokens[i].ID != wantToken.id || got.AddedTokens[i].Content != wantToken.content {
			t.Fatalf("kept token %d = (%d, %q), want (%d, %q)",
				i,
				got.AddedTokens[i].ID,
				got.AddedTokens[i].Content,
				wantToken.id,
				wantToken.content,
			)
		}
	}
}

func TestTokenizerPruningIsApertus1p5Only(t *testing.T) {
	data := []byte(`{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"added_tokens":[{"id":7,"content":"kept"},{"id":129,"content":"media"}]}`)

	v1, err := tokenizerDataForConfig(Config{Architecture: apertus1p0Architecture, OutputVocabSize: 128}, data)
	if err != nil {
		t.Fatal(err)
	}
	if string(v1) != string(data) {
		t.Fatalf("Apertus 1.0 tokenizer changed: %s", v1)
	}
	v15, err := tokenizerDataForConfig(Config{Architecture: apertus1p5Architecture, OutputVocabSize: 128}, data)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(v15), `"id":129`) || !strings.Contains(string(v15), `"id":7`) {
		t.Fatalf("Apertus 1.5 tokenizer pruning = %s", v15)
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
	tensors := map[string]*mlx.Array{
		"model.vision_tokenizer.present": mlx.New("vision"),
		"model.audio_tokenizer.present":  mlx.New("audio"),
	}
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
	if err := vision.validate(); err != nil {
		t.Fatalf("supported vision config rejected: %v", err)
	}
	if err := audio.validate(); err != nil {
		t.Fatalf("supported audio config rejected: %v", err)
	}
	if !canValidateVisionTokenizer(tensors, vision) || !canValidateAudioTokenizer(tensors, audio) {
		t.Fatal("supported independent media configs were suppressed")
	}
	if err := (VisionTokenizerConfig{}).validate(); err == nil {
		t.Fatal("missing vision config accepted")
	}
	invalidVision := vision
	invalidVision.Resolution = 512
	if err := invalidVision.validate(); err == nil {
		t.Fatal("invalid vision config accepted")
	}
	if canValidateVisionTokenizer(tensors, invalidVision) || !canValidateAudioTokenizer(tensors, audio) {
		t.Fatal("invalid vision config did not suppress only vision")
	}
	if err := (AudioTokenizerConfig{}).validate(); err == nil {
		t.Fatal("missing audio config accepted")
	}
	invalidAudio := audio
	invalidAudio.SamplingRate = 16000
	if err := invalidAudio.validate(); err == nil {
		t.Fatal("invalid audio config accepted")
	}
	if canValidateAudioTokenizer(tensors, invalidAudio) || !canValidateVisionTokenizer(tensors, vision) {
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
		"rms_norm_eps": 1e-5,
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
		"rms_norm_eps": 1e-5,
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

func TestParseConfigBoundsAndNonFiniteValues(t *testing.T) {
	valid := `{
		"architectures":["ApertusForCausalLM"],"hidden_size":16,"intermediate_size":32,
		"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
		"vocab_size":128,"max_position_embeddings":64,"rms_norm_eps":1e-5,"rope_theta":12000000,
		"rope_scaling":{"factor":8,"high_freq_factor":4,"low_freq_factor":1,"original_max_position_embeddings":8192,"rope_type":"llama3"},
		"hidden_act":"xielu","qk_norm":true}`
	for _, tc := range []struct {
		name, old, replacement, want string
	}{
		{"zero hidden", `"hidden_size":16`, `"hidden_size":0`, "hidden_size"},
		{"layers first rejected", `"num_hidden_layers":1`, `"num_hidden_layers":1025`, "num_hidden_layers"},
		{"heads first rejected", `"num_attention_heads":4`, `"num_attention_heads":1025`, "num_attention_heads"},
		{"negative kv heads", `"num_key_value_heads":2`, `"num_key_value_heads":-1`, "num_key_value_heads"},
		{"context first rejected", `"max_position_embeddings":64`, `"max_position_embeddings":16777217`, "max_position_embeddings"},
		{"vocab first rejected", `"vocab_size":128`, `"vocab_size":16777217`, "vocab_size"},
		{"zero norm", `"rms_norm_eps":1e-5`, `"rms_norm_eps":0`, "rms_norm_eps"},
		{"overflowing norm", `"rms_norm_eps":1e-5`, `"rms_norm_eps":1e100`, "rms_norm_eps"},
		{"zero theta", `"rope_theta":12000000`, `"rope_theta":0`, "rope_theta"},
		{"equal rope factors", `"high_freq_factor":4`, `"high_freq_factor":1`, "high_freq_factor"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseConfig([]byte(strings.Replace(valid, tc.old, tc.replacement, 1)))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("parseConfig error = %v, want %q", err, tc.want)
			}
		})
	}

	maxima := strings.NewReplacer(
		`"hidden_size":16`, `"hidden_size":2048`,
		`"intermediate_size":32`, `"intermediate_size":2`,
		`"num_hidden_layers":1`, `"num_hidden_layers":1024`,
		`"num_attention_heads":4`, `"num_attention_heads":1024`,
		`"num_key_value_heads":2`, `"num_key_value_heads":1024`,
		`"vocab_size":128`, `"vocab_size":2`,
		`"max_position_embeddings":64`, `"max_position_embeddings":16777216`,
	).Replace(valid)
	if _, err := parseConfig([]byte(maxima)); err != nil {
		t.Fatalf("documented maxima rejected: %v", err)
	}
	if got, err := checkedProduct("boundary", maxArrayElements, 1); err != nil || got != maxArrayElements {
		t.Fatalf("greatest array product = %d, %v", got, err)
	}
	if _, err := checkedProduct("boundary", maxArrayElements, 2); err == nil {
		t.Fatal("first overflowing array product accepted")
	}
	future70B := strings.NewReplacer(
		`"hidden_size":16`, `"hidden_size":8192`,
		`"intermediate_size":32`, `"intermediate_size":43008`,
		`"num_hidden_layers":1`, `"num_hidden_layers":80`,
		`"num_attention_heads":4`, `"num_attention_heads":64`,
		`"num_key_value_heads":2`, `"num_key_value_heads":8`,
		`"vocab_size":128`, `"vocab_size":266752`,
		`"max_position_embeddings":64`, `"max_position_embeddings":262144`,
	).Replace(valid)
	if _, err := parseConfig([]byte(future70B)); err != nil {
		t.Fatalf("documented 70B plus recorded 1.5 dimensions rejected: %v", err)
	}
}

func TestTensorValidationDenseAndPacked(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	cfg := tinyConfig()
	tensors := tinyDenseTensors(cfg)
	if err := validateTensors(tensors, cfg); err != nil {
		t.Fatalf("valid dense tensors rejected: %v", err)
	}

	for _, tc := range []struct {
		name string
		edit func(map[string]*mlx.Array)
	}{
		{"missing", func(ts map[string]*mlx.Array) { delete(ts, "model.layers.0.self_attn.k_proj.weight") }},
		{"wrong shape", func(ts map[string]*mlx.Array) {
			ts["model.layers.0.mlp.up_proj.weight"] = mlx.Zeros(mlx.DTypeBFloat16, 31, 16)
		}},
		{"wrong dtype", func(ts map[string]*mlx.Array) { ts["model.norm.weight"] = mlx.Zeros(mlx.DTypeUint32, 16) }},
		{"orphan scale", func(ts map[string]*mlx.Array) {
			ts["model.layers.0.self_attn.q_proj.weight_scale"] = mlx.Zeros(mlx.DTypeUint8, 16, 1)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ts := tinyDenseTensors(cfg)
			tc.edit(ts)
			if err := validateTensors(ts, cfg); err == nil {
				t.Fatal("malformed dense tensors accepted")
			}
		})
	}

	name := "model.layers.0.self_attn.q_proj.weight"
	tensors[name] = mlx.Zeros(mlx.DTypeUint32, 16, 2)
	tensors[name+"_scale"] = mlx.Zeros(mlx.DTypeUint8, 16, 1)
	cfg.TensorQuant = map[string]*model.TensorQuantInfo{name: {QuantType: "nvfp4", GroupSize: 16}}
	if err := validateTensors(tensors, cfg); err != nil {
		t.Fatalf("valid NVFP4 tensors rejected: %v", err)
	}
	for _, tc := range []struct {
		name string
		edit func(map[string]*mlx.Array)
	}{
		{"packed shape", func(ts map[string]*mlx.Array) { ts[name] = mlx.Zeros(mlx.DTypeUint32, 16, 3) }},
		{"packed dtype", func(ts map[string]*mlx.Array) { ts[name] = mlx.Zeros(mlx.DTypeUint8, 16, 2) }},
		{"scale shape", func(ts map[string]*mlx.Array) { ts[name+"_scale"] = mlx.Zeros(mlx.DTypeUint8, 16, 2) }},
		{"scale dtype", func(ts map[string]*mlx.Array) { ts[name+"_scale"] = mlx.Zeros(mlx.DTypeBFloat16, 16, 1) }},
		{"qbias", func(ts map[string]*mlx.Array) { ts[name+"_qbias"] = mlx.Zeros(mlx.DTypeUint8, 16, 1) }},
		{"duplicate global", func(ts map[string]*mlx.Array) {
			ts[name+".global_scale"] = mlx.Zeros(mlx.DTypeFloat32, 1)
			ts[name+"_scale_2"] = mlx.Zeros(mlx.DTypeFloat32, 1)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ts := tinyDenseTensors(cfg)
			ts[name] = mlx.Zeros(mlx.DTypeUint32, 16, 2)
			ts[name+"_scale"] = mlx.Zeros(mlx.DTypeUint8, 16, 1)
			tc.edit(ts)
			if err := validateTensors(ts, cfg); err == nil {
				t.Fatal("malformed packed tensors accepted")
			}
		})
	}
	if err := validateTensors(tensors, cfg); err != nil {
		t.Fatalf("valid packed tensors rejected after failures: %v", err)
	}
}

func TestMatrixQuantizationIdentitiesFailClosed(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	const name = "projection.weight"
	for _, quantType := range []string{"int4", "int8", "nvfp4", "mxfp4", "mxfp8", "FP4", "Q8"} {
		t.Run("supported model "+quantType, func(t *testing.T) {
			tensors, cfg := quantizedMatrix(name, quantType, 4, 64)
			if err := validateMatrix(tensors, "projection", 4, 64, cfg, false); err != nil {
				t.Fatalf("supported quantization %q rejected: %v", quantType, err)
			}
		})
		t.Run("supported tensor "+quantType, func(t *testing.T) {
			tensors, cfg := quantizedMatrix(name, quantType, 4, 64)
			cfg.QuantType, cfg.QuantMode, cfg.QuantBits, cfg.QuantGroupSize = "", "", 0, 0
			groupSize, _, _ := model.QuantizationParams(quantType)
			cfg.TensorQuant = map[string]*model.TensorQuantInfo{name: {QuantType: quantType, GroupSize: groupSize}}
			if err := validateMatrix(tensors, "projection", 4, 64, cfg, false); err != nil {
				t.Fatalf("supported per-tensor quantization %q rejected: %v", quantType, err)
			}
		})
	}

	t.Run("supported dense", func(t *testing.T) {
		if err := validateMatrix(map[string]*mlx.Array{name: mlx.Zeros(mlx.DTypeBFloat16, 4, 64)}, "projection", 4, 64, &Config{}, false); err != nil {
			t.Fatalf("dense matrix rejected: %v", err)
		}
	})

	for _, tc := range []struct {
		name string
		edit func(*Config)
		want string
	}{
		{
			name: "unknown model identity",
			edit: func(cfg *Config) { cfg.QuantType = "future8" },
			want: "unsupported model quantization type",
		},
		{
			name: "conflicting model identity",
			edit: func(cfg *Config) {
				cfg.QuantType = "nvfp4"
				cfg.QuantBits = 8
				cfg.QuantMode = "affine"
			},
			want: "conflicting model quantization metadata",
		},
		{
			name: "unknown tensor identity",
			edit: func(cfg *Config) {
				cfg.TensorQuant = map[string]*model.TensorQuantInfo{name: {QuantType: "future8", GroupSize: 32}}
			},
			want: "unsupported quantization type",
		},
		{
			name: "empty tensor identity",
			edit: func(cfg *Config) {
				cfg.TensorQuant = map[string]*model.TensorQuantInfo{name: {GroupSize: 32}}
			},
			want: "missing explicit quantization type",
		},
		{
			name: "nil tensor identity",
			edit: func(cfg *Config) { cfg.TensorQuant = map[string]*model.TensorQuantInfo{name: nil} },
			want: "missing explicit quantization type",
		},
		{
			name: "conflicting tensor group size",
			edit: func(cfg *Config) {
				cfg.TensorQuant = map[string]*model.TensorQuantInfo{name: {QuantType: "nvfp4", GroupSize: 32}}
			},
			want: "conflicting quantization group size",
		},
		{
			name: "negative tensor group size",
			edit: func(cfg *Config) {
				cfg.TensorQuant = map[string]*model.TensorQuantInfo{name: {QuantType: "int8", GroupSize: -1}}
			},
			want: "conflicting quantization group size",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tensors, cfg := quantizedMatrix(name, "int8", 4, 64)
			tc.edit(cfg)
			err := validateMatrix(tensors, "projection", 4, 64, cfg, false)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("validateMatrix error = %v, want %q", err, tc.want)
			}
		})
	}

	t.Run("unknown identity rejected before dense fallback", func(t *testing.T) {
		err := validateMatrix(
			map[string]*mlx.Array{name: mlx.Zeros(mlx.DTypeBFloat16, 4, 64)},
			"projection", 4, 64, &Config{QuantType: "future8", QuantBits: 8, QuantMode: "affine"}, false,
		)
		if err == nil || !strings.Contains(err.Error(), "unsupported model quantization type") {
			t.Fatalf("validateMatrix error = %v, want unknown identity rejection", err)
		}
	})
}

func TestNVFP4GlobalScaleShapes(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	const (
		linearName    = "projection.weight"
		embeddingName = "embedding.weight"
	)
	for _, tc := range []struct {
		name      string
		embedding bool
		shape     []int
		invalid   bool
		wantOK    bool
	}{
		{name: "linear scalar", shape: nil, wantOK: true},
		{name: "linear per row", shape: []int{4}, wantOK: true},
		{name: "linear invalid handle", invalid: true},
		{name: "linear wrong rank same size", shape: []int{4, 1}},
		{name: "linear transposed wrong rank same size", shape: []int{1, 4}},
		{name: "linear wrong length", shape: []int{3}},
		{name: "embedding scalar", embedding: true, shape: nil, wantOK: true},
		{name: "embedding per row", embedding: true, shape: []int{4}},
		{name: "embedding wrong rank same size", embedding: true, shape: []int{4, 1}},
		{name: "embedding single element vector is not scalar", embedding: true, shape: []int{1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			name := linearName
			path := "projection"
			if tc.embedding {
				name = embeddingName
				path = "embedding"
			}
			tensors, cfg := quantizedMatrix(name, "nvfp4", 4, 64)
			if tc.invalid {
				tensors[name+".global_scale"] = &mlx.Array{}
			} else {
				tensors[name+".global_scale"] = mlx.Zeros(mlx.DTypeFloat32, tc.shape...)
			}
			err := validateMatrix(tensors, path, 4, 64, cfg, tc.embedding)
			if tc.wantOK && err != nil {
				t.Fatalf("valid global scale rejected: %v", err)
			}
			if !tc.wantOK && err == nil {
				t.Fatal("invalid global scale accepted")
			}
		})
	}
}

func quantizedMatrix(name, quantType string, out, input int) (map[string]*mlx.Array, *Config) {
	groupSize, bits, mode := model.QuantizationParams(quantType)
	tensors := map[string]*mlx.Array{}
	switch mode {
	case "affine":
		tensors[name] = mlx.Zeros(mlx.DTypeUint32, out, input/(32/bits))
		tensors[name+"_scale"] = mlx.Zeros(mlx.DTypeBFloat16, out, input/groupSize)
		tensors[name+"_qbias"] = mlx.Zeros(mlx.DTypeBFloat16, out, input/groupSize)
	case "nvfp4", "mxfp4":
		tensors[name] = mlx.Zeros(mlx.DTypeUint32, out, input/8)
		tensors[name+"_scale"] = mlx.Zeros(mlx.DTypeUint8, out, input/groupSize)
	case "mxfp8":
		tensors[name] = mlx.Zeros(mlx.DTypeUint8, out, input)
		tensors[name+"_scale"] = mlx.Zeros(mlx.DTypeUint8, out, input/groupSize)
	}
	return tensors, &Config{QuantType: quantType, QuantGroupSize: groupSize, QuantBits: bits, QuantMode: mode}
}

func TestCacheCount(t *testing.T) {
	m := &Model{Config: &Config{}, Layers: make([]*Layer, 7)}
	if got := len(m.NewCaches()); got != 7 {
		t.Fatalf("cache count = %d, want 7", got)
	}
}

func tinyConfig() *Config {
	return &Config{
		Architecture: apertus1p0Architecture, prefix: "model.",
		HiddenSize: 16, IntermediateSize: 32, NumHiddenLayers: 1, NumAttentionHeads: 4,
		NumKeyValueHeads: 2, VocabSize: 128, OutputVocabSize: 128, HeadDim: 4, QuantMode: "", TensorQuant: map[string]*model.TensorQuantInfo{},
	}
}

func tinyDenseTensors(cfg *Config) map[string]*mlx.Array {
	hidden, intermediate, vocab := int(cfg.HiddenSize), int(cfg.IntermediateSize), int(cfg.VocabSize)
	outputVocab := int(cfg.OutputVocabSize)
	prefix, err := tensorPrefixForArchitecture(cfg.Architecture)
	if err != nil {
		panic(err)
	}
	qOut, kvOut, headDim := int(cfg.NumAttentionHeads*cfg.HeadDim), int(cfg.NumKeyValueHeads*cfg.HeadDim), int(cfg.HeadDim)
	tensors := map[string]*mlx.Array{
		prefix + "embed_tokens.weight": mlx.Zeros(mlx.DTypeBFloat16, vocab, hidden),
		prefix + "norm.weight":         mlx.Zeros(mlx.DTypeBFloat16, hidden),
		"lm_head.weight":               mlx.Zeros(mlx.DTypeBFloat16, outputVocab, hidden),
	}
	for i := range cfg.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("%slayers.%d", prefix, i)
		tensors[layerPrefix+".attention_layernorm.weight"] = mlx.Zeros(mlx.DTypeBFloat16, hidden)
		tensors[layerPrefix+".feedforward_layernorm.weight"] = mlx.Zeros(mlx.DTypeBFloat16, hidden)
		tensors[layerPrefix+".self_attn.q_norm.weight"] = mlx.Zeros(mlx.DTypeBFloat16, headDim)
		tensors[layerPrefix+".self_attn.k_norm.weight"] = mlx.Zeros(mlx.DTypeBFloat16, headDim)
		tensors[layerPrefix+".self_attn.q_proj.weight"] = mlx.Zeros(mlx.DTypeBFloat16, qOut, hidden)
		tensors[layerPrefix+".self_attn.k_proj.weight"] = mlx.Zeros(mlx.DTypeBFloat16, kvOut, hidden)
		tensors[layerPrefix+".self_attn.v_proj.weight"] = mlx.Zeros(mlx.DTypeBFloat16, kvOut, hidden)
		tensors[layerPrefix+".self_attn.o_proj.weight"] = mlx.Zeros(mlx.DTypeBFloat16, hidden, hidden)
		tensors[layerPrefix+".mlp.up_proj.weight"] = mlx.Zeros(mlx.DTypeBFloat16, intermediate, hidden)
		tensors[layerPrefix+".mlp.down_proj.weight"] = mlx.Zeros(mlx.DTypeBFloat16, hidden, intermediate)
		tensors[layerPrefix+".mlp.act_fn.alpha_p"] = mlx.Zeros(mlx.DTypeBFloat16, 1)
		tensors[layerPrefix+".mlp.act_fn.alpha_n"] = mlx.Zeros(mlx.DTypeBFloat16, 1)
		tensors[layerPrefix+".mlp.act_fn.beta"] = mlx.Zeros(mlx.DTypeBFloat16)
		tensors[layerPrefix+".mlp.act_fn.eps"] = mlx.Zeros(mlx.DTypeBFloat16)
	}
	return tensors
}

func TestTensorNamespaceSelectedByArchitecture(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	v1 := tinyConfig()
	if err := validateTensors(tinyDenseTensors(v1), v1); err != nil {
		t.Fatalf("valid Apertus 1.0 tensors rejected: %v", err)
	}
	v15 := *v1
	v15.Architecture = apertus1p5Architecture
	v15.prefix = "model.language_model."
	v15.VocabSize = 256
	v15.OutputVocabSize = 128
	if err := validateTensors(tinyDenseTensors(&v15), &v15); err != nil {
		t.Fatalf("valid Apertus 1.5 tensors rejected: %v", err)
	}

	t.Run("wrong namespace", func(t *testing.T) {
		err := validateTensors(tinyDenseTensors(v1), &v15)
		if err == nil || !strings.Contains(err.Error(), "unexpected Apertus 1.0 tensor namespace") {
			t.Fatalf("validateTensors error = %v, want explicit wrong-namespace rejection", err)
		}
	})
	t.Run("mixed namespace", func(t *testing.T) {
		tensors := tinyDenseTensors(&v15)
		tensors["model.norm.weight"] = mlx.Zeros(mlx.DTypeBFloat16, int(v15.HiddenSize))
		err := validateTensors(tensors, &v15)
		if err == nil || !strings.Contains(err.Error(), "unexpected Apertus 1.0 tensor namespace") {
			t.Fatalf("validateTensors error = %v, want mixed-namespace rejection", err)
		}
	})
	t.Run("missing selected namespace", func(t *testing.T) {
		tensors := tinyDenseTensors(&v15)
		delete(tensors, "model.language_model.norm.weight")
		err := validateTensors(tensors, &v15)
		if err == nil || !strings.Contains(err.Error(), `missing tensor "model.language_model.norm.weight"`) {
			t.Fatalf("validateTensors error = %v, want selected-namespace missing tensor", err)
		}
	})
	t.Run("output vocabulary", func(t *testing.T) {
		tensors := tinyDenseTensors(&v15)
		tensors["lm_head.weight"] = mlx.Zeros(mlx.DTypeBFloat16, int(v15.OutputVocabSize)+1, int(v15.HiddenSize))
		if err := validateTensors(tensors, &v15); err == nil || !strings.Contains(err.Error(), "lm_head.weight") {
			t.Fatalf("validateTensors error = %v, want output-vocabulary mismatch", err)
		}
	})
	t.Run("explicit quantization mismatch", func(t *testing.T) {
		tensors := tinyDenseTensors(&v15)
		name := "model.language_model.layers.0.self_attn.q_proj.weight"
		v15Quant := v15
		v15Quant.TensorQuant = map[string]*model.TensorQuantInfo{name: {QuantType: "nvfp4", GroupSize: 16}}
		err := validateTensors(tensors, &v15Quant)
		if err == nil || !strings.Contains(err.Error(), "missing quantization companions") {
			t.Fatalf("validateTensors error = %v, want explicit quantization mismatch", err)
		}
	})
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
