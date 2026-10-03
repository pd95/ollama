package create

import (
	"math"
	"strconv"
	"testing"
)

func TestApertusImportTransform(t *testing.T) {
	transform, err := newTensorImportTransform(Inventory{
		Config: sourceModelConfig{Architectures: []string{"ApertusForCausalLM"}},
	})
	if err != nil {
		t.Fatalf("newTensorImportTransform() error = %v", err)
	}
	if _, ok := transform.(apertusImportTransform); !ok {
		t.Fatalf("newTensorImportTransform() = %T, want apertusImportTransform", transform)
	}
	for _, architecture := range []string{"GptOssForCausalLM", "Qwen3_5ForCausalLM"} {
		if _, ok := tensorImportTransformRegistry[architecture]; !ok {
			t.Fatalf("registry lost %s", architecture)
		}
	}
}

func TestApertus1p5ImportTransform(t *testing.T) {
	transform, err := newTensorImportTransform(Inventory{
		Config: sourceModelConfig{Architectures: []string{"Apertus1p5ForConditionalGeneration"}},
	})
	if err != nil {
		t.Fatalf("newTensorImportTransform() error = %v", err)
	}
	if _, ok := transform.(apertusImportTransform); !ok {
		t.Fatalf("newTensorImportTransform() = %T, want apertusImportTransform", transform)
	}
}

func TestApertusQuantizationPolicy(t *testing.T) {
	transform := apertusImportTransform{}
	for _, tt := range []struct {
		name, tensor, quantize, want string
		shape                        []int32
	}{
		{"lm head retains requested nvfp4", "lm_head.weight", "NVFP4", "nvfp4", []int32{131072, 4096}},
		{"k projection retains requested nvfp4", "model.layers.0.self_attn.k_proj.weight", "nvfp4", "nvfp4", []int32{1024, 4096}},
		{"v projection retains requested nvfp4", "model.layers.0.self_attn.v_proj.weight", "nvfp4", "nvfp4", []int32{1024, 4096}},
		{"down projection retains requested nvfp4", "model.layers.0.mlp.down_proj.weight", "nvfp4", "nvfp4", []int32{4096, 21504}},
		{"stacked rank three projection retains requested nvfp4", "model.layers.0.mlp.experts.down_proj.weight", "nvfp4", "nvfp4", []int32{8, 4096, 21504}},
		{"embeddings remain source precision", "model.embed_tokens.weight", "nvfp4", "", []int32{131072, 4096}},
		{"routing remains source precision", "model.layers.0.mlp.gate.weight", "nvfp4", "", []int32{4096, 4096}},
		{"small remains source precision", "model.layers.0.mlp.down_proj.weight", "nvfp4", "", []int32{16, 16}},
		{"nonlinear remains source precision", "model.layers.0.mlp.act_fn.alpha.weight", "nvfp4", "", []int32{1024, 4096}},
		{"misaligned remains source precision", "model.layers.0.mlp.down_proj.weight", "nvfp4", "", []int32{1024, 4095}},
		{"unknown quantization remains source precision", "model.layers.0.mlp.down_proj.weight", "unknown", "", []int32{1024, 4096}},
		{"mismatched rank remains source precision", "model.layers.0.mlp.down_proj.weight", "nvfp4", "", []int32{2, 1024, 4096}},
		{"apertus 1.5 embeddings remain source precision", "model.language_model.embed_tokens.weight", "nvfp4", "", []int32{266752, 4096}},
		{"apertus 1.5 decoder weights retain requested nvfp4", "model.language_model.layers.0.self_attn.q_proj.weight", "nvfp4", "nvfp4", []int32{4096, 4096}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := transform.quantizationType(tt.tensor, tt.shape, tt.quantize); got != tt.want {
				t.Fatalf("quantizationType(%q, %v, %q) = %q, want %q", tt.tensor, tt.shape, tt.quantize, got, tt.want)
			}
		})
	}
}

func TestApertusMXFP8UsesGeneralQuantizationPolicy(t *testing.T) {
	transform := apertusImportTransform{}
	for _, tt := range []struct {
		name, tensor, want string
		shape              []int32
	}{
		{"70B lm head", "lm_head.weight", "mxfp8", []int32{131072, 8192}},
		{"70B attention projection", "model.layers.0.self_attn.q_proj.weight", "mxfp8", []int32{8192, 8192}},
		{"70B MLP projection", "model.layers.0.mlp.down_proj.weight", "mxfp8", []int32{8192, 28672}},
		{"embedding remains source precision", "model.embed_tokens.weight", "", []int32{131072, 8192}},
		{"routing remains source precision", "model.layers.0.mlp.gate.weight", "", []int32{8192, 8192}},
		{"Apertus 1.5 media remains source precision", "model.audio_tokenizer.encoder.layers.0.weight", "", []int32{8192, 8192}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := transform.quantizationType(tt.tensor, tt.shape, "mxfp8"); got != tt.want {
				t.Fatalf("quantizationType(%q, %v, mxfp8) = %q, want %q", tt.tensor, tt.shape, got, tt.want)
			}
		})
	}
}

func TestApertusElementCount(t *testing.T) {
	for _, tt := range []struct {
		name  string
		shape []int32
		want  uint64
		ok    bool
	}{
		{"positive rank two", []int32{1024, 4096}, 4194304, true},
		{"greatest accepted max-dimension product", []int32{math.MaxInt32, math.MaxInt32, 4}, 18446744056529682436, true},
		{"first rejected max-dimension product", []int32{math.MaxInt32, math.MaxInt32, 5}, 0, false},
		{"zero dimension", []int32{1024, 0}, 0, false},
		{"negative dimension", []int32{1024, -1}, 0, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := apertusElementCount(tt.shape)
			if got != tt.want || ok != tt.ok {
				t.Fatalf("apertusElementCount(%v) = (%d, %t), want (%d, %t)", tt.shape, got, ok, tt.want, tt.ok)
			}
		})
	}
}

func TestApertus1p5PlanKeepsEncodeOnlyMediaTensors(t *testing.T) {
	inv := apertus1p5Inventory()
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatal(err)
	}
	specs, err := Plan(inv, Classification{Kind: SourceFloat, Quantize: "nvfp4"}, policy)
	if err != nil {
		t.Fatal(err)
	}

	var tensorCount int
	seen := map[string]TensorSpec{}
	for _, spec := range specs {
		for _, tensor := range spec.Tensors {
			tensorCount++
			seen[tensor.Name] = tensor
			if !isApertus1p5RuntimeTensor(tensor.Name) {
				t.Fatalf("planned unsupported tensor %q", tensor.Name)
			}
			if tensor.Name != tensor.Sources[0].Name {
				t.Fatalf("tensor name %q rewrote source %q", tensor.Name, tensor.Sources[0].Name)
			}
		}
	}
	if tensorCount != 761 {
		t.Fatalf("planned tensor count = %d, want 761", tensorCount)
	}
	if got := len(inv.Tensors) - tensorCount; got != 163 {
		t.Fatalf("filtered tensor count = %d, want 163", got)
	}
	if seen["model.language_model.embed_tokens.weight"].Quantize != "" {
		t.Fatalf("embedding quantize = %q, want source precision", seen["model.language_model.embed_tokens.weight"].Quantize)
	}
	if seen["lm_head.weight"].Quantize != "nvfp4" {
		t.Fatalf("lm_head quantize = %q, want nvfp4", seen["lm_head.weight"].Quantize)
	}
	for name, tensor := range seen {
		if isApertus1p5MediaTensor(name) && tensor.Quantize != "" {
			t.Fatalf("media tensor %q quantize = %q, want source precision", name, tensor.Quantize)
		}
	}
}

func apertus1p5Inventory() Inventory {
	tensors := map[string]SourceTensor{
		"model.language_model.embed_tokens.weight": {Name: "model.language_model.embed_tokens.weight", Dtype: "BF16", Shape: []int32{266752, 4096}},
		"model.language_model.norm.weight":         {Name: "model.language_model.norm.weight", Dtype: "BF16", Shape: []int32{4096}},
		"lm_head.weight":                           {Name: "lm_head.weight", Dtype: "BF16", Shape: []int32{131072, 4096}},
	}
	for i := range 32 {
		for _, suffix := range []struct {
			name  string
			shape []int32
		}{
			{"attention_layernorm.weight", []int32{4096}},
			{"feedforward_layernorm.weight", []int32{4096}},
			{"self_attn.q_proj.weight", []int32{4096, 4096}},
			{"self_attn.k_proj.weight", []int32{1024, 4096}},
			{"self_attn.v_proj.weight", []int32{1024, 4096}},
			{"self_attn.o_proj.weight", []int32{4096, 4096}},
			{"self_attn.q_norm.weight", []int32{128}},
			{"self_attn.k_norm.weight", []int32{128}},
			{"mlp.up_proj.weight", []int32{21504, 4096}},
			{"mlp.down_proj.weight", []int32{4096, 21504}},
			{"mlp.act_fn.alpha_p", []int32{1}},
			{"mlp.act_fn.alpha_n", []int32{1}},
			{"mlp.act_fn.beta", []int32{1}},
			{"mlp.act_fn.eps", []int32{1}},
		} {
			name := "model.language_model.layers." + strconv.Itoa(i) + "." + suffix.name
			tensors[name] = SourceTensor{Name: name, Dtype: "BF16", Shape: suffix.shape}
		}
	}
	for i := range 247 {
		name := "model.vision_tokenizer.synthetic." + strconv.Itoa(i) + ".weight"
		tensors[name] = SourceTensor{Name: name, Dtype: "F32", Shape: []int32{32, 32}}
	}
	for i := range 62 {
		name := "model.audio_tokenizer.encoder.synthetic." + strconv.Itoa(i) + ".weight"
		tensors[name] = SourceTensor{Name: name, Dtype: "F32", Shape: []int32{32, 32}}
	}
	tensors["model.audio_tokenizer.quantizer.codebook.embed"] = SourceTensor{
		Name: "model.audio_tokenizer.quantizer.codebook.embed", Dtype: "F32", Shape: []int32{4096, 512},
	}
	for i := range 163 {
		name := "model.audio_tokenizer.decoder.synthetic." + strconv.Itoa(i) + ".weight"
		tensors[name] = SourceTensor{Name: name, Dtype: "F32", Shape: []int32{32, 32}}
	}
	return Inventory{
		Config:  sourceModelConfig{Architectures: []string{"Apertus1p5ForConditionalGeneration"}},
		Tensors: tensors,
	}
}
