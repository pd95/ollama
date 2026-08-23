package create

import (
	"math"
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
