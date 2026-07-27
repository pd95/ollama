package create

import (
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

	tests := []struct {
		name     string
		tensor   string
		shape    []int32
		quantize string
		want     string
	}{
		{
			name:     "lm head quantizes to requested fp4",
			tensor:   "lm_head.weight",
			shape:    []int32{131072, 4096},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "k projection is not promoted",
			tensor:   "model.layers.0.self_attn.k_proj.weight",
			shape:    []int32{1024, 4096},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "v projection is not promoted",
			tensor:   "model.layers.0.self_attn.v_proj.weight",
			shape:    []int32{1024, 4096},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "down projection is not promoted",
			tensor:   "model.layers.0.mlp.down_proj.weight",
			shape:    []int32{4096, 21504},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "embeddings stay source precision",
			tensor:   "model.embed_tokens.weight",
			shape:    []int32{131072, 4096},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "apertus 1.5 embeddings stay source precision",
			tensor:   "model.language_model.embed_tokens.weight",
			shape:    []int32{266752, 4096},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "apertus 1.5 decoder weights quantize",
			tensor:   "model.language_model.layers.0.self_attn.q_proj.weight",
			shape:    []int32{4096, 4096},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "small tensors stay source precision",
			tensor:   "model.layers.0.mlp.act_fn.alpha_n.weight",
			shape:    []int32{1},
			quantize: "nvfp4",
			want:     "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := transform.quantizationType(tt.tensor, tt.shape, tt.quantize)
			if got != tt.want {
				t.Errorf("quantizationType(%q, %v, %q) = %q, want %q", tt.tensor, tt.shape, tt.quantize, got, tt.want)
			}
		})
	}
}

func TestApertus1p5TextPlanFiltersMediaTensors(t *testing.T) {
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
			if !isApertus1p5TextTensor(tensor.Name) {
				t.Fatalf("planned media/non-text tensor %q", tensor.Name)
			}
			if tensor.Name != tensor.Sources[0].Name {
				t.Fatalf("tensor name %q rewrote source %q", tensor.Name, tensor.Sources[0].Name)
			}
		}
	}
	if tensorCount != 451 {
		t.Fatalf("planned tensor count = %d, want 451", tensorCount)
	}
	if got := len(inv.Tensors) - tensorCount; got != 473 {
		t.Fatalf("filtered tensor count = %d, want 473", got)
	}
	if seen["model.language_model.embed_tokens.weight"].Quantize != "" {
		t.Fatalf("embedding quantize = %q, want source precision", seen["model.language_model.embed_tokens.weight"].Quantize)
	}
	if seen["lm_head.weight"].Quantize != "nvfp4" {
		t.Fatalf("lm_head quantize = %q, want nvfp4", seen["lm_head.weight"].Quantize)
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
	for i := range 473 {
		name := "model.vision_tokenizer.synthetic." + strconv.Itoa(i) + ".weight"
		if i%2 == 1 {
			name = "model.audio_tokenizer.synthetic." + strconv.Itoa(i) + ".weight"
		}
		tensors[name] = SourceTensor{Name: name, Dtype: "F32", Shape: []int32{32, 32}}
	}
	return Inventory{
		Config:  sourceModelConfig{Architectures: []string{"Apertus1p5ForConditionalGeneration"}},
		Tensors: tensors,
	}
}
