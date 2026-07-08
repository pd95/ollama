package create

import "testing"

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
