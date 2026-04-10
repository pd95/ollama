package create

import (
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	st "github.com/ollama/ollama/x/safetensors"
)

func TestNewTensorImportTransform_GptOSSRegistered(t *testing.T) {
	transform, err := newTensorImportTransform(t.TempDir(), sourceModelConfig{
		Architectures: []string{"GptOssForCausalLM"},
	})
	if err != nil {
		t.Fatalf("newTensorImportTransform() error = %v", err)
	}

	if _, ok := transform.(gptossImportTransform); !ok {
		t.Fatalf("newTensorImportTransform() type = %T, want gptossImportTransform", transform)
	}
}

func TestGPTOSSImportTransformRenamesTensors(t *testing.T) {
	transform := gptossImportTransform{}

	tests := []struct {
		name string
		want string
	}{
		{name: "model.embed_tokens.weight", want: "embedding.weight"},
		{name: "model.norm.weight", want: "output_norm.weight"},
		{name: "lm_head.weight", want: "output.weight"},
		{name: "model.layers.2.input_layernorm.weight", want: "blocks.2.attn_norm.weight"},
		{name: "model.layers.2.self_attn.q_proj.weight", want: "blocks.2.q_proj.weight"},
		{name: "model.layers.2.self_attn.k_proj.bias", want: "blocks.2.k_proj.bias"},
		{name: "model.layers.2.self_attn.sinks", want: "blocks.2.attn_sinks"},
		{name: "model.layers.2.post_attention_layernorm.weight", want: "blocks.2.ffn_norm.weight"},
		{name: "model.layers.2.mlp.router.weight", want: "blocks.2.router.weight"},
		{name: "model.layers.2.mlp.experts.gate_up_proj_blocks", want: "blocks.2.experts.gate_up_proj.blocks"},
		{name: "model.layers.2.mlp.experts.gate_up_proj_scales", want: "blocks.2.experts.gate_up_proj.scales"},
		{name: "model.layers.2.mlp.experts.gate_up_proj_bias", want: "blocks.2.experts.gate_up_proj.bias"},
		{name: "model.layers.2.mlp.experts.down_proj_blocks", want: "blocks.2.experts.down_proj.blocks"},
		{name: "model.layers.2.mlp.experts.down_proj_scales", want: "blocks.2.experts.down_proj.scales"},
		{name: "model.layers.2.mlp.experts.down_proj_bias", want: "blocks.2.experts.down_proj.bias"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			td := st.NewTensorDataFromBytes(tt.name, "BF16", []int32{2, 2}, make([]byte, 8))
			out, err := transform.transformTensor(td)
			if err != nil {
				t.Fatalf("transformTensor() error = %v", err)
			}
			if len(out) != 1 {
				t.Fatalf("transformTensor() returned %d tensors, want 1", len(out))
			}
			if out[0].Name != tt.want {
				t.Fatalf("transformTensor() name = %q, want %q", out[0].Name, tt.want)
			}
		})
	}
}

func TestExpertGroupPrefix_GptOSSBlocksExperts(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{name: "blocks.0.experts.gate_up_proj.blocks", want: "blocks.0.experts"},
		{name: "blocks.0.experts.gate_up_proj.scales", want: "blocks.0.experts"},
		{name: "blocks.0.experts.down_proj.bias", want: "blocks.0.experts"},
		{name: "model.layers.1.mlp.experts.0.down_proj.weight", want: "model.layers.1.mlp.experts"},
		{name: "model.layers.1.mlp.shared_experts.down_proj.weight", want: "model.layers.1.mlp.shared_experts"},
		{name: "model.layers.1.mlp.switch_mlp.gate_proj.weight", want: "model.layers.1.mlp.switch_mlp"},
		{name: "model.layers.0.self_attn.q_proj.weight", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ExpertGroupPrefix(tt.name); got != tt.want {
				t.Fatalf("ExpertGroupPrefix(%q) = %q, want %q", tt.name, got, tt.want)
			}
		})
	}
}

func TestCreateSafetensorsModel_GptOSSPacksExperts(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"model_type": "test",
		"architectures": ["GptOssForCausalLM"],
		"quantization_config": {"quant_method": "mxfp4"}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{2, 2}, make([]byte, 8)),
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{2}, make([]byte, 4)),
		st.NewTensorDataFromBytes("lm_head.weight", "BF16", []int32{2, 2}, make([]byte, 8)),
		st.NewTensorDataFromBytes("model.layers.0.input_layernorm.weight", "BF16", []int32{2}, make([]byte, 4)),
		st.NewTensorDataFromBytes("model.layers.0.self_attn.q_proj.weight", "BF16", []int32{2, 2}, make([]byte, 8)),
		st.NewTensorDataFromBytes("model.layers.0.self_attn.sinks", "BF16", []int32{2}, make([]byte, 4)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.router.weight", "BF16", []int32{2, 2}, make([]byte, 8)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.gate_up_proj_blocks", "U8", []int32{2, 2}, make([]byte, 4)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.gate_up_proj_scales", "U8", []int32{2, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.gate_up_proj_bias", "BF16", []int32{2}, make([]byte, 4)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_blocks", "U8", []int32{2, 2}, make([]byte, 4)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_scales", "U8", []int32{2, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_bias", "BF16", []int32{2}, make([]byte, 4)),
	})

	var denseNames []string
	var packedGroupName string
	var packedTensors []PackedTensorInput

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		_, _ = io.ReadAll(r)
		return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
	}

	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		_, _ = io.ReadAll(r)
		denseNames = append(denseNames, name)
		return []LayerInfo{{Name: name, Digest: "sha256:" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}

	createPackedLayer := func(groupName string, tensors []PackedTensorInput) (LayerInfo, error) {
		packedGroupName = groupName
		packedTensors = append([]PackedTensorInput(nil), tensors...)
		return LayerInfo{Name: groupName, Digest: "sha256:" + groupName, MediaType: "application/vnd.ollama.image.tensor"}, nil
	}

	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }

	if err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, func(string) {}, createPackedLayer); err != nil {
		t.Fatalf("CreateSafetensorsModel() error = %v", err)
	}

	if packedGroupName != "blocks.0.experts" {
		t.Fatalf("packedGroupName = %q, want %q", packedGroupName, "blocks.0.experts")
	}
	if len(packedTensors) != 6 {
		t.Fatalf("packed tensor count = %d, want 6", len(packedTensors))
	}

	gotPackedNames := make([]string, 0, len(packedTensors))
	for _, tensor := range packedTensors {
		gotPackedNames = append(gotPackedNames, tensor.Name)
	}
	slices.Sort(gotPackedNames)

	wantPackedNames := []string{
		"blocks.0.experts.down_proj.bias",
		"blocks.0.experts.down_proj.blocks",
		"blocks.0.experts.down_proj.scales",
		"blocks.0.experts.gate_up_proj.bias",
		"blocks.0.experts.gate_up_proj.blocks",
		"blocks.0.experts.gate_up_proj.scales",
	}
	if !slices.Equal(gotPackedNames, wantPackedNames) {
		t.Fatalf("packed names = %v, want %v", gotPackedNames, wantPackedNames)
	}

	for _, name := range wantPackedNames {
		if slices.Contains(denseNames, name) {
			t.Fatalf("expert tensor %q unexpectedly routed through createTensorLayer", name)
		}
	}

	wantDenseNames := []string{
		"embedding.weight",
		"output_norm.weight",
		"output.weight",
		"blocks.0.attn_norm.weight",
		"blocks.0.q_proj.weight",
		"blocks.0.attn_sinks",
		"blocks.0.router.weight",
	}
	for _, name := range wantDenseNames {
		if !slices.Contains(denseNames, name) {
			t.Fatalf("dense tensor %q not seen in createTensorLayer; got %v", name, denseNames)
		}
	}
}
