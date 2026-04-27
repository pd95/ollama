package create

import (
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
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

	if _, ok := transform.(*gptossImportTransform); !ok {
		t.Fatalf("newTensorImportTransform() type = %T, want *gptossImportTransform", transform)
	}
}

func TestGPTOSSImportTransformRenamesTensors(t *testing.T) {
	transform := &gptossImportTransform{}

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
		{name: "model.layers.2.mlp.experts.gate_up_proj_blocks", want: "blocks.2.experts.gate_up_proj.weight"},
		{name: "model.layers.2.mlp.experts.gate_up_proj_scales", want: "blocks.2.experts.gate_up_proj.weight"},
		{name: "model.layers.2.mlp.experts.gate_up_proj_bias", want: "blocks.2.experts.gate_up_proj.bias"},
		{name: "model.layers.2.mlp.experts.down_proj_blocks", want: "blocks.2.experts.down_proj.weight"},
		{name: "model.layers.2.mlp.experts.down_proj_scales", want: "blocks.2.experts.down_proj.weight"},
		{name: "model.layers.2.mlp.experts.down_proj_bias", want: "blocks.2.experts.down_proj.bias"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := transform.canonicalTensorName(tt.name); got != tt.want {
				t.Fatalf("canonicalTensorName(%q) = %q, want %q", tt.name, got, tt.want)
			}
		})
	}
}

func TestGPTOSSImportTransformQuantizationType_Experts(t *testing.T) {
	transform := &gptossImportTransform{}
	shape := []int32{2, 64, 128}

	for _, quantize := range []string{"int8", "int4", "nvfp4", "mxfp4", "mxfp8"} {
		t.Run(quantize, func(t *testing.T) {
			if got := transform.quantizationType("blocks.0.experts.gate_proj.weight", shape, quantize); got != quantize {
				t.Fatalf("quantizationType(%s) = %q, want %q", quantize, got, quantize)
			}
		})
	}
}

func TestGPTOSSImportTransformQuantizationType_RouterKeptSourcePrecision(t *testing.T) {
	transform := &gptossImportTransform{}
	shape := []int32{64, 128}

	for _, quantize := range []string{"int4", "int8", "nvfp4", "mxfp4", "mxfp8"} {
		if got := transform.quantizationType("blocks.0.router.weight", shape, quantize); got != "" {
			t.Fatalf("quantizationType(%q) = %q for router weight, want empty (source precision)", quantize, got)
		}
	}
}

func TestGPTOSSImportTransformDequantizesExpertWeights(t *testing.T) {
	transform := &gptossImportTransform{
		pendingBlocks: make(map[string]*st.TensorData),
		pendingScales: make(map[string]*st.TensorData),
	}
	raw := []byte{
		0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
		0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
		0x11, 0x33, 0x55, 0x77, 0x99, 0xbb, 0xdd, 0xff,
		0x00, 0x22, 0x44, 0x66, 0x88, 0xaa, 0xcc, 0xee,
	}
	scales := []byte{0x00, 0x10}

	out, err := transform.transformTensor(st.NewTensorDataFromBytes("model.layers.0.mlp.experts.gate_up_proj_blocks", "U8", []int32{1, 2, 1, 16}, raw))
	if err != nil {
		t.Fatalf("transformTensor() error = %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("blocks transform returned %d tensors, want 0 until scales arrive", len(out))
	}

	out, err = transform.transformTensor(st.NewTensorDataFromBytes("model.layers.0.mlp.experts.gate_up_proj_scales", "U8", []int32{1, 2, 1}, scales))
	if err != nil {
		t.Fatalf("transformTensor() error = %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("scales transform returned %d tensors, want 2", len(out))
	}
	if out[0].Name != "blocks.0.experts.gate_proj.weight" {
		t.Fatalf("transformTensor() gate name = %q, want %q", out[0].Name, "blocks.0.experts.gate_proj.weight")
	}
	if out[1].Name != "blocks.0.experts.up_proj.weight" {
		t.Fatalf("transformTensor() up name = %q, want %q", out[1].Name, "blocks.0.experts.up_proj.weight")
	}
	if out[0].Dtype != "BF16" || out[1].Dtype != "BF16" {
		t.Fatalf("transformTensor() dtypes = %q/%q, want BF16/BF16", out[0].Dtype, out[1].Dtype)
	}
	if !slices.Equal(out[0].Shape, []int32{1, 1, 32}) {
		t.Fatalf("transformTensor() gate shape = %v, want [1 1 32]", out[0].Shape)
	}
	if !slices.Equal(out[1].Shape, []int32{1, 1, 32}) {
		t.Fatalf("transformTensor() up shape = %v, want [1 1 32]", out[1].Shape)
	}
	got, err := io.ReadAll(out[0].Reader())
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if len(got) != 32*2 {
		t.Fatalf("dequantized gate byte length = %d, want %d", len(got), 32*2)
	}
	if slices.Equal(got, make([]byte, len(got))) {
		t.Fatal("dequantized gate bytes are all zero, want non-zero output for non-zero source blocks")
	}
}

func TestGPTOSSDequantizeGateUpSplitParity(t *testing.T) {
	raw := []byte{
		0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
		0x11, 0x33, 0x55, 0x77, 0x99, 0xbb, 0xdd, 0xff, 0x00, 0x22, 0x44, 0x66, 0x88, 0xaa, 0xcc, 0xee,
		0x89, 0x67, 0x45, 0x23, 0x01, 0xef, 0xcd, 0xab, 0x98, 0x76, 0x54, 0x32, 0x10, 0xfe, 0xdc, 0xba,
		0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00,
	}
	scales := []byte{0x00, 0x10, 0x20, 0x30}

	whole, err := dequantizeGPTOSSMXFP4Tensor(
		"blocks.0.experts.gate_up_proj.weight",
		st.NewTensorDataFromBytes("blocks.0.experts.gate_up_proj.weight", "U8", []int32{1, 4, 1, 16}, raw),
		st.NewTensorDataFromBytes("blocks.0.experts.gate_up_proj.weight", "U8", []int32{1, 4, 1}, scales),
	)
	if err != nil {
		t.Fatalf("dequantize whole gate_up tensor: %v", err)
	}
	wholeVals := mustDecodeBF16Tensor(t, whole)
	if len(wholeVals) != 4*32 {
		t.Fatalf("whole gate_up decoded %d values, want %d", len(wholeVals), 4*32)
	}

	even, err := dequantizeGPTOSSMXFP4Tensor(
		"blocks.0.experts.gate_proj.weight",
		st.NewTensorDataFromBytes("blocks.0.experts.gate_proj.weight", "U8", []int32{1, 2, 1, 16}, append(append([]byte{}, raw[:16]...), raw[32:48]...)),
		st.NewTensorDataFromBytes("blocks.0.experts.gate_proj.weight", "U8", []int32{1, 2, 1}, []byte{scales[0], scales[2]}),
	)
	if err != nil {
		t.Fatalf("dequantize even gate rows: %v", err)
	}
	odd, err := dequantizeGPTOSSMXFP4Tensor(
		"blocks.0.experts.up_proj.weight",
		st.NewTensorDataFromBytes("blocks.0.experts.up_proj.weight", "U8", []int32{1, 2, 1, 16}, append(append([]byte{}, raw[16:32]...), raw[48:64]...)),
		st.NewTensorDataFromBytes("blocks.0.experts.up_proj.weight", "U8", []int32{1, 2, 1}, []byte{scales[1], scales[3]}),
	)
	if err != nil {
		t.Fatalf("dequantize odd up rows: %v", err)
	}

	evenVals := mustDecodeBF16Tensor(t, even)
	oddVals := mustDecodeBF16Tensor(t, odd)
	if len(evenVals) != 2*32 || len(oddVals) != 2*32 {
		t.Fatalf("split decode lengths = %d/%d, want %d/%d", len(evenVals), len(oddVals), 2*32, 2*32)
	}

	for i := range 2 {
		for j := range 32 {
			if wholeVals[(i*2)*32+j] != evenVals[i*32+j] {
				t.Fatalf("even row %d col %d mismatch: whole=%v split=%v", i, j, wholeVals[(i*2)*32+j], evenVals[i*32+j])
			}
			if wholeVals[(i*2+1)*32+j] != oddVals[i*32+j] {
				t.Fatalf("odd row %d col %d mismatch: whole=%v split=%v", i, j, wholeVals[(i*2+1)*32+j], oddVals[i*32+j])
			}
		}
	}
}

func mustDecodeBF16Tensor(t *testing.T, td *st.TensorData) []float32 {
	t.Helper()
	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		t.Fatalf("ReadAll(%q): %v", td.Name, err)
	}
	values, err := DecodeFloatTensor(td.Dtype, raw)
	if err != nil {
		t.Fatalf("DecodeFloatTensor(%q): %v", td.Name, err)
	}
	return values
}

func TestExpertGroupPrefix_GptOSSBlocksExperts(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{name: "blocks.0.experts.gate_up_proj.weight", want: "blocks.0.experts"},
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
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.gate_up_proj_blocks", "U8", []int32{2, 32, 1, 16}, make([]byte, 2*32*16)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.gate_up_proj_scales", "U8", []int32{2, 32, 1}, make([]byte, 2*32)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.gate_up_proj_bias", "BF16", []int32{2, 32}, make([]byte, 2*32*2)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_blocks", "U8", []int32{2, 16, 1, 16}, make([]byte, 2*16*16)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_scales", "U8", []int32{2, 16, 1}, make([]byte, 2*16)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_bias", "BF16", []int32{2, 16}, make([]byte, 2*16*2)),
		st.NewTensorDataFromBytes("model.layers.0.post_attention_layernorm.weight", "BF16", []int32{2}, make([]byte, 4)),
	})

	var denseNames []string
	var events []string
	var packedGroupName string
	var packedTensors []PackedTensorInput

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		_, _ = io.ReadAll(r)
		return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
	}

	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		_, _ = io.ReadAll(r)
		denseNames = append(denseNames, name)
		events = append(events, "tensor:"+name)
		return []LayerInfo{{Name: name, Digest: "sha256:" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}

	createPackedLayer := func(groupName string, tensors []PackedTensorInput) (LayerInfo, error) {
		events = append(events, "pack:"+groupName)
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
	packIdx := slices.Index(events, "pack:blocks.0.experts")
	postIdx := slices.Index(events, "tensor:blocks.0.ffn_norm.weight")
	if packIdx == -1 || postIdx == -1 || packIdx > postIdx {
		t.Fatalf("packed group was not flushed as soon as complete; events=%v", events)
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
		"blocks.0.experts.down_proj.weight",
		"blocks.0.experts.gate_proj.bias",
		"blocks.0.experts.gate_proj.weight",
		"blocks.0.experts.up_proj.bias",
		"blocks.0.experts.up_proj.weight",
	}
	if !slices.Equal(gotPackedNames, wantPackedNames) {
		t.Fatalf("packed names = %v, want %v", gotPackedNames, wantPackedNames)
	}

	for _, tensor := range packedTensors {
		t.Logf("packed tensor %s shape=%v dtype=%s quantize=%q", tensor.Name, tensor.Shape, tensor.Dtype, tensor.Quantize)
		if tensor.Reader == nil {
			t.Fatalf("packed tensor %q reader = nil, want safetensors reader", tensor.Name)
		}
		if strings.HasSuffix(tensor.Name, ".weight") {
			if tensor.Dtype != "BF16" {
				t.Fatalf("packed tensor %q dtype = %q, want BF16", tensor.Name, tensor.Dtype)
			}
			if tensor.Quantize != "" {
				t.Fatalf("packed tensor %q quantize = %q, want empty", tensor.Name, tensor.Quantize)
			}
			if !slices.Equal(tensor.Shape, []int32{2, 16, 32}) {
				t.Fatalf("packed tensor %q shape = %v, want [2 16 32]", tensor.Name, tensor.Shape)
			}
		}
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
		"blocks.0.ffn_norm.weight",
	}
	for _, name := range wantDenseNames {
		if !slices.Contains(denseNames, name) {
			t.Fatalf("dense tensor %q not seen in createTensorLayer; got %v", name, denseNames)
		}
	}
}

func TestCreateSafetensorsModel_GptOSSPacksExpertsQuantized(t *testing.T) {
	for _, quantize := range []string{"nvfp4", "mxfp4", "mxfp8"} {
		t.Run(quantize, func(t *testing.T) {
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
				st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
				st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{64}, make([]byte, 64*2)),
				st.NewTensorDataFromBytes("lm_head.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
				st.NewTensorDataFromBytes("model.layers.0.input_layernorm.weight", "BF16", []int32{64}, make([]byte, 64*2)),
				st.NewTensorDataFromBytes("model.layers.0.self_attn.q_proj.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
				st.NewTensorDataFromBytes("model.layers.0.self_attn.sinks", "BF16", []int32{64}, make([]byte, 64*2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.router.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.gate_up_proj_blocks", "U8", []int32{2, 32, 1, 16}, make([]byte, 2*32*16)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.gate_up_proj_scales", "U8", []int32{2, 32, 1}, make([]byte, 2*32)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.gate_up_proj_bias", "BF16", []int32{2, 32}, make([]byte, 2*32*2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_blocks", "U8", []int32{2, 16, 1, 16}, make([]byte, 2*16*16)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_scales", "U8", []int32{2, 16, 1}, make([]byte, 2*16)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_bias", "BF16", []int32{2, 16}, make([]byte, 2*16*2)),
			})

			tensorQuantize := make(map[string]string)
			var packedTensors []PackedTensorInput

			createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
				_, _ = io.ReadAll(r)
				return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
			}

			createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
				_, _ = io.ReadAll(r)
				tensorQuantize[name] = quantize
				return []LayerInfo{{Name: name, Digest: "sha256:" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
			}

			createPackedLayer := func(groupName string, tensors []PackedTensorInput) (LayerInfo, error) {
				if groupName != "blocks.0.experts" {
					t.Fatalf("packed group = %q, want %q", groupName, "blocks.0.experts")
				}
				packedTensors = append([]PackedTensorInput(nil), tensors...)
				return LayerInfo{Name: groupName, Digest: "sha256:" + groupName, MediaType: "application/vnd.ollama.image.tensor"}, nil
			}

			writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }

			if err := CreateSafetensorsModel("test-model", dir, quantize, createLayer, createTensorLayer, writeManifest, func(string) {}, createPackedLayer); err != nil {
				t.Fatalf("CreateSafetensorsModel() error = %v", err)
			}

			if len(packedTensors) != 6 {
				t.Fatalf("packed tensor count = %d, want 6", len(packedTensors))
			}

			for _, tensor := range packedTensors {
				if tensor.Reader == nil {
					t.Fatalf("packed tensor %q reader = nil, want safetensors reader", tensor.Name)
				}
				if strings.HasSuffix(tensor.Name, ".weight") {
					if tensor.Dtype != "BF16" {
						t.Fatalf("packed tensor %q dtype = %q, want BF16 before quantization", tensor.Name, tensor.Dtype)
					}
					if tensor.Quantize != quantize {
						t.Fatalf("packed tensor %q quantize = %q, want %q", tensor.Name, tensor.Quantize, quantize)
					}
					if !slices.Equal(tensor.Shape, []int32{2, 16, 32}) {
						t.Fatalf("packed tensor %q shape = %v, want [2 16 32]", tensor.Name, tensor.Shape)
					}
					continue
				}
				if tensor.Quantize != "" {
					t.Fatalf("packed tensor %q quantize = %q, want empty for bias", tensor.Name, tensor.Quantize)
				}
			}

			switch quantize {
			case "nvfp4":
				if got := tensorQuantize["output.weight"]; got != "int8" {
					t.Fatalf("output.weight quantize = %q, want int8 for nvfp4 GPT-OSS non-expert path", got)
				}
				if got := tensorQuantize["blocks.0.q_proj.weight"]; got != "int8" {
					t.Fatalf("blocks.0.q_proj.weight quantize = %q, want int8 for nvfp4 GPT-OSS non-expert path", got)
				}
			default:
				if got := tensorQuantize["output.weight"]; got != quantize {
					t.Fatalf("output.weight quantize = %q, want %q", got, quantize)
				}
				if got := tensorQuantize["blocks.0.q_proj.weight"]; got != quantize {
					t.Fatalf("blocks.0.q_proj.weight quantize = %q, want %q", got, quantize)
				}
			}
			if got := tensorQuantize["blocks.0.router.weight"]; got != "" {
				t.Fatalf("blocks.0.router.weight quantize = %q, want empty", got)
			}
		})
	}
}
