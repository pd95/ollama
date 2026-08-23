package create

import (
	"io"
	"maps"
	"path/filepath"
	"slices"
	"testing"

	st "github.com/ollama/ollama/x/safetensors"
)

func TestNewTensorImportTransform_GptOSSRegistered(t *testing.T) {
	inv := gptossInventory(nil)
	transform, err := newTensorImportTransform(inv)
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
		{name: "model.embed_tokens.scales", want: "embedding.weight.scale"},
		{name: "model.embed_tokens.biases", want: "embedding.weight.bias"},
		{name: "model.norm.weight", want: "output_norm.weight"},
		{name: "lm_head.weight", want: "output.weight"},
		{name: "lm_head.scales", want: "output.weight.scale"},
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

func TestGPTOSSImportTransformQuantizationType(t *testing.T) {
	transform := &gptossImportTransform{}
	if got := transform.quantizationType("blocks.0.experts.gate_proj.weight", []int32{2, 64, 128}, "mxfp4"); got != "" {
		t.Fatalf("expert quantizationType = %q, want empty for native expert weights", got)
	}
	if got := transform.quantizationType("blocks.0.router.weight", []int32{64, 128}, "mxfp4"); got != "" {
		t.Fatalf("router quantizationType = %q, want empty", got)
	}
	if got := transform.quantizationType("blocks.0.q_proj.weight", []int32{2048, 2048}, "mxfp4"); got != "mxfp4" {
		t.Fatalf("q_proj quantizationType = %q, want mxfp4", got)
	}
}

func TestClassify_GptOSSMatchingQuantizePreservesNative(t *testing.T) {
	inv := gptossInventory(map[string]SourceTensor{
		"model.layers.0.mlp.experts.down_proj_blocks": {Name: "model.layers.0.mlp.experts.down_proj_blocks", Dtype: "U8", Shape: []int32{2, 16, 1, 16}},
		"model.layers.0.mlp.experts.down_proj_scales": {Name: "model.layers.0.mlp.experts.down_proj_scales", Dtype: "U8", Shape: []int32{2, 16, 1}},
		"model.layers.0.mlp.experts.down_proj_bias":   {Name: "model.layers.0.mlp.experts.down_proj_bias", Dtype: "BF16", Shape: []int32{2, 16}},
	})

	class, err := Classify(inv, "mxfp4")
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if class.Kind != SourcePrequantized || class.Quantize != "mxfp4" {
		t.Fatalf("Classify() = %+v, want prequantized native preservation", class)
	}
}

func TestClassifyGPTOSSNativeMXFP4TrustBoundary(t *testing.T) {
	valid := map[string]SourceTensor{
		"model.layers.0.mlp.experts.down_proj_blocks": {Name: "model.layers.0.mlp.experts.down_proj_blocks", Dtype: "U8", Shape: []int32{2, 16, 1, 16}},
		"model.layers.0.mlp.experts.down_proj_scales": {Name: "model.layers.0.mlp.experts.down_proj_scales", Dtype: "U8", Shape: []int32{2, 16, 1}},
		"model.layers.0.mlp.experts.down_proj_bias":   {Name: "model.layers.0.mlp.experts.down_proj_bias", Dtype: "BF16", Shape: []int32{2, 16}},
	}
	for _, requested := range []string{"", "mxfp4"} {
		class, err := Classify(gptossInventory(valid), requested)
		if err != nil {
			t.Fatalf("Classify(requested=%q) error = %v", requested, err)
		}
		if class.Kind != SourcePrequantized || class.Quantize != "mxfp4" {
			t.Fatalf("Classify(requested=%q) = %+v", requested, class)
		}
	}

	// Native GPT-OSS checkpoints have shipped with quantization metadata in
	// more than one shape. The packed expert layout, rather than an optional
	// metadata spelling, is the import contract.
	withoutMetadata := gptossInventory(valid)
	withoutMetadata.Config.Quantization = sourceQuantization{}
	withoutMetadata.RawConfig = []byte(`{"architectures":["GptOssForCausalLM"]}`)
	class, err := Classify(withoutMetadata, "mxfp4")
	if err != nil {
		t.Fatalf("Classify(native layout without quantization metadata) error = %v", err)
	}
	if class.Kind != SourcePrequantized || class.Quantize != "mxfp4" {
		t.Fatalf("Classify(native layout without quantization metadata) = %+v", class)
	}

	for _, tt := range []struct {
		name           string
		mutate         func(*Inventory)
		checkNoRequest bool
	}{
		{"missing scales", func(inv *Inventory) { delete(inv.Tensors, "model.layers.0.mlp.experts.down_proj_scales") }, false},
		{"missing bias", func(inv *Inventory) { delete(inv.Tensors, "model.layers.0.mlp.experts.down_proj_bias") }, false},
		{"mismatched scales", func(inv *Inventory) {
			inv.Tensors["model.layers.0.mlp.experts.down_proj_scales"] = SourceTensor{Name: "model.layers.0.mlp.experts.down_proj_scales", Dtype: "U8", Shape: []int32{2, 15, 1}}
		}, false},
		{"wrong blocks dtype", func(inv *Inventory) {
			inv.Tensors["model.layers.0.mlp.experts.down_proj_blocks"] = SourceTensor{Name: "model.layers.0.mlp.experts.down_proj_blocks", Dtype: "BF16", Shape: []int32{2, 16, 1, 16}}
		}, false},
		{"architecture only", func(inv *Inventory) {
			clear(inv.Tensors)
			inv.Tensors["model.layers.0.weight"] = SourceTensor{Name: "model.layers.0.weight", Dtype: "U32", Shape: []int32{16, 4}}
			inv.Tensors["model.layers.0.scales"] = SourceTensor{Name: "model.layers.0.scales", Dtype: "U8", Shape: []int32{16, 1}}
		}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			inv := gptossInventory(valid)
			inv.Tensors = maps.Clone(inv.Tensors)
			tt.mutate(&inv)
			if _, err := Classify(inv, "mxfp4"); err == nil {
				t.Fatal("unproven native preservation accepted")
			}
			if tt.checkNoRequest {
				class, err := Classify(inv, "")
				if err != nil {
					t.Fatalf("no-request classification error = %v", err)
				}
				if class.Kind != SourcePrequantized || class.Quantize != "" {
					t.Fatalf("no-request classification = %+v, want unlabelled prequantized source", class)
				}
			}
		})
	}
}

func TestCreatePipelineReportsGPTOSSMXFP4FileType(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{
		"architectures":["GptOssForCausalLM"],
		"quantization":{"quant_method":"mxfp4","mode":"mxfp4","bits":4,"group_size":32}
	}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_blocks", "U8", []int32{1, 2, 1, 16}, make([]byte, 32)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_scales", "U8", []int32{1, 2, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.down_proj_bias", "BF16", []int32{1, 2}, make([]byte, 4)),
	})

	var got Classification
	err := Create("gptoss", dir, "mxfp4", newCaptureStore(), func(_ string, _ LayerInfo, _ []LayerInfo, class Classification) error {
		got = class
		return nil
	}, func(string) {})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if got.Kind != SourcePrequantized || got.Quantize != "mxfp4" {
		t.Fatalf("manifest classification = {%s %q}, want {prequantized mxfp4}", got.Kind, got.Quantize)
	}
}

func TestPlanGPTOSSNativeExperts(t *testing.T) {
	inv := gptossInventory(map[string]SourceTensor{
		"model.layers.0.mlp.experts.gate_up_proj_blocks": {Name: "model.layers.0.mlp.experts.gate_up_proj_blocks", Dtype: "U8", Shape: []int32{2, 32, 1, 16}},
		"model.layers.0.mlp.experts.gate_up_proj_scales": {Name: "model.layers.0.mlp.experts.gate_up_proj_scales", Dtype: "U8", Shape: []int32{2, 32, 1}},
		"model.layers.0.mlp.experts.gate_up_proj_bias":   {Name: "model.layers.0.mlp.experts.gate_up_proj_bias", Dtype: "BF16", Shape: []int32{2, 32}},
		"model.layers.0.mlp.experts.down_proj_blocks":    {Name: "model.layers.0.mlp.experts.down_proj_blocks", Dtype: "U8", Shape: []int32{2, 16, 1, 16}},
		"model.layers.0.mlp.experts.down_proj_scales":    {Name: "model.layers.0.mlp.experts.down_proj_scales", Dtype: "U8", Shape: []int32{2, 16, 1}},
		"model.layers.0.mlp.experts.down_proj_bias":      {Name: "model.layers.0.mlp.experts.down_proj_bias", Dtype: "BF16", Shape: []int32{2, 16}},
		"model.layers.0.mlp.router.weight":               {Name: "model.layers.0.mlp.router.weight", Dtype: "BF16", Shape: []int32{2, 2}},
	})
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatalf("newTensorImportTransform() error = %v", err)
	}

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, policy)
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}

	experts, ok := specByName(specs, "blocks.0.experts")
	if !ok {
		t.Fatalf("missing blocks.0.experts spec; got %v", specNames(specs))
	}
	if experts.Metadata["quant_type"] != "mxfp4" || experts.Metadata["group_size"] != "32" {
		t.Fatalf("expert metadata = %v, want mxfp4 group_size=32", experts.Metadata)
	}

	got := tensorNames(experts.Tensors)
	want := []string{
		"blocks.0.experts.down_proj.bias",
		"blocks.0.experts.down_proj.weight",
		"blocks.0.experts.down_proj.weight.scale",
		"blocks.0.experts.gate_proj.bias",
		"blocks.0.experts.gate_proj.weight",
		"blocks.0.experts.gate_proj.weight.scale",
		"blocks.0.experts.up_proj.bias",
		"blocks.0.experts.up_proj.weight",
		"blocks.0.experts.up_proj.weight.scale",
	}
	slices.Sort(got)
	if !slices.Equal(got, want) {
		t.Fatalf("expert tensor names = %v, want %v", got, want)
	}

	router, ok := specByName(specs, "blocks.0.router.weight")
	if !ok {
		t.Fatalf("missing router spec; got %v", specNames(specs))
	}
	if router.Tensors[0].Quantize != "" {
		t.Fatalf("router Quantize = %q, want empty", router.Tensors[0].Quantize)
	}
}

func TestPlanGPTOSSNativeDenseQuantizedBlob(t *testing.T) {
	inv := gptossInventory(map[string]SourceTensor{
		"model.layers.0.self_attn.q_proj.weight": {Name: "model.layers.0.self_attn.q_proj.weight", Dtype: "U32", Shape: []int32{64, 8}},
		"model.layers.0.self_attn.q_proj.scales": {Name: "model.layers.0.self_attn.q_proj.scales", Dtype: "U8", Shape: []int32{64, 2}},
		"model.layers.0.self_attn.q_proj.biases": {Name: "model.layers.0.self_attn.q_proj.biases", Dtype: "BF16", Shape: []int32{64, 2}},
	})
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatalf("newTensorImportTransform() error = %v", err)
	}

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, policy)
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	spec, ok := specByName(specs, "blocks.0.q_proj.weight")
	if !ok {
		t.Fatalf("missing q_proj spec; got %v", specNames(specs))
	}
	got := tensorNames(spec.Tensors)
	want := []string{"blocks.0.q_proj.weight", "blocks.0.q_proj.weight.bias", "blocks.0.q_proj.weight.scale"}
	slices.Sort(got)
	if !slices.Equal(got, want) {
		t.Fatalf("q_proj tensor names = %v, want %v", got, want)
	}
	if spec.Metadata["quant_type"] != "mxfp4" || spec.Metadata["group_size"] != "32" {
		t.Fatalf("q_proj metadata = %v, want mxfp4 group_size=32", spec.Metadata)
	}
}

func TestGPTOSSNativeGateUpSplitDequantizesLikeOriginal(t *testing.T) {
	raw := []byte{
		0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
		0x11, 0x33, 0x55, 0x77, 0x99, 0xbb, 0xdd, 0xff, 0x00, 0x22, 0x44, 0x66, 0x88, 0xaa, 0xcc, 0xee,
		0x89, 0x67, 0x45, 0x23, 0x01, 0xef, 0xcd, 0xab, 0x98, 0x76, 0x54, 0x32, 0x10, 0xfe, 0xdc, 0xba,
		0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00,
	}
	scales := []byte{0x10, 0x11, 0x12, 0x13}

	whole, err := dequantizeGPTOSSMXFP4Tensor(
		"blocks.0.experts.gate_up_proj.weight",
		st.NewTensorDataFromBytes("blocks.0.experts.gate_up_proj.weight", "U8", []int32{1, 4, 1, 16}, raw),
		st.NewTensorDataFromBytes("blocks.0.experts.gate_up_proj.weight", "U8", []int32{1, 4, 1}, scales),
	)
	if err != nil {
		t.Fatalf("dequantize whole gate_up tensor: %v", err)
	}
	wholeVals := mustDecodeBF16Tensor(t, whole)

	out, err := preserveAndSplitGateUpTensor(
		"blocks.0.experts.gate_up_proj.weight",
		st.NewTensorDataFromBytes("blocks.0.experts.gate_up_proj.weight", "U8", []int32{1, 4, 1, 16}, raw),
		st.NewTensorDataFromBytes("blocks.0.experts.gate_up_proj.weight", "U8", []int32{1, 4, 1}, scales),
	)
	if err != nil {
		t.Fatalf("preserveAndSplitGateUpTensor() error = %v", err)
	}

	gateRaw, err := io.ReadAll(out[0].Reader())
	if err != nil {
		t.Fatalf("read gate rows: %v", err)
	}
	gateScaleRaw, err := io.ReadAll(out[1].Reader())
	if err != nil {
		t.Fatalf("read gate scales: %v", err)
	}
	upRaw, err := io.ReadAll(out[2].Reader())
	if err != nil {
		t.Fatalf("read up rows: %v", err)
	}
	upScaleRaw, err := io.ReadAll(out[3].Reader())
	if err != nil {
		t.Fatalf("read up scales: %v", err)
	}

	evenVals := decodeNativePackedMXFP4Values(t, gateRaw, gateScaleRaw)
	oddVals := decodeNativePackedMXFP4Values(t, upRaw, upScaleRaw)
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

func gptossInventory(tensors map[string]SourceTensor) Inventory {
	if tensors == nil {
		tensors = map[string]SourceTensor{}
	}
	return Inventory{
		Config: sourceModelConfig{
			Architectures: []string{"GptOssForCausalLM"},
			Quantization:  sourceQuantization{QuantMethod: "mxfp4", Mode: "mxfp4", Bits: 4, GroupSize: 32},
		},
		RawConfig: []byte(`{
			"architectures": ["GptOssForCausalLM"],
			"quantization": {"quant_method": "mxfp4", "bits": 4, "group_size": 32}
		}`),
		Tensors: tensors,
	}
}

func tensorNames(tensors []TensorSpec) []string {
	names := make([]string, 0, len(tensors))
	for _, tensor := range tensors {
		names = append(names, tensor.Name)
	}
	return names
}

func decodeNativePackedMXFP4Values(t *testing.T, blocks, scales []byte) []float32 {
	t.Helper()
	if len(blocks)%16 != 0 {
		t.Fatalf("native block byte length = %d, want multiple of 16", len(blocks))
	}
	groups := len(blocks) / 16
	if len(scales) != groups {
		t.Fatalf("native scale byte length = %d, want %d", len(scales), groups)
	}
	values := make([]float32, groups*32)
	for i := range groups {
		scale := decodeGPTOSSMXFP4Scale(scales[i])
		for j, packed := range blocks[i*16 : (i+1)*16] {
			values[i*32+2*j] = gptossMXFP4Values[packed&0x0F] * scale
			values[i*32+2*j+1] = gptossMXFP4Values[packed>>4] * scale
		}
	}
	return values
}

func mustDecodeBF16Tensor(t *testing.T, td *st.TensorData) []float32 {
	t.Helper()
	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		t.Fatalf("read tensor: %v", err)
	}
	values, err := DecodeFloatTensor(td.Dtype, raw)
	if err != nil {
		t.Fatalf("decode tensor: %v", err)
	}
	return values
}
