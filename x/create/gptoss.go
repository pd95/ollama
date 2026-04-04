package create

import (
	"fmt"
	"io"
	"strings"

	"github.com/ollama/ollama/x/imagegen/safetensors"
)

type gptossImportTransform struct{}

var gptossTensorReplacer = strings.NewReplacer(
	"model.embed_tokens", "token_embd",
	"model.layers", "blk",
	"input_layernorm", "attn_norm",
	"self_attn.q_proj", "attn_q",
	"self_attn.k_proj", "attn_k",
	"self_attn.v_proj", "attn_v",
	"self_attn.o_proj", "attn_out",
	"self_attn.sinks", "attn_sinks",
	"post_attention_layernorm", "ffn_norm",
	"mlp.router", "ffn_gate_inp",
	"mlp.experts.gate_up_proj_", "ffn_gate_up_exps.",
	"mlp.experts.gate_up_proj", "ffn_gate_up_exps",
	"mlp.experts.down_proj_", "ffn_down_exps.",
	"mlp.experts.down_proj", "ffn_down_exps",
	"model.norm", "output_norm",
	"lm_head", "output",
)

func newGptOssImportTransform(modelDir string, cfg sourceModelConfig) (tensorImportTransform, error) {
	return gptossImportTransform{}, nil
}

func (gptossImportTransform) skipTensor(string) bool { return false }

func (gptossImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}

	name, ok := gptossTensorName(td.Name)
	if !ok {
		name, ok = gptossTensorBase(td.Name)
		if !ok {
			return []*safetensors.TensorData{td}, nil
		}
	}

	rewritten := td.WithName(name)
	if strings.Contains(name, "ffn_gate_up_exps") {
		return gptossSplitGateUpTensors(rewritten)
	}

	return []*safetensors.TensorData{rewritten}, nil
}

func (gptossImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	return GetTensorQuantization(name, shape, quantize)
}

func (gptossImportTransform) transformPrequantizedTensor(extractor *safetensors.TensorExtractor, td *safetensors.TensorData, tensorSet map[string]struct{}) ([]prequantizedTensorBlob, bool, error) {
	if td == nil {
		return nil, false, nil
	}

	rawBase, base, ok, kind := gptossRawCompanion(td.Name)
	switch {
	case !ok:
		return nil, false, nil
	case kind != "blocks":
		_, blocksOK := tensorSet[rawBase+"_blocks"]
		return nil, blocksOK, nil
	}

	scaleName := rawBase + "_scales"
	if _, ok := tensorSet[scaleName]; !ok {
		return nil, false, nil
	}

	scaleTD, err := extractor.GetTensor(scaleName)
	if err != nil {
		return nil, false, fmt.Errorf("get scale tensor %s: %w", scaleName, err)
	}

	var biasTD *safetensors.TensorData
	biasName := rawBase + "_bias"
	if _, ok := tensorSet[biasName]; ok {
		biasTD, err = extractor.GetTensor(biasName)
		if err != nil {
			return nil, false, fmt.Errorf("get bias tensor %s: %w", biasName, err)
		}
	}

	if strings.Contains(base, "ffn_gate_up_exps") {
		return gptossSplitGateUpBlobs(base, td, scaleTD, biasTD)
	}

	blob, err := gptossPrequantizedBlob(base+".weight", td, scaleTD, biasTD)
	if err != nil {
		return nil, false, err
	}
	return []prequantizedTensorBlob{blob}, true, nil
}

func gptossTensorName(name string) (string, bool) {
	if _, base, ok, kind := gptossRawCompanion(name); ok {
		switch kind {
		case "blocks":
			return base + ".weight", true
		case "scales":
			return base + ".weight.scale", true
		case "bias":
			return base + ".bias", true
		}
	}
	return "", false
}

func gptossRawCompanion(name string) (rawBase, canonicalBase string, ok bool, kind string) {
	for suffix, companionKind := range map[string]string{
		"_blocks": "blocks",
		"_scales": "scales",
		"_bias":   "bias",
		".blocks": "blocks",
		".scales": "scales",
		".biases": "bias",
	} {
		if !strings.HasSuffix(name, suffix) {
			continue
		}
		rawBase = strings.TrimSuffix(name, suffix)
		base, baseOK := gptossTensorBase(rawBase)
		return rawBase, base, baseOK, companionKind
	}
	return "", "", false, ""
}

func gptossTensorBase(name string) (string, bool) {
	rewritten := gptossTensorReplacer.Replace(name)
	if rewritten == name && !strings.HasPrefix(name, "blk.") && !strings.HasPrefix(name, "token_embd") && !strings.HasPrefix(name, "output") {
		return "", false
	}
	return rewritten, true
}

func gptossPrequantizedBlob(weightName string, blocksTD, scalesTD, biasTD *safetensors.TensorData) (prequantizedTensorBlob, error) {
	tensors := []*safetensors.TensorData{blocksTD.WithName(weightName), scalesTD.WithName(weightName + ".scale")}
	if biasTD != nil {
		tensors = append(tensors, biasTD.WithName(strings.TrimSuffix(weightName, ".weight")+".bias"))
	}
	return prequantizedTensorBlob{Name: weightName, Tensors: tensors}, nil
}

func gptossSplitGateUpBlobs(base string, blocksTD, scalesTD, biasTD *safetensors.TensorData) ([]prequantizedTensorBlob, bool, error) {
	gateBase := strings.Replace(base, "gate_up", "gate", 1)
	upBase := strings.Replace(base, "gate_up", "up", 1)

	gateBlocks, upBlocks, blockShape, err := splitTensorAxis1Raw(blocksTD)
	if err != nil {
		return nil, false, fmt.Errorf("split blocks tensor %s: %w", blocksTD.Name, err)
	}
	gateScales, upScales, scaleShape, err := splitTensorAxis1Raw(scalesTD)
	if err != nil {
		return nil, false, fmt.Errorf("split scales tensor %s: %w", scalesTD.Name, err)
	}

	var gateBias, upBias []byte
	var biasShape []int32
	if biasTD != nil {
		gateBias, upBias, biasShape, err = splitTensorAxis1Raw(biasTD)
		if err != nil {
			return nil, false, fmt.Errorf("split bias tensor %s: %w", biasTD.Name, err)
		}
	}

	gateBlob, err := gptossPrequantizedBlob(
		gateBase+".weight",
		safetensors.NewTensorDataFromBytes(gateBase+".weight", blocksTD.Dtype, blockShape, gateBlocks),
		safetensors.NewTensorDataFromBytes(gateBase+".weight.scale", scalesTD.Dtype, scaleShape, gateScales),
		gptossMaybeBiasTensor(gateBase+".bias", biasTD, biasShape, gateBias),
	)
	if err != nil {
		return nil, false, err
	}

	upBlob, err := gptossPrequantizedBlob(
		upBase+".weight",
		safetensors.NewTensorDataFromBytes(upBase+".weight", blocksTD.Dtype, blockShape, upBlocks),
		safetensors.NewTensorDataFromBytes(upBase+".weight.scale", scalesTD.Dtype, scaleShape, upScales),
		gptossMaybeBiasTensor(upBase+".bias", biasTD, biasShape, upBias),
	)
	if err != nil {
		return nil, false, err
	}

	return []prequantizedTensorBlob{gateBlob, upBlob}, true, nil
}

func gptossSplitGateUpTensors(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	switch {
	case strings.HasSuffix(td.Name, ".weight"), strings.HasSuffix(td.Name, ".weight.scale"), strings.HasSuffix(td.Name, ".bias"):
	default:
		return []*safetensors.TensorData{td}, nil
	}

	gateName := strings.Replace(td.Name, "gate_up", "gate", 1)
	upName := strings.Replace(td.Name, "gate_up", "up", 1)

	left, right, shape, err := splitTensorAxis1Raw(td)
	if err != nil {
		return nil, fmt.Errorf("split tensor %s: %w", td.Name, err)
	}

	return []*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes(gateName, td.Dtype, shape, left),
		safetensors.NewTensorDataFromBytes(upName, td.Dtype, shape, right),
	}, nil
}

func gptossMaybeBiasTensor(name string, source *safetensors.TensorData, shape []int32, raw []byte) *safetensors.TensorData {
	if source == nil {
		return nil
	}
	return safetensors.NewTensorDataFromBytes(name, source.Dtype, shape, raw)
}

func splitTensorAxis1Raw(td *safetensors.TensorData) ([]byte, []byte, []int32, error) {
	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		return nil, nil, nil, fmt.Errorf("read tensor %s: %w", td.Name, err)
	}

	shape := td.Shape
	if len(shape) < 2 {
		return nil, nil, nil, fmt.Errorf("expected rank >= 2, got shape %v", shape)
	}
	if shape[1]%2 != 0 {
		return nil, nil, nil, fmt.Errorf("axis 1 dim %d is not even", shape[1])
	}

	elemSize, err := DTypeSize(td.Dtype)
	if err != nil {
		return nil, nil, nil, err
	}

	outer := int(shape[0])
	axis1 := int(shape[1])
	tail := 1
	for _, dim := range shape[2:] {
		tail *= int(dim)
	}

	perOuterBytes := axis1 * tail * elemSize
	if len(raw) != outer*perOuterBytes {
		return nil, nil, nil, fmt.Errorf("raw byte length %d does not match shape %v and dtype %s", len(raw), shape, td.Dtype)
	}

	halfAxis1 := axis1 / 2
	halfOuterBytes := halfAxis1 * tail * elemSize
	left := make([]byte, outer*halfOuterBytes)
	right := make([]byte, outer*halfOuterBytes)
	for i := range outer {
		src := i * perOuterBytes
		dst := i * halfOuterBytes
		copy(left[dst:dst+halfOuterBytes], raw[src:src+halfOuterBytes])
		copy(right[dst:dst+halfOuterBytes], raw[src+halfOuterBytes:src+perOuterBytes])
	}

	outShape := append([]int32(nil), shape...)
	outShape[1] = int32(halfAxis1)
	return left, right, outShape, nil
}
