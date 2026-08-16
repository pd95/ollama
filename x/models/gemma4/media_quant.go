package gemma4

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	gemma4metadata "github.com/ollama/ollama/x/models/gemma4/metadata"
)

func validateGemma4LinearWeight(
	tensors map[string]*mlx.Array,
	path string,
	want []int,
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*model.TensorQuantInfo,
) error {
	weightName := path + ".weight"
	weight := tensors[weightName]
	scales := tensors[weightName+"_scale"]
	groupSize, bits, mode := defaultGroupSize, defaultBits, defaultMode
	if weight != nil && scales != nil {
		groupSize, bits, mode = model.ResolveLinearQuantParams(
			defaultGroupSize, defaultBits, defaultMode,
			tensorQuant, weightName, weight, scales,
		)
	}
	descriptors := gemma4RuntimeTensorDescriptors(tensors, tensorQuant, defaultGroupSize, defaultMode)
	if descriptor, ok := descriptors[weightName]; ok && descriptor.QuantType == "" && scales != nil {
		descriptor.QuantType = mode
		descriptor.GroupSize = groupSize
		descriptors[weightName] = descriptor
	}
	logical := make([]int32, len(want))
	for i, dim := range want {
		logical[i] = int32(dim)
	}
	cfg := gemma4metadata.ConfigFile{
		QuantizationConfig: gemma4metadata.Quantization{Bits: bits, GroupSize: groupSize, Mode: mode},
	}
	return gemma4metadata.ValidateRuntimeLinearDescriptor(cfg, descriptors, path, logical)
}

func validateGemma4PackedLinearShape(weight, scales, want []int, groupSize, bits int, mode string) error {
	descriptors := map[string]gemma4metadata.TensorDescriptor{
		"test.weight":       {Dtype: "U32", Shape: intShape32(weight), GroupSize: groupSize},
		"test.weight_scale": {Dtype: "U8", Shape: intShape32(scales)},
	}
	if mode == "affine" {
		descriptors["test.weight_scale"] = gemma4metadata.TensorDescriptor{Dtype: "BF16", Shape: intShape32(scales)}
		descriptors["test.weight_qbias"] = gemma4metadata.TensorDescriptor{Dtype: "BF16", Shape: intShape32(scales)}
	}
	return gemma4metadata.ValidateRuntimeLinearDescriptor(
		gemma4metadata.ConfigFile{QuantizationConfig: gemma4metadata.Quantization{Bits: bits, GroupSize: groupSize, Mode: mode}},
		descriptors, "test", intShape32(want),
	)
}

func intShape32(shape []int) []int32 {
	out := make([]int32, len(shape))
	for i, dim := range shape {
		out[i] = int32(dim)
	}
	return out
}

func gemma4RuntimeTensorDescriptors(
	tensors map[string]*mlx.Array,
	tensorQuant map[string]*model.TensorQuantInfo,
	defaultGroupSize int,
	defaultMode string,
) map[string]gemma4metadata.TensorDescriptor {
	descriptors := make(map[string]gemma4metadata.TensorDescriptor, len(tensors))
	for name, tensor := range tensors {
		if tensor == nil {
			continue
		}
		descriptor := gemma4metadata.TensorDescriptor{Dtype: tensor.DType().String(), Shape: intShape32(tensor.Dims())}
		if quant := tensorQuant[name]; quant != nil {
			descriptor.QuantType = quant.QuantType
			descriptor.GroupSize = quant.GroupSize
		} else if tensor.DType() == mlx.DTypeUint32 && defaultMode != "" {
			descriptor.QuantType = defaultMode
			descriptor.GroupSize = defaultGroupSize
		}
		descriptors[name] = descriptor
	}
	return descriptors
}

func isGemma4FloatingDType(dtype mlx.DType) bool {
	switch dtype {
	case mlx.DTypeBFloat16, mlx.DTypeFloat16, mlx.DTypeFloat32:
		return true
	default:
		return false
	}
}

func gemma4LinearComputeDType(tensors map[string]*mlx.Array, path string) mlx.DType {
	weightName := path + ".weight"
	weight := tensors[weightName]
	if weight != nil && tensors[weightName+"_scale"] == nil && isGemma4FloatingDType(weight.DType()) {
		return weight.DType()
	}
	return mlx.DTypeBFloat16
}
