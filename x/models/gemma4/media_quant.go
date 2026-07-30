package gemma4

import (
	"fmt"
	"slices"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
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
	if weight == nil {
		return fmt.Errorf("missing Gemma4 linear tensor %s", weightName)
	}
	scales := tensors[weightName+"_scale"]
	if scales == nil {
		if !slices.Equal(weight.Dims(), want) {
			return fmt.Errorf("Gemma4 linear tensor %s shape %v, want %v", weightName, weight.Dims(), want)
		}
		if !isGemma4FloatingDType(weight.DType()) {
			return fmt.Errorf("Gemma4 linear tensor %s has unsupported dtype %s", weightName, weight.DType())
		}
		return nil
	}

	groupSize, bits, mode := model.ResolveLinearQuantParams(
		defaultGroupSize,
		defaultBits,
		defaultMode,
		tensorQuant,
		weightName,
		weight,
		scales,
	)
	if err := validateGemma4PackedLinearShape(weight.Dims(), scales.Dims(), want, groupSize, bits, mode); err != nil {
		return fmt.Errorf("Gemma4 linear tensor %s: %w", weightName, err)
	}
	if weight.DType() != mlx.DTypeUint32 {
		return fmt.Errorf("Gemma4 linear tensor %s packed dtype %s, want uint32", weightName, weight.DType())
	}
	qbiasName := weightName + "_qbias"
	qbias := tensors[qbiasName]
	switch mode {
	case "affine":
		if !isGemma4FloatingDType(scales.DType()) {
			return fmt.Errorf("Gemma4 affine quantization tensor %s has unsupported dtype %s", weightName+"_scale", scales.DType())
		}
		if qbias == nil {
			return fmt.Errorf("missing Gemma4 affine quantization tensor %s", qbiasName)
		}
		if !slices.Equal(qbias.Dims(), scales.Dims()) {
			return fmt.Errorf("Gemma4 affine quantization tensor %s shape %v, want %v", qbiasName, qbias.Dims(), scales.Dims())
		}
		if !isGemma4FloatingDType(qbias.DType()) {
			return fmt.Errorf("Gemma4 affine quantization tensor %s has unsupported dtype %s", qbiasName, qbias.DType())
		}
	case "nvfp4", "mxfp4", "mxfp8":
		if scales.DType() != mlx.DTypeUint8 {
			return fmt.Errorf("Gemma4 %s quantization tensor %s dtype %s, want uint8", mode, weightName+"_scale", scales.DType())
		}
		if qbias != nil {
			return fmt.Errorf("unexpected Gemma4 %s quantization tensor %s", mode, qbiasName)
		}
	default:
		return fmt.Errorf("Gemma4 linear tensor %s has unsupported quantization mode %q", weightName, mode)
	}
	return nil
}

func validateGemma4PackedLinearShape(weight, scales, want []int, groupSize, bits int, mode string) error {
	if len(want) != 2 || len(weight) != 2 || len(scales) != 2 {
		return fmt.Errorf("packed weight/scales shapes %v/%v do not represent matrix %v", weight, scales, want)
	}
	if groupSize <= 0 || (bits != 4 && bits != 8) || mode == "" {
		return fmt.Errorf("invalid quantization parameters group_size=%d bits=%d mode=%q", groupSize, bits, mode)
	}
	valuesPerWord := 32 / bits
	if weight[0] != want[0] || weight[1]*valuesPerWord != want[1] {
		return fmt.Errorf("packed weight shape %v does not represent %v at %d bits", weight, want, bits)
	}
	if want[1]%groupSize != 0 || scales[0] != want[0] || scales[1] != want[1]/groupSize {
		return fmt.Errorf("scale shape %v does not represent %v with group size %d", scales, want, groupSize)
	}
	return nil
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
