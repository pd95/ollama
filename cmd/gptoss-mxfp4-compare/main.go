//go:build darwin

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"

	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/internal/gptossmxfp4"
	ggml "github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

func main() {
	var (
		modelDir = flag.String("model-dir", "", "directory containing raw gpt-oss safetensors")
		tensor   = flag.String("tensor", "", "tensor base name without _blocks/_scales suffix")
		part     = flag.String("part", "none", "none, gate, or up")
	)
	flag.Parse()

	if *modelDir == "" || *tensor == "" {
		fmt.Fprintln(os.Stderr, "usage: gptoss-mxfp4-compare --model-dir <dir> --tensor <base> [--part none|gate|up]")
		os.Exit(2)
	}

	blocksTD, err := findTensor(*modelDir, *tensor+"_blocks")
	if err != nil {
		panic(err)
	}
	scalesTD, err := findTensor(*modelDir, *tensor+"_scales")
	if err != nil {
		panic(err)
	}

	blocksRaw, err := io.ReadAll(blocksTD.Reader())
	if err != nil {
		panic(err)
	}
	scalesRaw, err := io.ReadAll(scalesTD.Reader())
	if err != nil {
		panic(err)
	}

	blockShape := slices.Clone(blocksTD.Shape)
	scaleShape := slices.Clone(scalesTD.Shape)

	switch *part {
	case "none":
	case "gate":
		blocksRaw, _, blockShape, err = splitTensorAxis1Raw(blocksRaw, blocksTD.Dtype, blockShape)
		if err != nil {
			panic(err)
		}
		scalesRaw, _, scaleShape, err = splitTensorAxis1Raw(scalesRaw, scalesTD.Dtype, scaleShape)
		if err != nil {
			panic(err)
		}
	case "up":
		_, blocksRaw, blockShape, err = splitTensorAxis1Raw(blocksRaw, blocksTD.Dtype, blockShape)
		if err != nil {
			panic(err)
		}
		_, scalesRaw, scaleShape, err = splitTensorAxis1Raw(scalesRaw, scalesTD.Dtype, scaleShape)
		if err != nil {
			panic(err)
		}
	default:
		panic(fmt.Errorf("unknown part %q", *part))
	}

	comp, err := gptossmxfp4.Compare(blocksRaw, blockShape, scalesRaw, scaleShape)
	if err != nil {
		panic(err)
	}

	stableFused, stableShape, err := gptossmxfp4.StableFused(blocksRaw, blockShape, scalesRaw, scaleShape)
	if err != nil {
		panic(err)
	}
	packed, err := gptossmxfp4.PackExperimental(blocksRaw, blockShape, scalesRaw, scaleShape)
	if err != nil {
		panic(err)
	}
	experimentalFused, _, err := gptossmxfp4.ExperimentalToStableFused(packed)
	if err != nil {
		panic(err)
	}

	nelements := uint64(product(stableShape))
	stableF32 := ggml.ConvertToF32(stableFused, uint32(fsggml.TensorTypeMXFP4), nelements)
	experimentalF32 := ggml.ConvertToF32(experimentalFused, uint32(fsggml.TensorTypeMXFP4), nelements)

	cos, maxAbs, meanAbs, stableMin, stableMax, expMin, expMax, stableSum, expSum := compareFloatSlices(stableF32, experimentalF32)

	fmt.Printf("TENSOR %s\n", *tensor)
	fmt.Printf("PART %s\n", *part)
	fmt.Printf("BLOCK_SHAPE %v\n", blockShape)
	fmt.Printf("SCALE_SHAPE %v\n", scaleShape)
	fmt.Printf("STABLE_SHAPE %v\n", stableShape)
	fmt.Printf("EXPERIMENTAL_SHAPE %v\n", comp.ExperimentalShape)
	fmt.Printf("STABLE_SHA256 %s\n", comp.StableSHA256)
	fmt.Printf("EXPERIMENTAL_SHA256 %s\n", comp.ExperimentalSHA256)
	fmt.Printf("STABLE_FLOAT_SHA256 %s\n", floatSHA256(stableF32))
	fmt.Printf("EXPERIMENTAL_FLOAT_SHA256 %s\n", floatSHA256(experimentalF32))
	fmt.Printf("COSINE_SIMILARITY %.9f\n", cos)
	fmt.Printf("MAX_ABS_DIFF %.9g\n", maxAbs)
	fmt.Printf("MEAN_ABS_DIFF %.9g\n", meanAbs)
	fmt.Printf("STABLE_MIN %.9g\n", stableMin)
	fmt.Printf("STABLE_MAX %.9g\n", stableMax)
	fmt.Printf("STABLE_SUM %.9g\n", stableSum)
	fmt.Printf("EXPERIMENTAL_MIN %.9g\n", expMin)
	fmt.Printf("EXPERIMENTAL_MAX %.9g\n", expMax)
	fmt.Printf("EXPERIMENTAL_SUM %.9g\n", expSum)
}

func findTensor(modelDir, name string) (*safetensors.TensorData, error) {
	paths, err := filepath.Glob(filepath.Join(modelDir, "*.safetensors"))
	if err != nil {
		return nil, err
	}
	for _, path := range paths {
		extractor, err := safetensors.OpenForExtraction(path)
		if err != nil {
			return nil, err
		}
		td, err := extractor.GetTensor(name)
		if err == nil {
			return td, nil
		}
		_ = extractor.Close()
	}
	return nil, fmt.Errorf("tensor %q not found in %s", name, modelDir)
}

func splitTensorAxis1Raw(raw []byte, dtype string, shape []int32) ([]byte, []byte, []int32, error) {
	if len(shape) < 2 {
		return nil, nil, nil, fmt.Errorf("expected rank >= 2, got shape %v", shape)
	}
	if shape[1]%2 != 0 {
		return nil, nil, nil, fmt.Errorf("axis 1 dim %d is not even", shape[1])
	}

	elemSize, err := dtypeSize(dtype)
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
		return nil, nil, nil, fmt.Errorf("raw byte length %d does not match shape %v and dtype %s", len(raw), shape, dtype)
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

func dtypeSize(dtype string) (int, error) {
	switch dtype {
	case "U8", "I8":
		return 1, nil
	case "U16", "I16", "F16", "BF16":
		return 2, nil
	case "U32", "I32", "F32":
		return 4, nil
	case "U64", "I64", "F64":
		return 8, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %q", dtype)
	}
}

func product(shape []int32) int {
	n := 1
	for _, dim := range shape {
		n *= int(dim)
	}
	return n
}

func floatSHA256(xs []float32) string {
	raw := make([]byte, 4*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(x))
	}
	return fmt.Sprintf("%x", sha256.Sum256(raw))
}

func compareFloatSlices(a, b []float32) (cosine float64, maxAbs float32, meanAbs float64, aMin float32, aMax float32, bMin float32, bMax float32, aSum float64, bSum float64) {
	aMin, bMin = float32(math.Inf(1)), float32(math.Inf(1))
	aMax, bMax = float32(math.Inf(-1)), float32(math.Inf(-1))
	var dot, aNorm, bNorm, sumAbs float64
	for i := range a {
		x, y := a[i], b[i]
		if x < aMin {
			aMin = x
		}
		if x > aMax {
			aMax = x
		}
		if y < bMin {
			bMin = y
		}
		if y > bMax {
			bMax = y
		}
		aSum += float64(x)
		bSum += float64(y)
		diff := float32(math.Abs(float64(x - y)))
		if diff > maxAbs {
			maxAbs = diff
		}
		sumAbs += float64(diff)
		dot += float64(x) * float64(y)
		aNorm += float64(x) * float64(x)
		bNorm += float64(y) * float64(y)
	}
	if len(a) > 0 {
		meanAbs = sumAbs / float64(len(a))
	}
	if aNorm > 0 && bNorm > 0 {
		cosine = dot / math.Sqrt(aNorm*bNorm)
	}
	return cosine, maxAbs, meanAbs, aMin, aMax, bMin, bMax, aSum, bSum
}
