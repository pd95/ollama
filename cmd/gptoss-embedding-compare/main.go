//go:build darwin

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type manifest struct {
	Layers []manifestLayer `json:"layers"`
}

type manifestLayer struct {
	Name   string `json:"name"`
	Digest string `json:"digest"`
}

func main() {
	var (
		sourceDir      = flag.String("source-dir", "", "directory containing raw HF safetensors")
		sourceTensor   = flag.String("source-tensor", "model.embed_tokens.weight", "source tensor name")
		modelsRoot     = flag.String("models-root", filepath.Join(os.Getenv("HOME"), ".ollama", "models"), "Ollama models root")
		modelName      = flag.String("model-name", "", "imported Ollama model name")
		importedTensor = flag.String("imported-tensor", "token_embd.weight", "imported tensor name")
		tokenID        = flag.Int("token-id", 173781, "token id / embedding row to compare")
	)
	flag.Parse()

	if *sourceDir == "" || *modelName == "" {
		fmt.Fprintln(os.Stderr, "usage: gptoss-embedding-compare --source-dir <dir> --model-name <name> [--token-id N]")
		os.Exit(2)
	}

	sourceFile, err := findTensorFile(*sourceDir, *sourceTensor)
	if err != nil {
		panic(err)
	}
	sourceSF, err := mlx.LoadSafetensorsNative(sourceFile)
	if err != nil {
		panic(err)
	}
	defer sourceSF.Free()

	sourceWeight := sourceSF.Get(*sourceTensor)
	if sourceWeight == nil {
		panic(fmt.Errorf("source tensor %q not found in %s", *sourceTensor, sourceFile))
	}

	importedFile, err := findImportedBlob(*modelsRoot, *modelName, *importedTensor)
	if err != nil {
		panic(err)
	}
	importedSF, err := mlx.LoadSafetensorsNative(importedFile)
	if err != nil {
		panic(err)
	}
	defer importedSF.Free()

	importedWeight := importedSF.Get(*importedTensor)
	if importedWeight == nil {
		panic(fmt.Errorf("imported tensor %q not found in %s", *importedTensor, importedFile))
	}
	importedScale := importedSF.Get(*importedTensor + ".scale")
	importedQBias := importedSF.Get(*importedTensor + ".qbias")

	index := mlx.FromValues([]int32{int32(*tokenID)}, 1)
	sourceRow := sourceWeight.TakeAxis(index, 0)
	var importedRow *mlx.Array
	if importedScale != nil {
		w := importedWeight.TakeAxis(index, 0)
		s := importedScale.TakeAxis(index, 0)
		var qb *mlx.Array
		if importedQBias != nil {
			qb = importedQBias.TakeAxis(index, 0)
		}
		groupSize, bits, mode := inferImportedQuantParams(importedSF, *importedTensor, importedWeight, importedScale)
		importedRow = mlx.Dequantize(w, s, qb, groupSize, bits, mode)
	} else {
		importedRow = importedWeight.TakeAxis(index, 0)
	}

	mlx.Eval(sourceRow, importedRow)
	sourceFloats := sourceRow.Floats()
	importedFloats := importedRow.Floats()

	sourceTD, _, err := findTensor(*sourceDir, *sourceTensor)
	if err != nil {
		panic(err)
	}
	sourceRaw, err := io.ReadAll(sourceTD.Reader())
	if err != nil {
		panic(err)
	}
	importedTD, _, err := findImportedTensor(*modelsRoot, *modelName, *importedTensor)
	if err != nil {
		panic(err)
	}
	importedRaw, err := io.ReadAll(importedTD.Reader())
	if err != nil {
		panic(err)
	}
	sourceRawRow, err := bf16RowFloats(sourceTD.Shape, *tokenID, sourceRaw)
	if err != nil {
		panic(err)
	}
	importedRawRow, err := bf16RowFloats(importedTD.Shape, *tokenID, importedRaw)
	if err != nil {
		panic(err)
	}

	cos, maxAbs, meanAbs, sourceMin, sourceMax, importedMin, importedMax, sourceSum, importedSum := compareFloatSlices(sourceFloats, importedFloats)
	sourceRawCos, sourceRawMaxAbs, sourceRawMeanAbs, _, _, _, _, _, _ := compareFloatSlices(sourceRawRow, sourceFloats)
	importedRawCos, importedRawMaxAbs, importedRawMeanAbs, _, _, _, _, _, _ := compareFloatSlices(importedRawRow, importedFloats)

	fmt.Printf("SOURCE_FILE %s\n", sourceFile)
	fmt.Printf("SOURCE_TENSOR %s\n", *sourceTensor)
	fmt.Printf("SOURCE_WEIGHT_DTYPE %s\n", sourceWeight.DType())
	fmt.Printf("SOURCE_WEIGHT_SHAPE %v\n", sourceWeight.Dims())
	fmt.Printf("IMPORTED_FILE %s\n", importedFile)
	fmt.Printf("IMPORTED_TENSOR %s\n", *importedTensor)
	fmt.Printf("IMPORTED_WEIGHT_DTYPE %s\n", importedWeight.DType())
	fmt.Printf("IMPORTED_WEIGHT_SHAPE %v\n", importedWeight.Dims())
	if importedScale != nil {
		fmt.Printf("IMPORTED_SCALE_DTYPE %s\n", importedScale.DType())
		fmt.Printf("IMPORTED_SCALE_SHAPE %v\n", importedScale.Dims())
	} else {
		fmt.Printf("IMPORTED_SCALE_DTYPE <none>\n")
		fmt.Printf("IMPORTED_SCALE_SHAPE <none>\n")
	}
	fmt.Printf("TOKEN_ID %d\n", *tokenID)
	fmt.Printf("SOURCE_ROW_SHAPE %v\n", sourceRow.Dims())
	fmt.Printf("IMPORTED_ROW_SHAPE %v\n", importedRow.Dims())
	fmt.Printf("SOURCE_RAW_ROW_SHA256 %s\n", floatSHA256(sourceRawRow))
	fmt.Printf("SOURCE_ROW_SHA256 %s\n", floatSHA256(sourceFloats))
	fmt.Printf("IMPORTED_RAW_ROW_SHA256 %s\n", floatSHA256(importedRawRow))
	fmt.Printf("IMPORTED_ROW_SHA256 %s\n", floatSHA256(importedFloats))
	fmt.Printf("SOURCE_RAW_VS_MLX_COSINE %.9f\n", sourceRawCos)
	fmt.Printf("SOURCE_RAW_VS_MLX_MAX_ABS_DIFF %.9g\n", sourceRawMaxAbs)
	fmt.Printf("SOURCE_RAW_VS_MLX_MEAN_ABS_DIFF %.9g\n", sourceRawMeanAbs)
	fmt.Printf("IMPORTED_RAW_VS_MLX_COSINE %.9f\n", importedRawCos)
	fmt.Printf("IMPORTED_RAW_VS_MLX_MAX_ABS_DIFF %.9g\n", importedRawMaxAbs)
	fmt.Printf("IMPORTED_RAW_VS_MLX_MEAN_ABS_DIFF %.9g\n", importedRawMeanAbs)
	fmt.Printf("COSINE_SIMILARITY %.9f\n", cos)
	fmt.Printf("MAX_ABS_DIFF %.9g\n", maxAbs)
	fmt.Printf("MEAN_ABS_DIFF %.9g\n", meanAbs)
	fmt.Printf("SOURCE_MIN %.9g\n", sourceMin)
	fmt.Printf("SOURCE_MAX %.9g\n", sourceMax)
	fmt.Printf("SOURCE_SUM %.9g\n", sourceSum)
	fmt.Printf("IMPORTED_MIN %.9g\n", importedMin)
	fmt.Printf("IMPORTED_MAX %.9g\n", importedMax)
	fmt.Printf("IMPORTED_SUM %.9g\n", importedSum)
}

func findTensorFile(dir, name string) (string, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return "", err
	}
	slices.Sort(paths)
	for _, path := range paths {
		extractor, err := safetensors.OpenForExtraction(path)
		if err != nil {
			return "", err
		}
		_, err = extractor.GetTensor(name)
		_ = extractor.Close()
		if err == nil {
			return path, nil
		}
	}
	return "", fmt.Errorf("tensor %q not found in %s", name, dir)
}

func findImportedBlob(modelsRoot, modelName, tensorName string) (string, error) {
	manifestPath, err := manifestPathFor(modelsRoot, modelName)
	if err != nil {
		return "", err
	}

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return "", err
	}
	var m manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return "", err
	}
	for _, layer := range m.Layers {
		if layer.Name != tensorName {
			continue
		}
		return filepath.Join(modelsRoot, "blobs", strings.Replace(layer.Digest, ":", "-", 1)), nil
	}
	return "", fmt.Errorf("tensor %q not found in manifest %s", tensorName, manifestPath)
}

func manifestPathFor(modelsRoot, modelName string) (string, error) {
	registry, namespace, name, tag := "registry.ollama.ai", "library", "", "latest"
	ref := modelName
	if i := strings.LastIndex(ref, ":"); i >= 0 && !strings.Contains(ref[i+1:], "/") {
		tag = ref[i+1:]
		ref = ref[:i]
	}
	parts := strings.Split(ref, "/")
	switch len(parts) {
	case 1:
		name = parts[0]
	case 2:
		namespace, name = parts[0], parts[1]
	case 3:
		registry, namespace, name = parts[0], parts[1], parts[2]
	default:
		return "", fmt.Errorf("unsupported model name %q", modelName)
	}
	return filepath.Join(modelsRoot, "manifests", registry, namespace, name, tag), nil
}

func inferImportedQuantParams(sf *mlx.SafetensorsFile, tensorName string, weight, scale *mlx.Array) (groupSize, bits int, mode string) {
	quantType := strings.ToUpper(sf.GetMetadata("quant_type"))
	switch quantType {
	case "MXFP4":
		return 32, 4, "mxfp4"
	case "NVFP4":
		return 16, 4, "nvfp4"
	case "MXFP8":
		return 32, 8, "mxfp8"
	case "FP4", "Q4", "INT4":
		return 64, 4, "affine"
	case "FP8", "Q8", "INT8":
		return 64, 8, "affine"
	}

	// Fallback for dense-or-affine-like shapes; enough for this debugging tool.
	wdims := weight.Dims()
	sdims := scale.Dims()
	if len(wdims) > 0 && len(sdims) > 0 {
		wcols := wdims[len(wdims)-1]
		scols := sdims[len(sdims)-1]
		if scols > 0 {
			if gs := wcols * 8 / scols; gs == 32 {
				return 32, 4, "mxfp4"
			}
			if gs := wcols * 4 / scols; gs == 64 {
				return 64, 8, "affine"
			}
		}
	}
	panic(fmt.Errorf("unable to infer quant params for %s", tensorName))
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

func bf16RowFloats(shape []int32, row int, raw []byte) ([]float32, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("expected rank-2 tensor, got %v", shape)
	}
	rows, cols := int(shape[0]), int(shape[1])
	if row < 0 || row >= rows {
		return nil, fmt.Errorf("row %d out of range for %d rows", row, rows)
	}
	rowSize := cols * 2
	start := row * rowSize
	end := start + rowSize
	if end > len(raw) {
		return nil, fmt.Errorf("row bytes exceed raw length")
	}
	bts := raw[start:end]
	out := make([]float32, cols)
	for i := range cols {
		bf16 := binary.LittleEndian.Uint16(bts[i*2:])
		f32bits := uint32(bf16) << 16
		out[i] = math.Float32frombits(f32bits)
	}
	return out, nil
}
