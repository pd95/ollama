//go:build darwin

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/x/imagegen/safetensors"
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
		rowIndex       = flag.Int("row-index", 173781, "row index to compare")
	)
	flag.Parse()

	if *sourceDir == "" || *modelName == "" {
		fmt.Fprintln(os.Stderr, "usage: gptoss-dense-blob-compare --source-dir <dir> --model-name <name> [--row-index N]")
		os.Exit(2)
	}

	sourceTD, sourcePath, err := findTensor(*sourceDir, *sourceTensor)
	if err != nil {
		panic(err)
	}
	importedTD, importedPath, err := findImportedTensor(*modelsRoot, *modelName, *importedTensor)
	if err != nil {
		panic(err)
	}

	sourceRaw, err := io.ReadAll(sourceTD.Reader())
	if err != nil {
		panic(err)
	}
	importedRaw, err := io.ReadAll(importedTD.Reader())
	if err != nil {
		panic(err)
	}

	fmt.Printf("SOURCE_FILE %s\n", sourcePath)
	fmt.Printf("SOURCE_TENSOR %s\n", sourceTD.Name)
	fmt.Printf("SOURCE_DTYPE %s\n", sourceTD.Dtype)
	fmt.Printf("SOURCE_SHAPE %v\n", sourceTD.Shape)
	fmt.Printf("SOURCE_SIZE %d\n", len(sourceRaw))
	fmt.Printf("SOURCE_SHA256 %s\n", sha256Hex(sourceRaw))
	fmt.Printf("SOURCE_FIRST64 %s\n", hex.EncodeToString(prefix(sourceRaw, 64)))
	fmt.Printf("SOURCE_LAST64 %s\n", hex.EncodeToString(suffix(sourceRaw, 64)))

	fmt.Printf("IMPORTED_FILE %s\n", importedPath)
	fmt.Printf("IMPORTED_TENSOR %s\n", importedTD.Name)
	fmt.Printf("IMPORTED_DTYPE %s\n", importedTD.Dtype)
	fmt.Printf("IMPORTED_SHAPE %v\n", importedTD.Shape)
	fmt.Printf("IMPORTED_SIZE %d\n", len(importedRaw))
	fmt.Printf("IMPORTED_SHA256 %s\n", sha256Hex(importedRaw))
	fmt.Printf("IMPORTED_FIRST64 %s\n", hex.EncodeToString(prefix(importedRaw, 64)))
	fmt.Printf("IMPORTED_LAST64 %s\n", hex.EncodeToString(suffix(importedRaw, 64)))

	fmt.Printf("PAYLOAD_EQUAL %t\n", slices.Equal(sourceRaw, importedRaw))

	rowBytes, err := bf16RowBytes(sourceTD.Shape, *rowIndex, sourceRaw)
	if err != nil {
		panic(fmt.Errorf("source row: %w", err))
	}
	importedRowBytes, err := bf16RowBytes(importedTD.Shape, *rowIndex, importedRaw)
	if err != nil {
		panic(fmt.Errorf("imported row: %w", err))
	}

	fmt.Printf("ROW_INDEX %d\n", *rowIndex)
	fmt.Printf("SOURCE_ROW_SHA256 %s\n", sha256Hex(rowBytes))
	fmt.Printf("IMPORTED_ROW_SHA256 %s\n", sha256Hex(importedRowBytes))
	fmt.Printf("ROW_EQUAL %t\n", slices.Equal(rowBytes, importedRowBytes))
	fmt.Printf("SOURCE_ROW_FIRST64 %s\n", hex.EncodeToString(prefix(rowBytes, 64)))
	fmt.Printf("IMPORTED_ROW_FIRST64 %s\n", hex.EncodeToString(prefix(importedRowBytes, 64)))
	fmt.Printf("SOURCE_ROW_LAST64 %s\n", hex.EncodeToString(suffix(rowBytes, 64)))
	fmt.Printf("IMPORTED_ROW_LAST64 %s\n", hex.EncodeToString(suffix(importedRowBytes, 64)))
}

func findTensor(dir, name string) (*safetensors.TensorData, string, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, "", err
	}
	slices.Sort(paths)
	for _, path := range paths {
		extractor, err := safetensors.OpenForExtraction(path)
		if err != nil {
			return nil, "", err
		}
		td, err := extractor.GetTensor(name)
		if err == nil {
			return td, path, nil
		}
		_ = extractor.Close()
	}
	return nil, "", fmt.Errorf("tensor %q not found in %s", name, dir)
}

func findImportedTensor(modelsRoot, modelName, tensorName string) (*safetensors.TensorData, string, error) {
	manifestPath, err := manifestPathFor(modelsRoot, modelName)
	if err != nil {
		return nil, "", err
	}
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, "", err
	}
	var m manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, "", err
	}
	for _, layer := range m.Layers {
		if layer.Name != tensorName {
			continue
		}
		blobPath := filepath.Join(modelsRoot, "blobs", strings.Replace(layer.Digest, ":", "-", 1))
		extractor, err := safetensors.OpenForExtraction(blobPath)
		if err != nil {
			return nil, "", err
		}
		td, err := extractor.GetTensor(tensorName)
		if err != nil {
			_ = extractor.Close()
			return nil, "", err
		}
		return td, blobPath, nil
	}
	return nil, "", fmt.Errorf("tensor %q not found in manifest", tensorName)
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

func bf16RowBytes(shape []int32, row int, raw []byte) ([]byte, error) {
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
	return raw[start:end], nil
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return fmt.Sprintf("%x", sum)
}

func prefix(b []byte, n int) []byte {
	if len(b) < n {
		n = len(b)
	}
	return b[:n]
}

func suffix(b []byte, n int) []byte {
	if len(b) < n {
		n = len(b)
	}
	return b[len(b)-n:]
}
