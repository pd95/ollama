package create

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/fs/safetensors"
	schema "github.com/ollama/ollama/mlxrunner/model/chatterbox/config"
)

const chatterboxArchitecture = schema.Architecture

func normalizeChatterboxConfig(raw []byte) ([]byte, error) {
	if err := schema.Validate(raw); err != nil {
		return nil, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, err
	}
	fields["architectures"], _ = json.Marshal([]string{chatterboxArchitecture})
	return json.Marshal(fields)
}

func chatterboxSource(dir string) (bool, error) {
	if _, err := os.Stat(filepath.Join(dir, "config.json")); os.IsNotExist(err) {
		return false, nil
	}
	cfg, _, err := readSourceModelConfig(dir)
	return cfg.Architecture() == chatterboxArchitecture, err
}

func chatterboxAsset(dir, name string) (string, error) {
	info, err := os.Stat(filepath.Join(dir, name))
	if err != nil || !info.Mode().IsRegular() {
		return "", fmt.Errorf("Chatterbox requires %s", name)
	}
	return name, nil
}

func prepareChatterboxInventory(inv Inventory) (Inventory, error) {
	if inv.Config.Architecture() != chatterboxArchitecture {
		return inv, nil
	}
	raw, err := os.ReadFile(filepath.Join(inv.Dir, "s3_tokenizer/config.json"))
	if err != nil {
		return Inventory{}, fmt.Errorf("Chatterbox requires s3_tokenizer/config.json: %w", err)
	}
	if err := schema.ValidateS3(raw); err != nil {
		return Inventory{}, err
	}
	file, err := chatterboxAsset(inv.Dir, "s3_tokenizer/model.safetensors")
	if err != nil {
		return Inventory{}, err
	}
	reader, err := safetensors.OpenForExtraction(filepath.Join(inv.Dir, file))
	if err != nil {
		return Inventory{}, err
	}
	defer reader.Close()
	if len(reader.ListTensors()) == 0 {
		return Inventory{}, fmt.Errorf("empty Chatterbox S3 tokenizer")
	}
	inv.Tensors = maps.Clone(inv.Tensors)
	for name, tensor := range inv.Tensors {
		if tensor.Dtype != "F32" {
			return Inventory{}, fmt.Errorf("Chatterbox v3 requires F32 weights: %s", name)
		}
	}
	for _, name := range reader.ListTensors() {
		key := "s3_tokenizer." + name
		if inv.Has(key) {
			return Inventory{}, fmt.Errorf("Chatterbox S3 tensor collision: %s", key)
		}
		tensor, err := reader.GetTensor(name)
		if err != nil {
			return Inventory{}, err
		}
		if tensor.Dtype != "F32" {
			return Inventory{}, fmt.Errorf("Chatterbox S3 tokenizer requires F32 weights")
		}
		inv.Tensors[key] = SourceTensor{Name: name, Dtype: tensor.Dtype, Shape: tensor.Shape, File: file}
	}
	return inv, nil
}

type chatterboxImportTransform struct{}

func newChatterboxImportTransform(raw json.RawMessage) (quantizePolicy, error) {
	if err := schema.Validate(raw); err != nil {
		return nil, err
	}
	return chatterboxImportTransform{}, nil
}
func (chatterboxImportTransform) quantizationType(string, []int32, string) string { return "" }
