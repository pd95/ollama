package create

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/safetensors"
	schema "github.com/ollama/ollama/mlxrunner/model/chatterbox/config"
	"github.com/ollama/ollama/types/model"
)

const chatterboxTurboArchitecture = schema.TurboArchitecture

func normalizeChatterboxTurboConfig(raw []byte) ([]byte, error) {
	if err := schema.ValidateTurbo(raw); err != nil {
		return nil, err
	}
	var c map[string]json.RawMessage
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	c["architectures"], _ = json.Marshal([]string{chatterboxTurboArchitecture})
	return json.Marshal(c)
}

func prepareChatterboxTurboInventory(inv Inventory) (Inventory, error) {
	if inv.Config.Architecture() != chatterboxTurboArchitecture {
		return inv, nil
	}
	for name, validate := range map[string]func([]byte) error{"s3_tokenizer/config.json": schema.ValidateS3, "reference_encoder/config.json": schema.ValidateReferenceEncoder} {
		raw, err := os.ReadFile(filepath.Join(inv.Dir, name))
		if err != nil {
			return Inventory{}, err
		}
		if err = validate(raw); err != nil {
			return Inventory{}, err
		}
	}
	inv.Tensors = maps.Clone(inv.Tensors)
	for _, asset := range []struct {
		file, prefix string
		filter       func(string) bool
	}{{"s3_tokenizer/model.safetensors", "s3_tokenizer.", func(string) bool { return true }}, {"reference_encoder/model.safetensors", "", func(name string) bool { return strings.HasPrefix(name, "s3gen.speaker_encoder.") }}, {"conds.safetensors", "default.", func(string) bool { return true }}} {
		reader, err := safetensors.OpenForExtraction(filepath.Join(inv.Dir, asset.file))
		if err != nil {
			return Inventory{}, err
		}
		count := 0
		for _, name := range reader.ListTensors() {
			if !asset.filter(name) {
				continue
			}
			count++
			key := asset.prefix + name
			if inv.Has(key) {
				reader.Close()
				return Inventory{}, fmt.Errorf("Turbo tensor collision %s", key)
			}
			tensor, err := reader.GetTensor(name)
			if err != nil {
				reader.Close()
				return Inventory{}, err
			}
			if tensor.Dtype != "F32" && !(asset.prefix == "default." && tensor.Dtype == "I32") {
				reader.Close()
				return Inventory{}, fmt.Errorf("Turbo dependency %s has unsupported precision", key)
			}
			inv.Tensors[key] = SourceTensor{Name: name, Dtype: tensor.Dtype, Shape: tensor.Shape, File: asset.file}
		}
		reader.Close()
		if count == 0 {
			return Inventory{}, fmt.Errorf("Turbo dependency %s is empty", asset.file)
		}
		if asset.prefix == "s3_tokenizer." {
			bank, hasBank := inv.Tensors["s3_tokenizer.mel_filters"]
			window, hasWindow := inv.Tensors["s3_tokenizer.window"]
			if hasBank != hasWindow || (hasBank && (!slices.Equal(bank.Shape, []int32{128, 201}) || !slices.Equal(window.Shape, []int32{400}))) {
				return Inventory{}, fmt.Errorf("Turbo original S3 frontend requires mel_filters [128,201] and window [400]")
			}
		}
	}
	for name, shape := range map[string][]int32{"default.t3.speaker_emb": {1, 256}, "default.t3.cond_prompt_speech_tokens": {1, 375}, "default.gen.embedding": {1, 192}, "default.gen.prompt_feat": {1, 500, 80}, "default.gen.prompt_token": {1, 250}} {
		if !slices.Equal(inv.Tensors[name].Shape, shape) {
			return Inventory{}, fmt.Errorf("Turbo cached conditioning %s has invalid shape", name)
		}
	}
	return inv, nil
}

type chatterboxTurboImportTransform struct{}

func newChatterboxTurboImportTransform(raw json.RawMessage) (quantizePolicy, error) {
	if err := schema.ValidateTurbo(raw); err != nil {
		return nil, err
	}
	return chatterboxTurboImportTransform{}, nil
}
func (chatterboxTurboImportTransform) quantizationType(string, []int32, string) string { return "" }
func (chatterboxTurboImportTransform) augmentConfig(c *model.ConfigV2) error {
	c.Voices = []string{"default"}
	c.Languages = []string{"en"}
	c.VoiceLanguages = map[string][]string{"default": {"en"}}
	return nil
}
