package create

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
)

func validateQwen3ASRInventory(inv Inventory) error {
	if inv.Config.Architecture() != qwen3asr.Architecture {
		return nil
	}
	shapes, err := qwen3asr.TensorShapes(inv.RawConfig)
	if err != nil {
		return err
	}
	if len(inv.Tensors) != len(shapes) {
		return fmt.Errorf("Qwen3-ASR requires %d trained tensors; got %d", len(shapes), len(inv.Tensors))
	}
	for name, shape := range shapes {
		tensor, ok := inv.Tensors[name]
		if !ok || !slices.Equal(tensor.Shape, shape) || tensor.Dtype != "BF16" {
			return fmt.Errorf("Qwen3-ASR tensor %s requires original BF16 shape %v", name, shape)
		}
	}
	for _, name := range []string{"preprocessor_config.json", "generation_config.json", "chat_template.json", "vocab.json", "merges.txt", "tokenizer_config.json"} {
		info, err := os.Stat(filepath.Join(inv.Dir, name))
		if err != nil || !info.Mode().IsRegular() {
			return fmt.Errorf("Qwen3-ASR requires %s", name)
		}
	}
	raw, err := os.ReadFile(filepath.Join(inv.Dir, "preprocessor_config.json"))
	if err != nil {
		return err
	}
	return qwen3asr.ValidatePreprocessor(raw)
}

func validateQwen3ASRTokenizer(inv Inventory, source *validatedTokenizerSource) error {
	if inv.Config.Architecture() != qwen3asr.Architecture {
		return nil
	}
	if source == nil {
		return fmt.Errorf("Qwen3-ASR requires its tokenizer")
	}
	generation, err := os.ReadFile(filepath.Join(inv.Dir, "generation_config.json"))
	if err != nil {
		return err
	}
	config, err := os.ReadFile(filepath.Join(inv.Dir, "tokenizer_config.json"))
	if err != nil {
		return err
	}
	_, err = qwen3asr.LoadTokenizer(inv.RawConfig, source.data, config, generation)
	return err
}
