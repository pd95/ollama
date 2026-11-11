package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
)

// qwen3ASRTokenizer keeps the checkpoint's Qwen2 normalization for ASR. The
// pinned original ForcedAligner wrapper explicitly passes fix_mistral_regex=True
// to Transformers4.57.6, which replaces only its first pretokenizer. Preserve
// that task-specific behavior without changing TTS or checkpoint source bytes.
func qwen3ASRTokenizer(inv Inventory) (*validatedTokenizerSource, error) {
	source, err := qwen3TTSTokenizer(inv.Dir)
	if err != nil || !qwen3asr.IsForcedAligner(inv.RawConfig) {
		return source, err
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(source.data, &doc); err != nil {
		return nil, err
	}
	var pre struct {
		Type  string            `json:"type"`
		Parts []json.RawMessage `json:"pretokenizers"`
	}
	if err := json.Unmarshal(doc["pre_tokenizer"], &pre); err != nil {
		return nil, err
	}
	if pre.Type != "Sequence" || len(pre.Parts) != 2 {
		return nil, fmt.Errorf("invalid aligner pretokenizer sequence")
	}
	const pattern = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	pre.Parts[0], err = json.Marshal(map[string]any{"type": "Split", "pattern": map[string]string{"Regex": pattern}, "behavior": "Isolated", "invert": false})
	if err != nil {
		return nil, err
	}
	doc["pre_tokenizer"], err = json.Marshal(pre)
	if err != nil {
		return nil, err
	}
	source.data, err = json.Marshal(doc)
	return source, err
}

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
