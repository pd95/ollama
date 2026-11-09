package create

import (
	"fmt"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/torch"
)

var kokoroStateKey = regexp.MustCompile(`^[A-Za-z0-9_.]{1,256}$`)

func prepareOriginalKokoroInventory(inv Inventory) (Inventory, error) {
	file := "kokoro-v1_0.pth"
	original := false
	for _, source := range inv.Tensors {
		original = original || source.File == file
	}
	if !original {
		return inv, nil
	}
	a, err := torch.Open(filepath.Join(inv.Dir, file))
	if err != nil {
		return Inventory{}, err
	}
	defer a.Close()
	modules := []string{"bert", "bert_encoder", "predictor", "decoder", "text_encoder"}
	root, ok := a.Root.(map[string]any)
	if !ok || len(root) != len(modules) {
		return Inventory{}, fmt.Errorf("unsupported original Kokoro module inventory")
	}
	out := make(map[string]SourceTensor, len(inv.Tensors))
	var elements int64
	for module, value := range root {
		state, ok := value.(map[string]any)
		if !ok || !slices.Contains(modules, module) || len(state) == 0 || len(state) > 1500 {
			return Inventory{}, fmt.Errorf("invalid original Kokoro module %s", module)
		}
		for key, value := range state {
			tensor, ok := value.(*torch.Tensor)
			if !ok || !kokoroStateKey.MatchString(key) || tensor.Dtype != "F32" || len(tensor.Shape) > 4 {
				return Inventory{}, fmt.Errorf("unsupported original Kokoro tensor %s.%s", module, key)
			}
			elements += tensor.Size / 4
			if elements > 100_000_000 || len(out) >= 1500 {
				return Inventory{}, fmt.Errorf("original Kokoro tensor inventory exceeds bounds")
			}
			name := module + "." + strings.TrimPrefix(key, "module.")
			source, exists := inv.Tensors[module+"."+key]
			if !exists || source.File != file || source.Dtype != tensor.Dtype || !slices.Equal(source.Shape, tensor.Shape) {
				return Inventory{}, fmt.Errorf("original Kokoro source metadata changed: %s", name)
			}
			if _, exists := out[name]; exists {
				return Inventory{}, fmt.Errorf("original Kokoro tensor alias collision: %s", name)
			}
			out[name] = source
		}
	}
	if len(out) != len(inv.Tensors) {
		return Inventory{}, fmt.Errorf("unexpected tensors outside original Kokoro modules")
	}
	inv.Tensors = out
	return inv, nil
}
