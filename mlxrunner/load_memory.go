package mlxrunner

import (
	"fmt"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
)

// estimatedLoadMemory is replaced by live allocations when the runner is ready.
// Qwen3-ASR retains original BF16 tensors while constructing F32 weights, so
// its load estimate must include both allocations rather than just disk bytes.
func estimatedLoadMemory(m *manifest.Manifest) (uint64, error) {
	var stored uint64
	for _, layer := range m.TensorLayers() {
		if layer.Size < 0 || uint64(layer.Size) > ^uint64(0)-stored {
			return 0, fmt.Errorf("invalid tensor size in model manifest")
		}
		stored += uint64(layer.Size)
	}
	if _, ok := m.ConfigLayer("config.json"); !ok {
		return stored, nil
	}
	var cfg struct {
		Architectures []string `json:"architectures"`
	}
	if err := m.ReadConfigJSON("config.json", &cfg); err != nil {
		return 0, err
	}
	if len(cfg.Architectures) != 1 || cfg.Architectures[0] != qwen3asr.Architecture {
		return stored, nil
	}
	raw, err := m.ReadConfig("config.json")
	if err != nil {
		return 0, err
	}
	promoted, err := qwen3asr.EstimatedLoadMemory(raw)
	if err != nil {
		return 0, err
	}
	return max(stored, promoted), nil
}
