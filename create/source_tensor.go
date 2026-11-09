package create

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/fs/torch"
)

// tensorSource keeps inventory metadata separate from deferred raw tensor reads.
// Original speech archives enter only through their model-specific file lists.
type tensorSource interface {
	ListTensors() []string
	GetTensor(string) (*safetensors.TensorData, error)
	Close() error
}

func openTensorSource(ctx context.Context, name string) (tensorSource, error) {
	switch filepath.Ext(name) {
	case ".pt", ".pth":
		a, err := torch.OpenContext(ctx, name)
		if err != nil {
			return nil, err
		}
		return &torchTensorSource{archive: a}, nil
	default:
		return safetensors.OpenForExtraction(name)
	}
}

type torchTensorSource struct{ archive *torch.Archive }

func (s *torchTensorSource) Close() error { return s.archive.Close() }

func (s *torchTensorSource) ListTensors() []string {
	names := make([]string, 0, len(s.archive.Tensors))
	for name := range s.archive.Tensors {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

func (s *torchTensorSource) GetTensor(name string) (*safetensors.TensorData, error) {
	t, ok := s.archive.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("PyTorch tensor %q not found", name)
	}
	return safetensors.NewTensorDataFromReaderAt(name, t.Dtype, slices.Clone(t.Shape), t, t.Size), nil
}

// speechTensorFile selects one supported representation without silently
// preferring a stale conversion when both original and converted files exist.
func speechTensorFile(dir, base string, extensions ...string) (string, error) {
	var selected string
	for _, extension := range extensions {
		name := base + extension
		info, err := os.Stat(filepath.Join(dir, name))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return "", err
		}
		if !info.Mode().IsRegular() {
			return "", fmt.Errorf("speech tensor %s must be a regular file", name)
		}
		if selected != "" {
			return "", fmt.Errorf("ambiguous speech tensors: %s and %s", selected, name)
		}
		selected = name
	}
	if selected == "" {
		return "", fmt.Errorf("missing speech tensor %s: %w", base, os.ErrNotExist)
	}
	return selected, nil
}
