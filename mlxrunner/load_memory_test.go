package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"os"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/ml"
)

func TestQwen3ASRLoadMemory(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_GPU_OVERHEAD", "0")
	raw, err := os.ReadFile("model/qwen3asr/testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	// Independent original checkpoint headers contain 938,008,576 BF16 values.
	const loadBytes = uint64(5628051456)
	for _, tc := range []struct {
		name string
		raw  []byte
		want uint64
	}{
		{"original ASR", raw, loadBytes},
		{"Qwen TTS unchanged", []byte(`{"architectures":["Qwen3TTSForConditionalGeneration"]}`), 1000},
		{"LFM3 unchanged", []byte(`{"architectures":["Lfm3ForCausalLM"]}`), 1000},
		{"text unchanged", []byte(`{"architectures":["Qwen3ForCausalLM"]}`), 1000},
		{"legacy no source config", nil, 1000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &manifest.Manifest{Layers: []manifest.Layer{{MediaType: manifest.MediaTypeImageTensor, Size: 1000}}}
			if tc.raw != nil {
				layer, err := manifest.NewLayer(bytes.NewReader(tc.raw), manifest.MediaTypeImageJSON)
				if err != nil {
					t.Fatal(err)
				}
				layer.Name = "config.json"
				m.Layers = append(m.Layers, layer)
			}
			got, err := estimatedLoadMemory(m)
			if err != nil || got != tc.want {
				t.Fatalf("estimate %d, want %d: %v", got, tc.want, err)
			}
			if tc.name == "original ASR" {
				c := &Client{}
				c.memory.Store(got)
				// The original disk-byte estimate admitted this load. Reject it
				// before opening a port or starting a process instead.
				_, err := c.Load(context.Background(), ml.SystemInfo{FreeMemory: 3 << 30}, []ml.DeviceInfo{{Integrated: true, FreeMemory: 3 << 30}}, true)
				if !errors.Is(err, llm.ErrLoadRequiredFull) || c.cmd != nil {
					t.Fatalf("insufficient-memory admission: %v, process=%v", err, c.cmd)
				}
			}
		})
	}
}

func TestQwen3ASRLoadMemoryInvalidConfig(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	layer, err := manifest.NewLayer(bytes.NewReader([]byte(`{"architectures":["Qwen3ASRForConditionalGeneration"]}`)), manifest.MediaTypeImageJSON)
	if err != nil {
		t.Fatal(err)
	}
	layer.Name = "config.json"
	if _, err := estimatedLoadMemory(&manifest.Manifest{Layers: []manifest.Layer{layer}}); err == nil {
		t.Fatal("incomplete ASR configuration fell back to a disk-byte estimate")
	}
}

func TestQwen3ASRImportedLoadMemory(t *testing.T) {
	name := os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED")
	if name == "" {
		t.Skip("set OLLAMA_QWEN3_ASR_IMPORTED to the original imported model")
	}
	c, err := NewClient(name, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := c.memory.Load(); got != 5628051456 {
		t.Fatalf("imported load estimate %d, want 5628051456", got)
	}
}
