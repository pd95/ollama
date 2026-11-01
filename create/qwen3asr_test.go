package create

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func TestQwen3ASRReferenceImport(t *testing.T) {
	dir := os.Getenv("OLLAMA_QWEN3_ASR_SOURCE")
	if dir == "" {
		t.Skip("set OLLAMA_QWEN3_ASR_SOURCE to the pinned original checkpoint")
	}
	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateQwen3ASRInventory(inv); err != nil {
		t.Fatal(err)
	}
	cfg, err := inferSafetensorsConfig(dir, inv.Config, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(cfg.Capabilities, []string{"completion", "audio"}) || cfg.Parser != "" || cfg.Renderer != "qwen3-asr" || cfg.ModelFamily != "qwen3_asr" {
		t.Fatalf("ASR import requires its native audio renderer: %+v", cfg)
	}
	source, err := qwen3TTSTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateQwen3ASRTokenizer(inv, source); err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadFromBytes(source.data)
	if err != nil {
		t.Fatal(err)
	}
	for id, text := range map[int32]string{151676: "<|audio_pad|>", 151704: "<asr_text>", 151645: "<|im_end|>"} {
		if got := tok.Encode(text, false); !slices.Equal(got, []int32{id}) {
			t.Fatalf("%q encoded as %v", text, got)
		}
	}
	if ref := os.Getenv("OLLAMA_QWEN3_ASR_REFERENCE"); ref != "" {
		data, err := os.ReadFile(filepath.Join(ref, "tokens.json"))
		if err != nil {
			t.Fatal(err)
		}
		var cases []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
		}
		if err := json.Unmarshal(data, &cases); err != nil {
			t.Fatal(err)
		}
		for _, c := range cases {
			if got := tok.Encode(c.Text, false); !slices.Equal(got, c.IDs) {
				t.Errorf("provider tokenizer mismatch for %q: %v != %v", c.Text, got, c.IDs)
			}
		}
	}
	class, err := Classify(inv, "")
	if err != nil {
		t.Fatal(err)
	}
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatal(err)
	}
	specs, err := Plan(inv, class, policy)
	if err != nil {
		t.Fatal(err)
	}
	want := 612
	if len(inv.Tensors) == 708 {
		want = 708
	}
	if len(specs) != want {
		t.Fatalf("planned %d tensor blobs, want %d", len(specs), want)
	}
	for _, spec := range specs {
		for _, tensor := range spec.Tensors {
			if tensor.Transform != TransformNone || tensor.Quantize != "" || tensor.OutDtype != "" {
				t.Fatalf("original tensor %s would be transformed: %+v", tensor.Name, tensor)
			}
		}
	}
	delete(inv.Tensors, "thinker.lm_head.weight")
	if err := validateQwen3ASRInventory(inv); err == nil {
		t.Fatal("missing separate output head accepted")
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); !os.IsNotExist(err) {
		t.Fatal("original split tokenizer source changed")
	}
}

func TestQwen3ASRInvalidSidecarsBeforeWrites(t *testing.T) {
	source := os.Getenv("OLLAMA_QWEN3_ASR_SOURCE")
	if source == "" {
		t.Skip("set original source to check pre-write rejection")
	}
	for _, name := range []string{"generation_config.json", "tokenizer_config.json"} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			files, err := os.ReadDir(source)
			if err != nil {
				t.Fatal(err)
			}
			for _, file := range files {
				if file.IsDir() || file.Name() == name {
					continue
				}
				if err := os.Symlink(filepath.Join(source, file.Name()), filepath.Join(dir, file.Name())); err != nil {
					t.Fatal(err)
				}
			}
			bad := []byte(`{"eos_token_id":[151645]}`)
			if name == "tokenizer_config.json" {
				raw, err := os.ReadFile(filepath.Join(source, name))
				if err != nil {
					t.Fatal(err)
				}
				var cfg map[string]any
				if err := json.Unmarshal(raw, &cfg); err != nil {
					t.Fatal(err)
				}
				cfg["added_tokens_decoder"].(map[string]any)["151676"].(map[string]any)["content"] = "<|wrong_audio_pad|>"
				bad, err = json.Marshal(cfg)
				if err != nil {
					t.Fatal(err)
				}
			}
			if err := os.WriteFile(filepath.Join(dir, name), bad, 0o600); err != nil {
				t.Fatal(err)
			}
			store := newCaptureStore()
			err = Create(t.Context(), "invalid-asr", dir, testPipelineOptions(), store, func(_ context.Context, _ string, _ ManifestInfo) error {
				t.Fatal("invalid sidecar reached manifest writes")
				return nil
			}, func(string) {})
			if !errors.Is(err, ErrInvalidTokenizer) || len(store.blobs) != 0 {
				t.Fatalf("invalid sidecar reached tensor writes: %v, blobs=%d", err, len(store.blobs))
			}
		})
	}
}
