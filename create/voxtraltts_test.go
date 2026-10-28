package create

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func TestVoxtralTTSMissingAssetsBeforeWrites(t *testing.T) {
	for _, broken := range []string{"missing", "invalid"} {
		t.Run(broken, func(t *testing.T) {
			dir := t.TempDir()
			raw, err := os.ReadFile("testdata/voxtraltts-config.json")
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o600); err != nil {
				t.Fatal(err)
			}
			createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{st.NewTensorDataFromBytes("language_model.model.model.embed_tokens.weight", "F32", []int32{1, 1}, make([]byte, 4))})
			if broken == "invalid" {
				if err = os.WriteFile(filepath.Join(dir, "tekken.json"), []byte(`{"config":{}}`), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			store := newCaptureStore()
			err = Create(t.Context(), "voxtral", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { t.Fatal("wrote invalid model"); return nil }, func(string) {})
			if err == nil || len(store.blobs) > 0 {
				t.Fatalf("invalid source reached writes: %v", err)
			}
			after, err := os.ReadFile(filepath.Join(dir, "config.json"))
			if err != nil || string(after) != string(raw) {
				t.Fatal("source config changed")
			}
		})
	}
}
