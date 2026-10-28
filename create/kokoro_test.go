package create

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func writeKokoroSource(t *testing.T, dir string) []byte {
	t.Helper()
	raw, err := os.ReadFile("testdata/kokoro-config.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	createTestSafetensors(t, filepath.Join(dir, "kokoro-v1_0.safetensors"), []*st.TensorData{st.NewTensorDataFromBytes("bert.embeddings.word_embeddings.weight", "F32", []int32{178, 4}, make([]byte, 178*4*4))})
	if err := os.Mkdir(filepath.Join(dir, "voices"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, voice := range []string{"af_heart", "am_adam", "bf_emma", "zf_xiaoxiao"} {
		createTestSafetensors(t, filepath.Join(dir, "voices", voice+".safetensors"), []*st.TensorData{st.NewTensorDataFromBytes("voice", "F32", []int32{510, 1, 256}, make([]byte, 510*256*4))})
	}
	return raw
}

func TestKokoroImport(t *testing.T) {
	dir := t.TempDir()
	original := writeKokoroSource(t, dir)
	files, err := SafetensorsWeightFiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(files, "voices/af_heart.safetensors") || slices.Contains(files, "voices/zf_xiaoxiao.safetensors") {
		t.Fatalf("English transfer voices: %v", files)
	}
	store := newCaptureStore()
	var info ManifestInfo
	err = Create(t.Context(), "kokoro", dir, testPipelineOptions(), store, func(_ context.Context, _ string, m ManifestInfo) error { info = m; return nil }, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(info.ModelConfig.Capabilities, []string{"speech"}) || !slices.Equal(info.ModelConfig.Voices, []string{"af_heart", "am_adam", "bf_emma"}) || info.ModelConfig.Parser != "" || info.ModelConfig.Renderer != "" {
		t.Fatalf("metadata: %+v", info.ModelConfig)
	}
	var normalized map[string]any
	if err := json.Unmarshal(store.blobs["config.json"], &normalized); err != nil {
		t.Fatal(err)
	}
	if normalized["model_type"] != "kokoro" {
		t.Fatalf("normalized config: %v", normalized)
	}
	if store.blobs["voices.af_heart"] == nil || store.blobs["voices.zf_xiaoxiao"] != nil {
		t.Fatal("wrong imported voices")
	}
	after, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil || string(after) != string(original) {
		t.Fatal("source config was changed")
	}
}

func TestKokoroVoiceValidationBeforeWrites(t *testing.T) {
	for _, broken := range []string{"missing", "wrong-shape"} {
		t.Run(broken, func(t *testing.T) {
			dir := t.TempDir()
			writeKokoroSource(t, dir)
			path := filepath.Join(dir, "voices/af_heart.safetensors")
			if broken == "missing" {
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
			} else {
				createTestSafetensors(t, path, []*st.TensorData{st.NewTensorDataFromBytes("voice", "F32", []int32{1, 256}, make([]byte, 256*4))})
			}
			store := newCaptureStore()
			err := Create(t.Context(), "bad-kokoro", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
			if err == nil || len(store.blobs) != 0 {
				t.Fatalf("accepted malformed voice: %v", err)
			}
		})
	}
}
