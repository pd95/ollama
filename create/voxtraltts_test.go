package create

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/types/model"
)

func TestVoxtralImportedLanguageMetadata(t *testing.T) {
	cfg := &model.ConfigV2{}
	if err := (voxtralTTSImportTransform{voices: []string{"casual_male", "fr_female", "hi_male"}}).augmentConfig(cfg); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(cfg.Languages, []string{"en", "fr", "hi"}) || !slices.Equal(cfg.VoiceLanguages["fr_female"], []string{"fr"}) {
		t.Fatalf("metadata: %+v", cfg)
	}
}

func TestVoxtralVoiceDiscoveryRetainsAllPresentPresets(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "voice_embedding"), 0o700); err != nil {
		t.Fatal(err)
	}
	for _, name := range voxtralEnglishGermanVoices {
		if err := os.WriteFile(filepath.Join(dir, "voice_embedding", name+".safetensors"), nil, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "voice_embedding", "fr_female.safetensors"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	files, err := voxtralVoiceFiles(dir)
	if err != nil || len(files) != 8 || !slices.Contains(files, "voice_embedding/fr_female.safetensors") {
		t.Fatalf("voice discovery: %v %v", files, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "params.json"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := voxtralVoiceFiles(dir); err == nil {
		t.Fatal("incomplete official voice set accepted")
	}
}

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
