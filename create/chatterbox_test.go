package create

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func TestChatterboxImport(t *testing.T) {
	for _, broken := range []string{"", "missing", "dtype", "collision"} {
		t.Run(broken, func(t *testing.T) {
			dir := t.TempDir()
			raw, err := os.ReadFile("testdata/chatterbox-config.json")
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o600); err != nil {
				t.Fatal(err)
			}
			name := "t3.speech_emb.weight"
			if broken == "collision" {
				name = "s3_tokenizer.encoder.conv1.weight"
			}
			createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{st.NewTensorDataFromBytes(name, "F32", []int32{2, 2}, make([]byte, 16))})
			if broken != "missing" {
				if err = os.Mkdir(filepath.Join(dir, "s3_tokenizer"), 0o700); err != nil {
					t.Fatal(err)
				}
				config := []byte(`{"model_type":"s3_tokenizer_v2","version":"2.0","sample_rate":16000,"token_rate":25,"codebook_size":6561}`)
				if err = os.WriteFile(filepath.Join(dir, "s3_tokenizer/config.json"), config, 0o600); err != nil {
					t.Fatal(err)
				}
				dtype := "F32"
				size := 16
				if broken == "dtype" {
					dtype = "F16"
					size = 8
				}
				createTestSafetensors(t, filepath.Join(dir, "s3_tokenizer/model.safetensors"), []*st.TensorData{st.NewTensorDataFromBytes("encoder.conv1.weight", dtype, []int32{2, 2}, make([]byte, size))})
			}
			store := newCaptureStore()
			var info ManifestInfo
			err = Create(t.Context(), "chatterbox", dir, testPipelineOptions(), store, func(_ context.Context, _ string, m ManifestInfo) error { info = m; return nil }, func(string) {})
			if broken != "" {
				if err == nil || len(store.blobs) != 0 {
					t.Fatalf("invalid input wrote blobs: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(info.ModelConfig.Capabilities, []string{"speech"}) || info.ModelConfig.Renderer != "" || info.ModelConfig.Parser != "" || len(info.ModelConfig.Voices) > 0 {
				t.Fatal(info.ModelConfig)
			}
			if store.blobs["s3_tokenizer.encoder.conv1.weight"] == nil || store.blobs["s3_tokenizer/config.json"] == nil {
				t.Fatal("missing S3 runtime assets")
			}
			files, err := SafetensorsWeightFiles(dir)
			if err != nil || !slices.Contains(files, "s3_tokenizer/model.safetensors") {
				t.Fatal(files, err)
			}
			after, _ := os.ReadFile(filepath.Join(dir, "config.json"))
			if string(after) != string(raw) {
				t.Fatal("source config modified")
			}
		})
	}
}
