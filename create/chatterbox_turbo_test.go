package create

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func TestTurboImport(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, data []byte) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile("testdata/chatterbox-turbo-config.json")
	if err != nil {
		t.Fatal(err)
	}
	write("config.json", raw)
	write("tokenizer_config.json", []byte(`{"tokenizer_class":"GPT2Tokenizer","added_tokens_decoder":{}}`))
	write("vocab.json", []byte(`{"H":0,"e":1,"l":2,"o":3,".":4}`))
	write("merges.txt", []byte("#version: 0.2\n"))
	tensor := func(name, dtype string, shape []int32) *st.TensorData {
		n := 4
		for _, d := range shape {
			n *= int(d)
		}
		return st.NewTensorDataFromBytes(name, dtype, shape, make([]byte, n))
	}
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{tensor("t3.speech_head.weight", "F32", []int32{2, 2})})
	for _, sub := range []string{"s3_tokenizer", "reference_encoder"} {
		if err := os.Mkdir(filepath.Join(dir, sub), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	write("s3_tokenizer/config.json", []byte(`{"model_type":"s3_tokenizer_v2","version":"2.0","sample_rate":16000,"token_rate":25,"codebook_size":6561}`))
	v3, err := os.ReadFile("testdata/chatterbox-config.json")
	if err != nil {
		t.Fatal(err)
	}
	write("reference_encoder/config.json", v3)
	createTestSafetensors(t, filepath.Join(dir, "s3_tokenizer/model.safetensors"), []*st.TensorData{tensor("encoder.conv1.weight", "F32", []int32{2, 2})})
	createTestSafetensors(t, filepath.Join(dir, "reference_encoder/model.safetensors"), []*st.TensorData{tensor("s3gen.speaker_encoder.head.conv1.weight", "F32", []int32{2, 2}), tensor("t3.unused.weight", "F32", []int32{2, 2})})
	var conds []*st.TensorData
	for name, shape := range map[string][]int32{"t3.speaker_emb": {1, 256}, "t3.cond_prompt_speech_tokens": {1, 375}, "gen.embedding": {1, 192}, "gen.prompt_feat": {1, 500, 80}, "gen.prompt_token": {1, 250}} {
		dtype := "F32"
		if name == "gen.prompt_token" || name == "t3.cond_prompt_speech_tokens" {
			dtype = "I32"
		}
		conds = append(conds, tensor(name, dtype, shape))
	}
	createTestSafetensors(t, filepath.Join(dir, "conds.safetensors"), conds)
	store := newCaptureStore()
	var info ManifestInfo
	err = Create(t.Context(), "turbo", dir, testPipelineOptions(), store, func(_ context.Context, _ string, m ManifestInfo) error { info = m; return nil }, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(info.ModelConfig.Capabilities, []string{"speech"}) || !slices.Equal(info.ModelConfig.Voices, []string{"default"}) {
		t.Fatal(info.ModelConfig)
	}
	if store.blobs["default.gen.prompt_feat"] == nil || store.blobs["tokenizer.json"] == nil || store.blobs["s3gen.speaker_encoder.head.conv1.weight"] == nil || store.blobs["t3.unused.weight"] != nil {
		t.Fatal("wrong imported dependencies")
	}
	if err := os.Remove(filepath.Join(dir, "reference_encoder/config.json")); err != nil {
		t.Fatal(err)
	}
	store = newCaptureStore()
	err = Create(t.Context(), "bad", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
	if err == nil || len(store.blobs) > 0 {
		t.Fatal("missing reference encoder was admitted")
	}
}
