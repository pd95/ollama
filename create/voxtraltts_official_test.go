package create

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func TestOfficialVoxtralInventory(t *testing.T) {
	dir := t.TempDir()
	raw, err := os.ReadFile("testdata/voxtraltts-official-params.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "params.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	inputs := map[string]string{"mm_audio_embeddings.tok_embeddings.weight": "language_model.model.model.embed_tokens.weight", "layers.0.attention.wq.weight": "language_model.model.model.layers.0.self_attn.q_proj.weight", "layers.0.feed_forward.w2.weight": "language_model.model.model.layers.0.mlp.down_proj.weight", "norm.weight": "language_model.model.model.norm.weight", "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight": "audio_codebook_embeddings.embeddings.weight", "acoustic_transformer.norm.weight": "acoustic_transformer.norm.weight", "audio_tokenizer.decoder.norm.weight": "audio_tokenizer.decoder.norm.weight"}
	var tensors []*st.TensorData
	for name := range inputs {
		tensors = append(tensors, st.NewTensorDataFromBytes(name, "F32", []int32{2, 2}, make([]byte, 16)))
	}
	createTestSafetensors(t, filepath.Join(dir, "consolidated.safetensors"), tensors)
	if !IsSafetensorsModelDir(dir) {
		t.Fatal("CLI did not recognize official Voxtral source")
	}
	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatal(err)
	}
	if inv.Config.Architecture() != voxtralTTSArchitecture {
		t.Fatal(inv.Config.Architecture())
	}
	normalized, err := prepareOfficialVoxtralTensors(inv)
	if err != nil {
		t.Fatal(err)
	}
	for old, name := range inputs {
		got, ok := normalized.Tensors[name]
		if !ok || got.Name != old || got.File != "consolidated.safetensors" {
			t.Fatalf("wrong source alias %s: %+v", name, got)
		}
	}
	specs, err := Plan(normalized, Classification{Kind: SourceFloat}, voxtralTTSImportTransform{})
	if err != nil {
		t.Fatal(err)
	}
	for _, spec := range specs {
		if len(spec.Tensors) != 1 || spec.Tensors[0].Transform != TransformNone || spec.Tensors[0].Quantize != "" {
			t.Fatal("official tensor bytes would change")
		}
	}
	after, err := os.ReadFile(filepath.Join(dir, "params.json"))
	if err != nil || !slices.Equal(after, raw) {
		t.Fatal("original parameters changed")
	}
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), tensors)
	if _, err := ReadInventory(dir); err == nil {
		t.Fatal("accepted ambiguous original/converted weights")
	}
}

func TestOfficialVoxtralAliases(t *testing.T) {
	inv := Inventory{Tensors: map[string]SourceTensor{"tok_embeddings.weight": {Name: "tok_embeddings.weight", Dtype: "BF16", Shape: []int32{2, 2}}, "mm_audio_embeddings.tok_embeddings.weight": {Name: "mm_audio_embeddings.tok_embeddings.weight", Dtype: "BF16", Shape: []int32{2, 2}}}}
	if _, err := prepareOfficialVoxtralTensors(inv); err == nil {
		t.Fatal("accepted colliding aliases")
	}
	inv.Tensors = map[string]SourceTensor{"layers.0.attention.unknown.weight": {Dtype: "BF16"}}
	if _, err := prepareOfficialVoxtralTensors(inv); err == nil {
		t.Fatal("accepted unknown attention mapping")
	}
}
