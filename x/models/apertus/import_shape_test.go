package apertus

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"reflect"
	"testing"

	imagemanifest "github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/tokenizer"
)

func TestImportedApertusTensorShapes(t *testing.T) {
	if os.Getenv("OLLAMA_MODELS") == "" {
		t.Skip("set OLLAMA_MODELS to the imported model cache to validate imported tensor shapes")
	}

	m, err := imagemanifest.LoadManifest("apertus-mlx")
	if err != nil {
		t.Fatalf("load imported manifest: %v", err)
	}

	got := map[string][]int{}
	for _, layer := range m.GetTensorLayers("") {
		header, err := readSafetensorsHeader(m.BlobPath(layer.Digest))
		if err != nil {
			t.Fatalf("read tensor layer %s: %v", layer.Name, err)
		}
		for name, meta := range header {
			if name == "__metadata__" {
				continue
			}
			var info struct {
				DType string `json:"dtype"`
				Shape []int  `json:"shape"`
			}
			if err := json.Unmarshal(meta, &info); err != nil {
				t.Fatalf("parse tensor %s metadata: %v", name, err)
			}
			got[name] = info.Shape
		}
	}

	want := map[string][]int{
		"model.embed_tokens.weight":                   {131072, 4096},
		"lm_head.weight":                              {131072, 4096},
		"model.norm.weight":                           {4096},
		"model.layers.0.attention_layernorm.weight":   {4096},
		"model.layers.0.feedforward_layernorm.weight": {4096},
		"model.layers.0.self_attn.q_proj.weight":      {4096, 4096},
		"model.layers.0.self_attn.k_proj.weight":      {1024, 4096},
		"model.layers.0.self_attn.v_proj.weight":      {1024, 4096},
		"model.layers.0.self_attn.o_proj.weight":      {4096, 4096},
		"model.layers.0.self_attn.q_norm.weight":      {128},
		"model.layers.0.self_attn.k_norm.weight":      {128},
		"model.layers.0.mlp.up_proj.weight":           {21504, 4096},
		"model.layers.0.mlp.down_proj.weight":         {4096, 21504},
		"model.layers.0.mlp.act_fn.alpha_p":           {1},
		"model.layers.0.mlp.act_fn.alpha_n":           {1},
		"model.layers.0.mlp.act_fn.beta":              {},
		"model.layers.0.mlp.act_fn.eps":               {},
		"model.layers.31.self_attn.q_proj.weight":     {4096, 4096},
		"model.layers.31.self_attn.k_proj.weight":     {1024, 4096},
		"model.layers.31.mlp.down_proj.weight":        {4096, 21504},
		"model.layers.31.mlp.act_fn.eps":              {},
	}

	for name, wantShape := range want {
		gotShape, ok := got[name]
		if !ok {
			t.Fatalf("imported tensor %s missing", name)
		}
		if !reflect.DeepEqual(gotShape, wantShape) {
			t.Fatalf("imported tensor %s shape = %v, want %v", name, gotShape, wantShape)
		}
	}
	if len(got) != 451 {
		t.Fatalf("imported tensor count = %d, want 451", len(got))
	}
}

func TestImportedApertusEOSTokens(t *testing.T) {
	if os.Getenv("OLLAMA_MODELS") == "" {
		t.Skip("set OLLAMA_MODELS to the imported model cache to validate imported EOS tokens")
	}

	m, err := imagemanifest.LoadManifest("apertus-mlx")
	if err != nil {
		t.Fatalf("load imported manifest: %v", err)
	}

	tokData, err := m.ReadConfig("tokenizer.json")
	if err != nil {
		t.Fatalf("read tokenizer.json: %v", err)
	}
	configData, err := m.ReadConfig("config.json")
	if err != nil {
		t.Fatalf("read config.json: %v", err)
	}
	tokConfig := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	if genConfigData, err := m.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = genConfigData
	}
	if tokConfigData, err := m.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = tokConfigData
	}
	if specialTokensMapData, err := m.ReadConfig("special_tokens_map.json"); err == nil {
		tokConfig.SpecialTokensMapJSON = specialTokensMapData
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	want := []int32{2, 68, 72}
	if got := tok.EOSTokens(); !reflect.DeepEqual(got, want) {
		t.Fatalf("EOSTokens() = %v, want %v", got, want)
	}
}

func readSafetensorsHeader(path string) (map[string]json.RawMessage, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, err
	}
	header := make([]byte, headerSize)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, err
	}

	var out map[string]json.RawMessage
	if err := json.Unmarshal(header, &out); err != nil {
		return nil, err
	}
	return out, nil
}
