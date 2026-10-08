package apertus

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"

	imagemanifest "github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
	modeltypes "github.com/ollama/ollama/types/model"
)

func TestImportedApertusTensorShapes(t *testing.T) {
	if os.Getenv("OLLAMA_MODELS") == "" {
		t.Skip("set OLLAMA_MODELS to the imported model cache to validate imported tensor shapes")
	}

	m, err := imagemanifest.ParseNamedManifest(modeltypes.ParseName(importedApertusModelName()))
	if err != nil {
		t.Fatalf("load imported manifest: %v", err)
	}

	got := map[string][]int{}
	rawCount, quantizedCount := 0, 0
	for _, layer := range m.TensorLayers() {
		blobPath, err := imagemanifest.BlobsPath(layer.Digest)
		if err != nil {
			t.Fatal(err)
		}
		header, err := readSafetensorsHeader(blobPath)
		if err != nil {
			t.Fatalf("read tensor layer %s: %v", layer.Name, err)
		}
		var quant struct {
			QuantType string `json:"quant_type"`
			GroupSize string `json:"group_size"`
		}
		if metadata, ok := header["__metadata__"]; ok {
			if err := json.Unmarshal(metadata, &quant); err != nil {
				t.Fatal(err)
			}
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
			rawCount++
			if strings.HasSuffix(name, ".scale") {
				continue
			}
			if _, exists := got[name]; exists {
				t.Fatalf("duplicate tensor %s", name)
			}
			shape := append([]int{}, info.Shape...)
			if quant.QuantType != "" {
				packing, groupSize := 0, 0
				switch quant.QuantType {
				case "nvfp4":
					packing, groupSize = 8, 16
				case "mxfp8":
					packing, groupSize = 4, 32
				default:
					t.Fatalf("unsupported imported quantization %q", quant.QuantType)
				}
				if info.DType != "U32" || len(shape) != 2 {
					t.Fatalf("invalid packed tensor %s: dtype=%s shape=%v", name, info.DType, shape)
				}
				if quant.GroupSize != fmt.Sprint(groupSize) {
					t.Fatalf("invalid quantization group size %q", quant.GroupSize)
				}
				var scale struct {
					DType string `json:"dtype"`
					Shape []int  `json:"shape"`
				}
				if err := json.Unmarshal(header[name+".scale"], &scale); err != nil {
					t.Fatalf("missing or invalid scale for %s: %v", name, err)
				}
				shape[1] *= packing
				if scale.DType != "U8" || !reflect.DeepEqual(scale.Shape, []int{shape[0], shape[1] / groupSize}) {
					t.Fatalf("invalid scale for %s: dtype=%s shape=%v", name, scale.DType, scale.Shape)
				}
				quantizedCount++
			} else if info.DType != "BF16" {
				t.Fatalf("dense tensor %s dtype=%s, want BF16", name, info.DType)
			}
			got[name] = shape
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
	if quantizedCount != 0 && quantizedCount != 193 {
		t.Fatalf("quantized tensor count=%d, want 0 or 193", quantizedCount)
	}
	if rawCount != 451+quantizedCount {
		t.Fatalf("header tensor count=%d, want %d", rawCount, 451+quantizedCount)
	}
}

func importedApertusModelName() string {
	if name := os.Getenv("PORTING_APERTUS_MODEL_NAME"); name != "" {
		return name
	}
	return "apertus-mlx:8b-nvfp4"
}

func TestImportedApertusEOSTokens(t *testing.T) {
	if os.Getenv("OLLAMA_MODELS") == "" {
		t.Skip("set OLLAMA_MODELS to the imported model cache to validate imported EOS tokens")
	}

	m, err := imagemanifest.ParseNamedManifest(modeltypes.ParseName(importedApertusModelName()))
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
