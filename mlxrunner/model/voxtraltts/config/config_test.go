package config

import (
	"encoding/json"
	"os"
	"testing"
)

func TestSupportedConfiguration(t *testing.T) {
	raw, err := os.ReadFile("../../../../create/testdata/voxtraltts-config.json")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Parse(raw); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(map[string]any){
		func(c map[string]any) { c["quantization"] = map[string]any{"bits": 8, "group_size": 64} },
		func(c map[string]any) { c["n_layers"] = 1 },
		func(c map[string]any) {
			c["multimodal"].(map[string]any)["audio_tokenizer_args"].(map[string]any)["qk_norm"] = false
		},
		func(c map[string]any) {
			c["multimodal"].(map[string]any)["audio_model_args"].(map[string]any)["acoustic_transformer_args"].(map[string]any)["sigma_max"] = 2
		},
	} {
		var fields map[string]any
		if err := json.Unmarshal(raw, &fields); err != nil {
			t.Fatal(err)
		}
		mutate(fields)
		data, err := json.Marshal(fields)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Parse(data); err == nil {
			t.Fatal("accepted unsupported geometry/options")
		}
	}
}
