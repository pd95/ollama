package config

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestSupportedConfiguration(t *testing.T) {
	raw, err := os.ReadFile("../../../../create/testdata/kokoro-config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name      string
		mutate    func(map[string]any)
		errorText string
	}{
		{"float", func(map[string]any) {}, ""},
		{"affine4", func(c map[string]any) { c["quantization"] = map[string]any{"bits": 4, "group_size": 64} }, ""},
		{"other-family", func(c map[string]any) { c["model_type"] = "unrelated" }, "not a Kokoro"},
		{"wrong-context", func(c map[string]any) { c["plbert"].(map[string]any)["max_position_embeddings"] = 256 }, "text/prosody"},
		{"wrong-codec", func(c map[string]any) { c["istftnet"].(map[string]any)["gen_istft_hop_size"] = 6 }, "waveform"},
		{"wrong-quantization", func(c map[string]any) { c["quantization"] = map[string]any{"bits": 8, "group_size": 64} }, "4-bit"},
		{"boundary-token", func(c map[string]any) { c["vocab"].(map[string]any)["x"] = 0 }, "invalid Kokoro phoneme"},
		{"duplicate-token", func(c map[string]any) { c["vocab"].(map[string]any)["x"] = 16 }, "invalid Kokoro phoneme"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var fields map[string]any
			if err := json.Unmarshal(raw, &fields); err != nil {
				t.Fatal(err)
			}
			tc.mutate(fields)
			encoded, err := json.Marshal(fields)
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := Parse(encoded)
			if tc.errorText != "" {
				if err == nil || !strings.Contains(err.Error(), tc.errorText) {
					t.Fatalf("error %v, want %s", err, tc.errorText)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if cfg.Quant.Bits == 4 && cfg.Quant.Mode != "affine" {
				t.Fatal("default affine mode not resolved")
			}
		})
	}
}
