package config

import (
	"encoding/json"
	"os"
	"testing"
)

func TestConfiguration(t *testing.T) {
	raw, err := os.ReadFile("../../../../create/testdata/chatterbox-config.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(raw); err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	_ = json.Unmarshal(raw, &fields)
	for key, value := range map[string]any{"t3_model": "turbo", "multilingual": false, "sample_rate": 22050, "quantization": map[string]any{"bits": 4}} {
		copy := map[string]any{}
		for k, v := range fields {
			copy[k] = v
		}
		copy[key] = value
		changed, _ := json.Marshal(copy)
		if Validate(changed) == nil {
			t.Errorf("accepted %s", key)
		}
	}
}
