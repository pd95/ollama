package gptoss

import "testing"

func TestParseConfigDefaultsRopeScalingFactor(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"hidden_size": 2880,
		"num_hidden_layers": 24,
		"num_attention_heads": 64,
		"num_key_value_heads": 8,
		"head_dim": 64,
		"num_experts": 32,
		"experts_per_token": 4
	}`))
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	if cfg.RopeScalingFactor != 1 {
		t.Fatalf("RopeScalingFactor = %v, want 1", cfg.RopeScalingFactor)
	}
}

func TestParseConfigKeepsRopeScalingFactor(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"hidden_size": 2880,
		"num_hidden_layers": 24,
		"num_attention_heads": 64,
		"num_key_value_heads": 8,
		"head_dim": 64,
		"rope_scaling_factor": 32,
		"num_experts": 32,
		"experts_per_token": 4
	}`))
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	if cfg.RopeScalingFactor != 32 {
		t.Fatalf("RopeScalingFactor = %v, want 32", cfg.RopeScalingFactor)
	}
}

func TestParseConfigUsesNestedRopeScalingFactor(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"hidden_size": 2880,
		"num_hidden_layers": 24,
		"num_attention_heads": 64,
		"num_key_value_heads": 8,
		"head_dim": 64,
		"rope_scaling": {
			"factor": 32
		},
		"num_experts": 32,
		"experts_per_token": 4
	}`))
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	if cfg.RopeScalingFactor != 32 {
		t.Fatalf("RopeScalingFactor = %v, want 32", cfg.RopeScalingFactor)
	}
}
