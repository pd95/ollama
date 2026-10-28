package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Official sources retain params.json/consolidated.safetensors. Configuration
// and tensor aliases are virtual; the provider files remain unchanged.
func officialVoxtralLayout(dir string) bool {
	if _, err := os.Stat(filepath.Join(dir, "config.json")); !os.IsNotExist(err) {
		return false
	}
	info, err := os.Stat(filepath.Join(dir, "params.json"))
	return err == nil && info.Mode().IsRegular()
}

func readOfficialVoxtralConfig(dir string) (sourceModelConfig, json.RawMessage, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "params.json"))
	if err != nil {
		return sourceModelConfig{}, nil, err
	}
	normalized, err := normalizeVoxtralTTSConfig(raw)
	if err != nil {
		return sourceModelConfig{}, nil, fmt.Errorf("official Voxtral params.json: %w", err)
	}
	var cfg sourceModelConfig
	if err := json.Unmarshal(normalized, &cfg); err != nil {
		return sourceModelConfig{}, nil, err
	}
	return cfg, normalized, nil
}

var officialVoxtralLayer = regexp.MustCompile(`^layers\.([0-9]+)\.(.+)$`)

func officialVoxtralTensorName(name string) (string, error) {
	switch name {
	case "tok_embeddings.weight", "mm_audio_embeddings.tok_embeddings.weight":
		return "language_model.model.model.embed_tokens.weight", nil
	case "norm.weight":
		return "language_model.model.model.norm.weight", nil
	}
	if suffix, ok := strings.CutPrefix(name, "mm_audio_embeddings.audio_codebook_embeddings."); ok {
		return "audio_codebook_embeddings." + suffix, nil
	}
	if parts := officialVoxtralLayer.FindStringSubmatch(name); parts != nil {
		suffixes := map[string]string{"attention.wq.weight": "self_attn.q_proj.weight", "attention.wk.weight": "self_attn.k_proj.weight", "attention.wv.weight": "self_attn.v_proj.weight", "attention.wo.weight": "self_attn.o_proj.weight", "feed_forward.w1.weight": "mlp.gate_proj.weight", "feed_forward.w2.weight": "mlp.down_proj.weight", "feed_forward.w3.weight": "mlp.up_proj.weight", "attention_norm.weight": "input_layernorm.weight", "ffn_norm.weight": "post_attention_layernorm.weight"}
		suffix, ok := suffixes[parts[2]]
		if !ok {
			return "", fmt.Errorf("unsupported official Voxtral layer tensor %s", name)
		}
		return "language_model.model.model.layers." + parts[1] + "." + suffix, nil
	}
	if strings.HasPrefix(name, "acoustic_transformer.") || strings.HasPrefix(name, "audio_tokenizer.") {
		return name, nil
	}
	return "", fmt.Errorf("unsupported official Voxtral tensor %s", name)
}

func prepareOfficialVoxtralTensors(inv Inventory) (Inventory, error) {
	out := make(map[string]SourceTensor, len(inv.Tensors))
	for name, source := range inv.Tensors {
		if source.Dtype != "BF16" && source.Dtype != "F32" {
			return Inventory{}, fmt.Errorf("official Voxtral requires floating-point tensors: %s", name)
		}
		key, err := officialVoxtralTensorName(name)
		if err != nil {
			return Inventory{}, err
		}
		if key == "" {
			continue
		}
		if _, ok := out[key]; ok {
			return Inventory{}, fmt.Errorf("official Voxtral tensor collision: %s", key)
		}
		out[key] = source
	}
	inv.Tensors = out
	return inv, nil
}
