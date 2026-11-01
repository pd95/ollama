package qwen3asr

import (
	"encoding/json"
	"testing"
)

func TestQwen3ASRTokenizerSidecars(t *testing.T) {
	var added []map[string]any
	for id, text := range map[int32]string{151643: "<|endoftext|>", 151644: "<|im_start|>", 151645: "<|im_end|>", 151669: "<|audio_start|>", 151670: "<|audio_end|>", 151676: "<|audio_pad|>", 151704: "<asr_text>"} {
		added = append(added, map[string]any{"id": id, "content": text, "special": true})
	}
	data, err := json.Marshal(map[string]any{"model": map[string]any{"type": "BPE", "vocab": map[string]int{"a": 0}, "merges": []string{}}, "added_tokens": added})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, generation, config string
		valid                    bool
	}{
		{"original EOS", `{"eos_token_id":[151643,151645]}`, `{}`, true},
		{"missing EOS", `{}`, `{}`, false},
		{"single EOS", `{"eos_token_id":151645}`, `{}`, false},
		{"duplicate EOS", `{"eos_token_id":[151645,151645]}`, `{}`, false},
		{"malformed generation", `{`, `{}`, false},
		{"malformed tokenizer config", `{"eos_token_id":[151643,151645]}`, `{`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadTokenizer([]byte(`{}`), data, []byte(tc.config), []byte(tc.generation))
			if (err == nil) != tc.valid {
				t.Fatalf("sidecar acceptance=%v: %v", err == nil, err)
			}
		})
	}
}
