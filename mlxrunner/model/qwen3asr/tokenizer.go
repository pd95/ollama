package qwen3asr

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

// LoadTokenizer applies identical ASR sidecar checks during import and startup.
func LoadTokenizer(config, data, tokenConfig, generation []byte) (*tokenizer.Tokenizer, error) {
	var gen struct {
		EOS []int32 `json:"eos_token_id"`
	}
	if err := json.Unmarshal(generation, &gen); err != nil {
		return nil, fmt.Errorf("Qwen3-ASR generation config: %w", err)
	}
	if len(gen.EOS) != 2 || !slices.Contains(gen.EOS, 151643) || !slices.Contains(gen.EOS, 151645) {
		return nil, fmt.Errorf("Qwen3-ASR requires the original two generation EOS tokens")
	}
	if !json.Valid(tokenConfig) {
		return nil, fmt.Errorf("invalid Qwen3-ASR tokenizer config")
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(data, &tokenizer.TokenizerConfig{
		ConfigJSON: config, TokenizerConfigJSON: tokenConfig, GenerationConfigJSON: generation,
	})
	if err != nil {
		return nil, err
	}
	for id, text := range map[int32]string{151643: "<|endoftext|>", 151644: "<|im_start|>", 151645: "<|im_end|>", 151669: "<|audio_start|>", 151670: "<|audio_end|>", audioPad: "<|audio_pad|>", 151704: "<asr_text>"} {
		if tok.Decode([]int32{id}) != text || !slices.Equal(tok.Encode(text, false), []int32{id}) {
			return nil, fmt.Errorf("Qwen3-ASR tokenizer mismatch at %d", id)
		}
	}
	if len(tok.EOSTokens()) != 2 || !tok.IsEOS(151643) || !tok.IsEOS(151645) {
		return nil, fmt.Errorf("Qwen3-ASR requires the original two generation EOS tokens")
	}
	return tok, nil
}
