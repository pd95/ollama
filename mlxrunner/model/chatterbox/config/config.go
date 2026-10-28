// Package config validates the supplied Chatterbox multilingual-v3 geometry.
package config

import (
	"encoding/json"
	"fmt"
)

const Architecture = "ChatterboxForConditionalGeneration"

func Validate(raw []byte) error {
	var c struct {
		Type          string          `json:"model_type"`
		Version       string          `json:"version"`
		T3            string          `json:"t3_model"`
		Multilingual  bool            `json:"multilingual"`
		Vocab         int             `json:"vocab_size"`
		Preprocessing string          `json:"text_preprocessing"`
		Rate          int             `json:"sample_rate"`
		Quant         json.RawMessage `json:"quantization"`
		T3Config      map[string]any  `json:"t3_config"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return err
	}
	if c.Type != "chatterbox" || c.Version != "1.0" || c.T3 != "v3" || !c.Multilingual || c.Vocab != 2454 || c.Preprocessing != "NFKD,fullcase" || (c.Rate != 0 && c.Rate != 24000) || (len(c.Quant) > 0 && string(c.Quant) != "null") {
		return fmt.Errorf("unsupported Chatterbox multilingual-v3 configuration")
	}
	expected := map[string]any{"text_tokens_dict_size": float64(2454), "start_text_token": float64(255), "stop_text_token": float64(0), "max_text_tokens": float64(2048), "speech_tokens_dict_size": float64(8194), "start_speech_token": float64(6561), "stop_speech_token": float64(6562), "max_speech_tokens": float64(4096), "llama_config_name": "Llama_520M", "input_pos_emb": "learned", "speech_cond_prompt_len": float64(150), "encoder_type": "voice_encoder", "speaker_embed_size": float64(256), "use_perceiver_resampler": true, "emotion_adv": true}
	for k, v := range expected {
		if c.T3Config[k] != v {
			return fmt.Errorf("unsupported Chatterbox t3_config.%s", k)
		}
	}
	return nil
}

func ValidateS3(raw []byte) error {
	var c struct {
		Type    string `json:"model_type"`
		Version string `json:"version"`
		Rate    int    `json:"sample_rate"`
		Tokens  int    `json:"token_rate"`
		Codes   int    `json:"codebook_size"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return err
	}
	if c.Type != "s3_tokenizer_v2" || c.Version != "2.0" || c.Rate != 16000 || c.Tokens != 25 || c.Codes != 6561 {
		return fmt.Errorf("Chatterbox requires S3TokenizerV2 (16 kHz, 25 Hz, 6561 codes)")
	}
	return nil
}
