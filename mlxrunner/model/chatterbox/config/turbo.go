package config

import (
	"encoding/json"
	"fmt"
	"reflect"
)

const TurboArchitecture = "ChatterboxTurboForConditionalGeneration"

func ValidateTurbo(raw []byte) error {
	var c map[string]any
	if err := json.Unmarshal(raw, &c); err != nil {
		return err
	}
	check := func(fields map[string]any, expected map[string]any) bool {
		for key, value := range expected {
			if !reflect.DeepEqual(fields[key], value) {
				return false
			}
		}
		return true
	}
	get := func(key string) map[string]any { x, _ := c[key].(map[string]any); return x }
	if !check(c, map[string]any{"model_type": "chatterbox_turbo", "architecture": "chatterbox_turbo", "sample_rate": float64(24000), "enc_cond_len_seconds": float64(15), "dec_cond_len_seconds": float64(10)}) || !check(get("gpt2"), map[string]any{"n_embd": float64(1024), "n_head": float64(16), "n_layer": float64(24), "n_positions": float64(8196), "vocab_size": float64(50276), "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5}) || !check(get("quantization"), map[string]any{"bits": float64(4), "group_size": float64(64), "mode": "affine"}) {
		return fmt.Errorf("unsupported Chatterbox Turbo checkpoint configuration")
	}
	if !check(get("t3"), map[string]any{"text_tokens_dict_size": float64(50276), "speech_tokens_dict_size": float64(6563), "start_speech_token": float64(6561), "stop_speech_token": float64(6562), "max_text_tokens": float64(2048), "speech_cond_prompt_len": float64(375), "speaker_embed_size": float64(256), "llama_config_name": "GPT2_medium", "input_pos_emb": nil, "use_perceiver_resampler": false, "emotion_adv": false}) || !check(get("s3gen"), map[string]any{"meanflow": true, "output_sample_rate": float64(24000), "input_sample_rate": float64(16000), "silence_token": float64(4299), "speech_vocab_size": float64(6561), "token_embedding_dim": float64(512), "encoder_num_blocks": float64(6), "decoder_in_channels": float64(320), "decoder_out_channels": float64(80), "decoder_num_mid_blocks": float64(12)}) {
		return fmt.Errorf("unsupported Turbo T3/mean-flow geometry")
	}
	if !check(get("s3gen"), map[string]any{"encoder_attention_heads": float64(8), "encoder_linear_units": float64(2048), "decoder_channels": []any{float64(256)}, "decoder_attention_head_dim": float64(64), "decoder_n_blocks": float64(4), "decoder_num_heads": float64(8), "cfm_sigma_min": 1e-6, "cfm_t_scheduler": "cosine", "cfm_inference_cfg_rate": .7}) || !check(get("voice_encoder"), map[string]any{"num_mels": float64(40), "sample_rate": float64(16000), "speaker_embed_size": float64(256), "ve_hidden_size": float64(256), "n_fft": float64(400), "hop_size": float64(160), "win_size": float64(400), "fmax": float64(8000), "fmin": float64(0), "preemphasis": float64(0), "mel_power": float64(2), "mel_type": "amp", "normalized_mels": false, "ve_partial_frames": float64(160), "ve_final_relu": true}) {
		return fmt.Errorf("unsupported Turbo codec/reference frontend")
	}

	return nil
}
