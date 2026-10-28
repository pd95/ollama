// Package config validates the supplied Voxtral 2603 TTS model geometry.
package config

import (
	"encoding/json"
	"fmt"
)

const Architecture = "VoxtralTTSForConditionalGeneration"

type Dimensions struct {
	Dim     int     `json:"dim"`
	Layers  int     `json:"n_layers"`
	HeadDim int     `json:"head_dim"`
	Hidden  int     `json:"hidden_dim"`
	Heads   int     `json:"n_heads"`
	KVHeads int     `json:"n_kv_heads"`
	Norm    float32 `json:"norm_eps"`
	Rope    float32 `json:"rope_theta"`
}

type Config struct {
	Dimensions
	ModelType string `json:"model_type"`
	Vocab     int    `json:"vocab_size"`
	Context   int    `json:"max_position_embeddings"`
	Causal    bool   `json:"causal"`
	Bias      bool   `json:"use_biases"`
	Tied      bool   `json:"tied_embeddings"`
	Quant     struct {
		Bits  int    `json:"bits"`
		Group int    `json:"group_size"`
		Mode  string `json:"mode"`
	} `json:"quantization"`
	Multimodal struct {
		BOS   int `json:"bos_token_id"`
		Audio struct {
			Semantic int `json:"semantic_codebook_size"`
			Acoustic int `json:"acoustic_codebook_size"`
			Books    int `json:"n_acoustic_codebook"`
			AudioID  int `json:"audio_token_id"`
			BeginID  int `json:"begin_audio_token_id"`
			Flow     struct {
				Dimensions
				Input    int     `json:"input_dim"`
				Sigma    float32 `json:"sigma"`
				SigmaMax float32 `json:"sigma_max"`
			} `json:"acoustic_transformer_args"`
			Encoding struct {
				Rate      int     `json:"sampling_rate"`
				FrameRate float64 `json:"frame_rate"`
				Books     int     `json:"num_codebooks"`
				Pattern   string  `json:"codebook_pattern"`
			} `json:"audio_encoding_args"`
		} `json:"audio_model_args"`
		Codec struct {
			Dimensions
			SemanticDim int            `json:"semantic_dim"`
			AcousticDim int            `json:"acoustic_dim"`
			Patch       int            `json:"pretransform_patch_size"`
			Kernel      int            `json:"patch_proj_kernel_size"`
			Semantic    int            `json:"semantic_codebook_size"`
			Acoustic    int            `json:"acoustic_codebook_size"`
			Lengths     string         `json:"decoder_transformer_lengths_str"`
			Kernels     string         `json:"decoder_convs_kernels_str"`
			Strides     string         `json:"decoder_convs_strides_str"`
			Voices      map[string]int `json:"voice"`
			QKNorm      bool           `json:"qk_norm"`
			QKEpsilon   float32        `json:"qk_norm_eps"`
			LayerScale  bool           `json:"layer_scale"`
			Bias        bool           `json:"use_biases"`
		} `json:"audio_tokenizer_args"`
	} `json:"multimodal"`
	Voices map[string]int `json:"ollama_voice_num_audio_tokens,omitempty"`
}

func Parse(raw []byte) (*Config, error) {
	var c Config
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	if c.ModelType != "voxtral_tts" || c.Dim != 3072 || c.Layers != 26 || c.HeadDim != 128 || c.Heads != 32 || c.KVHeads != 8 || c.Hidden != 9216 || c.Vocab != 131072 || c.Norm != 1e-5 || c.Rope != 1e6 || c.Context != 128000 || !c.Causal || c.Bias || !c.Tied {
		return nil, fmt.Errorf("unsupported Voxtral TTS backbone configuration")
	}
	a := c.Multimodal.Audio
	if c.Multimodal.BOS != 1 || a.AudioID != 24 || a.BeginID != 25 || a.Semantic != 8192 || a.Acoustic != 21 || a.Books != 36 || a.Encoding.Rate != 24000 || a.Encoding.FrameRate != 12.5 || a.Encoding.Books != 37 || a.Encoding.Pattern != "parallel" {
		return nil, fmt.Errorf("unsupported Voxtral TTS codebook configuration")
	}
	if a.Flow.Dim != 3072 || a.Flow.Layers != 3 || a.Flow.HeadDim != 128 || a.Flow.Hidden != 9216 || a.Flow.Heads != 32 || a.Flow.KVHeads != 8 || a.Flow.Input != 3072 || a.Flow.Sigma != 1e-5 || a.Flow.SigmaMax != 1 {
		return nil, fmt.Errorf("unsupported Voxtral TTS acoustic transformer")
	}
	t := c.Multimodal.Codec
	if t.Dim != 1024 || t.Heads != 8 || t.KVHeads != 8 || t.HeadDim != 128 || t.Hidden != 4096 || t.Norm != .01 || t.SemanticDim != 256 || t.AcousticDim != 36 || t.Patch != 240 || t.Kernel != 7 || t.Semantic != 8192 || t.Acoustic != 21 || t.Lengths != "2,2,2,2" || t.Kernels != "3,4,4,4" || t.Strides != "1,2,2,2" || !t.QKNorm || t.QKEpsilon != 1e-6 || !t.LayerScale || t.Bias {
		return nil, fmt.Errorf("unsupported Voxtral TTS codec configuration")
	}
	if c.Quant.Bits != 0 && ((c.Quant.Bits != 4 && c.Quant.Bits != 6) || c.Quant.Group != 64 || (c.Quant.Mode != "" && c.Quant.Mode != "affine")) {
		return nil, fmt.Errorf("Voxtral TTS requires floating-point or affine 4/6-bit weights")
	}
	if c.Quant.Bits != 0 {
		c.Quant.Mode = "affine"
	}
	return &c, nil
}
