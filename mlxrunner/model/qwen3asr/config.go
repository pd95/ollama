// Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
// SPDX-License-Identifier: Apache-2.0
// Native Go/MLX adaptations for Ollama.

// Package qwen3asr implements the original Qwen3-ASR audio encoder and thinker.
package qwen3asr

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/mlxrunner/model/qwen3"
)

const (
	Architecture          = "Qwen3ASRForConditionalGeneration"
	audioPad        int32 = 151676
	maxAudioSeconds       = 60
)

type audioConfig struct {
	Width       int    `json:"d_model"`
	Layers      int    `json:"encoder_layers"`
	Heads       int    `json:"encoder_attention_heads"`
	FFN         int    `json:"encoder_ffn_dim"`
	Channels    int    `json:"downsample_hidden_size"`
	Output      int    `json:"output_dim"`
	Mel         int    `json:"num_mel_bins"`
	Window      int    `json:"n_window"`
	InferWindow int    `json:"n_window_infer"`
	Activation  string `json:"activation_function"`
}

type config struct {
	Audio audioConfig
	Text  qwen3.Config
}

func parseConfig(raw []byte) (*config, error) {
	var source struct {
		Architecture []string `json:"architectures"`
		Type         string   `json:"model_type"`
		Thinker      struct {
			Audio audioConfig     `json:"audio_config"`
			Text  json.RawMessage `json:"text_config"`
			Pad   int32           `json:"audio_token_id"`
			Start int32           `json:"audio_start_token_id"`
			End   int32           `json:"audio_end_token_id"`
		} `json:"thinker_config"`
	}
	if err := json.Unmarshal(raw, &source); err != nil {
		return nil, err
	}
	var d qwen3.Config
	var details struct {
		Bias bool   `json:"attention_bias"`
		Act  string `json:"hidden_act"`
		Rope struct {
			Type     string `json:"rope_type"`
			Sections []int  `json:"mrope_section"`
		} `json:"rope_scaling"`
	}
	if err := json.Unmarshal(source.Thinker.Text, &d); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(source.Thinker.Text, &details); err != nil {
		return nil, err
	}
	a := source.Thinker.Audio
	if source.Type != "qwen3_asr" || len(source.Architecture) != 1 || source.Architecture[0] != Architecture ||
		a.Width != 896 || a.Layers != 18 || a.Heads != 14 || a.FFN != 3584 || a.Channels != 480 || a.Output != 1024 || a.Mel != 128 || a.Window != 50 || a.InferWindow != 800 || a.Activation != "gelu" ||
		d.HiddenSize != 1024 || d.NumHiddenLayers != 28 || d.NumAttentionHeads != 16 || d.NumKeyValueHeads != 8 || d.HeadDim != 128 || d.IntermediateSize != 3072 || d.VocabSize != 151936 ||
		d.RMSNormEps != 1e-6 || d.RopeTheta != 1000000 || d.MaxPositionEmbeddings != 65536 || details.Bias || details.Act != "silu" || details.Rope.Type != "default" ||
		source.Thinker.Pad != audioPad || source.Thinker.Start != 151669 || source.Thinker.End != 151670 {
		return nil, fmt.Errorf("Qwen3-ASR currently requires the original 0.6B configuration")
	}
	// All three provider position axes are identical for this audio/text path.
	// Interleaving their frequencies therefore reduces to ordinary half-rotation.
	d.Scale = float32(1 / math.Sqrt(float64(d.HeadDim)))
	d.QKNormEps = d.RMSNormEps
	// The thinker has a separate trained output head, regardless of the nested
	// text-model tie_word_embeddings metadata (that text model has no LM head).
	d.TieWordEmbeddings = false
	return &config{Audio: a, Text: d}, nil
}

func audioTokens(frames int) int {
	n := frames % 100
	for range 3 {
		n = (n + 1) / 2
	}
	return frames/100*13 + n
}

// ValidatePreprocessor rejects metadata whose DSP would differ from the
// qualified Whisper frontend. Missing sampling_rate uses the provider's 16 kHz
// default, as in the original snapshot.
func ValidatePreprocessor(raw []byte) error {
	var p struct {
		Extractor string  `json:"feature_extractor_type"`
		Processor string  `json:"processor_class"`
		Features  int     `json:"feature_size"`
		FFT       int     `json:"n_fft"`
		Hop       int     `json:"hop_length"`
		Rate      int     `json:"sampling_rate"`
		Dither    float64 `json:"dither"`
		Padding   float64 `json:"padding_value"`
		Side      string  `json:"padding_side"`
		Mask      bool    `json:"return_attention_mask"`
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return err
	}
	if p.Extractor != "WhisperFeatureExtractor" || p.Processor != "Qwen3ASRProcessor" || p.Features != 128 || p.FFT != 400 || p.Hop != 160 || (p.Rate != 0 && p.Rate != 16000) || p.Dither != 0 || p.Padding != 0 || p.Side != "right" || !p.Mask {
		return fmt.Errorf("unsupported Qwen3-ASR audio preprocessor")
	}
	return nil
}
