// Package config describes the supported Kokoro v1 speech checkpoints.
package config

import (
	"encoding/json"
	"fmt"
	"slices"
	"unicode/utf8"
)

const Architecture = "KokoroForSpeechSynthesis"

type Config struct {
	Architectures []string       `json:"architectures,omitempty"`
	ModelType     string         `json:"model_type,omitempty"`
	Hidden        int            `json:"hidden_dim"`
	Style         int            `json:"style_dim"`
	Layers        int            `json:"n_layer"`
	Tokens        int            `json:"n_token"`
	MaxDuration   int            `json:"max_dur"`
	TextKernel    int            `json:"text_encoder_kernel_size"`
	Vocab         map[string]int `json:"vocab"`
	Voices        []string       `json:"ollama_voices,omitempty"`
	BERT          struct {
		Hidden       int `json:"hidden_size"`
		Heads        int `json:"num_attention_heads"`
		Layers       int `json:"num_hidden_layers"`
		Intermediate int `json:"intermediate_size"`
		Context      int `json:"max_position_embeddings"`
	} `json:"plbert"`
	ISTFT struct {
		FFT       int     `json:"gen_istft_n_fft"`
		Hop       int     `json:"gen_istft_hop_size"`
		Rates     []int   `json:"upsample_rates"`
		Kernels   []int   `json:"upsample_kernel_sizes"`
		Residuals []int   `json:"resblock_kernel_sizes"`
		Dilations [][]int `json:"resblock_dilation_sizes"`
		Channels  int     `json:"upsample_initial_channel"`
	} `json:"istftnet"`
	Quant struct {
		Bits  int    `json:"bits"`
		Group int    `json:"group_size"`
		Mode  string `json:"mode"`
	} `json:"quantization"`
}

func Parse(raw []byte) (*Config, error) {
	var c Config
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	if c.ModelType != "" && c.ModelType != "kokoro" {
		return nil, fmt.Errorf("not a Kokoro configuration")
	}
	if c.Hidden != 512 || c.Style != 128 || c.Layers != 3 || c.Tokens != 178 || c.MaxDuration != 50 || c.TextKernel != 5 || c.BERT.Hidden != 768 || c.BERT.Heads != 12 || c.BERT.Layers != 12 || c.BERT.Intermediate != 2048 || c.BERT.Context != 512 {
		return nil, fmt.Errorf("unsupported Kokoro text/prosody configuration")
	}
	if c.ISTFT.FFT != 20 || c.ISTFT.Hop != 5 || c.ISTFT.Channels != 512 || !slices.Equal(c.ISTFT.Rates, []int{10, 6}) || !slices.Equal(c.ISTFT.Kernels, []int{20, 12}) || !slices.Equal(c.ISTFT.Residuals, []int{3, 7, 11}) || len(c.ISTFT.Dilations) != 3 {
		return nil, fmt.Errorf("unsupported Kokoro waveform configuration")
	}
	for _, d := range c.ISTFT.Dilations {
		if !slices.Equal(d, []int{1, 3, 5}) {
			return nil, fmt.Errorf("unsupported Kokoro residual dilation")
		}
	}
	if len(c.Vocab) == 0 || c.Vocab[" "] != 16 || c.Vocab["."] != 4 {
		return nil, fmt.Errorf("unsupported Kokoro phoneme vocabulary")
	}
	seen := map[int]bool{}
	for phoneme, id := range c.Vocab {
		if utf8.RuneCountInString(phoneme) != 1 || id <= 0 || id >= c.Tokens || seen[id] {
			return nil, fmt.Errorf("invalid Kokoro phoneme %q or ID %d", phoneme, id)
		}
		seen[id] = true
	}
	if c.Quant.Bits != 0 && (c.Quant.Bits != 4 || c.Quant.Group != 64 || (c.Quant.Mode != "" && c.Quant.Mode != "affine")) {
		return nil, fmt.Errorf("Kokoro requires floating-point or affine 4-bit weights")
	}
	if c.Quant.Bits != 0 {
		c.Quant.Mode = "affine"
	}
	return &c, nil
}
