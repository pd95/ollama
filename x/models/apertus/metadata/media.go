package metadata

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

const (
	visionPrefix  = "model.vision_tokenizer."
	audioPrefix   = "model.audio_tokenizer.encoder."
	AudioCodebook = "model.audio_tokenizer.quantizer.codebook.embed"
)

type TensorDescriptor struct {
	Dtype string
	Shape []int32
}

type VisionConfig struct {
	CodebookSize      int32   `json:"codebook_size"`
	EmbedDim          int32   `json:"embed_dim"`
	InChannels        int32   `json:"in_channels"`
	ChannelMultiplier []int32 `json:"channel_multiplier"`
}

type AudioConfig struct {
	CodebookSize     int32   `json:"codebook_size"`
	CodebookDim      int32   `json:"codebook_dim"`
	AudioChannels    int32   `json:"audio_channels"`
	SamplingRate     int32   `json:"sampling_rate"`
	UpsamplingRatios []int32 `json:"upsampling_ratios"`
}

type Config struct {
	Architectures    []string     `json:"architectures"`
	ModelType        string       `json:"model_type"`
	ImageTokenID     int32        `json:"image_token_id"`
	AudioTokenID     int32        `json:"audio_token_id"`
	ImageTokenOffset int32        `json:"image_token_offset"`
	AudioTokenOffset int32        `json:"audio_token_offset"`
	Vision           VisionConfig `json:"vision_tokenizer_config"`
	Audio            AudioConfig  `json:"audio_tokenizer_config"`
}

func ParseConfig(data []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse Apertus media config: %w", err)
	}
	found := false
	for _, architecture := range cfg.Architectures {
		if architecture == "Apertus1p5ForConditionalGeneration" {
			found = true
		}
	}
	if !found {
		return Config{}, fmt.Errorf("unsupported Apertus media architecture %q", cfg.Architectures)
	}
	return cfg, nil
}

func ValidateVisionInventory(cfg Config, tensors map[string]TensorDescriptor) error {
	if cfg.ImageTokenID != 131079 || cfg.ImageTokenOffset != 131272 || cfg.Vision.CodebookSize != 131072 ||
		cfg.Vision.EmbedDim != 256 || cfg.Vision.InChannels != 3 ||
		!slices.Equal(cfg.Vision.ChannelMultiplier, []int32{1, 1, 2, 2, 4}) {
		return fmt.Errorf("unsupported Apertus 1.5 vision tokenizer configuration")
	}
	if err := requireInventory(tensors, visionPrefix, 247, visionRequiredNames()); err != nil {
		return fmt.Errorf("Apertus 1.5 vision inventory: %w", err)
	}
	return nil
}

func ValidateAudioInventory(cfg Config, tensors map[string]TensorDescriptor) error {
	if cfg.AudioTokenID != 131085 || cfg.AudioTokenOffset != 262344 || cfg.Audio.CodebookSize != 4096 ||
		cfg.Audio.CodebookDim != 512 || cfg.Audio.AudioChannels != 1 || cfg.Audio.SamplingRate != 24000 ||
		!slices.Equal(cfg.Audio.UpsamplingRatios, []int32{6, 5, 5, 4}) {
		return fmt.Errorf("unsupported Apertus 1.5 audio tokenizer configuration")
	}
	if err := requireInventory(tensors, audioPrefix, 62, audioRequiredNames()); err != nil {
		return fmt.Errorf("Apertus 1.5 audio inventory: %w", err)
	}
	codebook := tensors[AudioCodebook]
	if !strings.EqualFold(codebook.Dtype, "F32") || !slices.Equal(codebook.Shape, []int32{4096, 512}) {
		return fmt.Errorf("Apertus 1.5 audio codebook descriptor is %s %v, want F32 [4096 512]", codebook.Dtype, codebook.Shape)
	}
	return nil
}

func visionRequiredNames() []string {
	var names []string
	pair := func(path string) { names = append(names, path+".weight", path+".bias") }
	pair(visionPrefix + "encoder.conv_in")
	pair(visionPrefix + "encoder.conv_out")
	pair(visionPrefix + "encoder.norm_out")
	pair(visionPrefix + "quant_conv")
	names = append(names, visionPrefix+"quantize.embedding.weight")
	for level := range 5 {
		for block := range 4 {
			path := fmt.Sprintf("%sencoder.down.%d.block.%d", visionPrefix, level, block)
			pair(path + ".norm1")
			pair(path + ".conv1")
			pair(path + ".norm2")
			pair(path + ".conv2")
			if block == 0 && (level == 2 || level == 4) {
				pair(path + ".nin_shortcut")
			}
			if level == 4 {
				attn := fmt.Sprintf("%sencoder.down.%d.attn.%d", visionPrefix, level, block)
				for _, suffix := range []string{"norm", "q", "k", "v", "proj_out"} {
					pair(attn + "." + suffix)
				}
			}
		}
		if level < 4 {
			pair(fmt.Sprintf("%sencoder.down.%d.downsample.conv", visionPrefix, level))
		}
	}
	for _, block := range []string{"block_1", "block_2"} {
		path := visionPrefix + "encoder.mid." + block
		for _, suffix := range []string{"norm1", "conv1", "norm2", "conv2"} {
			pair(path + "." + suffix)
		}
	}
	for _, suffix := range []string{"norm", "q", "k", "v", "proj_out"} {
		pair(visionPrefix + "encoder.mid.attn_1." + suffix)
	}
	return names
}

// VisionRequiredTensorNames returns the exact released Apertus 1.5 vision inventory.
func VisionRequiredTensorNames() []string { return visionRequiredNames() }

func audioRequiredNames() []string {
	var names []string
	conv := func(path string) {
		names = append(names, path+".conv.bias", path+".conv.parametrizations.weight.original0", path+".conv.parametrizations.weight.original1")
	}
	p := audioPrefix + "layers."
	conv(p + "0")
	conv(p + "15")
	for _, index := range []int{1, 4, 7, 10} {
		path := fmt.Sprintf("%s%d", p, index)
		conv(path + ".block.1")
		conv(path + ".block.3")
		conv(path + ".shortcut")
	}
	for _, index := range []int{3, 6, 9, 12} {
		conv(fmt.Sprintf("%s%d", p, index))
	}
	for layer := range 2 {
		for _, kind := range []string{"weight_ih", "weight_hh", "bias_ih", "bias_hh"} {
			names = append(names, fmt.Sprintf("%s13.lstm.%s_l%d", p, kind, layer))
		}
	}
	names = append(names, AudioCodebook)
	return names
}

// AudioRequiredTensorNames returns the exact released Apertus 1.5 audio inventory.
func AudioRequiredTensorNames() []string { return audioRequiredNames() }

func requireInventory(tensors map[string]TensorDescriptor, prefix string, count int, required []string) error {
	got := 0
	for name, descriptor := range tensors {
		if strings.HasPrefix(name, prefix) {
			got++
			if descriptor.Dtype == "" || len(descriptor.Shape) == 0 {
				return fmt.Errorf("invalid descriptor %q", name)
			}
		}
	}
	if got != count {
		return fmt.Errorf("tensor count %d, want %d", got, count)
	}
	for _, name := range required {
		if _, ok := tensors[name]; !ok {
			return fmt.Errorf("missing tensor %q", name)
		}
	}
	return nil
}
