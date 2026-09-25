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
	if err := requireInventory(tensors, visionPrefix, 247, []string{
		visionPrefix + "encoder.conv_in.weight",
		visionPrefix + "quant_conv.weight",
		visionPrefix + "quantize.embedding.weight",
	}); err != nil {
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
	if err := requireInventory(tensors, audioPrefix, 62, []string{AudioCodebook}); err != nil {
		return fmt.Errorf("Apertus 1.5 audio inventory: %w", err)
	}
	codebook := tensors[AudioCodebook]
	if !strings.EqualFold(codebook.Dtype, "F32") || !slices.Equal(codebook.Shape, []int32{4096, 512}) {
		return fmt.Errorf("Apertus 1.5 audio codebook descriptor is %s %v, want F32 [4096 512]", codebook.Dtype, codebook.Shape)
	}
	return nil
}

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
