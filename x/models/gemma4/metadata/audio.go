package metadata

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/ollama/ollama/x/tokenizer"
)

const (
	gemma4AudioFeatureSize = 128
	maxAudioHiddenSize     = 8192
	maxAudioLayers         = 128
	maxAudioHeads          = 256
	maxAudioOutputDims     = 16384
	maxAudioConvChannels   = 4096
	maxAudioKernelSize     = 255
	maxAudioContextSize    = 4096
	maxTextHiddenSize      = 65536
)

type AudioConfig struct {
	ModelType               string  `json:"model_type"`
	AudioEmbedDim           int     `json:"audio_embed_dim"`
	AudioSamplesPerToken    int     `json:"audio_samples_per_token"`
	AttentionChunkSize      int     `json:"attention_chunk_size"`
	AttentionContextLeft    int     `json:"attention_context_left"`
	AttentionContextRight   int     `json:"attention_context_right"`
	AttentionInvalidLogit   float32 `json:"attention_invalid_logits_value"`
	AttentionLogitCap       float32 `json:"attention_logit_cap"`
	ConvKernelSize          int     `json:"conv_kernel_size"`
	GradientClipping        float32 `json:"gradient_clipping"`
	HiddenSize              int     `json:"hidden_size"`
	NumAttentionHeads       int     `json:"num_attention_heads"`
	NumHiddenLayers         int     `json:"num_hidden_layers"`
	OutputProjDims          int     `json:"output_proj_dims"`
	ResidualWeight          float32 `json:"residual_weight"`
	RMSNormEps              float32 `json:"rms_norm_eps"`
	SubsamplingConvChannels []int   `json:"subsampling_conv_channels"`
	UseClippedLinears       bool    `json:"use_clipped_linears"`
}

type audioProcessorConfig struct {
	AudioSequenceLength int `json:"audio_seq_length"`
	FeatureExtractor    struct {
		Type                 string    `json:"feature_extractor_type"`
		AudioSamplesPerToken int       `json:"audio_samples_per_token"`
		Dither               float64   `json:"dither"`
		FeatureSize          int       `json:"feature_size"`
		FFTLength            int       `json:"fft_length"`
		FFTOverdrive         bool      `json:"fft_overdrive"`
		FrameLength          int       `json:"frame_length"`
		HopLength            int       `json:"hop_length"`
		InputScaleFactor     float64   `json:"input_scale_factor"`
		MaxFrequency         float64   `json:"max_frequency"`
		MelFloor             float64   `json:"mel_floor"`
		MinFrequency         float64   `json:"min_frequency"`
		PaddingSide          string    `json:"padding_side"`
		PerBinMean           []float64 `json:"per_bin_mean"`
		PerBinStddev         []float64 `json:"per_bin_stddev"`
		Preemphasis          float64   `json:"preemphasis"`
		SamplingRate         int       `json:"sampling_rate"`
	} `json:"feature_extractor"`
}

// ValidateAudioRuntimeMetadata verifies the processor and tokenizer metadata
// required by the native Gemma 4 audio path. It deliberately accepts only the
// released processor contract implemented by the runner.
func ValidateAudioRuntimeMetadata(cfg ConfigFile, processorData, tokenizerConfigData, tokenizerData []byte) error {
	if err := validateAudioConfig(cfg); err != nil {
		return err
	}
	var processor audioProcessorConfig
	if len(processorData) == 0 {
		return fmt.Errorf("missing processor_config.json")
	}
	if err := json.Unmarshal(processorData, &processor); err != nil {
		return fmt.Errorf("parse processor_config.json: %w", err)
	}
	f := processor.FeatureExtractor
	if isUnifiedAudioConfig(cfg) {
		if processor.AudioSequenceLength != 750 || f.Type != "Gemma4UnifiedAudioFeatureExtractor" ||
			f.FeatureSize != 640 || f.SamplingRate != 16000 || f.AudioSamplesPerToken != 640 ||
			f.PaddingSide != "right" {
			return fmt.Errorf("unsupported Gemma 4 unified audio processor configuration")
		}
	} else if f.Type != "Gemma4AudioFeatureExtractor" || processor.AudioSequenceLength != 750 ||
		f.FeatureSize != 128 || f.SamplingRate != 16000 ||
		f.FrameLength != 320 || f.HopLength != 160 || f.FFTLength != 512 || f.FFTOverdrive ||
		f.Dither != 0 || f.InputScaleFactor != 1 || f.MinFrequency != 0 || f.MaxFrequency != 8000 ||
		f.MelFloor != 1e-3 || f.Preemphasis != 0 || f.PaddingSide != "right" ||
		len(f.PerBinMean) != 0 || len(f.PerBinStddev) != 0 {
		return fmt.Errorf("unsupported Gemma 4 audio processor configuration")
	}
	if cfg.AudioTokenID <= 0 || cfg.TextConfig.VocabSize <= cfg.AudioTokenID {
		return fmt.Errorf("invalid Gemma 4 audio token id %d", cfg.AudioTokenID)
	}
	if len(tokenizerConfigData) == 0 {
		return fmt.Errorf("missing tokenizer_config.json")
	}
	var tokens struct {
		BOA   string `json:"boa_token"`
		Audio string `json:"audio_token"`
		EOA   string `json:"eoa_token"`
	}
	if err := json.Unmarshal(tokenizerConfigData, &tokens); err != nil {
		return fmt.Errorf("parse tokenizer_config.json: %w", err)
	}
	if tokens.BOA == "" || tokens.Audio == "" || tokens.EOA == "" {
		return fmt.Errorf("missing Gemma 4 audio tokenizer tokens")
	}
	if len(tokenizerData) == 0 {
		return fmt.Errorf("missing tokenizer.json")
	}
	tok, err := tokenizer.LoadFromBytes(tokenizerData)
	if err != nil {
		return fmt.Errorf("parse tokenizer.json: %w", err)
	}
	for _, token := range []string{tokens.BOA, tokens.Audio, tokens.EOA} {
		ids := tok.Encode(token, false)
		if len(ids) != 1 || ids[0] < 0 || int(ids[0]) >= cfg.TextConfig.VocabSize {
			return fmt.Errorf("audio tokenizer token %q is not a valid singleton token", token)
		}
		if token == tokens.Audio && int(ids[0]) != cfg.AudioTokenID {
			return fmt.Errorf("audio tokenizer token id %d, want %d", ids[0], cfg.AudioTokenID)
		}
	}
	return nil
}

// ValidateAudioTensors verifies the normalized tensor names required by the
// native Gemma 4 MLX audio loader.
func ValidateAudioTensors(cfg ConfigFile, names []string) error {
	if err := validateAudioConfig(cfg); err != nil {
		return err
	}
	for name := range requiredAudioShapes(cfg) {
		if !slices.Contains(names, name) {
			return fmt.Errorf("missing %s", name)
		}
	}
	return nil
}

// ValidateAudioSourceInventory additionally validates released tensor shapes
// and floating-point dtypes.
func ValidateAudioSourceInventory(cfg ConfigFile, tensors map[string]TensorDescriptor) error {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	if err := ValidateAudioTensors(cfg, names); err != nil {
		return err
	}
	for name, shape := range requiredAudioShapes(cfg) {
		desc := tensors[name]
		if !slices.Equal(desc.Shape, shape) {
			return fmt.Errorf("%s shape %v, want %v", name, desc.Shape, shape)
		}
		if !isFloatingDtype(desc.Dtype) {
			return fmt.Errorf("%s dtype %s is not floating point", name, desc.Dtype)
		}
	}
	return nil
}

// RequiredAudioTensorShapes returns a copy of the normalized audio tensor
// contract derived from config.
func RequiredAudioTensorShapes(cfg ConfigFile) (map[string][]int32, error) {
	if err := validateAudioConfig(cfg); err != nil {
		return nil, err
	}
	shapes := requiredAudioShapes(cfg)
	out := make(map[string][]int32, len(shapes))
	for name, shape := range shapes {
		out[name] = append([]int32(nil), shape...)
	}
	return out, nil
}

func validateAudioConfig(cfg ConfigFile) error {
	ac := cfg.AudioConfig
	if ac == nil {
		return fmt.Errorf("missing audio_config")
	}
	if isUnifiedAudioConfig(cfg) {
		if ac.ModelType != "gemma4_unified_audio" || ac.AudioEmbedDim != 640 ||
			ac.AudioSamplesPerToken != 640 || ac.HiddenSize != 640 || ac.OutputProjDims != 640 ||
			ac.RMSNormEps <= 0 || cfg.TextConfig.HiddenSize <= 0 || cfg.TextConfig.HiddenSize > maxTextHiddenSize {
			return fmt.Errorf("invalid Gemma 4 unified audio dimensions")
		}
		return nil
	}
	if ac.ModelType != "gemma4_audio" {
		return fmt.Errorf("unsupported Gemma 4 audio model type %q", ac.ModelType)
	}
	if ac.HiddenSize <= 0 || ac.HiddenSize > maxAudioHiddenSize ||
		ac.NumHiddenLayers <= 0 || ac.NumHiddenLayers > maxAudioLayers ||
		ac.NumAttentionHeads <= 0 || ac.NumAttentionHeads > maxAudioHeads || ac.HiddenSize%ac.NumAttentionHeads != 0 ||
		ac.OutputProjDims <= 0 || ac.OutputProjDims > maxAudioOutputDims ||
		ac.ConvKernelSize <= 0 || ac.ConvKernelSize > maxAudioKernelSize || ac.ConvKernelSize%2 == 0 ||
		ac.AttentionChunkSize <= 0 || ac.AttentionChunkSize > maxAudioContextSize ||
		ac.AttentionContextLeft <= 0 || ac.AttentionContextLeft > maxAudioContextSize ||
		ac.AttentionContextRight < 0 || ac.AttentionContextRight > maxAudioContextSize ||
		ac.AttentionInvalidLogit >= 0 || ac.AttentionLogitCap <= 0 ||
		ac.GradientClipping <= 0 || ac.ResidualWeight <= 0 || ac.RMSNormEps <= 0 ||
		len(ac.SubsamplingConvChannels) != 2 ||
		ac.SubsamplingConvChannels[0] <= 0 || ac.SubsamplingConvChannels[0] > maxAudioConvChannels ||
		ac.SubsamplingConvChannels[1] <= 0 || ac.SubsamplingConvChannels[1] > maxAudioConvChannels ||
		cfg.TextConfig.HiddenSize <= 0 || cfg.TextConfig.HiddenSize > maxTextHiddenSize {
		return fmt.Errorf("invalid Gemma 4 audio dimensions")
	}
	return nil
}

func requiredAudioShapes(cfg ConfigFile) map[string][]int32 {
	ac := cfg.AudioConfig
	if isUnifiedAudioConfig(cfg) {
		return map[string][]int32{
			"model.embed_audio.embedding_projection.weight": {int32(cfg.TextConfig.HiddenSize), int32(ac.OutputProjDims)},
		}
	}
	hidden := int32(ac.HiddenSize)
	output := int32(ac.OutputProjDims)
	headDim := hidden / int32(ac.NumAttentionHeads)
	c0 := int32(ac.SubsamplingConvChannels[0])
	c1 := int32(ac.SubsamplingConvChannels[1])
	freq0 := int32((gemma4AudioFeatureSize + 1) / 2)
	freq1 := (freq0 + 1) / 2

	required := map[string][]int32{
		"model.audio_tower.subsample_conv_projection.layer0.conv.weight":       {c0, 1, 3, 3},
		"model.audio_tower.subsample_conv_projection.layer0.norm.weight":       {c0},
		"model.audio_tower.subsample_conv_projection.layer1.conv.weight":       {c1, c0, 3, 3},
		"model.audio_tower.subsample_conv_projection.layer1.norm.weight":       {c1},
		"model.audio_tower.subsample_conv_projection.input_proj_linear.weight": {hidden, freq1 * c1},
		"model.audio_tower.output_proj.weight":                                 {output, hidden},
		"model.audio_tower.output_proj.bias":                                   {output},
		"model.embed_audio.embedding_projection.weight":                        {int32(cfg.TextConfig.HiddenSize), output},
	}

	addLinear := func(path string, shape []int32) {
		required[path+".linear.weight"] = shape
		if ac.UseClippedLinears {
			for _, suffix := range []string{".input_min", ".input_max", ".output_min", ".output_max"} {
				required[path+suffix] = []int32{}
			}
		}
	}

	for i := range ac.NumHiddenLayers {
		layer := fmt.Sprintf("model.audio_tower.layers.%d", i)
		for _, ff := range []string{"feed_forward1", "feed_forward2"} {
			required[layer+"."+ff+".pre_layer_norm.weight"] = []int32{hidden}
			required[layer+"."+ff+".post_layer_norm.weight"] = []int32{hidden}
			addLinear(layer+"."+ff+".ffw_layer_1", []int32{4 * hidden, hidden})
			addLinear(layer+"."+ff+".ffw_layer_2", []int32{hidden, 4 * hidden})
		}

		required[layer+".norm_pre_attn.weight"] = []int32{hidden}
		required[layer+".norm_post_attn.weight"] = []int32{hidden}
		required[layer+".norm_out.weight"] = []int32{hidden}
		for _, projection := range []string{"q_proj", "k_proj", "v_proj", "post"} {
			addLinear(layer+".self_attn."+projection, []int32{hidden, hidden})
		}
		required[layer+".self_attn.per_dim_scale"] = []int32{headDim}
		required[layer+".self_attn.relative_k_proj.weight"] = []int32{hidden, hidden}

		required[layer+".lconv1d.pre_layer_norm.weight"] = []int32{hidden}
		required[layer+".lconv1d.conv_norm.weight"] = []int32{hidden}
		required[layer+".lconv1d.depthwise_conv1d.weight"] = []int32{hidden, 1, int32(ac.ConvKernelSize)}
		addLinear(layer+".lconv1d.linear_start", []int32{2 * hidden, hidden})
		addLinear(layer+".lconv1d.linear_end", []int32{hidden, hidden})
	}

	return required
}

func isUnifiedAudioConfig(cfg ConfigFile) bool {
	return cfg.AudioConfig != nil && cfg.AudioConfig.ModelType == "gemma4_unified_audio"
}
