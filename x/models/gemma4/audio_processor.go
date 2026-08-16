package gemma4

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"strings"

	"github.com/ollama/ollama/llm"
	mlxmedia "github.com/ollama/ollama/x/mlxrunner/media"
	"gonum.org/v1/gonum/dsp/fourier"
)

const (
	maxGemma4AudioBytes      = 32 << 20
	maxGemma4AudioChannels   = 8
	minGemma4AudioSampleRate = 8_000
	maxGemma4AudioSampleRate = 192_000
	maxGemma4AudioSamples    = 480_000
	maxGemma4AudioFrames     = 3_000
	maxGemma4AudioFeatures   = maxGemma4AudioFrames * 128
)

type AudioProcessorConfig struct {
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

type gemma4AudioInput struct {
	Features    []float32
	FeatureMask []bool
	FeatureSize int
	Frames      int
	SoftTokens  int
}

func defaultAudioProcessorConfig() AudioProcessorConfig {
	var cfg AudioProcessorConfig
	cfg.AudioSequenceLength = 750
	cfg.FeatureExtractor.Type = "Gemma4AudioFeatureExtractor"
	cfg.FeatureExtractor.FeatureSize = 128
	cfg.FeatureExtractor.FFTLength = 512
	cfg.FeatureExtractor.FrameLength = 320
	cfg.FeatureExtractor.HopLength = 160
	cfg.FeatureExtractor.InputScaleFactor = 1
	cfg.FeatureExtractor.MaxFrequency = 8000
	cfg.FeatureExtractor.MelFloor = 1e-3
	cfg.FeatureExtractor.PaddingSide = "right"
	cfg.FeatureExtractor.SamplingRate = 16000
	return cfg
}

func parseAudioProcessorConfig(data []byte) (*AudioProcessorConfig, error) {
	cfg := defaultAudioProcessorConfig()
	if len(data) > 0 {
		if err := json.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("parse Gemma4 audio processor config: %w", err)
		}
	}
	if err := validateReleasedAudioProcessorConfig(&cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func validateReleasedAudioProcessorConfig(cfg *AudioProcessorConfig) error {
	if cfg == nil {
		return errors.New("Gemma4 MLX model has no supported audio processor configuration")
	}
	f := cfg.FeatureExtractor
	if f.Type == "Gemma4UnifiedAudioFeatureExtractor" {
		if cfg.AudioSequenceLength != 750 || f.FeatureSize != 640 || f.SamplingRate != 16000 ||
			f.AudioSamplesPerToken != 640 || f.PaddingSide != "right" {
			return errors.New("unsupported Gemma4 unified audio processor configuration")
		}
		return nil
	}
	if f.Type != "Gemma4AudioFeatureExtractor" || cfg.AudioSequenceLength != 750 ||
		f.FeatureSize != 128 || f.SamplingRate != 16000 ||
		f.FrameLength != 320 || f.HopLength != 160 || f.FFTLength != 512 || f.FFTOverdrive ||
		f.Dither != 0 || f.InputScaleFactor != 1 || f.MinFrequency != 0 || f.MaxFrequency != 8000 ||
		f.MelFloor != 1e-3 || f.Preemphasis != 0 || f.PaddingSide != "right" ||
		len(f.PerBinMean) != 0 || len(f.PerBinStddev) != 0 {
		return errors.New("unsupported Gemma4 audio processor configuration")
	}
	return nil
}

func preprocessGemma4Audio(ctx context.Context, data []byte, cfg *AudioProcessorConfig) (*gemma4AudioInput, error) {
	if err := validateReleasedAudioProcessorConfig(cfg); err != nil {
		return nil, err
	}
	var samples []float32
	var err error
	format, _ := llm.AudioFormat(data)
	if format == "mp3" {
		samples, err = mlxmedia.DecodeMP3(ctx, data, mlxmedia.MP3DecodeOptions{
			TargetSampleRate: cfg.FeatureExtractor.SamplingRate,
			MaxInputBytes:    maxGemma4AudioBytes,
			MaxSamples:       maxGemma4AudioSamples,
			Overflow:         mlxmedia.AudioOverflowReject,
		})
	} else {
		samples, err = decodeGemma4WAV(ctx, data, cfg.FeatureExtractor.SamplingRate)
	}
	if err != nil {
		if format == "mp3" && strings.Contains(err.Error(), "exceeds limit of") {
			return nil, errors.New("Gemma4 audio exceeds the maximum duration of 30 seconds")
		}
		return nil, err
	}
	if cfg.FeatureExtractor.Type == "Gemma4UnifiedAudioFeatureExtractor" {
		return preprocessGemma4UnifiedAudio(samples, cfg)
	}
	features, featureMask, err := computeGemma4LogMel(ctx, samples, cfg)
	if err != nil {
		return nil, err
	}
	outputMask := downsampleGemma4AudioMask(featureMask, 2)
	softTokens := 0
	seenPadding := false
	for _, valid := range outputMask {
		if valid {
			if seenPadding {
				return nil, errors.New("Gemma4 audio validity mask is not a prefix")
			}
			softTokens++
		} else {
			seenPadding = true
		}
	}
	if softTokens == 0 {
		return nil, errors.New("Gemma4 audio is too short to encode")
	}
	if softTokens > cfg.AudioSequenceLength {
		return nil, fmt.Errorf("Gemma4 audio token count %d exceeds limit %d", softTokens, cfg.AudioSequenceLength)
	}
	return &gemma4AudioInput{
		Features: features, FeatureMask: featureMask, FeatureSize: cfg.FeatureExtractor.FeatureSize,
		Frames: len(featureMask), SoftTokens: softTokens,
	}, nil
}

func preprocessGemma4UnifiedAudio(samples []float32, cfg *AudioProcessorConfig) (*gemma4AudioInput, error) {
	frameSize := cfg.FeatureExtractor.AudioSamplesPerToken
	if frameSize <= 0 {
		return nil, errors.New("invalid Gemma4 unified audio frame size")
	}
	softTokens := (len(samples) + frameSize - 1) / frameSize
	if softTokens == 0 {
		return nil, errors.New("Gemma4 audio is too short to encode")
	}
	if softTokens > cfg.AudioSequenceLength {
		return nil, fmt.Errorf("Gemma4 audio token count %d exceeds limit %d", softTokens, cfg.AudioSequenceLength)
	}
	features := make([]float32, softTokens*frameSize)
	copy(features, samples)
	mask := make([]bool, softTokens)
	for i := range mask {
		mask[i] = true
	}
	return &gemma4AudioInput{
		Features: features, FeatureMask: mask, FeatureSize: frameSize,
		Frames: softTokens, SoftTokens: softTokens,
	}, nil
}

func decodeGemma4WAV(ctx context.Context, data []byte, targetRate int) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(data) > maxGemma4AudioBytes {
		return nil, fmt.Errorf("Gemma4 audio is %d bytes, limit %d", len(data), maxGemma4AudioBytes)
	}
	if targetRate != 16000 {
		return nil, fmt.Errorf("unsupported Gemma4 audio target sample rate %d", targetRate)
	}
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, errors.New("Gemma4 audio must be a RIFF/WAVE file")
	}

	var format, channels, bits, blockAlign uint16
	var sampleRate uint32
	var pcm []byte
	seenFormat := false
	for offset := uint64(12); offset+8 <= uint64(len(data)); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		start := int(offset)
		size := uint64(binary.LittleEndian.Uint32(data[start+4 : start+8]))
		chunkStart := offset + 8
		chunkEnd := chunkStart + size
		if chunkEnd < chunkStart || chunkEnd > uint64(len(data)) {
			return nil, fmt.Errorf("truncated WAV chunk %q", string(data[start:start+4]))
		}
		chunk := data[int(chunkStart):int(chunkEnd)]
		switch string(data[start : start+4]) {
		case "fmt ":
			if seenFormat {
				return nil, errors.New("WAV file contains duplicate fmt chunks")
			}
			seenFormat = true
			if len(chunk) < 16 {
				return nil, errors.New("WAV fmt chunk is too short")
			}
			format = binary.LittleEndian.Uint16(chunk[0:2])
			channels = binary.LittleEndian.Uint16(chunk[2:4])
			sampleRate = binary.LittleEndian.Uint32(chunk[4:8])
			blockAlign = binary.LittleEndian.Uint16(chunk[12:14])
			bits = binary.LittleEndian.Uint16(chunk[14:16])
			if format == 0xfffe {
				if len(chunk) < 40 {
					return nil, errors.New("WAV extensible fmt chunk is too short")
				}
				cbSize := binary.LittleEndian.Uint16(chunk[16:18])
				if cbSize != 22 || len(chunk) != int(cbSize)+18 {
					return nil, errors.New("invalid WAV extensible fmt size")
				}
				validBits := binary.LittleEndian.Uint16(chunk[18:20])
				if validBits != bits {
					return nil, fmt.Errorf("unsupported WAV valid bits %d for %d-bit samples", validBits, bits)
				}
				pcmGUID := [16]byte{1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71}
				floatGUID := pcmGUID
				floatGUID[0] = 3
				var subformat [16]byte
				copy(subformat[:], chunk[24:40])
				switch subformat {
				case pcmGUID:
					format = 1
				case floatGUID:
					format = 3
				default:
					return nil, errors.New("unsupported WAV extensible subformat")
				}
			}
		case "data":
			if pcm == nil {
				pcm = chunk
			}
		}
		offset = chunkEnd + size%2
	}

	if format == 0 || pcm == nil {
		return nil, errors.New("WAV file is missing fmt or data chunk")
	}
	if channels == 0 || channels > maxGemma4AudioChannels {
		return nil, fmt.Errorf("unsupported WAV channel count %d", channels)
	}
	if sampleRate < minGemma4AudioSampleRate || sampleRate > maxGemma4AudioSampleRate {
		return nil, fmt.Errorf("unsupported WAV sample rate %d", sampleRate)
	}
	validEncoding := format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32) || format == 3 && bits == 32
	if !validEncoding {
		return nil, fmt.Errorf("unsupported WAV encoding format=%d bits=%d", format, bits)
	}
	bytesPerSample := int(bits / 8)
	wantBlockAlign := int(channels) * bytesPerSample
	if int(blockAlign) != wantBlockAlign || len(pcm)%wantBlockAlign != 0 {
		return nil, errors.New("invalid WAV block alignment")
	}

	frames := len(pcm) / wantBlockAlign
	maxSourceFrames := int64(maxGemma4AudioSamples) * int64(sampleRate) / int64(targetRate)
	if int64(frames) > maxSourceFrames {
		return nil, errors.New("Gemma4 audio exceeds the maximum duration of 30 seconds")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	samples := make([]float32, frames)
	for i := range frames {
		if i&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		var sum float64
		for channel := range int(channels) {
			offset := (i*int(channels) + channel) * bytesPerSample
			switch {
			case format == 1 && bits == 8:
				sum += (float64(pcm[offset]) - 128) / 128
			case format == 1 && bits == 16:
				sum += float64(int16(binary.LittleEndian.Uint16(pcm[offset:offset+2]))) / 32768
			case format == 1 && bits == 24:
				value := int32(pcm[offset]) | int32(pcm[offset+1])<<8 | int32(pcm[offset+2])<<16
				if value&0x800000 != 0 {
					value |= ^int32(0xffffff)
				}
				sum += float64(value) / 8388608
			case format == 1 && bits == 32:
				sum += float64(int32(binary.LittleEndian.Uint32(pcm[offset:offset+4]))) / 2147483648
			case format == 3 && bits == 32:
				value := math.Float32frombits(binary.LittleEndian.Uint32(pcm[offset : offset+4]))
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return nil, errors.New("WAV contains a non-finite float sample")
				}
				sum += float64(value)
			}
		}
		samples[i] = float32(sum / float64(channels))
	}
	if int(sampleRate) != targetRate {
		resampled, err := resampleGemma4Audio(ctx, samples, int(sampleRate), targetRate)
		if err != nil {
			return nil, err
		}
		samples = resampled
	}
	if len(samples) > maxGemma4AudioSamples {
		return nil, errors.New("Gemma4 audio exceeds the maximum duration of 30 seconds")
	}
	return samples, nil
}

func resampleGemma4Audio(ctx context.Context, samples []float32, sourceRate, targetRate int) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if sourceRate < minGemma4AudioSampleRate || sourceRate > maxGemma4AudioSampleRate || targetRate != 16000 {
		return nil, fmt.Errorf("unsupported Gemma4 audio resampling rate %d to %d", sourceRate, targetRate)
	}
	if sourceRate == targetRate {
		if len(samples) > maxGemma4AudioSamples {
			return samples[:maxGemma4AudioSamples], nil
		}
		return samples, nil
	}
	if len(samples) == 0 {
		return nil, nil
	}
	maxSourceFrames := int64(maxGemma4AudioSamples) * int64(sourceRate) / int64(targetRate)
	if int64(len(samples)) > maxSourceFrames {
		samples = samples[:maxSourceFrames]
	}
	length64 := int64(len(samples)) * int64(targetRate) / int64(sourceRate)
	if length64 > maxGemma4AudioSamples {
		length64 = maxGemma4AudioSamples
	}
	if length64 <= 0 {
		return nil, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([]float32, int(length64))
	for i := range out {
		if i&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		position := float64(i) * float64(sourceRate) / float64(targetRate)
		left := int(position)
		if left >= len(samples)-1 {
			out[i] = samples[len(samples)-1]
			continue
		}
		fraction := float32(position - float64(left))
		out[i] = samples[left]*(1-fraction) + samples[left+1]*fraction
	}
	return out, nil
}

func computeGemma4LogMel(ctx context.Context, samples []float32, cfg *AudioProcessorConfig) ([]float32, []bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if err := validateReleasedAudioProcessorConfig(cfg); err != nil {
		return nil, nil, err
	}
	if len(samples) > maxGemma4AudioSamples {
		return nil, nil, fmt.Errorf("Gemma4 audio has %d samples, limit %d", len(samples), maxGemma4AudioSamples)
	}
	f := cfg.FeatureExtractor
	validSamples := len(samples)
	paddedLength := validSamples
	if remainder := paddedLength % 128; remainder != 0 {
		var ok bool
		paddedLength, ok = checkedGemma4AudioAdd(paddedLength, 128-remainder)
		if !ok {
			return nil, nil, errors.New("Gemma4 audio padded sample count overflows")
		}
	}
	leftPadding := f.FrameLength / 2
	unfoldSize, ok := checkedGemma4AudioAdd(f.FrameLength, 1)
	if !ok {
		return nil, nil, errors.New("Gemma4 audio frame size overflows")
	}
	paddedWithLeft, ok := checkedGemma4AudioAdd(paddedLength, leftPadding)
	if !ok {
		return nil, nil, errors.New("Gemma4 audio padded frame count overflows")
	}
	if paddedWithLeft < unfoldSize {
		return nil, nil, errors.New("Gemma4 audio is too short to frame")
	}
	frames := (paddedWithLeft-unfoldSize)/f.HopLength + 1
	if frames <= 0 || frames > maxGemma4AudioFrames {
		return nil, nil, fmt.Errorf("Gemma4 audio frame count %d exceeds limit %d", frames, maxGemma4AudioFrames)
	}
	featureValues, ok := checkedGemma4AudioMul(frames, f.FeatureSize)
	if !ok || featureValues > maxGemma4AudioFeatures {
		return nil, nil, errors.New("Gemma4 audio feature dimensions exceed limit")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	features := make([]float32, featureValues)
	mask := make([]bool, frames)
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	window := make([]float32, f.FrameLength)
	for i := range window {
		window[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(f.FrameLength)))
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	melFilters := gemma4MelFilterBank(f.FFTLength/2+1, f.FeatureSize, f.MinFrequency, f.MaxFrequency, f.SamplingRate)
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	fft := fourier.NewFFT(f.FFTLength)
	sequence := make([]float64, f.FFTLength)
	coefficients := make([]complex128, f.FFTLength/2+1)
	magnitudes := make([]float64, len(coefficients))

	for frame := range frames {
		if frame&31 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, nil, err
			}
		}
		clear(sequence)
		start := frame*f.HopLength - leftPadding
		for i := range f.FrameLength {
			index := start + i
			if index >= 0 && index < validSamples {
				sequence[i] = float64(samples[index] * window[i])
			}
		}
		coefficients = fft.Coefficients(coefficients, sequence)
		for i, value := range coefficients {
			magnitudes[i] = cmplx.Abs(value)
		}
		valid := frame*f.HopLength+f.FrameLength < leftPadding+validSamples
		mask[frame] = valid
		if !valid {
			continue
		}
		for mel := range f.FeatureSize {
			value := 0.0
			for bin, magnitude := range magnitudes {
				value += magnitude * melFilters[bin*f.FeatureSize+mel]
			}
			features[frame*f.FeatureSize+mel] = float32(math.Log(value + f.MelFloor))
		}
	}
	return features, mask, nil
}

func gemma4MelFilterBank(frequencyBins, melBins int, minFrequency, maxFrequency float64, sampleRate int) []float64 {
	hzToMel := func(value float64) float64 { return 2595 * math.Log10(1+value/700) }
	melToHz := func(value float64) float64 { return 700 * (math.Pow(10, value/2595) - 1) }
	melMin, melMax := hzToMel(minFrequency), hzToMel(maxFrequency)
	centers := make([]float64, melBins+2)
	for i := range centers {
		centers[i] = melToHz(melMin + float64(i)*(melMax-melMin)/float64(melBins+1))
	}
	filters := make([]float64, frequencyBins*melBins)
	for bin := range frequencyBins {
		frequency := float64(bin) * float64(sampleRate/2) / float64(frequencyBins-1)
		for mel := range melBins {
			down := (frequency - centers[mel]) / (centers[mel+1] - centers[mel])
			up := (centers[mel+2] - frequency) / (centers[mel+2] - centers[mel+1])
			filters[bin*melBins+mel] = math.Max(0, math.Min(down, up))
		}
	}
	return filters
}

func checkedGemma4AudioAdd(a, b int) (int, bool) {
	if a < 0 || b < 0 || a > int(^uint(0)>>1)-b {
		return 0, false
	}
	return a + b, true
}

func checkedGemma4AudioMul(a, b int) (int, bool) {
	if a < 0 || b < 0 || a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}

func downsampleGemma4AudioMask(mask []bool, layers int) []bool {
	if len(mask) == 0 {
		return nil
	}
	for range layers {
		outputLength := (len(mask)-1)/2 + 1
		output := make([]bool, outputLength)
		for i := range output {
			output[i] = mask[i*2]
		}
		mask = output
	}
	return mask
}
