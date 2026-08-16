package gemma4

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

func TestParseReleasedAudioProcessorConfig(t *testing.T) {
	data := []byte(`{
			"audio_seq_length":750,
			"feature_extractor":{
				"feature_extractor_type":"Gemma4AudioFeatureExtractor",
			"dither":0.0,"feature_size":128,"fft_length":512,"fft_overdrive":false,
			"frame_length":320,"hop_length":160,"input_scale_factor":1.0,
			"max_frequency":8000.0,"mel_floor":0.001,"min_frequency":0.0,
			"padding_side":"right","per_bin_mean":null,"per_bin_stddev":null,
			"preemphasis":0.0,"sampling_rate":16000
		}
	}`)
	cfg, err := parseAudioProcessorConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.FeatureExtractor.FFTLength != 512 || cfg.AudioSequenceLength != 750 {
		t.Fatalf("processor config = %#v", cfg)
	}

	bad := bytes.Replace(data, []byte(`"fft_length":512`), []byte(`"fft_length":1024`), 1)
	if _, err := parseAudioProcessorConfig(bad); err == nil {
		t.Fatal("1024-point FFT processor: error = nil")
	}
	unknown := bytes.Replace(data, []byte(`"Gemma4AudioFeatureExtractor"`), []byte(`"FutureAudioFeatureExtractor"`), 1)
	if _, err := parseAudioProcessorConfig(unknown); err == nil {
		t.Fatal("unknown tower processor: error = nil")
	}
}

func TestReleasedAudioProcessorRejectsMalformedDimensions(t *testing.T) {
	base := defaultAudioProcessorConfig()
	base.FeatureExtractor.Type = "Gemma4AudioFeatureExtractor"
	tests := []struct {
		name   string
		mutate func(*AudioProcessorConfig)
	}{
		{"sample rate", func(cfg *AudioProcessorConfig) { cfg.FeatureExtractor.SamplingRate = 0 }},
		{"frame length", func(cfg *AudioProcessorConfig) { cfg.FeatureExtractor.FrameLength = -1 }},
		{"hop length", func(cfg *AudioProcessorConfig) { cfg.FeatureExtractor.HopLength = 0 }},
		{"feature size", func(cfg *AudioProcessorConfig) { cfg.FeatureExtractor.FeatureSize = int(^uint(0) >> 1) }},
		{"FFT length", func(cfg *AudioProcessorConfig) { cfg.FeatureExtractor.FFTLength = 0 }},
		{"sequence length", func(cfg *AudioProcessorConfig) { cfg.AudioSequenceLength = int(^uint(0) >> 1) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := base
			tt.mutate(&cfg)
			if _, _, err := computeGemma4LogMel(context.Background(), make([]float32, 400), &cfg); err == nil || !strings.Contains(err.Error(), "configuration") {
				t.Fatalf("malformed configuration error = %v", err)
			}
		})
	}

	if _, _, err := computeGemma4LogMel(context.Background(), make([]float32, maxGemma4AudioSamples+1), &base); err == nil || !strings.Contains(err.Error(), "samples") {
		t.Fatalf("oversized sample input error = %v", err)
	}
	maxInt := int(^uint(0) >> 1)
	if _, ok := checkedGemma4AudioAdd(maxInt, 1); ok {
		t.Fatal("checked add accepted overflow")
	}
	if _, ok := checkedGemma4AudioMul(maxInt, 2); ok {
		t.Fatal("checked multiply accepted overflow")
	}
}

func TestParseReleasedUnifiedAudioProcessorConfig(t *testing.T) {
	data := []byte(`{
		"audio_seq_length":750,
		"feature_extractor":{
			"audio_samples_per_token":640,
			"feature_extractor_type":"Gemma4UnifiedAudioFeatureExtractor",
			"feature_size":640,"padding_side":"right","padding_value":0,
			"return_attention_mask":true,"sampling_rate":16000
		}
	}`)
	cfg, err := parseAudioProcessorConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.FeatureExtractor.AudioSamplesPerToken != 640 || cfg.FeatureExtractor.FeatureSize != 640 {
		t.Fatalf("processor config = %#v", cfg)
	}

	bad := bytes.Replace(data, []byte(`"audio_samples_per_token":640`), []byte(`"audio_samples_per_token":320`), 1)
	if _, err := parseAudioProcessorConfig(bad); err == nil {
		t.Fatal("invalid unified processor: error = nil")
	}
}

func TestPreprocessUnifiedAudioFrames(t *testing.T) {
	data := []byte(`{
		"audio_seq_length":750,
		"feature_extractor":{
			"audio_samples_per_token":640,
			"feature_extractor_type":"Gemma4UnifiedAudioFeatureExtractor",
			"feature_size":640,"padding_side":"right","sampling_rate":16000
		}
	}`)
	cfg, err := parseAudioProcessorConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	frames := make([][]float64, 641)
	for i := range frames {
		frames[i] = []float64{0.25}
	}
	input, err := preprocessGemma4Audio(context.Background(), makeTestWAV(t, 1, 16, 16000, frames), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if input.SoftTokens != 2 || input.Frames != 2 || input.FeatureSize != 640 || len(input.Features) != 1280 {
		t.Fatalf("unified input = %+v, feature count %d", input, len(input.Features))
	}
	if input.Features[640] != 0.25 || input.Features[641] != 0 {
		t.Fatalf("second frame values = %v, %v; want 0.25, 0", input.Features[640], input.Features[641])
	}
}

func TestGemma4LogMelReference(t *testing.T) {
	cfg := defaultAudioProcessorConfig()
	samples := make([]float32, 4000)
	for i := range samples {
		samples[i] = float32(0.25 * math.Sin(2*math.Pi*440*float64(i)/16000))
	}
	features, mask, err := computeGemma4LogMel(context.Background(), samples, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(mask) != 25 || len(features) != 25*128 {
		t.Fatalf("feature shape = (%d, %d), want (25, 128)", len(mask), len(features)/len(mask))
	}
	valid := 0
	for _, value := range mask {
		if value {
			valid++
		}
	}
	if valid != 24 {
		t.Fatalf("valid frames = %d, want 24", valid)
	}
	selected := map[int]float64{
		0*128 + 0:   -6.907755374908447,
		0*128 + 1:   0.11585413664579391,
		0*128 + 10:  -0.6425663232803345,
		0*128 + 64:  -1.9309545755386353,
		0*128 + 127: -2.837610960006714,
		1*128 + 10:  -4.558506965637207,
		12*128 + 10: -4.529771327972412,
		23*128 + 10: -4.529770374298096,
		24*128 + 10: 0,
	}
	for index, want := range selected {
		if got := float64(features[index]); math.Abs(got-want) > 1e-5 {
			t.Errorf("features[%d] = %.9f, want %.9f", index, got, want)
		}
	}
	sum := 0.0
	for _, value := range features {
		sum += float64(value)
	}
	if math.Abs(sum-(-15998.611463362351)) > 0.05 {
		t.Errorf("feature sum = %.9f, want -15998.611463362351", sum)
	}
	if got := len(downsampleGemma4AudioMask(mask, 2)); got != 7 {
		t.Fatalf("downsampled mask length = %d, want 7", got)
	}
	softTokens := 0
	for _, value := range downsampleGemma4AudioMask(mask, 2) {
		if value {
			softTokens++
		}
	}
	if softTokens != 6 {
		t.Fatalf("soft tokens = %d, want 6", softTokens)
	}
}

func TestDecodeGemma4WAVEncodings(t *testing.T) {
	for _, tt := range []struct {
		name   string
		format uint16
		bits   uint16
		tol    float64
	}{
		{"pcm8", 1, 8, 1.0 / 128},
		{"pcm16", 1, 16, 1.0 / 32768},
		{"pcm24", 1, 24, 1.0 / 8388608},
		{"pcm32", 1, 32, 1.0 / 2147483648},
		{"float32", 3, 32, 1e-6},
	} {
		t.Run(tt.name, func(t *testing.T) {
			data := makeTestWAV(t, tt.format, tt.bits, 16000, [][]float64{{-0.5}, {0}, {0.5}})
			samples, err := decodeGemma4WAV(context.Background(), data, 16000)
			if err != nil {
				t.Fatal(err)
			}
			for i, want := range []float64{-0.5, 0, 0.5} {
				if math.Abs(float64(samples[i])-want) > tt.tol {
					t.Errorf("sample %d = %v, want %v", i, samples[i], want)
				}
			}
		})
	}
}

func TestDecodeGemma4WAVExtensible(t *testing.T) {
	pcmGUID := [16]byte{1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71}
	floatGUID := pcmGUID
	floatGUID[0] = 3
	for _, tt := range []struct {
		name   string
		format uint16
		bits   uint16
		guid   [16]byte
	}{
		{"pcm", 1, 16, pcmGUID},
		{"float", 3, 32, floatGUID},
	} {
		t.Run(tt.name, func(t *testing.T) {
			data := makeTestExtensibleWAV(t, tt.format, tt.bits, tt.guid)
			samples, err := decodeGemma4WAV(context.Background(), data, 16000)
			if err != nil {
				t.Fatal(err)
			}
			if len(samples) != 3 || math.Abs(float64(samples[0])+0.5) > 1e-5 || math.Abs(float64(samples[2])-0.5) > 1e-5 {
				t.Fatalf("extensible samples = %v", samples)
			}
		})
	}

	valid := makeTestExtensibleWAV(t, 1, 16, pcmGUID)
	tests := []struct {
		name   string
		mutate func([]byte) []byte
		want   string
	}{
		{"small extension", func(data []byte) []byte {
			binary.LittleEndian.PutUint16(data[36:38], 21)
			return data
		}, "fmt size"},
		{"large extension", func(data []byte) []byte {
			binary.LittleEndian.PutUint16(data[36:38], 23)
			return data
		}, "fmt size"},
		{"valid bits", func(data []byte) []byte {
			binary.LittleEndian.PutUint16(data[38:40], 12)
			return data
		}, "valid bits"},
		{"GUID", func(data []byte) []byte {
			data[59] ^= 1
			return data
		}, "subformat"},
		{"duplicate fmt", func(data []byte) []byte {
			duplicate := make([]byte, 0, len(data)+48)
			duplicate = append(duplicate, data[:60]...)
			duplicate = append(duplicate, data[12:60]...)
			duplicate = append(duplicate, data[60:]...)
			binary.LittleEndian.PutUint32(duplicate[4:8], uint32(len(duplicate)-8))
			return duplicate
		}, "duplicate fmt"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := tt.mutate(append([]byte(nil), valid...))
			if samples, err := decodeGemma4WAV(context.Background(), data, 16000); err == nil || !strings.Contains(err.Error(), tt.want) || samples != nil {
				t.Fatalf("malformed extensible WAV = (%v, %v), want nil and %q error", samples, err, tt.want)
			}
		})
	}
}

func TestDecodeGemma4WAVDownmixResampleAndTruncate(t *testing.T) {
	frames := make([][]float64, 8000)
	for i := range frames {
		frames[i] = []float64{-0.5, 0.5}
	}
	data := makeTestWAV(t, 1, 16, 8000, frames)
	samples, err := decodeGemma4WAV(context.Background(), data, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 16000 {
		t.Fatalf("resampled length = %d, want 16000", len(samples))
	}
	for _, index := range []int{0, 7999, 15999} {
		if math.Abs(float64(samples[index])) > 1e-6 {
			t.Errorf("downmixed sample %d = %v, want 0", index, samples[index])
		}
	}

	limitFrames := make([][]float64, maxGemma4AudioSamples)
	for i := range limitFrames {
		limitFrames[i] = []float64{0}
	}
	limitData := makeTestWAV(t, 1, 16, 16000, limitFrames)
	limited, err := decodeGemma4WAV(context.Background(), limitData, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if len(limited) != maxGemma4AudioSamples {
		t.Fatalf("exact-boundary length = %d, want %d", len(limited), maxGemma4AudioSamples)
	}
	overFrames := make([][]float64, len(limitFrames)+1)
	copy(overFrames, limitFrames)
	overFrames[len(limitFrames)] = []float64{0}
	overData := makeTestWAV(t, 1, 16, 16000, overFrames)
	truncated, err := decodeGemma4WAV(context.Background(), overData, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if len(truncated) != maxGemma4AudioSamples {
		t.Fatalf("boundary+1 truncated length = %d, want %d", len(truncated), maxGemma4AudioSamples)
	}
}

func TestPreprocessGemma4AudioSequentialIsolation(t *testing.T) {
	cfg := defaultAudioProcessorConfig()
	firstFrames := make([][]float64, 4000)
	secondFrames := make([][]float64, 8000)
	for i := range firstFrames {
		firstFrames[i] = []float64{0.25 * math.Sin(2*math.Pi*440*float64(i)/16000)}
	}
	for i := range secondFrames {
		secondFrames[i] = []float64{0.25 * math.Sin(2*math.Pi*880*float64(i)/16000)}
	}
	first, err := preprocessGemma4Audio(context.Background(), makeTestWAV(t, 1, 16, 16000, firstFrames), &cfg)
	if err != nil {
		t.Fatal(err)
	}
	firstSnapshot := append([]float32(nil), first.Features...)
	second, err := preprocessGemma4Audio(context.Background(), makeTestWAV(t, 1, 16, 16000, secondFrames), &cfg)
	if err != nil {
		t.Fatal(err)
	}
	if first.Frames != 25 || second.Frames != 50 {
		t.Fatalf("sequential frame counts = (%d, %d), want (25, 50)", first.Frames, second.Frames)
	}
	if len(first.Features) != len(firstSnapshot) || !equalFloat32s(first.Features, firstSnapshot) {
		t.Fatal("second preprocess mutated or appended to first output")
	}
	if len(first.Features) > 0 && len(second.Features) > 0 && &first.Features[0] == &second.Features[0] {
		t.Fatal("sequential preprocess calls share feature storage")
	}
}

func equalFloat32s(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestGemma4AudioInputFailures(t *testing.T) {
	cfg := defaultAudioProcessorConfig()
	for _, tt := range []struct {
		name string
		data []byte
		want string
	}{
		{"mp3", []byte("ID3not-wav"), "RIFF/WAVE"},
		{"truncated chunk", append([]byte("RIFF\x00\x00\x00\x00WAVEdata\xff\xff\xff\x7f"), 0), "truncated"},
		{"too short", makeTestWAV(t, 1, 16, 16000, make([][]float64, 100)), "too short"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := preprocessGemma4Audio(context.Background(), tt.data, &cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want containing %q", err, tt.want)
			}
		})
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := decodeGemma4WAV(ctx, []byte("anything"), 16000); err != context.Canceled {
		t.Fatalf("canceled decode error = %v, want %v", err, context.Canceled)
	}
}

func TestGemma4WAVValidationLimits(t *testing.T) {
	if _, err := decodeGemma4WAV(context.Background(), make([]byte, maxGemma4AudioBytes+1), 16000); err == nil || !strings.Contains(err.Error(), "limit") {
		t.Fatalf("oversized WAV error = %v", err)
	}
	tooManyChannels := makeTestWAV(t, 1, 16, 16000, [][]float64{make([]float64, maxGemma4AudioChannels+1)})
	if _, err := decodeGemma4WAV(context.Background(), tooManyChannels, 16000); err == nil || !strings.Contains(err.Error(), "channel count") {
		t.Fatalf("channel-limit error = %v", err)
	}
	badRate := makeTestWAV(t, 1, 16, minGemma4AudioSampleRate-1, [][]float64{{0}})
	if _, err := decodeGemma4WAV(context.Background(), badRate, 16000); err == nil || !strings.Contains(err.Error(), "sample rate") {
		t.Fatalf("sample-rate error = %v", err)
	}
	badAlignment := makeTestWAV(t, 1, 16, 16000, [][]float64{{0}})
	binary.LittleEndian.PutUint16(badAlignment[32:34], 1)
	if _, err := decodeGemma4WAV(context.Background(), badAlignment, 16000); err == nil || !strings.Contains(err.Error(), "block alignment") {
		t.Fatalf("block-alignment error = %v", err)
	}
	nonFinite := makeTestWAV(t, 3, 32, 16000, [][]float64{{math.NaN()}})
	if _, err := decodeGemma4WAV(context.Background(), nonFinite, 16000); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("non-finite error = %v", err)
	}
}

func TestGemma4AudioCancellationBeforeAllocations(t *testing.T) {
	cfg := defaultAudioProcessorConfig()
	frames := make([][]float64, 400)
	for i := range frames {
		frames[i] = []float64{0.25}
	}
	data := makeTestWAV(t, 1, 16, 16000, frames)

	decodeCtx := &cancelAfterErrChecks{remaining: 4}
	if samples, err := decodeGemma4WAV(decodeCtx, data, 16000); err != context.Canceled || samples != nil {
		t.Fatalf("decode cancellation = (%v, %v), want (nil, context.Canceled)", samples, err)
	}

	resampleCtx := &cancelAfterErrChecks{remaining: 2}
	if samples, err := resampleGemma4Audio(resampleCtx, make([]float32, 400), 8000, 16000); err != context.Canceled || samples != nil {
		t.Fatalf("resample cancellation = (%v, %v), want (nil, context.Canceled)", samples, err)
	}

	for _, checks := range []int{2, 3, 4, 5} {
		t.Run(fmt.Sprintf("log-mel check %d", checks), func(t *testing.T) {
			ctx := &cancelAfterErrChecks{remaining: checks}
			features, mask, err := computeGemma4LogMel(ctx, make([]float32, 400), &cfg)
			if err != context.Canceled || features != nil || mask != nil {
				t.Fatalf("log-mel cancellation = (%v, %v, %v), want nil outputs and context.Canceled", features, mask, err)
			}
		})
	}

	preprocessCtx := &cancelAfterErrChecks{remaining: 6}
	if input, err := preprocessGemma4Audio(preprocessCtx, data, &cfg); err != context.Canceled || input != nil {
		t.Fatalf("preprocess cancellation = (%v, %v), want (nil, context.Canceled)", input, err)
	}
}

type cancelAfterErrChecks struct {
	remaining int
}

func (*cancelAfterErrChecks) Deadline() (time.Time, bool) { return time.Time{}, false }

func (*cancelAfterErrChecks) Done() <-chan struct{} { return nil }

func (c *cancelAfterErrChecks) Err() error {
	c.remaining--
	if c.remaining <= 0 {
		return context.Canceled
	}
	return nil
}

func (*cancelAfterErrChecks) Value(any) any { return nil }

func makeTestWAV(t *testing.T, format, bits uint16, sampleRate uint32, frames [][]float64) []byte {
	t.Helper()
	channels := 1
	if len(frames) > 0 && len(frames[0]) > 0 {
		channels = len(frames[0])
	}
	bytesPerSample := int(bits / 8)
	var pcm bytes.Buffer
	for _, frame := range frames {
		if len(frame) == 0 {
			frame = make([]float64, channels)
		}
		if len(frame) != channels {
			t.Fatalf("inconsistent channel count")
		}
		for _, value := range frame {
			switch {
			case format == 1 && bits == 8:
				pcm.WriteByte(byte(math.Round(value*128 + 128)))
			case format == 1 && bits == 16:
				_ = binary.Write(&pcm, binary.LittleEndian, int16(math.Round(value*32768)))
			case format == 1 && bits == 24:
				v := int32(math.Round(value * 8388608))
				pcm.Write([]byte{byte(v), byte(v >> 8), byte(v >> 16)})
			case format == 1 && bits == 32:
				_ = binary.Write(&pcm, binary.LittleEndian, int32(math.Round(value*2147483648)))
			case format == 3 && bits == 32:
				_ = binary.Write(&pcm, binary.LittleEndian, float32(value))
			default:
				t.Fatalf("unsupported test WAV encoding")
			}
		}
	}
	blockAlign := uint16(channels * bytesPerSample)
	byteRate := sampleRate * uint32(blockAlign)
	dataSize := pcm.Len()
	var out bytes.Buffer
	out.WriteString("RIFF")
	_ = binary.Write(&out, binary.LittleEndian, uint32(36+dataSize))
	out.WriteString("WAVEfmt ")
	_ = binary.Write(&out, binary.LittleEndian, uint32(16))
	_ = binary.Write(&out, binary.LittleEndian, format)
	_ = binary.Write(&out, binary.LittleEndian, uint16(channels))
	_ = binary.Write(&out, binary.LittleEndian, sampleRate)
	_ = binary.Write(&out, binary.LittleEndian, byteRate)
	_ = binary.Write(&out, binary.LittleEndian, blockAlign)
	_ = binary.Write(&out, binary.LittleEndian, bits)
	out.WriteString("data")
	_ = binary.Write(&out, binary.LittleEndian, uint32(dataSize))
	out.Write(pcm.Bytes())
	return out.Bytes()
}

func makeTestExtensibleWAV(t *testing.T, format, bits uint16, subformat [16]byte) []byte {
	t.Helper()
	base := makeTestWAV(t, format, bits, 16000, [][]float64{{-0.5}, {0}, {0.5}})
	pcm := base[44:]
	var out bytes.Buffer
	out.WriteString("RIFF")
	_ = binary.Write(&out, binary.LittleEndian, uint32(60+len(pcm)))
	out.WriteString("WAVEfmt ")
	_ = binary.Write(&out, binary.LittleEndian, uint32(40))
	_ = binary.Write(&out, binary.LittleEndian, uint16(0xfffe))
	_ = binary.Write(&out, binary.LittleEndian, uint16(1))
	_ = binary.Write(&out, binary.LittleEndian, uint32(16000))
	_ = binary.Write(&out, binary.LittleEndian, uint32(16000*int(bits/8)))
	_ = binary.Write(&out, binary.LittleEndian, bits/8)
	_ = binary.Write(&out, binary.LittleEndian, bits)
	_ = binary.Write(&out, binary.LittleEndian, uint16(22))
	_ = binary.Write(&out, binary.LittleEndian, bits)
	_ = binary.Write(&out, binary.LittleEndian, uint32(0))
	out.Write(subformat[:])
	out.WriteString("data")
	_ = binary.Write(&out, binary.LittleEndian, uint32(len(pcm)))
	out.Write(pcm)
	return out.Bytes()
}
