package gemma4

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

func TestParseReleasedAudioProcessorConfig(t *testing.T) {
	data := []byte(`{
		"audio_seq_length":750,
		"feature_extractor":{
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

	longFrames := make([][]float64, 31*16000)
	for i := range longFrames {
		longFrames[i] = []float64{0}
	}
	longData := makeTestWAV(t, 1, 16, 16000, longFrames)
	truncated, err := decodeGemma4WAV(context.Background(), longData, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if len(truncated) != maxGemma4AudioSamples {
		t.Fatalf("truncated length = %d, want %d", len(truncated), maxGemma4AudioSamples)
	}
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
