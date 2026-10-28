package audio

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestReferenceWAVAdmission(t *testing.T) {
	for _, rate := range []int{16000, 24000, 44100, 48000} {
		wav := EncodeWAV(make([]float32, rate*15), rate)
		if err := ValidateReferenceWAV(wav); err != nil {
			t.Fatalf("15 seconds at %d Hz: %v", rate, err)
		}
		binary.LittleEndian.PutUint32(wav[4:], math.MaxUint32)
		binary.LittleEndian.PutUint32(wav[40:], math.MaxUint32)
		samples, actualRate, err := DecodeReferenceWAV(wav)
		if err != nil || actualRate != rate || len(samples) != 15*rate {
			t.Fatalf("streamed WAV at %d Hz: %d samples, %d Hz, %v", rate, len(samples), actualRate, err)
		}
	}
	for _, tc := range []struct {
		name   string
		change func([]byte) []byte
	}{
		{"truncated", func(b []byte) []byte { return b[:len(b)-1] }},
		{"empty", func([]byte) []byte { return EncodeWAV(nil, 24000) }},
		{"long", func([]byte) []byte { return EncodeWAV(make([]float32, 24000*15+1), 24000) }},
		{"low rate", func([]byte) []byte { return EncodeWAV(make([]float32, 8000), 8000) }},
		{"high rate", func([]byte) []byte { return EncodeWAV(make([]float32, 96000), 96000) }},
		{"stereo", func(b []byte) []byte { binary.LittleEndian.PutUint16(b[22:], 2); return b }},
		{"float", func(b []byte) []byte { binary.LittleEndian.PutUint16(b[20:], 3); return b }},
		{"24 bit", func(b []byte) []byte { binary.LittleEndian.PutUint16(b[34:], 24); return b }},
		{"wrong alignment", func(b []byte) []byte { binary.LittleEndian.PutUint16(b[32:], 4); return b }},
		{"wrong byte rate", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[28:], 24000); return b }},
		{"odd PCM", func(b []byte) []byte {
			b = b[:len(b)-1]
			binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
			binary.LittleEndian.PutUint32(b[40:], uint32(len(b)-44))
			return b
		}},
		{"duplicate data", func(b []byte) []byte {
			b = append(b, b[36:]...)
			binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
			return b
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateReferenceWAV(tc.change(EncodeWAV(make([]float32, 24000), 24000))); err == nil {
				t.Fatal("invalid reference was admitted")
			}
		})
	}
}
