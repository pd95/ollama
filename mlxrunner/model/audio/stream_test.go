package audio

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func saveStreamFixture(t *testing.T, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "input")
	if err := os.WriteFile(p, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestWAVSampleStream(t *testing.T) {
	pcm := make([]float32, 16000*65)
	for i := range pcm {
		pcm[i] = float32(i%129-64) / 64
	}
	wav, err := EncodeFloatWAVContext(t.Context(), pcm, 16000)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := OpenSampleStream(t.Context(), saveStreamFixture(t, wav))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	buf := make([]float32, 1024)
	offset := 0
	for {
		n, err := stream.Read(buf)
		for i, v := range buf[:n] {
			if v != pcm[offset+i] {
				t.Fatal("stream sample mismatch")
			}
		}
		offset += n
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if offset != len(pcm) || stream.SampleRate() != 16000 {
		t.Fatal("missing samples")
	}
}

func TestWAVStreamCancellationAndInvalid(t *testing.T) {
	path := saveStreamFixture(t, EncodeWAV(make([]float32, 16000), 16000))
	ctx, cancel := context.WithCancel(t.Context())
	s, err := OpenSampleStream(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	if _, err := s.Read(make([]float32, 4096)); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenSampleStream(ctx, path); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	for _, kind := range []string{"truncated", "partial frame", "channels", "format"} {
		t.Run(kind, func(t *testing.T) {
			wav := EncodeWAV([]float32{0, 0}, 16000)
			switch kind {
			case "truncated":
				wav = wav[:len(wav)-1]
			case "partial frame":
				binary.LittleEndian.PutUint32(wav[40:], 3)
				wav = wav[:len(wav)-1]
			case "channels":
				binary.LittleEndian.PutUint16(wav[22:], 65535)
			case "format":
				binary.LittleEndian.PutUint16(wav[20:], 4)
			}
			if s, err := OpenSampleStream(t.Context(), saveStreamFixture(t, wav)); err == nil {
				_ = s.Close()
				t.Fatal("invalid WAV accepted")
			}
		})
	}
}
