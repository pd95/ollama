package audio

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// ErrInvalidData identifies unsupported or malformed encoded audio and samples,
// separately from caller cancellation and file-system I/O failures.
var ErrInvalidData = errors.New("invalid audio data")

func invalidStreamData(message string) error { return fmt.Errorf("%w: %s", ErrInvalidData, message) }

// SampleStream decodes mono samples incrementally at the source rate. The
// caller owns Close, including on cancellation. Read returns at most 4096 frames.
type SampleStream interface {
	Read([]float32) (int, error)
	SampleRate() int
	Close() error
}

// OpenSampleStream opens a file-backed WAV or MP3. It does not retain the
// compressed file or allocate PCM proportional to the recording duration.
func OpenSampleStream(ctx context.Context, path string) (SampleStream, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var header [12]byte
	n, err := f.ReadAt(header[:], 0)
	if err != nil && !errors.Is(err, io.EOF) {
		_ = f.Close()
		return nil, err
	}
	if n >= 3 && isMP3(header[:n]) {
		_ = f.Close()
		return openMP3Stream(ctx, path)
	}
	if err != nil || string(header[:4]) != "RIFF" || string(header[8:]) != "WAVE" {
		_ = f.Close()
		return nil, ErrUnrecognized
	}
	s, err := openWAVStream(ctx, f)
	if err != nil {
		_ = f.Close()
	}
	return s, err
}

type wavStream struct {
	ctx    context.Context //nolint:containedctx // Decoder lifetime belongs to one request.
	file   *os.File
	pcm    *io.SectionReader
	header []byte
	bytes  []byte
	rate   int
	stride int
}

func (s *wavStream) SampleRate() int { return s.rate }
func (s *wavStream) Close() error    { return s.file.Close() }
func openWAVStream(ctx context.Context, f *os.File) (SampleStream, error) {
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	var format []byte
	var dataOffset, dataSize int64
	for offset := int64(12); offset+8 <= stat.Size(); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var chunk [8]byte
		if _, err := f.ReadAt(chunk[:], offset); err != nil {
			return nil, err
		}
		size := int64(binary.LittleEndian.Uint32(chunk[4:]))
		if size > stat.Size()-offset-8 {
			return nil, invalidStreamData("wav: truncated chunk")
		}
		switch string(chunk[:4]) {
		case "fmt ":
			if size < 16 {
				return nil, invalidStreamData("wav: fmt chunk too short")
			}
			format = make([]byte, min(size, 40))
			if _, err := f.ReadAt(format, offset+8); err != nil {
				return nil, err
			}
		case "data":
			dataOffset, dataSize = offset+8, size
		}
		offset += 8 + size + size%2
	}
	if len(format) < 16 || dataOffset == 0 {
		return nil, invalidStreamData("wav: missing fmt or data chunk")
	}
	code := binary.LittleEndian.Uint16(format)
	if code == 0xfffe && len(format) >= 26 {
		code = binary.LittleEndian.Uint16(format[24:])
	}
	channels := int(binary.LittleEndian.Uint16(format[2:]))
	rate := int(binary.LittleEndian.Uint32(format[4:]))
	bits := int(binary.LittleEndian.Uint16(format[14:]))
	if channels < 1 || channels > 32 || rate < 1 || rate > maxSampleRate {
		return nil, invalidStreamData("wav: unsupported channels or sample rate")
	}
	if !(code == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32) || code == 3 && bits == 32) {
		return nil, invalidStreamData("wav: unsupported sample format")
	}
	stride := channels * (bits / 8)
	if dataSize%int64(stride) != 0 {
		return nil, invalidStreamData("wav: partial PCM frame")
	}
	header := WAVHeader(rate, 0)
	binary.LittleEndian.PutUint16(header[20:], code)
	binary.LittleEndian.PutUint16(header[22:], uint16(channels))
	binary.LittleEndian.PutUint16(header[32:], uint16(stride))
	binary.LittleEndian.PutUint16(header[34:], uint16(bits))
	return &wavStream{ctx: ctx, file: f, pcm: io.NewSectionReader(f, dataOffset, dataSize), header: header, bytes: make([]byte, 44+4096*stride), rate: rate, stride: stride}, nil
}

func (s *wavStream) Read(p []float32) (int, error) {
	if err := s.ctx.Err(); err != nil {
		return 0, err
	}
	if len(p) == 0 {
		return 0, nil
	}
	want := min(len(p), 4096) * s.stride
	n, err := io.ReadFull(s.pcm, s.bytes[44:44+want])
	if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
		return 0, err
	}
	if n == 0 {
		return 0, io.EOF
	}
	if n%s.stride != 0 {
		return 0, invalidStreamData("wav: partial PCM frame")
	}
	copy(s.bytes, s.header)
	binary.LittleEndian.PutUint32(s.bytes[4:], uint32(n+36))
	binary.LittleEndian.PutUint32(s.bytes[40:], uint32(n))
	samples, _, err := decodeWAVContext(s.ctx, s.bytes[:44+n], 4096)
	if err != nil {
		if s.ctx.Err() != nil {
			return 0, s.ctx.Err()
		}
		return 0, fmt.Errorf("%w: %v", ErrInvalidData, err)
	}
	for _, v := range samples {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, invalidStreamData("wav: non-finite sample")
		}
	}
	return copy(p, samples), nil
}

// EncodeFloatWAVContext preserves decoded samples without a second lossy PCM16
// quantization. The existing WAV decoder supports this IEEE float container.
func EncodeFloatWAVContext(ctx context.Context, samples []float32, rate int) ([]byte, error) {
	if len(samples) > int((math.MaxUint32-36)/4) {
		return nil, fmt.Errorf("WAV sample count is too large")
	}
	out := make([]byte, 44+4*len(samples))
	copy(out, WAVHeader(rate, uint32(4*len(samples))))
	binary.LittleEndian.PutUint16(out[20:], 3)
	binary.LittleEndian.PutUint32(out[28:], uint32(rate*4))
	binary.LittleEndian.PutUint16(out[32:], 4)
	binary.LittleEndian.PutUint16(out[34:], 32)
	for i, v := range samples {
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		binary.LittleEndian.PutUint32(out[44+i*4:], math.Float32bits(v))
	}
	return out, nil
}
