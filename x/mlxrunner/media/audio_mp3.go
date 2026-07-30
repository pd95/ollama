//go:build cgo

package media

/*
#cgo CFLAGS: -std=c99
#cgo linux LDFLAGS: -lm
#include <stdlib.h>
#include "miniaudio_wrapper.h"
*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// AudioOverflowPolicy controls what happens when decoded audio exceeds MaxSamples.
type AudioOverflowPolicy uint8

const (
	AudioOverflowReject AudioOverflowPolicy = iota
	AudioOverflowTruncate
)

// MP3DecodeOptions bounds and configures MP3 decoding.
type MP3DecodeOptions struct {
	TargetSampleRate int
	MaxInputBytes    int
	MaxSamples       int
	Overflow         AudioOverflowPolicy
}

// DecodeMP3 decodes MP3 bytes to mono float32 PCM at the requested sample rate.
// WAV intentionally remains model-owned until output-parity requirements for a
// shared decoder have been established.
func DecodeMP3(ctx context.Context, data []byte, opts MP3DecodeOptions) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if opts.TargetSampleRate <= 0 || opts.MaxInputBytes <= 0 || opts.MaxSamples <= 0 {
		return nil, errors.New("invalid MP3 decode limits")
	}
	if len(data) == 0 {
		return nil, errors.New("MP3 input is empty")
	}
	if len(data) > opts.MaxInputBytes {
		return nil, fmt.Errorf("MP3 audio is %d bytes, limit %d", len(data), opts.MaxInputBytes)
	}

	encoded := C.CBytes(data)
	defer C.free(encoded)
	var sourceChannels, sourceRate C.uint32_t
	decoder := C.ollama_miniaudio_mp3_init(encoded, C.size_t(len(data)), C.uint32_t(opts.TargetSampleRate), &sourceChannels, &sourceRate)
	if decoder == nil {
		return nil, errors.New("failed to decode MP3 audio")
	}
	defer C.ollama_miniaudio_mp3_uninit(decoder)
	if sourceChannels == 0 || sourceChannels > 8 {
		return nil, fmt.Errorf("unsupported MP3 channel count %d", uint32(sourceChannels))
	}
	if sourceRate < 8000 || sourceRate > 192000 {
		return nil, fmt.Errorf("unsupported MP3 sample rate %d", uint32(sourceRate))
	}

	const chunkFrames = 4096
	chunk := make([]float32, chunkFrames)
	samples := make([]float32, 0, min(opts.MaxSamples, chunkFrames))
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		want := chunkFrames
		remaining := opts.MaxSamples - len(samples)
		if remaining < want {
			want = remaining + 1 // Read one frame past the cap to detect overflow.
		}
		if want <= 0 {
			want = 1
		}
		var read C.uint64_t
		result := C.ollama_miniaudio_mp3_read(decoder, unsafe.Pointer(&chunk[0]), C.uint64_t(want), &read)
		if result != 0 {
			return nil, errors.New("failed while decoding MP3 audio")
		}
		count := int(read)
		if count > remaining {
			if opts.Overflow == AudioOverflowTruncate {
				samples = append(samples, chunk[:remaining]...)
				return validateDecodedMP3(samples)
			}
			return nil, fmt.Errorf("MP3 audio exceeds limit of %d samples", opts.MaxSamples)
		}
		for _, sample := range chunk[:count] {
			if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
				return nil, errors.New("MP3 contains a non-finite sample")
			}
		}
		samples = append(samples, chunk[:count]...)
		if count < want {
			break
		}
	}
	return validateDecodedMP3(samples)
}

func validateDecodedMP3(samples []float32) ([]float32, error) {
	if len(samples) == 0 {
		return nil, errors.New("MP3 audio contains no decodable samples")
	}
	return samples, nil
}
