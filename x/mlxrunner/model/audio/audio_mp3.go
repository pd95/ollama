//go:build cgo

package audio

/*
#cgo CFLAGS: -std=c99
#cgo linux LDFLAGS: -lm
#include <stdlib.h>
#include "miniaudio_wrapper.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// decodeMP3 decodes MP3 bytes to mono float32 PCM at the source sample rate.
// maxSamples is a test hook; zero applies maxAudioSeconds after the decoder
// reports the source rate.
func decodeMP3(data []byte, maxInputBytes, maxSamples int) ([]float32, int, error) {
	if maxInputBytes <= 0 || maxSamples < 0 {
		return nil, 0, errors.New("invalid MP3 decode limits")
	}
	if len(data) == 0 {
		return nil, 0, errors.New("MP3 input is empty")
	}
	if len(data) > maxInputBytes {
		return nil, 0, fmt.Errorf("MP3 audio is %d bytes, limit %d", len(data), maxInputBytes)
	}

	encoded := C.CBytes(data)
	defer C.free(encoded)
	var sourceChannels, sourceRate C.uint32_t
	decoder := C.ollama_miniaudio_mp3_init(encoded, C.size_t(len(data)), &sourceChannels, &sourceRate)
	if decoder == nil {
		return nil, 0, errors.New("failed to decode MP3 audio")
	}
	defer C.ollama_miniaudio_mp3_uninit(decoder)
	if sourceChannels == 0 || sourceChannels > 8 {
		return nil, 0, fmt.Errorf("unsupported MP3 channel count %d", uint32(sourceChannels))
	}
	if sourceRate < 8000 || sourceRate > 192000 {
		return nil, 0, fmt.Errorf("unsupported MP3 sample rate %d", uint32(sourceRate))
	}
	rate := int(sourceRate)
	if maxSamples == 0 {
		maxSamples = maxAudioSeconds * rate
	}

	const chunkFrames = 4096
	chunk := make([]float32, chunkFrames)
	samples := make([]float32, 0, min(maxSamples, chunkFrames))
	for {
		want := chunkFrames
		remaining := maxSamples - len(samples)
		if remaining < want {
			want = remaining + 1 // Read one frame past the cap to detect overflow.
		}
		if want <= 0 {
			want = 1
		}
		var read C.uint64_t
		result := C.ollama_miniaudio_mp3_read(decoder, unsafe.Pointer(&chunk[0]), C.uint64_t(want), &read)
		if result != 0 {
			return nil, 0, errors.New("failed while decoding MP3 audio")
		}
		count := int(read)
		for _, sample := range chunk[:count] {
			if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
				return nil, 0, errors.New("MP3 contains a non-finite sample")
			}
		}
		if count > remaining {
			if maxSamples == maxAudioSeconds*rate {
				return nil, 0, fmt.Errorf("audio longer than %d seconds", maxAudioSeconds)
			}
			return nil, 0, fmt.Errorf("MP3 audio exceeds limit of %d samples", maxSamples)
		}
		samples = append(samples, chunk[:count]...)
		if count < want {
			break
		}
	}
	if len(samples) == 0 {
		return nil, 0, errors.New("MP3 audio contains no decodable samples")
	}
	return samples, rate, nil
}
