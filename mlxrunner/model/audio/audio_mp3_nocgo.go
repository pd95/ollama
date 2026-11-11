//go:build !cgo

package audio

import (
	"context"
	"errors"
)

func decodeMP3([]byte, int, int) ([]float32, int, error) {
	return nil, 0, errors.New("MP3 decoding requires cgo")
}

func decodeMP3Context(ctx context.Context, _ []byte, _, _ int) ([]float32, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	return nil, 0, errors.New("MP3 decoding requires cgo")
}

func decodeMP3ContextDuration(ctx context.Context, data []byte, maxInputBytes, maxSamples, _ int) ([]float32, int, error) {
	return decodeMP3Context(ctx, data, maxInputBytes, maxSamples)
}

func openMP3Stream(ctx context.Context, _ string) (SampleStream, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return nil, errors.New("MP3 decoding requires cgo")
}
