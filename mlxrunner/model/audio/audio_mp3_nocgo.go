//go:build !cgo

package audio

import (
	"context"
	"errors"
)

func decodeMP3([]byte, int, int) ([]float32, int, error) {
	return nil, 0, errors.New("MP3 decoding requires cgo")
}

func decodeMP3Duration(data []byte, maxInputBytes, _ int) ([]float32, int, error) {
	return decodeMP3(data, maxInputBytes, 0)
}

func decodeMP3Context(ctx context.Context, _ []byte, _, _ int) ([]float32, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	return nil, 0, errors.New("MP3 decoding requires cgo")
}
