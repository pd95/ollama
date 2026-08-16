//go:build !cgo

package media

import (
	"context"
	"errors"
)

type AudioOverflowPolicy uint8

const (
	AudioOverflowReject AudioOverflowPolicy = iota
	AudioOverflowTruncate
)

type MP3DecodeOptions struct {
	TargetSampleRate int
	MaxInputBytes    int
	MaxSamples       int
	Overflow         AudioOverflowPolicy
}

func DecodeMP3(context.Context, []byte, MP3DecodeOptions) ([]float32, error) {
	return nil, errors.New("MP3 decoding requires cgo")
}
