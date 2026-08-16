//go:build !cgo

package audio

import "errors"

func decodeMP3([]byte, int, int) ([]float32, int, error) {
	return nil, 0, errors.New("MP3 decoding requires cgo")
}
