//go:build !cgo

package audio

import (
	"strings"
	"testing"
)

func TestDecodeMP3RequiresCGO(t *testing.T) {
	_, _, err := Decode([]byte("ID3\x04\x00rest"))
	if err == nil || !strings.Contains(err.Error(), "requires cgo") {
		t.Fatalf("MP3 decode error = %v", err)
	}
}
