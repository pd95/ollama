//go:build cgo

package audio

import (
	"encoding/base64"
	"os"
	"strings"
	"testing"
)

func testMP3(t *testing.T) []byte {
	t.Helper()
	encoded, err := os.ReadFile("testdata/bcn_weather_first_10_frames.mp3.base64")
	if err != nil {
		t.Fatal(err)
	}
	data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(string(encoded)))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestDecodeMP3(t *testing.T) {
	data := testMP3(t)
	samples, rate, err := Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) == 0 || len(samples) > maxAudioSeconds*rate {
		t.Fatalf("decoded %d samples at %d Hz", len(samples), rate)
	}
	if rate < 8000 || rate > 192000 {
		t.Fatalf("decoded sample rate = %d", rate)
	}
}

func TestDecodeMP3LimitsAndErrors(t *testing.T) {
	data := testMP3(t)

	if _, _, err := decodeMP3(data, len(data)-1, 10000); err == nil || !strings.Contains(err.Error(), "bytes") {
		t.Fatalf("input limit error = %v", err)
	}
	if _, _, err := decodeMP3([]byte("ID3not-an-mp3"), len(data), 10000); err == nil || !strings.Contains(err.Error(), "decode") {
		t.Fatalf("malformed error = %v", err)
	}
	if _, _, err := decodeMP3(data, len(data), 100); err == nil || !strings.Contains(err.Error(), "samples") {
		t.Fatalf("sample limit error = %v", err)
	}
	if _, _, err := decodeMP3(data, 0, 0); err == nil || !strings.Contains(err.Error(), "limits") {
		t.Fatalf("invalid options error = %v", err)
	}
}
