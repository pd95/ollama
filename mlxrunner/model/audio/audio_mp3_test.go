//go:build cgo

package audio

import (
	"bytes"
	"context"
	"encoding/base64"
	"os"
	"strings"
	"testing"
	"time"
)

func testMP3(t *testing.T) []byte {
	t.Helper()
	encoded, err := os.ReadFile("testdata/synthetic_silence.mp3.base64")
	if err != nil {
		t.Fatal(err)
	}
	data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(string(encoded)))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestDecodeMP3StereoDownmix(t *testing.T) {
	data := testMP3(t)
	for offset := 3; offset < len(data); offset += 768 {
		data[offset] &= 0x3f // MPEG channel mode 00: stereo.
	}
	samples, rate, err := Decode(data)
	if err != nil || len(samples) == 0 || rate == 0 {
		t.Fatalf("stereo MP3 decode = %d samples at %d Hz, %v", len(samples), rate, err)
	}
}

type cancelAfterErrChecks struct {
	checks int
	limit  int
}

func (c *cancelAfterErrChecks) Err() error {
	c.checks++
	if c.checks > c.limit {
		return context.Canceled
	}
	return nil
}

func (c *cancelAfterErrChecks) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *cancelAfterErrChecks) Done() <-chan struct{}       { return nil }
func (c *cancelAfterErrChecks) Value(any) any               { return nil }

func TestDecodeMP3MidStreamCancellation(t *testing.T) {
	data := bytes.Repeat(testMP3(t), 2)
	ctx := &cancelAfterErrChecks{limit: 2}
	_, _, err := decodeMP3Context(ctx, data, len(data), 10000)
	if err != context.Canceled || ctx.checks < 3 {
		t.Fatalf("mid-stream cancellation = %v after %d checks", err, ctx.checks)
	}
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
