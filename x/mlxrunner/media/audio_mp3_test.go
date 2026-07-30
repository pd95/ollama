//go:build cgo

package media

import (
	"bytes"
	"context"
	"encoding/base64"
	"os"
	"strconv"
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

func TestDecodeMP3(t *testing.T) {
	data := testMP3(t)
	for _, rate := range []int{16000, 24000} {
		t.Run(strconv.Itoa(rate), func(t *testing.T) {
			samples, err := DecodeMP3(context.Background(), data, MP3DecodeOptions{
				TargetSampleRate: rate, MaxInputBytes: len(data), MaxSamples: 10000,
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(samples) == 0 || len(samples) > 10000 {
				t.Fatalf("decoded sample count = %d", len(samples))
			}
		})
	}
}

func TestDecodeMP3StereoDownmix(t *testing.T) {
	data := testMP3(t)
	for offset := 3; offset < len(data); offset += 768 {
		data[offset] &= 0x3f // MPEG channel mode 00: stereo.
	}
	samples, err := DecodeMP3(context.Background(), data, MP3DecodeOptions{
		TargetSampleRate: 16000, MaxInputBytes: len(data), MaxSamples: 10000,
	})
	if err != nil || len(samples) == 0 {
		t.Fatalf("stereo MP3 decode = %d samples, %v", len(samples), err)
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
	_, err := DecodeMP3(ctx, data, MP3DecodeOptions{
		TargetSampleRate: 16000, MaxInputBytes: len(data), MaxSamples: 10000,
	})
	if err != context.Canceled || ctx.checks < 3 {
		t.Fatalf("mid-stream cancellation = %v after %d checks", err, ctx.checks)
	}
}

func TestDecodeMP3LimitsAndErrors(t *testing.T) {
	data := testMP3(t)
	base := MP3DecodeOptions{TargetSampleRate: 16000, MaxInputBytes: len(data), MaxSamples: 10000}

	if _, err := DecodeMP3(context.Background(), data, MP3DecodeOptions{
		TargetSampleRate: 16000, MaxInputBytes: len(data) - 1, MaxSamples: 10000,
	}); err == nil || !strings.Contains(err.Error(), "bytes") {
		t.Fatalf("input limit error = %v", err)
	}
	if _, err := DecodeMP3(context.Background(), []byte("ID3not-an-mp3"), base); err == nil || !strings.Contains(err.Error(), "decode") {
		t.Fatalf("malformed error = %v", err)
	}

	reject := base
	reject.MaxSamples = 100
	if _, err := DecodeMP3(context.Background(), data, reject); err == nil || !strings.Contains(err.Error(), "samples") {
		t.Fatalf("sample limit error = %v", err)
	}
	truncate := reject
	truncate.Overflow = AudioOverflowTruncate
	samples, err := DecodeMP3(context.Background(), data, truncate)
	if err != nil || len(samples) != truncate.MaxSamples {
		t.Fatalf("truncated samples = %d, %v", len(samples), err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := DecodeMP3(ctx, data, base); err != context.Canceled {
		t.Fatalf("canceled error = %v", err)
	}
	if _, err := DecodeMP3(context.Background(), data, MP3DecodeOptions{}); err == nil || !strings.Contains(err.Error(), "limits") {
		t.Fatalf("invalid options error = %v", err)
	}
}
