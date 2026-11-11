//go:build cgo

package audio

import (
	"bytes"
	"context"
	"errors"
	"io"
	"testing"
)

func TestMP3SampleStream(t *testing.T) {
	data := testMP3(t)
	want, rate, err := Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := OpenSampleStream(t.Context(), saveStreamFixture(t, data))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	got := []float32{}
	buf := make([]float32, 257)
	for {
		n, err := stream.Read(buf)
		got = append(got, buf[:n]...)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(got) != len(want) || stream.SampleRate() != rate {
		t.Fatalf("stream %d at %d versus %d at %d", len(got), stream.SampleRate(), len(want), rate)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MP3 changed sample%d", i)
		}
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Read(buf); err == nil {
		t.Fatal("read after close")
	}
}

func TestMP3LongStreamAndCancel(t *testing.T) {
	data := bytes.Repeat(testMP3(t), 2000)
	path := saveStreamFixture(t, data)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	stream, err := OpenSampleStream(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	total := 0
	buf := make([]float32, 4096)
	for {
		n, err := stream.Read(buf)
		total += n
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if total > 61*stream.SampleRate() {
			cancel()
			break
		}
	}
	if total <= 60*stream.SampleRate() {
		t.Fatal("long MP3 did not decode past60seconds")
	}
	if _, err := stream.Read(buf); !errors.Is(err, context.Canceled) {
		t.Fatal("cancelled MP3 kept decoding", err)
	}
}
