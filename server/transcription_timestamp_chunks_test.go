package server

import (
	"context"
	"errors"
	"io"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/openai"
)

type timestampTestSamples struct {
	remaining, reads, largest int
	ctx                       context.Context //nolint:containedctx // Test decoder follows caller cancellation.
}

func (s *timestampTestSamples) SampleRate() int { return 16000 }
func (s *timestampTestSamples) Close() error    { return nil }
func (s *timestampTestSamples) Read(p []float32) (int, error) {
	if s.ctx != nil && s.ctx.Err() != nil {
		return 0, s.ctx.Err()
	}
	if s.remaining == 0 {
		return 0, io.EOF
	}
	n := min(len(p), 4096, s.remaining)
	clear(p[:n])
	s.remaining -= n
	s.reads++
	s.largest = max(s.largest, len(p))
	return n, nil
}

func TestTimestampLongGlobalOffsets(t *testing.T) {
	source := &timestampTestSamples{remaining: 95 * 16000}
	calls := 0
	result, err := timestampRecording(t.Context(), source, nil, true, func(ctx context.Context, wav []byte, d time.Duration) (openai.TranscriptionTimestampResponse, error) {
		calls++
		if len(wav) > 34*16000*4+44 || d > 34*time.Second {
			t.Fatal("unbounded window")
		}
		return openai.TranscriptionTimestampResponse{Text: "Hello!", Language: "english", Words: []openai.TranscriptionWord{{Word: "Hello", Start: 3, End: 4}}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Duration != 95 || calls != 4 || len(result.Words) != 4 || result.Words[1].Start != 31 || result.Words[3].Start != 91 || result.Text != "Hello! Hello! Hello! Hello!" {
		t.Fatalf("wrong global output %+v calls%d", result, calls)
	}
	if result.Segments == nil || len(*result.Segments) != 4 || (*result.Segments)[3].ID != 3 {
		t.Fatal("segment IDs")
	}
	if source.largest > 4096 {
		t.Fatal("decoder read not bounded")
	}
}

func TestTimestampShortPreservesWAVAndText(t *testing.T) {
	original := []byte("original encoded WAV")
	result, err := timestampRecording(t.Context(), &timestampTestSamples{remaining: 60 * 16000}, original, false, func(ctx context.Context, wav []byte, d time.Duration) (openai.TranscriptionTimestampResponse, error) {
		if string(wav) != string(original) || d != 60*time.Second {
			t.Fatal("short request changed")
		}
		return openai.TranscriptionTimestampResponse{Text: "  Original, punctuation!", Duration: 60, Words: []openai.TranscriptionWord{}}, nil
	})
	if err != nil || result.Text != "  Original, punctuation!" {
		t.Fatalf("%+v %v", result, err)
	}
}

func TestTimestampBoundaryOwnership(t *testing.T) {
	out := openai.TranscriptionTimestampResponse{Words: []openai.TranscriptionWord{}}
	first := openai.TranscriptionTimestampResponse{Text: "One boundary!", Language: "english", Words: []openai.TranscriptionWord{{Word: "One", Start: 28, End: 29}, {Word: "boundary", Start: 29.7, End: 30.5}}}
	if err := appendTimestampWindow(&out, first, 0, 0, 30, false); err != nil {
		t.Fatal(err)
	}
	if len(out.Words) != 1 {
		t.Fatal("crossing word must belong to next core by midpoint")
	}
	next := openai.TranscriptionTimestampResponse{Text: "boundary! Next.", Language: "english", Words: []openai.TranscriptionWord{{Word: "boundary", Start: 1.7, End: 2.5}, {Word: "Next", Start: 3, End: 4}}}
	if err := appendTimestampWindow(&out, next, 28, 30, 60, false); err != nil {
		t.Fatal(err)
	}
	if len(out.Words) != 3 || out.Words[1].Start != 29.7 || strings.Count(strings.ToLower(out.Text), "boundary") != 1 {
		t.Fatalf("boundary lost/duplicated: %+v", out)
	}
}

func TestTimestampBoundaryDedupAndConflicts(t *testing.T) {
	for _, same := range []bool{true, false} {
		out := openai.TranscriptionTimestampResponse{Words: []openai.TranscriptionWord{{Word: "word", Start: 29.6, End: 30.3}}}
		text := "word"
		if !same {
			text = "other"
		}
		in := openai.TranscriptionTimestampResponse{Text: text, Words: []openai.TranscriptionWord{{Word: text, Start: 1.9, End: 2.5}}}
		err := appendTimestampWindow(&out, in, 28, 30, 60, false)
		if same && (err != nil || len(out.Words) != 1) {
			t.Fatalf("same interval not deduplicated: %v", err)
		}
		if !same && err == nil {
			t.Fatal("ambiguous intersecting words silently accepted")
		}
	}
	out := openai.TranscriptionTimestampResponse{Words: []openai.TranscriptionWord{{Word: "go", Start: 29, End: 29.5}}}
	if err := appendTimestampWindow(&out, openai.TranscriptionTimestampResponse{Text: "go go", Words: []openai.TranscriptionWord{{Word: "go", Start: 2, End: 2.3}, {Word: "go", Start: 2.4, End: 2.8}}}, 28, 30, 60, false); err != nil || len(out.Words) != 3 {
		t.Fatalf("legitimate repetitions lost: %v", err)
	}
}

func TestTimestampChunkCancellationAndLimits(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	source := &timestampTestSamples{remaining: 120 * 16000, ctx: ctx}
	calls := 0
	_, err := timestampRecording(ctx, source, nil, false, func(context.Context, []byte, time.Duration) (openai.TranscriptionTimestampResponse, error) {
		calls++
		cancel()
		return openai.TranscriptionTimestampResponse{}, nil
	})
	if !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatalf("cancellation %v calls%d", err, calls)
	}
	// Huge duration is represented by a reader, never a complete in-memory clip.
	source = &timestampTestSamples{remaining: (maxAudioSeconds + 1) * 16000}
	_, err = timestampRecording(t.Context(), source, nil, false, func(context.Context, []byte, time.Duration) (openai.TranscriptionTimestampResponse, error) {
		return openai.TranscriptionTimestampResponse{}, nil
	})
	if err == nil {
		t.Fatal("duration cap not enforced")
	}
}

func TestTimestampMP3LongDecoder(t *testing.T) {
	// Actual compressed-frame tests live beside the cgo decoder. This checks that
	// generated chunk WAVs preserve F32 samples and exact duration for the runner.
	values := []float32{.1234567, -.003, 0, 1}
	wav, err := audio.EncodeFloatWAVContext(t.Context(), values, 16000)
	if err != nil {
		t.Fatal(err)
	}
	decoded, rate, err := audio.Decode(wav)
	if err != nil || rate != 16000 || len(decoded) != len(values) {
		t.Fatal(err)
	}
	for i, v := range values {
		if math.Float32bits(v) != math.Float32bits(decoded[i]) {
			t.Fatal("chunk quantized samples")
		}
	}
}
