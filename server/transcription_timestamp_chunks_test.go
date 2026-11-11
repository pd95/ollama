package server

import (
	"context"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
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
	if err := appendTimestampWindow(&out, first, 0, 0, 30, false, nil); err != nil {
		t.Fatal(err)
	}
	if len(out.Words) != 1 {
		t.Fatal("crossing word must belong to next core by midpoint")
	}
	next := openai.TranscriptionTimestampResponse{Text: "boundary! Next.", Language: "english", Words: []openai.TranscriptionWord{{Word: "boundary", Start: 1.7, End: 2.5}, {Word: "Next", Start: 3, End: 4}}}
	if err := appendTimestampWindow(&out, next, 28, 30, 60, false, nil); err != nil {
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
		err := appendTimestampWindow(&out, in, 28, 30, 60, false, nil)
		if same && (err != nil || len(out.Words) != 1) {
			t.Fatalf("same interval not deduplicated: %v", err)
		}
		if !same && err == nil {
			t.Fatal("ambiguous intersecting words silently accepted")
		}
	}
	out := openai.TranscriptionTimestampResponse{Words: []openai.TranscriptionWord{{Word: "go", Start: 29, End: 29.5}}}
	if err := appendTimestampWindow(&out, openai.TranscriptionTimestampResponse{Text: "go go", Words: []openai.TranscriptionWord{{Word: "go", Start: 2, End: 2.3}, {Word: "go", Start: 2.4, End: 2.8}}}, 28, 30, 60, false, nil); err != nil || len(out.Words) != 3 {
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

func TestTimestampBoundaryUsesOneAlignment(t *testing.T) {
	out := openai.TranscriptionTimestampResponse{Text: "Why sky", Words: []openai.TranscriptionWord{{Word: "Why", Start: 29, End: 29.5}, {Word: "sky", Start: 29.6, End: 30.08}}}
	clip := openai.TranscriptionTimestampResponse{Text: "Why sky blue?", Words: []openai.TranscriptionWord{{Word: "Why", Start: .99, End: 1.52}, {Word: "sky", Start: 1.59, End: 2.05}, {Word: "blue", Start: 2.06, End: 2.5}}}
	if err := appendTimestampWindow(&out, clip, 28, 30, 60, false, nil); err != nil {
		t.Fatal(err)
	}
	if len(out.Words) != 3 || out.Words[1].End != 30.05 || out.Words[2].Start != 30.06 || out.Text != "Why sky blue?" {
		t.Fatalf("expected original next-window predictions, got %+v", out)
	}
}

func TestTimestampLongSilenceMetadata(t *testing.T) {
	result, err := timestampRecording(t.Context(), &timestampTestSamples{remaining: 61 * 16000}, nil, true, func(context.Context, []byte, time.Duration) (openai.TranscriptionTimestampResponse, error) {
		return openai.TranscriptionTimestampResponse{Language: "english", Words: []openai.TranscriptionWord{}}, nil
	})
	if err != nil || result.Duration != 61 || result.Language != "english" || result.Words == nil || len(result.Words) != 0 || result.Segments == nil || len(*result.Segments) != 0 {
		t.Fatalf("long silence metadata %+v %v", result, err)
	}
}

func TestTimestampRecordingDeferredWordShift(t *testing.T) {
	calls := 0
	out, err := timestampRecording(t.Context(), &timestampTestSamples{remaining: 65 * 16000}, nil, false, func(context.Context, []byte, time.Duration) (openai.TranscriptionTimestampResponse, error) {
		calls++
		switch calls {
		case 1:
			return openai.TranscriptionTimestampResponse{Text: "One boundary", Language: "english", Words: []openai.TranscriptionWord{{Word: "One", Start: 27, End: 27.5}, {Word: "boundary", Start: 29.8, End: 30.4}}}, nil
		case 2:
			return openai.TranscriptionTimestampResponse{Text: "boundary Next", Language: "english", Words: []openai.TranscriptionWord{{Word: "boundary", Start: 1.5, End: 2.1}, {Word: "Next", Start: 3, End: 4}}}, nil
		default:
			return openai.TranscriptionTimestampResponse{Language: "english", Words: []openai.TranscriptionWord{}}, nil
		}
	})
	if err != nil || len(out.Words) != 3 || out.Text != "One boundary Next" {
		t.Fatalf("deferred boundary lost: %+v %v", out, err)
	}
}

func TestTimestampDeferredAmbiguityFails(t *testing.T) {
	deferred := []openai.TranscriptionWord{{Word: "boundary", Start: 29.8, End: 30.4}}
	for _, clip := range []openai.TranscriptionTimestampResponse{
		{},
		{Text: "different Next", Words: []openai.TranscriptionWord{{Word: "different", Start: 1.5, End: 2.1}, {Word: "Next", Start: 3, End: 4}}},
		{Text: "boundary boundary", Words: []openai.TranscriptionWord{{Word: "boundary", Start: 1.8, End: 2.1}, {Word: "boundary", Start: 2.2, End: 2.3}}},
	} {
		out := openai.TranscriptionTimestampResponse{Text: "One", Words: []openai.TranscriptionWord{{Word: "One", Start: 27, End: 27.5}}}
		err := appendTimestampWindow(&out, clip, 28, 30, 60, false, deferred)
		if audioTextErrorStatus(err, 500) != 422 {
			t.Fatalf("unresolved deferred word succeeded: %+v %v", out, err)
		}
	}
}

func TestTimestampSilenceBeforeDetectedSpeech(t *testing.T) {
	calls := 0
	out, err := timestampRecording(t.Context(), &timestampTestSamples{remaining: 65 * 16000}, nil, false, func(context.Context, []byte, time.Duration) (openai.TranscriptionTimestampResponse, error) {
		calls++
		if calls == 2 {
			return openai.TranscriptionTimestampResponse{Text: "Hallo", Language: "german", Words: []openai.TranscriptionWord{{Word: "Hallo", Start: 3, End: 4}}}, nil
		}
		return openai.TranscriptionTimestampResponse{Language: "english", Words: []openai.TranscriptionWord{}}, nil
	})
	if err != nil || out.Language != "german" {
		t.Fatalf("silence chose language: %+v %v", out, err)
	}
}

func TestTimestampInvalidStreamDataStatus(t *testing.T) {
	pcm := make([]float32, 65*16000)
	pcm[0] = float32(math.NaN())
	wav, err := audio.EncodeFloatWAVContext(t.Context(), pcm, 16000)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "input.wav")
	if err := os.WriteFile(path, wav, 0o600); err != nil {
		t.Fatal(err)
	}
	s := &Server{}
	_, err = s.transcribeWithTimestamps(t.Context(), audioTextForm{timestampUpload: path}, nil, nil)
	if audioTextErrorStatus(err, 500) != 400 {
		t.Fatalf("bad samples classified as server failure: %v", err)
	}
	_, err = s.transcribeWithTimestamps(t.Context(), audioTextForm{timestampUpload: path + "-missing"}, nil, nil)
	if audioTextErrorStatus(err, 500) != 500 || !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("file IO failure reclassified: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = s.transcribeWithTimestamps(ctx, audioTextForm{timestampUpload: path}, nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation reclassified: %v", err)
	}
}

func TestTimestampDeferredPointAndOwnedPrefix(t *testing.T) {
	for _, point := range []bool{false, true} {
		end := 31.2
		currentEnd := 3.2
		if point {
			end = 31
			currentEnd = 3
		}
		out := openai.TranscriptionTimestampResponse{Text: "One", Words: []openai.TranscriptionWord{{Word: "One", Start: 27, End: 27.5}}}
		deferred := []openai.TranscriptionWord{{Word: "boundary", Start: 31, End: end}}
		clip := openai.TranscriptionTimestampResponse{Text: "This boundary Next", Words: []openai.TranscriptionWord{{Word: "This", Start: 2.2, End: 2.4}, {Word: "boundary", Start: 3, End: currentEnd}, {Word: "Next", Start: 4, End: 5}}}
		if err := appendTimestampWindow(&out, clip, 28, 30, 60, false, deferred); err != nil {
			t.Fatal(err)
		}
		if out.Text != "One This boundary Next" || len(out.Words) != 4 {
			t.Fatalf("current owned prefix lost: %+v", out)
		}
	}
}

func TestTimestampPointCorrespondence(t *testing.T) {
	for _, tc := range []struct {
		a, b openai.TranscriptionWord
		want bool
	}{
		{openai.TranscriptionWord{Start: 1, End: 1}, openai.TranscriptionWord{Start: 1, End: 1}, true},
		{openai.TranscriptionWord{Start: 1, End: 1}, openai.TranscriptionWord{Start: 1.1, End: 1.1}, false},
		{openai.TranscriptionWord{Start: 1, End: 1}, openai.TranscriptionWord{Start: .9, End: 1.1}, true},
		{openai.TranscriptionWord{Start: .9, End: 1.1}, openai.TranscriptionWord{Start: 1, End: 1}, true},
		{openai.TranscriptionWord{Start: 1, End: 2}, openai.TranscriptionWord{Start: 2, End: 3}, false},
	} {
		if got := timestampWordCorresponds(tc.a, tc.b); got != tc.want {
			t.Fatalf("correspondence %+v %+v =%v", tc.a, tc.b, got)
		}
	}
}

func TestTimestampOwnedPrefixBeforeCommittedAnchorFails(t *testing.T) {
	out := openai.TranscriptionTimestampResponse{Text: "sky", Words: []openai.TranscriptionWord{{Word: "sky", Start: 29.6, End: 30.1}}}
	clip := openai.TranscriptionTimestampResponse{Text: "This sky Next", Words: []openai.TranscriptionWord{{Word: "This", Start: 2, End: 2.04}, {Word: "sky", Start: 2.05, End: 2.65}, {Word: "Next", Start: 3, End: 4}}}
	if err := appendTimestampWindow(&out, clip, 28, 30, 60, false, nil); audioTextErrorStatus(err, 500) != 422 {
		t.Fatalf("new owned prefix silently lost: %+v %v", out, err)
	}
}

func TestTimestampFuturePreviewMayBeRevised(t *testing.T) {
	out := openai.TranscriptionTimestampResponse{Text: "One", Words: []openai.TranscriptionWord{{Word: "One", Start: 27, End: 27.5}}}
	deferred := []openai.TranscriptionWord{{Word: "blue", Start: 31.7, End: 32}}
	clip := openai.TranscriptionTimestampResponse{Text: "blue", Language: "english", Words: []openai.TranscriptionWord{{Word: "blue", Start: 5.6, End: 6.1}}}
	if err := appendTimestampWindow(&out, clip, 28, 30, 60, false, deferred); err != nil {
		t.Fatal(err)
	}
	if len(out.Words) != 2 || out.Text != "One blue" {
		t.Fatalf("owned revised preview lost: %+v", out)
	}
}

func TestTimestampFuturePreviewMayBecomeSilenceOrRepetitions(t *testing.T) {
	deferred := []openai.TranscriptionWord{{Word: "blue", Start: 31.7, End: 32.1}}
	for _, clip := range []openai.TranscriptionTimestampResponse{
		{},
		{Text: "blue blue", Words: []openai.TranscriptionWord{{Word: "blue", Start: 3.7, End: 3.8}, {Word: "blue", Start: 3.9, End: 4.1}}},
	} {
		out := openai.TranscriptionTimestampResponse{Text: "One", Words: []openai.TranscriptionWord{{Word: "One", Start: 27, End: 27.5}}}
		if err := appendTimestampWindow(&out, clip, 28, 30, 60, false, deferred); err != nil {
			t.Fatal(err)
		}
		if len(out.Words) != 1+len(clip.Words) {
			t.Fatalf("revised repetitions lost: %+v", out)
		}
	}
}

func TestTimestampMillisecondGlobalCoordinates(t *testing.T) {
	a := timestampGlobalWord(openai.TranscriptionWord{Word: "point", Start: 30.24, End: 30.24}, 0)
	b := timestampGlobalWord(openai.TranscriptionWord{Word: "point", Start: 2.24, End: 2.24}, 28)
	if a != b || !timestampWordCorresponds(a, b) {
		t.Fatalf("same millisecond differs: %+v %+v", a, b)
	}
	c := timestampGlobalWord(openai.TranscriptionWord{Word: "point", Start: 2.241, End: 2.241}, 28)
	if timestampWordCorresponds(a, c) {
		t.Fatal("distinct milliseconds merged")
	}
	left := timestampGlobalWord(openai.TranscriptionWord{Word: "left", Start: 29, End: 30.24}, 0)
	right := timestampGlobalWord(openai.TranscriptionWord{Word: "right", Start: 2.24, End: 3}, 28)
	if left.End != right.Start || timestampWordCorresponds(left, right) {
		t.Fatal("touching positive intervals merged")
	}
}
