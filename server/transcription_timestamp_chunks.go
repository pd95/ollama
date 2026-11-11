package server

import (
	"context"
	"errors"
	"io"
	"math"
	"strings"
	"time"

	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
	"github.com/ollama/ollama/openai"
)

const (
	timestampCoreSeconds    = 30
	timestampOverlapSeconds = 2
)

type timestampClipFunc func(context.Context, []byte, time.Duration) (openai.TranscriptionTimestampResponse, error)

// timestampRecording owns a rolling source buffer. At most60 seconds plus one
// source frame are buffered, independent of recording duration. Each native
// ASR->align stage drains before another stage acquires a scheduler lease.
func timestampRecording(ctx context.Context, source audio.SampleStream, original []byte, segments bool, transcribe timestampClipFunc) (openai.TranscriptionTimestampResponse, error) {
	out := openai.TranscriptionTimestampResponse{Task: "transcribe", Words: make([]openai.TranscriptionWord, 0)}
	if segments {
		v := make([]openai.TranscriptionSegment, 0)
		out.Segments = &v
	}
	rate := source.SampleRate()
	if rate < 1 || rate > 384000 {
		return out, timestampFailure(400, "unsupported timestamp sample rate")
	}
	buffer := make([]float32, 60*rate+1)
	count, start := 0, 0
	eof := false
	fill := func(target int) error {
		for !eof && start+count < target {
			if err := ctx.Err(); err != nil {
				return err
			}
			want := min(4096, target-start-count, len(buffer)-count)
			if want <= 0 {
				return errors.New("timestamp source buffer exhausted")
			}
			n, err := source.Read(buffer[count : count+want])
			if n < 0 || n > want {
				return errors.New("invalid audio decoder read")
			}
			count += n
			if start+count > maxAudioSeconds*rate {
				return timestampFailure(400, "word timestamps accept recordings up to 2 hours")
			}
			if err != nil && !errors.Is(err, io.EOF) {
				return err
			}
			eof = errors.Is(err, io.EOF)
			if n == 0 && !eof {
				return io.ErrNoProgress
			}
		}
		return nil
	}
	if err := fill(len(buffer)); err != nil {
		return out, err
	}
	if count == 0 {
		return out, timestampFailure(400, "audio file contains no samples")
	}
	encode := func(samples []float32) ([]byte, error) {
		pcm, err := audio.ResampleContext(ctx, samples, rate, 16000)
		if err != nil {
			return nil, err
		}
		return audio.EncodeFloatWAVContext(ctx, pcm, 16000)
	}
	if eof && count <= 60*rate {
		wav := original
		if wav == nil {
			var err error
			wav, err = encode(buffer[:count])
			if err != nil {
				return out, err
			}
		}
		return transcribe(ctx, wav, time.Duration(count)*time.Second/time.Duration(rate))
	}
	for core := 0; ; core += timestampCoreSeconds * rate {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		windowStart := max(0, core-timestampOverlapSeconds*rate)
		discard := windowStart - start
		copy(buffer, buffer[discard:count])
		count -= discard
		start = windowStart
		if err := fill(core + (timestampCoreSeconds+timestampOverlapSeconds)*rate); err != nil {
			return out, err
		}
		total := start + count
		if eof && core >= total {
			break
		}
		end := min(total, core+(timestampCoreSeconds+timestampOverlapSeconds)*rate)
		wav, err := encode(buffer[:end-start])
		if err != nil {
			return out, err
		}
		clip, err := transcribe(ctx, wav, time.Duration(end-start)*time.Second/time.Duration(rate))
		if err != nil {
			return out, err
		}
		final := eof && total <= core+timestampCoreSeconds*rate
		coreEnd := min(total, core+timestampCoreSeconds*rate)
		if err := appendTimestampWindow(&out, clip, float64(start)/float64(rate), float64(core)/float64(rate), float64(coreEnd)/float64(rate), final); err != nil {
			return out, err
		}
		out.Duration = float64(total) / float64(rate)
		if final {
			break
		}
	}
	return out, nil
}

// Ownership is decided by aligned word midpoints in nonoverlapping cores, not
// by matching arbitrary transcript suffixes. Context on both sides lets words
// crossing a cut be recognized and aligned whole. Times are never clamped.
func appendTimestampWindow(out *openai.TranscriptionTimestampResponse, clip openai.TranscriptionTimestampResponse, offset, coreStart, coreEnd float64, final bool) error {
	if clip.Text == "" {
		return nil
	}
	name := "English"
	if clip.Language == "german" {
		name = "German"
	}
	units, _, err := qwen3asr.AlignmentPrompt(clip.Text, name)
	if err != nil || len(units) != len(clip.Words) {
		return timestampFailure(502, "chunk alignment does not cover transcript")
	}
	first, last := -1, -1
	firstOutput := len(out.Words)
	for i, word := range clip.Words {
		word.Start += offset
		word.End += offset
		middle := (word.Start + word.End) / 2
		if math.IsNaN(middle) || math.IsInf(middle, 0) || word.End < word.Start {
			return timestampFailure(502, "invalid chunk word interval")
		}
		if middle < coreStart || middle >= coreEnd && !(final && middle == coreEnd) {
			continue
		}
		if len(out.Words) > 0 {
			previous := out.Words[len(out.Words)-1]
			if word.Start < previous.End {
				// Only a matching word at intersecting times may be the same boundary
				// event. Equal text at disjoint times is a legitimate repetition.
				if strings.EqualFold(previous.Word, word.Word) && word.End > previous.Start {
					continue
				}
				return timestampFailure(422, "ambiguous word alignment at chunk boundary")
			}
		}
		if first < 0 {
			first = i
		}
		last = i
		out.Words = append(out.Words, word)
	}
	if first < 0 {
		return nil
	}
	from, to := units[first].ByteStart, len(clip.Text)
	if first == 0 {
		from = 0
	}
	if last+1 < len(units) {
		to = units[last+1].ByteStart
	}
	text := strings.TrimSpace(clip.Text[from:to])
	if out.Text != "" && needsSpace(out.Text, text) {
		out.Text += " "
	}
	out.Text += text
	if out.Language == "" {
		out.Language = clip.Language
	}
	if out.Segments != nil {
		*out.Segments = append(*out.Segments, openai.TranscriptionSegment{ID: len(*out.Segments), Start: out.Words[firstOutput].Start, End: out.Words[len(out.Words)-1].End, Text: text})
	}
	return nil
}
