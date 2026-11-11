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

// timestampRecording owns a rolling source buffer. At most 60 seconds plus one
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
				if errors.Is(err, audio.ErrInvalidData) {
					return timestampFailure(400, err.Error())
				}
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
	var deferred []openai.TranscriptionWord
	var fallbackLanguage string
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
		if err := appendTimestampWindow(&out, clip, float64(start)/float64(rate), float64(core)/float64(rate), float64(coreEnd)/float64(rate), final, deferred); err != nil {
			return out, err
		}
		if fallbackLanguage == "" {
			fallbackLanguage = clip.Language
		}
		deferred = timestampDeferredWords(clip, float64(start)/float64(rate), float64(coreEnd)/float64(rate), final)
		out.Duration = float64(total) / float64(rate)
		if final {
			break
		}
	}
	if out.Language == "" {
		out.Language = fallbackLanguage
	}
	return out, nil
}

// Ownership is decided by aligned word midpoints in nonoverlapping cores, not
// by matching arbitrary transcript suffixes. Context on both sides lets words
// crossing a cut be recognized and aligned whole. Times are never clamped.
func appendTimestampWindow(out *openai.TranscriptionTimestampResponse, clip openai.TranscriptionTimestampResponse, offset, coreStart, coreEnd float64, final bool, deferred []openai.TranscriptionWord) error {
	if clip.Text == "" {
		if len(deferred) > 0 {
			return timestampFailure(422, "deferred boundary words disappeared in the next window")
		}
		return nil
	}
	if out.Language == "" {
		out.Language = clip.Language
	}
	name := "English"
	if clip.Language == "german" {
		name = "German"
	}
	units, _, err := qwen3asr.AlignmentPrompt(clip.Text, name)
	if err != nil || len(units) != len(clip.Words) {
		return timestampFailure(502, "chunk alignment does not cover transcript")
	}
	firstDeferred, err := timestampDeferredMatch(deferred, clip.Words, offset)
	if err != nil {
		return err
	}
	// Join at a shared time-local word, using one window's actual alignment
	// for the transition. This avoids mixing two independently jittered word
	// intervals on opposite sides of an artificial core cut.
	anchor, err := timestampWindowAnchor(out, clip, offset, coreStart)
	if err != nil {
		return err
	}
	if firstDeferred >= 0 && firstDeferred <= anchor {
		return timestampFailure(422, "deferred word conflicts with committed boundary")
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
		owned := middle >= coreStart && (middle < coreEnd || final && middle == coreEnd)
		if anchor >= 0 && i <= anchor {
			if owned {
				committed := false
				for j := len(out.Words) - 1; j >= 0; j-- {
					previous := out.Words[j]
					if previous.End < word.Start {
						break
					}
					if strings.EqualFold(previous.Word, word.Word) && timestampWordCorresponds(previous, word) {
						committed = true
						break
					}
				}
				if !committed {
					return timestampFailure(422, "new owned word conflicts with committed boundary")
				}
			}
			continue
		}
		if middle >= coreEnd && !(final && middle == coreEnd) || anchor < 0 && !owned && (firstDeferred < 0 || i < firstDeferred) {
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
	if out.Segments != nil {
		*out.Segments = append(*out.Segments, openai.TranscriptionSegment{ID: len(*out.Segments), Start: out.Words[firstOutput].Start, End: out.Words[len(out.Words)-1].End, Text: text})
	}
	return nil
}

// timestampWindowAnchor only matches equal words at intersecting audio times.
// It cannot collapse repeated words in separate intervals. Any replacement
// times are complete native predictions from the new window, never clamps or
// averages. Matching predecessors are included when needed to preserve ordering.
func timestampWindowAnchor(out *openai.TranscriptionTimestampResponse, clip openai.TranscriptionTimestampResponse, offset, coreStart float64) (int, error) {
	if len(out.Words) == 0 {
		return -1, nil
	}
	last := out.Words[len(out.Words)-1]
	anchor := -1
	global := func(w openai.TranscriptionWord) openai.TranscriptionWord {
		w.Start += offset
		w.End += offset
		return w
	}
	for i, word := range clip.Words {
		word = global(word)
		if word.Start > coreStart+timestampOverlapSeconds {
			break
		}
		if strings.EqualFold(word.Word, last.Word) && timestampWordCorresponds(word, last) {
			if anchor >= 0 {
				return -1, timestampFailure(422, "ambiguous repeated word at chunk boundary")
			}
			anchor = i
		}
	}
	if anchor < 0 {
		return -1, nil
	}
	oldIndex, newIndex := len(out.Words)-1, anchor
	for oldIndex > 0 && global(clip.Words[newIndex]).Start < out.Words[oldIndex-1].End {
		if newIndex == 0 {
			return -1, timestampFailure(422, "inconsistent alignment at chunk boundary")
		}
		previous := global(clip.Words[newIndex-1])
		if !strings.EqualFold(previous.Word, out.Words[oldIndex-1].Word) || !timestampWordCorresponds(previous, out.Words[oldIndex-1]) {
			return -1, timestampFailure(422, "conflicting words at chunk boundary")
		}
		oldIndex--
		newIndex--
	}
	for i := newIndex; i <= anchor; i++ {
		out.Words[oldIndex+i-newIndex] = global(clip.Words[i])
	}
	if out.Segments != nil && len(*out.Segments) > 0 {
		seg := &(*out.Segments)[len(*out.Segments)-1]
		seg.Start = min(seg.Start, out.Words[oldIndex].Start)
		seg.End = out.Words[len(out.Words)-1].End
	}
	return anchor, nil
}

// Keep only the bounded right-context tail until the next window accounts for
// it. A shifted midpoint must not make a word fall through both ownership cores.
func timestampDeferredWords(clip openai.TranscriptionTimestampResponse, offset, cut float64, final bool) []openai.TranscriptionWord {
	if final {
		return nil
	}
	var words []openai.TranscriptionWord
	for _, word := range clip.Words {
		word.Start += offset
		word.End += offset
		if (word.Start+word.End)/2 >= cut {
			words = append(words, word)
		}
	}
	return words
}

func timestampDeferredMatch(deferred, current []openai.TranscriptionWord, offset float64) (int, error) {
	first, previous := -1, -1
	for _, old := range deferred {
		match := -1
		for i, word := range current {
			word.Start += offset
			word.End += offset
			if strings.EqualFold(old.Word, word.Word) && timestampWordCorresponds(old, word) {
				if match >= 0 {
					return -1, timestampFailure(422, "ambiguous deferred word at chunk boundary")
				}
				match = i
			}
		}
		if match < 0 || match <= previous {
			return -1, timestampFailure(422, "unresolved deferred word at chunk boundary")
		}
		if first < 0 {
			first = match
		}
		previous = match
	}
	return first, nil
}

// Native repair permits zero-duration words. Equal points and a unique point
// inside another prediction can correspond; two positive intervals that only
// touch remain separate events. Callers reject multiple candidate matches.
func timestampWordCorresponds(a, b openai.TranscriptionWord) bool {
	if a.Start == a.End {
		return b.Start <= a.Start && a.Start <= b.End
	}
	if b.Start == b.End {
		return a.Start <= b.Start && b.Start <= a.End
	}
	return a.Start < b.End && b.Start < a.End
}
