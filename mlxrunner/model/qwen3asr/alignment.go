// Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
// SPDX-License-Identifier: Apache-2.0
// Native Go/MLX adaptations for Ollama.

package qwen3asr

import (
	"context"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model/audio"
)

// AlignmentUnit retains the supplied UTF-8 byte span independently of provider
// normalization. Text is the provider's cleaned word or CJK character.
type AlignmentUnit struct {
	Text      string `json:"text"`
	ByteStart int    `json:"byte_start"`
	ByteEnd   int    `json:"byte_end"`
}

type AlignmentItem struct {
	AlignmentUnit
	RawStartMS   int     `json:"raw_start_ms"`
	RawEndMS     int     `json:"raw_end_ms"`
	Start        float64 `json:"start"`
	End          float64 `json:"end"`
	Repaired     bool    `json:"repaired"`
	OutsideAudio bool    `json:"outside_audio"`
}

// AlignmentResult separates raw classification, provider repair, and source text.
// Timestamps are predictions, without an annotated alignment accuracy claim.
type AlignmentResult struct {
	SourceText string          `json:"source_text"`
	Classes    []int32         `json:"raw_classes"`
	Items      []AlignmentItem `json:"items"`
}

func cjk(r rune) bool {
	return r >= 0x4e00 && r <= 0x9fff || r >= 0x3400 && r <= 0x4dbf ||
		r >= 0x20000 && r <= 0x2a6df || r >= 0x2a700 && r <= 0x2b73f ||
		r >= 0x2b740 && r <= 0x2b81f || r >= 0x2b820 && r <= 0x2ceaf || r >= 0xf900 && r <= 0xfaff
}

// AlignmentPrompt implements the pinned provider's space-language tokenizer.
// Only English/German are admitted here; Japanese/Korean need separate assets.
func AlignmentPrompt(text, language string) ([]AlignmentUnit, string, error) {
	if !strings.EqualFold(language, "English") && !strings.EqualFold(language, "German") {
		return nil, "", fmt.Errorf("native alignment currently requires English or German")
	}
	if !utf8.ValidString(text) || len(text) > 65536 {
		return nil, "", fmt.Errorf("alignment text must be valid UTF-8 within64KiB")
	}
	var units []AlignmentUnit
	var word strings.Builder
	start, end := 0, 0
	flush := func() {
		if word.Len() > 0 {
			units = append(units, AlignmentUnit{Text: word.String(), ByteStart: start, ByteEnd: end})
			word.Reset()
		}
	}
	for offset, r := range text {
		if unicode.IsSpace(r) || r == 0x1c || r == 0x1d || r == 0x1e || r == 0x1f {
			flush()
			continue
		}
		if r != '\'' && !unicode.IsLetter(r) && !unicode.IsNumber(r) {
			continue
		}
		if cjk(r) {
			flush()
			units = append(units, AlignmentUnit{Text: string(r), ByteStart: offset, ByteEnd: offset + utf8.RuneLen(r)})
			continue
		}
		if word.Len() == 0 {
			start = offset
		}
		word.WriteRune(r)
		end = offset + utf8.RuneLen(r)
	}
	flush()
	if len(units) == 0 || len(units) > 1024 {
		return nil, "", fmt.Errorf("alignment requires1..1024 normalized units")
	}
	var prompt strings.Builder
	prompt.WriteString("<|audio_start|><|audio_pad|><|audio_end|>")
	for _, unit := range units {
		prompt.WriteString(unit.Text)
		prompt.WriteString("<timestamp><timestamp>")
	}
	return units, prompt.String(), nil
}

// repairTimestamps mirrors the pinned provider's earliest longest nondecreasing
// subsequence and anomaly repair, including integer truncation after interpolation.
func repairTimestamps(data []int) []int {
	n := len(data)
	result := append([]int(nil), data...)
	if n == 0 {
		return result
	}
	dp, parent, normal := make([]int, n), make([]int, n), make([]bool, n)
	best := 0
	for i := range n {
		dp[i], parent[i] = 1, -1
		for j := range i {
			if data[j] <= data[i] && dp[j]+1 > dp[i] {
				dp[i], parent[i] = dp[j]+1, j
			}
		}
		if dp[i] > dp[best] {
			best = i
		}
	}
	for i := best; i >= 0; i = parent[i] {
		normal[i] = true
	}
	for i := 0; i < n; {
		if normal[i] {
			i++
			continue
		}
		j := i
		for j < n && !normal[j] {
			j++
		}
		for k := i; k < j; k++ {
			switch {
			case i == 0:
				result[k] = result[j]
			case j == n:
				result[k] = result[i-1]
			case j-i <= 2:
				if k-(i-1) <= j-k {
					result[k] = result[i-1]
				} else {
					result[k] = result[j]
				}
			default:
				step := float64(result[j]-result[i-1]) / float64(j-i+1)
				result[k] = int(float64(result[i-1]) + step*float64(k-i+1))
			}
		}
		i = j
	}
	return result
}

func alignmentResult(text string, units []AlignmentUnit, classes []int32, seconds float64) (*AlignmentResult, error) {
	if len(classes) != len(units)*2 {
		return nil, fmt.Errorf("timestamp slot count mismatch")
	}
	ms := make([]int, len(classes))
	for i, c := range classes {
		if c < 0 || c >= 5000 {
			return nil, fmt.Errorf("invalid timestamp class%d", c)
		}
		ms[i] = int(c) * 80
	}
	fixed := repairTimestamps(ms)
	result := &AlignmentResult{SourceText: text, Classes: append([]int32(nil), classes...)}
	for i, u := range units {
		start, end := float64(fixed[2*i])/1000, float64(fixed[2*i+1])/1000
		result.Items = append(result.Items, AlignmentItem{
			AlignmentUnit: u, RawStartMS: ms[2*i], RawEndMS: ms[2*i+1],
			Start: start, End: end, Repaired: fixed[2*i] != ms[2*i] || fixed[2*i+1] != ms[2*i+1], OutsideAudio: start > seconds || end > seconds,
		})
	}
	return result, nil
}

// AlignContext performs one bounded native classification, never generation.
// The caller must own a serialized MLX worker, just as for model.Forward. CPU
// preparation honors cancellation; device work drains before return. Cancellation
// does not promise GPU preemption. No KV or request state survives this call.
func (m *Model) AlignContext(ctx context.Context, wav []byte, text, language string) (*AlignmentResult, error) {
	if m.cfg == nil || !m.cfg.Aligner {
		return nil, fmt.Errorf("model is not Qwen3-ForcedAligner")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	units, prompt, err := AlignmentPrompt(text, language)
	if err != nil {
		return nil, err
	}
	pcm, rate, err := audio.DecodeLimitContext(ctx, wav, maxAudioSeconds)
	if err != nil {
		return nil, err
	}
	if len(pcm) > maxAudioSeconds*rate {
		return nil, fmt.Errorf("alignment accepts at most60s")
	}
	pcm, err = audio.ResampleContext(ctx, pcm, rate, 16000)
	if err != nil {
		return nil, err
	}
	f, err := extractFeaturesContext(ctx, pcm)
	if err != nil {
		return nil, err
	}
	ids := m.tok.Encode(prompt, false)
	expanded := make([]int32, 0, len(ids)+audioTokens(f.Frames))
	start := -1
	var slots []int
	for _, id := range ids {
		if id == audioPad {
			start = len(expanded)
			for range audioTokens(f.Frames) {
				expanded = append(expanded, audioPad)
			}
		} else {
			if id == 151705 {
				slots = append(slots, len(expanded))
			}
			expanded = append(expanded, id)
		}
	}
	if start < 0 || len(slots) != 2*len(units) || len(expanded) > int(m.cfg.Text.MaxPositionEmbeddings) {
		return nil, fmt.Errorf("invalid alignment prompt/context length")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var classes []int32
	mlx.Scoped(func() {
		encoded := m.encoder.forward(mlx.FromValues(f.Data, 128, f.Frames))
		b := &batch.Batch{InputIDs: mlx.FromValues(expanded, 1, len(expanded)), SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(len(expanded))}, Media: []batch.MediaItem{{Pos: start, Features: encoded}}}
		hidden, _ := m.Forward(b, nil)
		indices := make([]int32, len(slots))
		for i, s := range slots {
			indices[i] = int32(s)
		}
		selected := mlx.Take(hidden, mlx.FromValues(indices, len(indices)), 1)
		classes = m.decoder.Unembed(selected).Argmax(-1, false).AsType(mlx.DTypeInt32).Ints()
		mlx.DefaultStream().Synchronize()
	})
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return alignmentResult(text, units, classes, float64(len(pcm))/16000)
}
