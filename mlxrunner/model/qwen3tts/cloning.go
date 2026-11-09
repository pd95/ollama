package qwen3tts

import (
	"context"
	"fmt"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model/audio"
)

// cloneProfile lives for one request only. It contains no model-global cache
// and is reused only across that request's bounded target-text chunks.
type cloneProfile struct {
	speaker *mlx.Array
	codes   []int32 // time-major, 16 groups per frame
	text    []int32
}

func (m *Model) referenceProfile(ctx context.Context, req llm.SpeechRequest, held *mlx.Scope) (*cloneProfile, error) {
	pcm, rate, err := audio.DecodeReferenceWAV(req.ReferenceAudio.Data)
	if err != nil {
		return nil, speechError("%s", err)
	}
	pcm = audio.Resample(pcm, rate, 24000)
	if len(pcm) < 1280 {
		return nil, speechError("Qwen3-TTS reference audio must contain at least 1280 samples at 24000 Hz")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	profile := &cloneProfile{text: m.tok.Encode("<|im_start|>assistant\n"+req.ReferenceText+"<|im_end|>\n", false)}
	if len(profile.text) < 5 {
		return nil, speechError("reference_text has no transcript tokens")
	}
	profile.text = profile.text[3 : len(profile.text)-2]
	mlx.Scoped(func() {
		x := mlx.FromValues(pcm, len(pcm))
		profile.speaker = m.speaker.forward(m.speaker.mels(x))
		codes := m.encoder.forward(mlx.Reshape(x, 1, 1, int32(len(pcm))))
		codes = mlx.Contiguous(mlx.Transpose(codes, 0, 2, 1).AsType(mlx.DTypeInt32), false)
		mlx.Eval(profile.speaker, codes)
		profile.codes = append([]int32(nil), codes.Ints()...)
		held.Attach(profile.speaker)
	})
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return profile, nil
}

func (n *network) prepareICL(ids []int32, language string, profile *cloneProfile) (*mlx.Array, *mlx.Array, error) {
	if len(ids) < 8 || len(profile.codes) == 0 || len(profile.codes)%16 != 0 {
		return nil, nil, fmt.Errorf("invalid Qwen3-TTS reference conditioning")
	}
	c := n.cfg.Talker
	special := n.project([]int32{n.cfg.TextBOS, n.cfg.TextEOS, n.cfg.TextPad})
	bos, eos, pad := timeSlice(special, 0, 1), timeSlice(special, 1, 2), timeSlice(special, 2, 3)
	combinedIDs := append(append([]int32(nil), profile.text...), ids[3:len(ids)-5]...)
	text := mlx.Concatenate([]*mlx.Array{n.project(combinedIDs), eos}, 1)
	frames := len(profile.codes) / 16
	codes := mlx.FromValues(profile.codes, 1, frames, 16)
	code := func(i int) *mlx.Array {
		return mlx.Squeeze(codes.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(i, i+1)), 2)
	}
	ref := n.codec.Forward(code(0))
	for i := range 15 {
		ref = mlx.Add(ref, n.codeEmb[i].Forward(code(i+1)))
	}
	codec := mlx.Concatenate([]*mlx.Array{n.codec.Forward(mlx.FromValues([]int32{c.BOS}, 1, 1)), ref}, 1)
	codecPad := n.codec.Forward(mlx.FromValues([]int32{c.Pad}, 1, 1))
	text = mlx.Add(text, mlx.BroadcastTo(codecPad, 1, int32(text.Dim(1)), int32(c.Hidden)))
	codec = mlx.Add(codec, mlx.BroadcastTo(pad, 1, int32(codec.Dim(1)), int32(c.Hidden)))
	language = languageName(language)
	prefix := []int32{c.NoThink, c.ThinkBOS, c.ThinkEOS}
	if language != "auto" {
		id, ok := c.Languages[language]
		if !ok {
			return nil, nil, fmt.Errorf("unsupported Qwen3-TTS language %q", language)
		}
		prefix = []int32{c.Think, c.ThinkBOS, id, c.ThinkEOS}
	}
	codecPrefix := n.codec.Forward(mlx.FromValues(prefix, 1, len(prefix)))
	speaker := mlx.Reshape(profile.speaker.AsType(codecPrefix.DType()), 1, 1, int32(c.Hidden))
	suffix := n.codec.Forward(mlx.FromValues([]int32{c.Pad, c.BOS}, 1, 2))
	codecPrefix = mlx.Concatenate([]*mlx.Array{codecPrefix, speaker, suffix}, 1)
	count := codecPrefix.Dim(1)
	combined := mlx.Concatenate([]*mlx.Array{mlx.BroadcastTo(pad, 1, int32(count-2), int32(c.Hidden)), bos}, 1)
	combined = mlx.Add(combined, timeSlice(codecPrefix, 0, count-1))
	return mlx.Concatenate([]*mlx.Array{n.project(ids[:3]), combined, text, codec}, 1), pad, nil
}
