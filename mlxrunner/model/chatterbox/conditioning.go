package chatterbox

import (
	"context"
	"math"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type referenceMel struct {
	bank, window *mlx.Array
	turbo        bool
}

func newReferenceMel() referenceMel {
	window := make([]float32, 1920)
	for i := range window {
		window[i] = float32(.5 * (1 - math.Cos(2*math.Pi*float64(i)/1919)))
	}
	return referenceMel{bank: nn.SlaneyMelFilters(24000, 1920, 80, 8000), window: mlx.FromValues(window, 1920)}
}

func (m referenceMel) forward(pcm *mlx.Array) *mlx.Array {
	x := mlx.Pad(pcm, []int{0}, []int{720}, []int{720}, mlx.FromValue(float32(0)), "reflect")
	count := 1 + (x.Size()-1920)/480
	frames := x.AsStrided([]int{count, 1920}, []int{480, 1}, 0)
	spectrum := mlx.RFFT(mlx.Mul(frames, m.window), 1920, -1).Abs()
	if m.turbo {
		spectrum = mlx.RFFTCPU(mlx.Mul(frames, m.window), 1920, -1).Abs()
		return mlx.ExpandDims(mlx.Log(mlx.Maximum(mlx.Transpose(mlx.MatmulCPU(m.bank, mlx.Transpose(spectrum, 1, 0)), 1, 0), mlx.FromValue(float32(1e-5)))), 0)
	}
	return mlx.ExpandDims(mlx.Log(mlx.Maximum(mlx.Matmul(spectrum, mlx.Transpose(m.bank, 1, 0)), mlx.FromValue(float32(1e-5)))), 0)
}

type cloneProfile struct{ conditioning, tokens, mels, speaker *mlx.Array }

func (m *Model) referenceProfile(ctx context.Context, req llm.SpeechRequest, held *mlx.Scope) (*cloneProfile, error) {
	pcm, rate, err := audio.DecodeReferenceWAV(req.ReferenceAudio.Data)
	if err != nil {
		return nil, badRequest("%s", err)
	}
	if len(pcm) < rate/10 {
		return nil, badRequest("Chatterbox reference audio must be at least 0.1 seconds")
	}
	pcm24 := audio.ResampleKaiserBest(pcm, rate, 24000)
	pcm24 = pcm24[:min(len(pcm24), 10*24000)]
	gen16 := audio.ResampleKaiserBest(pcm24, 24000, 16000)
	voice16 := audio.ResampleKaiserBest(pcm, rate, 16000)
	prompt16 := voice16[:min(len(voice16), 6*16000)]
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	profile := &cloneProfile{}
	mlx.Scoped(func() {
		tokenize := func(pcm []float32) *mlx.Array {
			return m.s3.quantize(m.s3.features(m.s3.mels(mlx.FromValues(pcm, len(pcm)))))
		}
		profile.tokens = tokenize(gen16)
		profile.mels = m.mel.forward(mlx.FromValues(pcm24, len(pcm24)))
		expected := profile.mels.Dim(1) / 2
		if profile.tokens.Dim(1) < expected {
			profile.mels = profile.mels.Slice(mlx.Slice(), mlx.Slice(0, 2*profile.tokens.Dim(1)), mlx.Slice())
		} else if profile.tokens.Dim(1) > expected {
			profile.tokens = profile.tokens.Slice(mlx.Slice(), mlx.Slice(0, expected))
		}
		features := m.camp.fbank(mlx.FromValues(gen16, len(gen16)))
		features = mlx.ExpandDims(mlx.Sub(features, mlx.Mean(features, 0, true)), 0)
		profile.speaker = m.camp.forward(features)
		trimmed := trimVoiceReference(voice16)
		speaker := m.voice.embedding(m.voice.mels(mlx.FromValues(trimmed, len(trimmed))))
		codes := tokenize(prompt16)
		codes = codes.Slice(mlx.Slice(), mlx.Slice(0, min(150, codes.Dim(1))))
		profile.conditioning = m.t3.conditioning(speaker, codes, mlx.FromValues([]float32{.1}, 1, 1, 1))
		mlx.Eval(profile.tokens, profile.mels, profile.speaker, profile.conditioning)
		held.Attach(profile.tokens, profile.mels, profile.speaker, profile.conditioning)
	})
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return profile, nil
}
