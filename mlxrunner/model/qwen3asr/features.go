// Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
// SPDX-License-Identifier: Apache-2.0
// Native Go/MLX adaptations for Ollama.

package qwen3asr

import (
	"context"
	"fmt"
	"math"
	"math/cmplx"
	"sync"
)

type features struct {
	Data   []float32
	Frames int
}

// Whisper uses a periodic Hann, centered reflection padding, power spectrum,
// Slaney-normalized filters, log10 floor and a clip-wide 80 dB dynamic range.
// This CPU path keeps PrepareMedia independent of the MLX execution thread.
var frontend = sync.OnceValue(func() struct {
	window, filters []float64
	twiddle         []complex128
} {
	out := struct {
		window, filters []float64
		twiddle         []complex128
	}{make([]float64, 400), make([]float64, 128*201), make([]complex128, 400)}
	for i := range 400 {
		out.window[i] = .5 - .5*math.Cos(2*math.Pi*float64(i)/400)
		out.twiddle[i] = cmplx.Rect(1, -2*math.Pi*float64(i)/400)
	}
	top := 15 + math.Log(8)/(math.Log(6.4)/27)
	edges := make([]float64, 130)
	for i := range edges {
		mel := top * float64(i) / 129
		if mel < 15 {
			edges[i] = mel * 200 / 3
		} else {
			edges[i] = 1000 * math.Exp((mel-15)*math.Log(6.4)/27)
		}
	}
	for m := range 128 {
		for k := range 201 {
			hz := float64(k) * 40
			out.filters[m*201+k] = max(0, min((hz-edges[m])/(edges[m+1]-edges[m]), (edges[m+2]-hz)/(edges[m+2]-edges[m+1]))) * 2 / (edges[m+2] - edges[m])
		}
	}
	return out
})

// Mixed-radix Cooley-Tukey; the 400-point transform is not radix two.
func fft(out, in []complex128, stride int, twiddle []complex128, scratch []complex128) {
	n := len(out)
	if n == 1 {
		out[0] = in[0]
		return
	}
	radix := 2
	for n%radix != 0 {
		radix++
	}
	width := n / radix
	for r := range radix {
		fft(out[r*width:(r+1)*width], in[r*stride:], stride*radix, twiddle, scratch)
	}
	for k := range width {
		for r := range radix {
			scratch[r] = out[r*width+k]
		}
		for q := range radix {
			i := k + q*width
			var sum complex128
			for r := range radix {
				sum += scratch[r] * twiddle[(r*i*(400/n))%400]
			}
			out[i] = sum
		}
	}
}

func extractFeatures(samples []float32) (features, error) {
	return extractFeaturesContext(context.Background(), samples)
}

func extractFeaturesContext(ctx context.Context, samples []float32) (features, error) {
	if err := ctx.Err(); err != nil {
		return features{}, err
	}
	// Bound eager encoder attention and decoder prefill for complete recordings.
	if len(samples) < 201 || len(samples) > maxAudioSeconds*16000 {
		return features{}, fmt.Errorf("Qwen3-ASR requires audio between 201 samples and %d seconds at 16 kHz", maxAudioSeconds)
	}
	peak := float32(1)
	for i, v := range samples {
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return features{}, err
			}
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return features{}, fmt.Errorf("audio contains non-finite samples")
		}
		peak = max(peak, float32(math.Abs(float64(v))))
	}
	f := features{Frames: len(samples) / 160}
	f.Data = make([]float32, 128*f.Frames)
	setup := frontend()
	in, out := make([]complex128, 400), make([]complex128, 400)
	scratch := make([]complex128, 5)
	power := make([]float64, 201)
	maximum := float32(-10)
	for frame := range f.Frames {
		if err := ctx.Err(); err != nil {
			return features{}, err
		}
		for i := range 400 {
			idx := frame*160 + i - 200
			if idx < 0 {
				idx = -idx
			}
			if idx >= len(samples) {
				idx = 2*len(samples) - 2 - idx
			}
			in[i] = complex(float64(samples[idx]/peak)*setup.window[i], 0)
		}
		fft(out, in, 1, setup.twiddle, scratch)
		for k := range power {
			power[k] = real(out[k])*real(out[k]) + imag(out[k])*imag(out[k])
		}
		for m := range 128 {
			var mel float64
			for k, p := range power {
				mel += setup.filters[m*201+k] * p
			}
			v := float32(math.Log10(max(mel, 1e-10)))
			f.Data[m*f.Frames+frame] = v
			maximum = max(maximum, v)
		}
	}
	for i, v := range f.Data {
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return features{}, err
			}
		}
		f.Data[i] = (max(v, maximum-8) + 4) / 4
	}
	return f, nil
}
