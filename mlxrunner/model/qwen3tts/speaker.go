package qwen3tts

import (
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type speakerConfig struct {
	Dimension int   `json:"enc_dim"`
	Rate      int   `json:"sample_rate"`
	Mel       int   `json:"mel_dim"`
	Channels  []int `json:"enc_channels"`
	Kernels   []int `json:"enc_kernel_sizes"`
	Dilations []int `json:"enc_dilations"`
	Attention int   `json:"enc_attention_channels"`
	Scale     int   `json:"enc_res2net_scale"`
	SE        int   `json:"enc_se_channels"`
}

func defaultSpeakerConfig() speakerConfig {
	return speakerConfig{Dimension: 1024, Rate: 24000, Mel: 128, Channels: []int{512, 512, 512, 512, 1536}, Kernels: []int{5, 3, 3, 3, 1}, Dilations: []int{1, 2, 3, 4, 1}, Attention: 128, Scale: 8, SE: 128}
}

func (c speakerConfig) validate(hidden int) error {
	d := defaultSpeakerConfig()
	if (c.Dimension != 1024 && c.Dimension != 2048) || c.Dimension != hidden || c.Rate != d.Rate || c.Mel != d.Mel || c.Attention != d.Attention || c.Scale != d.Scale || c.SE != d.SE || !slices.Equal(c.Channels, d.Channels) || !slices.Equal(c.Kernels, d.Kernels) || !slices.Equal(c.Dilations, d.Dilations) {
		return fmt.Errorf("unsupported Qwen3-TTS speaker encoder configuration")
	}
	return nil
}

type tdnn struct {
	conv    *convolution
	padding int
}

func (b tdnn) forward(x *mlx.Array) *mlx.Array {
	x = mlx.Transpose(x, 0, 2, 1)
	if b.padding > 0 {
		x = mlx.Pad(x, []int{1}, []int{b.padding}, []int{b.padding}, mlx.FromValue(float32(0)), "reflect")
	}
	x = mlx.Conv1d(x, b.conv.weight, b.conv.bias, 1, 0, int32(b.conv.dilation), 1)
	return mlx.ReLU(mlx.Transpose(x, 0, 2, 1))
}

type speakerResidual struct {
	one, two     tdnn
	parts        []tdnn
	seOne, seTwo *convolution
}

func (b speakerResidual) forward(x *mlx.Array) *mlx.Array {
	residual := x
	x = b.one.forward(x)
	parts := make([]*mlx.Array, 8)
	for i := range parts {
		part := x.Slice(mlx.Slice(), mlx.Slice(i*64, (i+1)*64), mlx.Slice())
		if i > 0 {
			if i > 1 {
				part = mlx.Add(part, parts[i-1])
			}
			part = b.parts[i-1].forward(part)
		}
		parts[i] = part
	}
	x = b.two.forward(mlx.Concatenate(parts, 1))
	se := mlx.Transpose(mlx.Mean(x, 2, true), 0, 2, 1)
	se = mlx.ReLU(mlx.Conv1d(se, b.seOne.weight, b.seOne.bias, 1, 0, 1, 1))
	se = mlx.Sigmoid(mlx.Conv1d(se, b.seTwo.weight, b.seTwo.bias, 1, 0, 1, 1))
	return mlx.Add(mlx.Mul(x, mlx.Transpose(se, 0, 2, 1)), residual)
}

type speakerEncoder struct {
	initial         tdnn
	blocks          []speakerResidual
	mfa, attention  tdnn
	pool, fc        *convolution
	melBank, window *mlx.Array
	derived         []*mlx.Array
}

func loadSpeaker(cfg speakerConfig, tensors map[string]*mlx.Array) (*speakerEncoder, error) {
	if err := cfg.validate(cfg.Dimension); err != nil {
		return nil, err
	}
	w := &weights{tensors: tensors}
	var derived []*mlx.Array
	conv := func(name string, input, output, kernel, dilation int) *convolution {
		p := "speaker_encoder." + name
		a, b := w.get(p+".weight"), w.get(p+".bias")
		if a == nil || b == nil {
			return nil
		}
		// Original PyTorch checkpoints retain OIK. The MLX conversion uses
		// OKI; the fixed architecture's channel/kernel sizes distinguish them.
		if slices.Equal(a.Dims(), []int{output, input, kernel}) {
			a = mlx.Contiguous(mlx.Transpose(a, 0, 2, 1), false)
			derived = append(derived, a)
		}
		if !slices.Equal(a.Dims(), []int{output, kernel, input}) || !slices.Equal(b.Dims(), []int{output}) {
			w.err = fmt.Errorf("Qwen3-TTS speaker convolution %s has incompatible shape", p)
			return nil
		}
		return &convolution{weight: a, bias: b, dilation: dilation, groups: 1, stride: 1}
	}
	td := func(name string, input, output, kernel, dilation int) tdnn {
		return tdnn{conv(name+".conv", input, output, kernel, dilation), (kernel - 1) * dilation / 2}
	}
	s := &speakerEncoder{initial: td("blocks.0", 128, 512, 5, 1)}
	for i, dilation := range []int{2, 3, 4} {
		p := fmt.Sprintf("blocks.%d", i+1)
		b := speakerResidual{one: td(p+".tdnn1", 512, 512, 1, 1), two: td(p+".tdnn2", 512, 512, 1, 1), seOne: conv(p+".se_block.conv1", 512, 128, 1, 1), seTwo: conv(p+".se_block.conv2", 128, 512, 1, 1)}
		for j := range 7 {
			b.parts = append(b.parts, td(fmt.Sprintf("%s.res2net_block.blocks.%d", p, j), 64, 64, 3, dilation))
		}
		s.blocks = append(s.blocks, b)
	}
	s.mfa = td("mfa", 1536, 1536, 1, 1)
	s.attention = td("asp.tdnn", 4608, 128, 1, 1)
	s.pool = conv("asp.conv", 128, 1536, 1, 1)
	s.fc = conv("fc", 3072, cfg.Dimension, 1, 1)
	if w.err != nil {
		return nil, w.err
	}
	s.melBank = speakerMelBank()
	window := make([]float32, 1024)
	for i := range window {
		window[i] = float32(.5 * (1 - math.Cos(2*math.Pi*float64(i)/1023)))
	}
	s.window = mlx.FromValues(window, 1024)
	s.derived = append(derived, s.melBank, s.window)
	return s, nil
}

// Match the float32 Slaney construction used by the reference DSP module.
func speakerMelBank() *mlx.Array { return nn.SlaneyMelFilters(24000, 1024, 128, 12000) }

func (s *speakerEncoder) mels(audio *mlx.Array) *mlx.Array {
	length := audio.Size()
	padded := mlx.Pad(audio, []int{0}, []int{384}, []int{384}, mlx.FromValue(float32(0)), "reflect")
	frames := padded.AsStrided([]int{length / 256, 1024}, []int{256, 1}, 0)
	spec := mlx.RFFT(mlx.Mul(frames, s.window), 1024, -1)
	mag := mlx.Add(spec.Abs().Power(mlx.FromValue(2)), mlx.FromValue(float32(1e-9))).Sqrt()
	mel := mlx.Matmul(mag, mlx.Transpose(s.melBank, 1, 0))
	return mlx.ExpandDims(mlx.Log(mlx.Maximum(mel, mlx.FromValue(float32(1e-5)))), 0)
}

func (s *speakerEncoder) features(mels *mlx.Array) *mlx.Array {
	x := s.initial.forward(mlx.Transpose(mels, 0, 2, 1))
	parts := make([]*mlx.Array, 0, 3)
	for _, b := range s.blocks {
		x = b.forward(x)
		parts = append(parts, x)
	}
	return s.mfa.forward(mlx.Concatenate(parts, 1))
}

func (s *speakerEncoder) statistics(x *mlx.Array) *mlx.Array {
	mean := mlx.Mean(x, 2, true)
	std := mlx.Add(mlx.Variance(x, 2, true), mlx.FromValue(float32(1e-12))).Sqrt()
	shape := []int32{int32(x.Dim(0)), int32(x.Dim(1)), int32(x.Dim(2))}
	a := s.attention.forward(mlx.Concatenate([]*mlx.Array{x, mlx.BroadcastTo(mean, shape...), mlx.BroadcastTo(std, shape...)}, 1)).Tanh()
	a = mlx.Transpose(mlx.Conv1d(mlx.Transpose(a, 0, 2, 1), s.pool.weight, s.pool.bias, 1, 0, 1, 1), 0, 2, 1)
	a = mlx.SoftmaxAxis(a, 2, false)
	mean = mlx.Sum(mlx.Mul(a, x), 2, true)
	variance := mlx.Sum(mlx.Mul(a, mlx.Sub(x, mean).Power(mlx.FromValue(2))), 2, true)
	std = mlx.Maximum(variance, mlx.FromValue(float32(1e-12))).Sqrt()
	return mlx.Concatenate([]*mlx.Array{mean, std}, 1)
}

func (s *speakerEncoder) forward(mels *mlx.Array) *mlx.Array {
	x := mlx.Transpose(s.statistics(s.features(mels)), 0, 2, 1)
	return mlx.Squeeze(mlx.Conv1d(x, s.fc.weight, s.fc.bias, 1, 0, 1, 1), 1)
}
