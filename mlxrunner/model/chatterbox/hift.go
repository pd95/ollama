package chatterbox

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type hifiSnake struct{ alpha, inverse *mlx.Array }

func loadHifiSnake(w *weights, p string) hifiSnake {
	alpha := w.get(p + ".alpha")
	if alpha == nil {
		return hifiSnake{}
	}
	alpha = mlx.Reshape(alpha, 1, int32(alpha.Size()), 1)
	abs := alpha.Abs()
	clamped := mlx.Mul(alpha.Sign(), mlx.Maximum(abs, mlx.FromValue(float32(1e-4)).AsType(alpha.DType())))
	clamped = mlx.Where(abs.Less(mlx.FromValue(float32(1e-9)).AsType(alpha.DType())), mlx.FromValue(float32(1e-4)).AsType(alpha.DType()), clamped)
	return hifiSnake{alpha, mlx.Div(mlx.FromValue(float32(1)).AsType(alpha.DType()), clamped)}
}

func (s hifiSnake) forward(x *mlx.Array) *mlx.Array {
	return mlx.Add(x, mlx.Mul(s.inverse, mlx.Sin(mlx.Mul(x, s.alpha)).Power(mlx.FromValue(2))))
}

type hifiUnit struct {
	first, second conv
	one, two      hifiSnake
}
type hifiResidual []hifiUnit

func loadHifiResidual(w *weights, p string, kernel int) hifiResidual {
	var layers hifiResidual
	for i, dilation := range []int32{1, 3, 5} {
		layers = append(layers, hifiUnit{first: w.conv(fmt.Sprintf("%s.convs1.%d", p, i), 1, int32(kernel-1)/2*dilation, dilation, 1), second: w.conv(fmt.Sprintf("%s.convs2.%d", p, i), 1, int32(kernel-1)/2, 1, 1), one: loadHifiSnake(w, fmt.Sprintf("%s.activations1.%d", p, i)), two: loadHifiSnake(w, fmt.Sprintf("%s.activations2.%d", p, i))})
	}
	return layers
}

func (r hifiResidual) forward(x *mlx.Array) *mlx.Array {
	for _, u := range r {
		y := mlx.Transpose(u.first.forward(mlx.Transpose(u.one.forward(x), 0, 2, 1)), 0, 2, 1)
		y = mlx.Transpose(u.second.forward(mlx.Transpose(u.two.forward(y), 0, 2, 1)), 0, 2, 1)
		x = mlx.Add(y, x)
	}
	return x
}

type hiFT struct {
	turbo             bool
	pre, post         conv
	ups, sourceDown   []conv
	sourceBlocks      []hifiResidual
	blocks            [][]hifiResidual
	f0                []conv
	f0Head, sourceMix nn.LinearLayer
	window            *mlx.Array
	derived           []*mlx.Array
}

func loadHiFT(w *weights) *hiFT {
	p := "s3gen.mel2wav"
	h := &hiFT{pre: w.conv(p+".conv_pre", 1, 3, 1, 1), post: w.conv(p+".conv_post", 1, 3, 1, 1), f0Head: w.linear(p + ".f0_predictor.classifier"), sourceMix: w.linear(p + ".m_source.l_linear")}
	for i := range 5 {
		h.f0 = append(h.f0, w.conv(fmt.Sprintf("%s.f0_predictor.condnet.%d", p, i), 1, 1, 1, 1))
	}
	for i, rate := range []int32{8, 5, 3} {
		kernel := []int32{16, 11, 7}[i]
		h.ups = append(h.ups, w.conv(fmt.Sprintf("%s.ups.%d", p, i), rate, (kernel-rate)/2, 1, 1))
		down := []int32{15, 3, 1}[i]
		padding := down / 2
		if down == 1 {
			padding = 0
		}
		h.sourceDown = append(h.sourceDown, w.conv(fmt.Sprintf("%s.source_downs.%d", p, i), down, padding, 1, 1))
		h.sourceBlocks = append(h.sourceBlocks, loadHifiResidual(w, fmt.Sprintf("%s.source_resblocks.%d", p, i), []int{7, 7, 11}[i]))
		var blocks []hifiResidual
		for j, k := range []int{3, 7, 11} {
			blocks = append(blocks, loadHifiResidual(w, fmt.Sprintf("%s.resblocks.%d", p, i*3+j), k))
		}
		h.blocks = append(h.blocks, blocks)
	}
	window := make([]float32, 16)
	for i := range window {
		window[i] = float32(.5 * (1 - math.Cos(2*math.Pi*float64(i)/16)))
	}
	h.window = mlx.FromValues(window, 16)
	h.derived = append(h.derived, h.window)
	for _, blocks := range h.blocks {
		for _, r := range blocks {
			for _, u := range r {
				h.derived = append(h.derived, u.one.alpha, u.one.inverse, u.two.alpha, u.two.inverse)
			}
		}
	}
	for _, r := range h.sourceBlocks {
		for _, u := range r {
			h.derived = append(h.derived, u.one.alpha, u.one.inverse, u.two.alpha, u.two.inverse)
		}
	}
	return h
}

var hifiELU = mlx.Compile1("ChatterboxHiFTELU", func(x *mlx.Array) *mlx.Array {
	return mlx.Where(x.Greater(mlx.FromValue(float32(0))), x, mlx.Sub(mlx.Exp(x), mlx.FromValue(float32(1))))
}, mlx.Shapeless())

func (h *hiFT) pitch(mels *mlx.Array) *mlx.Array {
	x := mlx.Transpose(mels, 0, 2, 1)
	for _, c := range h.f0 {
		x = hifiELU(c.forward(x))
	}
	return mlx.Squeeze(h.f0Head.Forward(x), -1).Abs()
}

func (h *hiFT) harmonics(f0, phase, noise *mlx.Array) *mlx.Array {
	length := f0.Dim(1)
	f0 = mlx.Reshape(mlx.BroadcastTo(mlx.ExpandDims(f0, -1), 1, int32(length), 480), 1, 1, int32(length*480))
	multipliers := mlx.Reshape(mlx.Arange(1, 10, 1, mlx.DTypeInt32), 1, 9, 1)
	frequencies := mlx.Div(mlx.Mul(f0, multipliers), mlx.FromValue(24000))
	theta := mlx.Mul(mlx.FromValue(float32(2*math.Pi)), frequencies.Cumsum(-1, false, true).Remainder(mlx.FromValue(1)))
	if h.turbo {
		theta = mlx.Mul(mlx.FromValue(float32(2*math.Pi)), frequencies.Cumsum(-1, false, true)).Remainder(mlx.FromValue(float32(2 * math.Pi)))
		phase = mlx.Concatenate([]*mlx.Array{mlx.Zeros(mlx.DTypeFloat32, 1, 1, 1), phase}, 1)
	}
	phase = mlx.Where(mlx.Reshape(mlx.Arange(0, 9, 1, mlx.DTypeInt32), 1, 9, 1).Greater(mlx.FromValue(0)), phase, mlx.FromValue(float32(0)))
	sines := mlx.Mul(mlx.FromValue(float32(.1)), mlx.Sin(mlx.Add(theta, phase)))
	voiced := f0.Greater(mlx.FromValue(float32(10))).AsType(mlx.DTypeFloat32)
	amplitude := mlx.Add(mlx.Mul(voiced, mlx.FromValue(float32(.003))), mlx.Div(mlx.Mul(mlx.Sub(mlx.FromValue(float32(1)), voiced), mlx.FromValue(float32(.1))), mlx.FromValue(3)))
	return mlx.Add(mlx.Mul(sines, voiced), mlx.Mul(amplitude, noise))
}

func (h *hiFT) excitation(f0, phase, noise *mlx.Array) *mlx.Array {
	x := mlx.Transpose(h.harmonics(f0, phase, noise), 0, 2, 1)
	return mlx.Transpose(h.sourceMix.Forward(x).Tanh(), 0, 2, 1)
}

func (h *hiFT) sourceSTFT(source *mlx.Array) (*mlx.Array, *mlx.Array) {
	x := mlx.Squeeze(source, 1)
	if !h.turbo {
		x = mlx.Pad(x, []int{1}, []int{8}, []int{8}, mlx.FromValue(float32(0)), "reflect")
	}
	count := 1 + (x.Dim(1)-16)/4
	indices := make([]int32, count*16)
	for i := range count {
		for j := range 16 {
			indices[i*16+j] = int32(i*4 + j)
		}
	}
	frames := mlx.Transpose(mlx.Reshape(mlx.Take(x, mlx.FromValues(indices, len(indices)), 1), 1, int32(count), 16), 0, 2, 1)
	frames = mlx.Mul(frames, mlx.Reshape(h.window, 1, 16, 1))
	if h.turbo {
		return turboSourceFFT(frames)
	}
	fft := mlx.FFT(frames, 16, 1).Slice(mlx.Slice(), mlx.Slice(0, 9), mlx.Slice())
	return mlx.Real(fft), mlx.Imag(fft)
}

func hifiLeaky(x *mlx.Array) *mlx.Array {
	return mlx.Maximum(x, mlx.Mul(x, mlx.FromValue(float32(.1))))
}

func (h *hiFT) stage(x, source *mlx.Array, index int) *mlx.Array {
	x = mlx.Transpose(hifiLeaky(x), 0, 2, 1)
	c := h.ups[index]
	x = mlx.ConvTranspose1d(x, c.weight, c.bias, c.stride, c.padding, 1, 0, 1)
	x = mlx.Transpose(x, 0, 2, 1)
	if index == 2 {
		if h.turbo {
			x = mlx.PadConstant(x, []int{2}, []int{1}, []int{0})
		} else {
			x = mlx.Concatenate([]*mlx.Array{x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(1, 2)), x}, 2)
		}
	}
	part := mlx.Transpose(h.sourceDown[index].forward(mlx.Transpose(source, 0, 2, 1)), 0, 2, 1)
	part = h.sourceBlocks[index].forward(part)
	if h.turbo {
		length := min(x.Dim(2), part.Dim(2))
		x = x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, length))
		part = part.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, length))
	}
	x = mlx.Add(x, part)
	outputs := make([]*mlx.Array, 3)
	for i, b := range h.blocks[index] {
		outputs[i] = b.forward(x)
	}
	if h.turbo {
		return mlx.Div(mlx.Add(mlx.Add(outputs[0], outputs[1]), outputs[2]), mlx.FromValue(float32(3)))
	}
	return mlx.Mean(mlx.Stack(outputs, 0), 0, false)
}

func (h *hiFT) spectrum(mels, source *mlx.Array) *mlx.Array {
	real, imag := h.sourceSTFT(source)
	spec := mlx.Concatenate([]*mlx.Array{real, imag}, 1)
	x := mlx.Transpose(h.pre.forward(mlx.Transpose(mels, 0, 2, 1)), 0, 2, 1)
	for i := range h.ups {
		x = h.stage(x, spec, i)
	}
	if h.turbo {
		x = mlx.Maximum(x, mlx.Mul(x, mlx.FromValue(float32(.01))))
	} else {
		x = hifiLeaky(x)
	}
	return mlx.Transpose(h.post.forward(mlx.Transpose(x, 0, 2, 1)), 0, 2, 1)
}

func hifiOverlap(frames *mlx.Array) *mlx.Array {
	count := frames.Dim(2)
	frames = mlx.Transpose(frames, 0, 2, 1)
	var sum *mlx.Array
	for part := range 4 {
		x := mlx.Reshape(frames.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(part*4, (part+1)*4)), 1, int32(count*4))
		x = mlx.PadConstant(x, []int{1}, []int{part * 4}, []int{(3 - part) * 4})
		if sum == nil {
			sum = x
		} else {
			sum = mlx.Add(sum, x)
		}
	}
	return sum
}

func (h *hiFT) inverse(spectrum *mlx.Array) *mlx.Array {
	count := spectrum.Dim(2)
	magnitude := mlx.Minimum(mlx.Exp(spectrum.Slice(mlx.Slice(), mlx.Slice(0, 9), mlx.Slice())), mlx.FromValue(float32(100)))
	phase := mlx.Sin(spectrum.Slice(mlx.Slice(), mlx.Slice(9, 18), mlx.Slice()))
	real, imag := mlx.Mul(magnitude, mlx.Cos(phase)), mlx.Mul(magnitude, mlx.Sin(phase))
	mirror := func(x *mlx.Array) *mlx.Array { return x.Slice(mlx.Slice(), mlx.Slice(7, 0, -1), mlx.Slice()) }
	real = mlx.Concatenate([]*mlx.Array{real, mirror(real)}, 1)
	imag = mlx.Concatenate([]*mlx.Array{imag, mlx.Neg(mirror(imag))}, 1)
	complex := mlx.Add(real, mlx.Mul(mlx.FromValue(complex64(1i)), imag))
	frames := mlx.Real(mlx.IFFT(complex, 16, 1))
	if h.turbo {
		frames = mlx.Transpose(mlx.IRFFT(mlx.Transpose(mlx.Add(real.Slice(mlx.Slice(), mlx.Slice(0, 9), mlx.Slice()), mlx.Mul(mlx.FromValue(complex64(1i)), imag.Slice(mlx.Slice(), mlx.Slice(0, 9), mlx.Slice()))), 0, 2, 1), 16, -1), 0, 2, 1)
	}
	window := mlx.Reshape(h.window, 1, 16, 1)
	frames = mlx.Mul(frames, window)
	output := hifiOverlap(frames)
	square := h.window.Power(mlx.FromValue(2))
	if h.turbo {
		square = square.AsType(mlx.DTypeFloat32)
	}
	norm := hifiOverlap(mlx.BroadcastTo(square.Reshape(1, 16, 1), 1, 16, int32(count)))
	output = mlx.Div(output, mlx.Maximum(norm, mlx.FromValue(float32(1e-8))))
	output = output.Slice(mlx.Slice(), mlx.Slice(8, output.Dim(1)-8))
	return mlx.Clip(output, mlx.FromValue(float32(-.99)), mlx.FromValue(float32(.99)))
}

func (h *hiFT) decode(mels, source *mlx.Array) *mlx.Array { return h.inverse(h.spectrum(mels, source)) }

func fadeHiFT(wave *mlx.Array) *mlx.Array {
	if wave.Dim(1) < 960 {
		return wave
	}
	fade := mlx.Div(mlx.Add(mlx.Cos(mlx.Linspace(math.Pi, 0, 480, mlx.DTypeFloat32)), mlx.FromValue(float32(1))), mlx.FromValue(float32(2)))
	fade = mlx.Concatenate([]*mlx.Array{mlx.Zeros(mlx.DTypeFloat32, 480), fade}, 0)
	return mlx.Concatenate([]*mlx.Array{mlx.Mul(wave.Slice(mlx.Slice(), mlx.Slice(0, 960)), fade), wave.Slice(mlx.Slice(), mlx.Slice(960, wave.Dim(1)))}, 1)
}

// Turbo's reference computes the 16-point source FFT with NumPy's double
// intermediates before rounding to float32. Keep that DSP boundary explicit.
func turboSourceFFT(frames *mlx.Array) (*mlx.Array, *mlx.Array) {
	length := frames.Dim(2)
	input := mlx.Contiguous(frames.AsType(mlx.DTypeFloat32), false).Floats()
	real, imag := make([]float32, 9*length), make([]float32, 9*length)
	for bin := range 9 {
		var cosine, sine [16]float64
		a, b, c := math.Cos(math.Pi/8), math.Sqrt(.5), math.Sin(math.Pi/8)
		table := [16]float64{1, a, b, c, 0, -c, -b, -a, -1, -a, -b, -c, 0, c, b, a}
		for n := range 16 {
			k := bin * n % 16
			cosine[n], sine[n] = table[k], -table[(4-k+16)%16]
		}

		for i := range length {
			var r, j float64
			for n := range 16 {
				v := float64(input[n*length+i])
				r += v * cosine[n]
				j += v * sine[n]
			}
			real[bin*length+i], imag[bin*length+i] = float32(r), float32(j)
		}
	}
	return mlx.FromValues(real, 1, 9, length), mlx.FromValues(imag, 1, 9, length)
}
