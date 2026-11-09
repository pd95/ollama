package kokoro

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/kokoro/config"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// All waveform networks use [batch, time, channels].
type snakeBlock struct {
	one, two           [3]*convolution
	normOne, normTwo   [3]nn.LinearLayer
	alphaOne, alphaTwo [3]*mlx.Array
}

func loadSnakeBlock(w *weights, p string, kernel int) snakeBlock {
	var b snakeBlock
	for i, dilation := range []int{1, 3, 5} {
		suffix := fmt.Sprint(i)
		b.one[i] = loadWNConv(w, p+".convs1."+suffix, 1, (kernel-1)*dilation/2, dilation, 1, false)
		b.two[i] = loadWNConv(w, p+".convs2."+suffix, 1, (kernel-1)/2, 1, 1, false)
		b.normOne[i] = w.linear(p + ".adain1." + suffix + ".fc")
		b.normTwo[i] = w.linear(p + ".adain2." + suffix + ".fc")
		for _, entry := range []struct {
			path string
			dst  **mlx.Array
		}{{p + ".alpha1." + suffix, &b.alphaOne[i]}, {p + ".alpha2." + suffix, &b.alphaTwo[i]}} {
			if a := w.get(entry.path); a != nil {
				*entry.dst = mlx.Transpose(a, 0, 2, 1)
				w.derived = append(w.derived, *entry.dst)
			}
		}
	}
	return b
}

func snake(x, alpha *mlx.Array) *mlx.Array {
	return mlx.Add(x, mlx.Mul(mlx.Div(mlx.FromValue(float32(1)), alpha), mlx.Sin(mlx.Mul(alpha, x)).Power(mlx.FromValue(float32(2)))))
}

func (b snakeBlock) run(x, style *mlx.Array) *mlx.Array {
	for i := range b.one {
		h := b.one[i].run(snake(styleNorm(x, style, b.normOne[i]), b.alphaOne[i]))
		h = b.two[i].run(snake(styleNorm(h, style, b.normTwo[i]), b.alphaTwo[i]))
		x = addChannels(h, x)
	}
	return x
}

type waveformGenerator struct {
	ups, noiseConvs [2]*convolution
	noise           [2]snakeBlock
	residual        [2][3]snakeBlock
	post            *convolution
	window          *mlx.Array
}

func periodicHann() *mlx.Array {
	// Original torch.hann_window(20), CPU float32. Preserve its coefficients
	// exactly: float64 cosine followed by a cast changes learned phase inputs.
	return mlx.FromValues([]float32{
		0, .024471759796142578, .09549152851104736, .20610737800598145,
		.34549152851104736, .5, .6545085310935974, .7938926219940186,
		.9045085310935974, .9755282998085022, 1, .9755282402038574,
		.9045084714889526, .793892502784729, .6545085906982422, .5,
		.3454914391040802, .2061072587966919, .09549140930175781, .024471759796142578,
	}, 20)
}

func loadGenerator(w *weights) waveformGenerator {
	p := "decoder.generator"
	g := waveformGenerator{post: loadWNConv(w, p+".conv_post", 1, 3, 1, 1, false), window: periodicHann()}
	w.derived = append(w.derived, g.window)
	for i, spec := range [][3]int{{10, 20, 5}, {6, 12, 3}} {
		suffix := fmt.Sprint(i)
		g.ups[i] = loadWNConv(w, p+".ups."+suffix, spec[0], spec[2], 1, 1, true)
		if i == 0 {
			g.noiseConvs[i] = loadConv(w, p+".noise_convs."+suffix, 6, 3, 1, 1)
		} else {
			g.noiseConvs[i] = loadConv(w, p+".noise_convs."+suffix, 1, 0, 1, 1)
		}
		g.noise[i] = loadSnakeBlock(w, p+".noise_res."+suffix, 7+4*i)
		for j, kernel := range []int{3, 7, 11} {
			g.residual[i][j] = loadSnakeBlock(w, fmt.Sprintf("%s.resblocks.%d", p, i*3+j), kernel)
		}
	}
	return g
}

// sourceSpectrum analyzes a single excitation using centered, reflected framing.
func sourceSpectrum(excitation, window *mlx.Array) (*mlx.Array, *mlx.Array) {
	x := mlx.Reshape(excitation, int32(excitation.Size()))
	length := x.Dim(0)
	left, right := make([]int32, 10), make([]int32, 10)
	for i := range left {
		left[i] = int32(10 - i)
		right[i] = int32(length - 2 - i)
	}
	x = mlx.Concatenate([]*mlx.Array{mlx.Take(x, mlx.FromValues(left, 10), 0), x, mlx.Take(x, mlx.FromValues(right, 10), 0)}, 0)
	frames := 1 + (x.Dim(0)-20)/5
	framed := x.AsStrided([]int{frames, 20}, []int{5, 1}, 0)
	transformed := mlx.RFFT(mlx.Mul(framed, window), 20, -1)
	// Real-input FFT endpoints are real. The original backend supplies +0
	// imaginary parts there; GPU signed zeros otherwise choose -pi instead
	// of +pi at negative real endpoints, altering the learned phase input.
	imag := mlx.Imag(transformed)
	imag = mlx.Concatenate([]*mlx.Array{
		mlx.Zeros(mlx.DTypeFloat32, frames, 1),
		imag.Slice(mlx.Slice(), mlx.Slice(1, 10)),
		mlx.Zeros(mlx.DTypeFloat32, frames, 1),
	}, 1)
	return transformed.Abs(), mlx.Atan2(imag, mlx.Real(transformed))
}

// inverseSpectrum receives the generator's sin-bounded phase. Its adjacent
// differences are < pi, so reference phase unwrapping is the identity here.
func inverseSpectrum(magnitude, phase, window *mlx.Array) *mlx.Array {
	spectrum := mlx.Add(mlx.Mul(magnitude, mlx.Cos(phase)), mlx.Mul(mlx.FromValue(complex64(1i)), mlx.Mul(magnitude, mlx.Sin(phase))))
	frames := mlx.IRFFT(spectrum, 20, -1)
	count := frames.Dim(0)
	length := (count-1)*5 + 20
	signal := overlapAdd(mlx.Mul(frames, window))
	norm := mlx.BroadcastTo(mlx.Mul(window, window), int32(count), 20)
	denominator := overlapAdd(norm)
	denominator = mlx.Where(denominator.Greater(mlx.FromValue(float32(1e-10))), denominator, mlx.FromValue(float32(1)))
	return mlx.Div(signal, denominator).Slice(mlx.Slice(10, length-10))
}

// Four window quarters overlap at each hop. Adding them in fixed order avoids
// atomic scatter-add rounding that can change a seeded request's final PCM bit.
func overlapAdd(frames *mlx.Array) *mlx.Array {
	var sum *mlx.Array
	for quarter := range 4 {
		piece := mlx.Reshape(frames.Slice(mlx.Slice(), mlx.Slice(quarter*5, (quarter+1)*5)), -1)
		padded := mlx.Concatenate([]*mlx.Array{mlx.Zeros(mlx.DTypeFloat32, quarter*5), piece, mlx.Zeros(mlx.DTypeFloat32, (3-quarter)*5)}, 0)
		if sum == nil {
			sum = padded
		} else {
			sum = mlx.Add(sum, padded)
		}
	}
	return sum
}

func (g waveformGenerator) spectrum(x, style, excitation *mlx.Array) (*mlx.Array, *mlx.Array) {
	mag, phase := sourceSpectrum(excitation, g.window)
	har := concatChannels(mlx.ExpandDims(mag, 0), mlx.ExpandDims(phase, 0))
	for i := range g.ups {
		source := g.noise[i].run(g.noiseConvs[i].run(har), style)
		x = g.ups[i].run(leaky(x, .1))
		if i == 1 {
			// MLX-Audio 0.6.2's ReflectionPad1d uses constant padding.
			x = padGenerator(x)
		}
		x = addChannels(x, source)
		h := g.residual[i][0].run(x, style)
		for j := 1; j < 3; j++ {
			h = addChannels(h, g.residual[i][j].run(x, style))
		}
		x = mlx.Div(h, mlx.FromValue(float32(3)))
	}
	x = g.post.run(leaky(x, .01))
	magnitude := mlx.Exp(mlx.Squeeze(x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, 11)), 0))
	phase = mlx.Sin(mlx.Squeeze(x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(11, 22)), 0))
	return magnitude, phase
}

func padGenerator(x *mlx.Array) *mlx.Array {
	x = mlx.Transpose(x, 0, 2, 1)
	x = mlx.Concatenate([]*mlx.Array{mlx.Zeros(x.DType(), x.Dim(0), x.Dim(1), 1), x}, 2)
	return mlx.Transpose(x, 0, 2, 1)
}

func (g waveformGenerator) run(x, style, excitation *mlx.Array) *mlx.Array {
	magnitude, phase := g.spectrum(x, style, excitation)
	return inverseSpectrum(magnitude, phase, g.window)
}

// Decoder maps aligned text, pitch, noise and a fixed excitation into 24 kHz PCM.
type Decoder struct {
	encode            resBlock
	blocks            [4]resBlock
	pitch, noise, asr *convolution
	generator         waveformGenerator
	source            nn.LinearLayer
	Derived           []*mlx.Array
}

func LoadDecoder(cfg *config.Config, tensors map[string]*mlx.Array, meta map[string]*model.TensorQuantInfo) (*Decoder, error) {
	w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, cfg.Quant.Group, cfg.Quant.Bits, cfg.Quant.Mode, meta), packed: cfg.Quant.Bits != 0}
	d := &Decoder{encode: loadResBlock(w, "decoder.encode", 514, 1024, false), pitch: loadWNConv(w, "decoder.F0_conv", 2, 1, 1, 1, false), noise: loadWNConv(w, "decoder.N_conv", 2, 1, 1, 1, false), asr: loadWNConv(w, "decoder.asr_res.0", 1, 0, 1, 1, false)}
	for i := range d.blocks {
		output := 1024
		if i == 3 {
			output = 512
		}
		d.blocks[i] = loadResBlock(w, fmt.Sprintf("decoder.decode.%d", i), 1090, output, i == 3)
	}
	d.generator = loadGenerator(w)
	d.source = w.linear("decoder.generator.m_source.l_linear")
	d.Derived = w.derived
	return d, w.err
}

func (d *Decoder) features(asr, pitch, noise, style *mlx.Array) *mlx.Array {
	f := d.pitch.run(mlx.ExpandDims(pitch, 2))
	n := d.noise.run(mlx.ExpandDims(noise, 2))
	x := d.encode.run(concatChannels(asr, f, n), style)
	residual := d.asr.run(asr)
	for _, block := range d.blocks {
		x = block.run(concatChannels(x, residual, f, n), style)
	}
	return x
}

// Match the reference channel-first concatenation layout before reductions.
func concatChannels(arrays ...*mlx.Array) *mlx.Array {
	channelsFirst := make([]*mlx.Array, len(arrays))
	for i, a := range arrays {
		channelsFirst[i] = mlx.Transpose(a, 0, 2, 1)
	}
	return mlx.Transpose(mlx.Concatenate(channelsFirst, 1), 0, 2, 1)
}

func addChannels(x, y *mlx.Array) *mlx.Array {
	return mlx.Transpose(mlx.Add(mlx.Transpose(x, 0, 2, 1), mlx.Transpose(y, 0, 2, 1)), 0, 2, 1)
}

func (d *Decoder) run(asr, pitch, noise, style, excitation *mlx.Array) *mlx.Array {
	return d.generator.run(d.features(asr, pitch, noise, style), style, excitation)
}
