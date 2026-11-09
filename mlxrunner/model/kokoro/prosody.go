package kokoro

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/kokoro/config"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func loadConv(w *weights, p string, stride, padding, dilation, groups int) *convolution {
	weight := w.get(p + ".weight")
	if weight == nil {
		return nil
	}
	if !w.packed {
		weight = mlx.Transpose(weight, 0, 2, 1)
		w.derived = append(w.derived, weight)
	}
	return &convolution{weight: weight, bias: w.tensors[p+".bias"], stride: int32(stride), padding: int32(padding), dilation: int32(dilation), groups: int32(groups)}
}

func styleNorm(x, style *mlx.Array, projection nn.LinearLayer) *mlx.Array {
	channels := x.Dim(2)
	x = mlx.Transpose(x, 0, 2, 1)
	h := projection.Forward(style)
	gamma := mlx.Reshape(h.Slice(mlx.Slice(), mlx.Slice(0, channels)), int32(x.Dim(0)), int32(channels), 1)
	beta := mlx.Reshape(h.Slice(mlx.Slice(), mlx.Slice(channels, 2*channels)), int32(x.Dim(0)), int32(channels), 1)
	norm := mlx.Div(mlx.Sub(x, mlx.Mean(x, 2, true)), mlx.Add(mlx.Variance(x, 2, true), mlx.FromValue(float32(1e-5))).Sqrt())
	return mlx.Transpose(mlx.Add(mlx.Mul(mlx.Add(mlx.FromValue(float32(1)), gamma), norm), beta), 0, 2, 1)
}

func repeatTime(x *mlx.Array, factor int) *mlx.Array {
	b, l, c := x.Dim(0), x.Dim(1), x.Dim(2)
	return mlx.Reshape(mlx.BroadcastTo(mlx.ExpandDims(x, 2), int32(b), int32(l), int32(factor), int32(c)), int32(b), int32(l*factor), int32(c))
}

type resBlock struct {
	one, two, shortcut, pool *convolution
	normOne, normTwo         nn.LinearLayer
	up                       bool
}

func loadResBlock(w *weights, p string, input, output int, up bool) resBlock {
	r := resBlock{one: loadWNConv(w, p+".conv1", 1, 1, 1, 1, false), two: loadWNConv(w, p+".conv2", 1, 1, 1, 1, false), normOne: w.linear(p + ".norm1.fc"), normTwo: w.linear(p + ".norm2.fc"), up: up}
	if input != output {
		r.shortcut = loadWNConv(w, p+".conv1x1", 1, 0, 1, 1, false)
	}
	if up {
		r.pool = loadWNConv(w, p+".pool", 2, 0, 1, input, true)
	}
	return r
}

func (r resBlock) run(x, style *mlx.Array) *mlx.Array {
	shortcut := x
	if r.up {
		shortcut = repeatTime(shortcut, 2)
	}
	if r.shortcut != nil {
		shortcut = r.shortcut.run(shortcut)
	}
	h := leaky(styleNorm(x, style, r.normOne), .2)
	if r.up {
		h = r.pool.run(h).Slice(mlx.Slice(), mlx.Slice(1, mlx.End), mlx.Slice())
	}
	h = r.one.run(h)
	h = r.two.run(leaky(styleNorm(h, style, r.normTwo), .2))
	return mlx.Div(mlx.Add(h, shortcut), mlx.FromValue(float32(math.Sqrt2)))
}

// Prosody predicts the pitch and noise curves from aligned phoneme features.
type Prosody struct {
	f0, noise       []resBlock
	f0Out, noiseOut *convolution
	Derived         []*mlx.Array
}

func LoadProsody(cfg *config.Config, tensors map[string]*mlx.Array, meta map[string]*model.TensorQuantInfo) (*Prosody, error) {
	w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, cfg.Quant.Group, cfg.Quant.Bits, cfg.Quant.Mode, meta), packed: cfg.Quant.Bits != 0}
	p := &Prosody{f0Out: loadConv(w, "predictor.F0_proj", 1, 0, 1, 1), noiseOut: loadConv(w, "predictor.N_proj", 1, 0, 1, 1)}
	for _, branch := range []string{"F0", "N"} {
		var blocks []resBlock
		for i, dims := range [][2]int{{512, 512}, {512, 256}, {256, 256}} {
			blocks = append(blocks, loadResBlock(w, fmt.Sprintf("predictor.%s.%d", branch, i), dims[0], dims[1], i == 1))
		}
		if branch == "F0" {
			p.f0 = blocks
		} else {
			p.noise = blocks
		}
	}
	p.Derived = w.derived
	return p, w.err
}

func (p *Prosody) run(shared, style *mlx.Array) (*mlx.Array, *mlx.Array) {
	f0, n := shared, shared
	for _, block := range p.f0 {
		f0 = block.run(f0, style)
	}
	for _, block := range p.noise {
		n = block.run(n, style)
	}
	return mlx.Squeeze(p.f0Out.run(f0), 2), mlx.Squeeze(p.noiseOut.run(n), 2)
}
