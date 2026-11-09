package qwen3tts

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type decoderConfig struct {
	stackConfig
	Latent     int   `json:"latent_dim"`
	Codebook   int   `json:"codebook_dim"`
	Channels   int   `json:"decoder_dim"`
	Quantizers int   `json:"num_quantizers"`
	Semantic   int   `json:"num_semantic_quantizers"`
	Rates      []int `json:"upsample_rates"`
	Ratios     []int `json:"upsampling_ratios"`
}
type convolution struct {
	weight, bias             *mlx.Array
	dilation, groups, stride int
	transpose                bool
}

func loadConv(w *weights, p string, dilation, groups, stride int, transpose bool) *convolution {
	weight := w.get(p + ".weight")
	if weight == nil {
		return nil
	}
	if weight.NumDims() != 3 {
		w.err = fmt.Errorf("Qwen3-TTS convolution %s must have rank 3", p)
		return nil
	}
	// Published speech-tokenizer convolutions use PyTorch OIK (normal) or IOK
	// (transposed) layout. Keep the import lossless and normalize at load time.
	if transpose {
		weight = mlx.Transpose(weight, 1, 2, 0)
	} else {
		weight = mlx.Transpose(weight, 0, 2, 1)
	}
	w.derived = append(w.derived, weight)
	return &convolution{weight: weight, bias: w.tensors[p+".bias"], dilation: dilation, groups: groups, stride: stride, transpose: transpose}
}

func (c *convolution) forward(x *mlx.Array) *mlx.Array {
	kernel := c.weight.Dim(1)
	if c.transpose {
		l := x.Dim(1)
		x = mlx.ConvTranspose1d(x, c.weight, c.bias, int32(c.stride), 0, 1, 0, 1)
		return timeSlice(x, 0, l*c.stride)
	}
	padding := (kernel - 1) * c.dilation
	if padding > 0 {
		x = mlx.PadConstant(x, []int{1}, []int{padding}, []int{0})
	}
	return mlx.Conv1d(x, c.weight, c.bias, 1, 0, int32(c.dilation), int32(c.groups))
}

type snake struct{ alpha, beta *mlx.Array }

func loadSnake(w *weights, p string) snake {
	a, b := w.get(p+".alpha"), w.get(p+".beta")
	if a == nil || b == nil {
		return snake{}
	}
	s := snake{mlx.Exp(a), mlx.Add(mlx.Exp(b), mlx.FromValue(float32(1e-9)))}
	w.derived = append(w.derived, s.alpha, s.beta)
	return s
}

func (s snake) forward(x *mlx.Array) *mlx.Array {
	sin := mlx.Sin(mlx.Mul(x, s.alpha))
	return mlx.Add(x, mlx.Mul(mlx.Div(mlx.FromValue(float32(1)), s.beta), sin.Power(mlx.FromValue(2))))
}

type residual struct {
	a, b     snake
	one, two *convolution
}

func (r residual) forward(x *mlx.Array) *mlx.Array {
	return mlx.Add(x, r.two.forward(r.b.forward(r.one.forward(r.a.forward(x)))))
}

type upBlock struct {
	activation snake
	up         *convolution
	units      []residual
}

func (b upBlock) forward(x *mlx.Array) *mlx.Array {
	x = b.up.forward(b.activation.forward(x))
	for _, unit := range b.units {
		x = unit.forward(x)
	}
	return x
}

type convNext struct {
	depth    *convolution
	norm     *nn.LayerNorm
	one, two nn.LinearLayer
	gamma    *mlx.Array
}

func (c convNext) forward(x *mlx.Array) *mlx.Array {
	return mlx.Add(x, mlx.Mul(c.gamma, c.two.Forward(speechGELU(c.one.Forward(c.norm.Forward(c.depth.forward(x)))))))
}

// Preserve MLX-Audio's operation order in the sensitive vocoder.
func speechGELU(x *mlx.Array) *mlx.Array {
	return mlx.Div(mlx.Mul(x, mlx.Add(mlx.FromValue(float32(1)), mlx.Erf(mlx.Div(x, mlx.FromValue(float32(math.Sqrt2)))))), mlx.FromValue(float32(2)))
}

type auxiliaryUp struct {
	up    *convolution
	block convNext
}
type decoder struct {
	cfg                     decoderConfig
	books                   []*mlx.Array
	semantic, acoustic, pre *convolution
	transformer             *stack
	in, out                 nn.LinearLayer
	ups                     []auxiliaryUp
	initial, final          *convolution
	blocks                  []upBlock
	last                    snake
	derived                 []*mlx.Array
}

func loadDecoder(raw []byte, tensors map[string]*mlx.Array) (*decoder, error) {
	var cfg struct {
		Decoder decoderConfig `json:"decoder_config"`
		Rate    int           `json:"output_sample_rate"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	c := cfg.Decoder
	if cfg.Rate != 24000 || c.Quantizers != 16 || c.Semantic != 1 || !slices.Equal(c.Rates, []int{8, 5, 4, 3}) || !slices.Equal(c.Ratios, []int{2, 2}) || c.Layers <= 0 || c.Head <= 0 || c.KVHeads <= 0 || c.Heads <= 0 || c.Hidden <= 0 || c.Eps <= 0 {
		return nil, fmt.Errorf("unsupported Qwen3-TTS speech decoder configuration")
	}
	w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, 0, 0, "", nil)}
	d := &decoder{cfg: c, pre: loadConv(w, "decoder.pre_conv.conv", 1, 1, 1, false), in: w.linear("decoder.pre_transformer.input_proj"), out: w.linear("decoder.pre_transformer.output_proj"), transformer: loadStack(w, "decoder.pre_transformer", c.stackConfig, false, true)}
	for _, bank := range []string{"rvq_first", "rvq_rest"} {
		count := c.Quantizers - 1
		if bank == "rvq_first" {
			count = 1
		}
		for i := range count {
			p := fmt.Sprintf("decoder.quantizer.%s.vq.layers.%d._codebook", bank, i)
			sum, usage := w.get(p+".embedding_sum"), w.get(p+".cluster_usage")
			if sum != nil && usage != nil {
				d.books = append(d.books, mlx.Div(sum, mlx.ExpandDims(mlx.Maximum(usage, mlx.FromValue(float32(1e-5))), 1)))
			}
		}
	}
	d.semantic = loadConv(w, "decoder.quantizer.rvq_first.output_proj", 1, 1, 1, false)
	d.acoustic = loadConv(w, "decoder.quantizer.rvq_rest.output_proj", 1, 1, 1, false)
	for i, ratio := range c.Ratios {
		p := fmt.Sprintf("decoder.upsample.%d", i)
		d.ups = append(d.ups, auxiliaryUp{up: loadConv(w, p+".0.conv", 1, 1, ratio, true), block: convNext{depth: loadConv(w, p+".1.dwconv.conv", 1, c.Latent, 1, false), norm: &nn.LayerNorm{Weight: w.get(p + ".1.norm.weight"), Bias: w.get(p + ".1.norm.bias"), Eps: 1e-6}, one: w.linear(p + ".1.pwconv1"), two: w.linear(p + ".1.pwconv2"), gamma: w.get(p + ".1.gamma")}})
	}
	d.initial = loadConv(w, "decoder.decoder.0.conv", 1, 1, 1, false)
	for i, rate := range c.Rates {
		p := fmt.Sprintf("decoder.decoder.%d.block", i+1)
		b := upBlock{activation: loadSnake(w, p+".0"), up: loadConv(w, p+".1.conv", 1, 1, rate, true)}
		for j, dilation := range []int{1, 3, 9} {
			q := fmt.Sprintf("%s.%d", p, j+2)
			b.units = append(b.units, residual{loadSnake(w, q+".act1"), loadSnake(w, q+".act2"), loadConv(w, q+".conv1.conv", dilation, 1, 1, false), loadConv(w, q+".conv2.conv", 1, 1, 1, false)})
		}
		d.blocks = append(d.blocks, b)
	}
	d.last = loadSnake(w, "decoder.decoder.5")
	d.final = loadConv(w, "decoder.decoder.6.conv", 1, 1, 1, false)
	d.derived = append(w.derived, d.books...)
	return d, w.err
}

func (d *decoder) decode(codes *mlx.Array) *mlx.Array {
	return mlx.ScopedEval(func() []*mlx.Array { return []*mlx.Array{d.forward(codes)} })[0]
}

func (d *decoder) forward(codes *mlx.Array) *mlx.Array {
	code := func(i int) *mlx.Array {
		return mlx.Squeeze(codes.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(i, i+1)), 2)
	}
	semantic := mlx.Take(d.books[0], code(0), 0)
	acoustic := mlx.Zeros(mlx.DTypeFloat32, codes.Dim(0), codes.Dim(1), d.books[1].Dim(1))
	for i := 1; i < len(d.books); i++ {
		acoustic = mlx.Add(acoustic, mlx.Take(d.books[i], code(i), 0))
	}
	x := d.pre.forward(mlx.Add(d.semantic.forward(semantic), d.acoustic.forward(acoustic)))
	x = d.out.Forward(d.transformer.forward(d.in.Forward(x), nil))
	for _, up := range d.ups {
		x = up.block.forward(up.up.forward(x))
	}
	x = d.initial.forward(x)
	for _, block := range d.blocks {
		x = block.forward(x)
	}
	x = d.final.forward(d.last.forward(x))
	x = mlx.Clip(x, mlx.FromValue(float32(-1)), mlx.FromValue(float32(1)))
	return mlx.Squeeze(x, 2)
}
