package voxtraltts

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
)

type causalConv struct {
	weight             *mlx.Array
	kernel, stride     int
	transpose, reflect bool
}

func loadConv(w *weights, p string, stride int, transpose, reflect bool) *causalConv {
	gain := w.get(p + ".parametrizations.weight.original0")
	direction := w.get(p + ".parametrizations.weight.original1")
	if gain == nil || direction == nil {
		return nil
	}
	norm := mlx.Add(mlx.SumAxes(mlx.Mul(direction, direction), []int{1, 2}, true), mlx.FromValue(float32(1e-12)).AsType(direction.DType())).Sqrt()
	weight := mlx.Div(mlx.Mul(gain, direction), norm)
	if transpose {
		weight = mlx.Transpose(weight, 1, 2, 0)
	} else {
		weight = mlx.Transpose(weight, 0, 2, 1)
	}
	w.derived = append(w.derived, weight)
	return &causalConv{weight: weight, kernel: direction.Dim(2), stride: stride, transpose: transpose, reflect: reflect}
}

func (c *causalConv) run(x *mlx.Array) *mlx.Array {
	if c.transpose {
		length := x.Dim(1) * c.stride
		return mlx.ConvTranspose1d(x, c.weight, nil, int32(c.stride), 0, 1, 0, 1).Slice(mlx.Slice(), mlx.Slice(0, length), mlx.Slice())
	}
	pad := c.kernel - 1
	if pad > 0 {
		var prefix *mlx.Array
		if c.reflect {
			indices := make([]int32, pad)
			for i := range indices {
				indices[i] = int32(pad - i)
			}
			prefix = mlx.Take(x, mlx.FromValues(indices, pad), 1)
		} else {
			prefix = mlx.BroadcastTo(x.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice()), int32(x.Dim(0)), int32(pad), int32(x.Dim(2)))
		}
		x = mlx.Concatenate([]*mlx.Array{prefix, x}, 1)
	}
	return mlx.Conv1d(x, c.weight, nil, 1, 0, 1, 1)
}

type codec struct {
	table        *mlx.Array
	convs        [4]*causalConv
	transformers [4]*transformer
	output       *causalConv
}

func loadCodec(w *weights) *codec {
	p := "audio_tokenizer"
	sum := w.get(p + ".quantizer.semantic_codebook.embedding_sum")
	usage := w.get(p + ".quantizer.semantic_codebook.cluster_usage")
	c := &codec{}
	if sum != nil && usage != nil {
		c.table = mlx.Div(sum.AsType(mlx.DTypeFloat32), mlx.Maximum(mlx.ExpandDims(usage.AsType(mlx.DTypeFloat32), 1), mlx.FromValue(float32(1e-8))))
		w.derived = append(w.derived, c.table)
	}
	dims := w.cfg.Multimodal.Codec.Dimensions
	dims.Layers = 2
	for i, stride := range []int{1, 2, 2, 2} {
		c.convs[i] = loadConv(w, fmt.Sprintf("%s.decoder_blocks.%d.conv", p, i*2), stride, i != 0, false)
		c.transformers[i] = loadTransformer(w, fmt.Sprintf("%s.decoder_blocks.%d", p, i*2+1), dims, "codec")
	}
	c.output = loadConv(w, p+".output_proj.conv", 1, false, true)
	return c
}

func (c *codec) features(codes *mlx.Array) *mlx.Array {
	semantic := mlx.Sub(mlx.Squeeze(codes.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, 1)), 2), mlx.FromValue(2))
	acoustic := mlx.Sub(codes.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(1, 37)), mlx.FromValue(2)).AsType(mlx.DTypeFloat32)
	decoded := mlx.Sub(mlx.Div(mlx.Mul(mlx.FromValue(float32(2)), acoustic), mlx.FromValue(float32(20))), mlx.FromValue(float32(1)))
	return mlx.Concatenate([]*mlx.Array{mlx.Take(c.table, semantic, 0), decoded}, 2)
}

func (c *codec) decode(codes *mlx.Array) *mlx.Array {
	x := c.features(codes)
	for i, conv := range c.convs {
		x = conv.run(x)
		x = c.transformers[i].forward(x, nil, 2<<i)
	}
	x = c.output.run(x)
	return mlx.Reshape(x, int32(x.Dim(0)), -1)
}
