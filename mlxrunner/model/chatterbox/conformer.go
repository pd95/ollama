package chatterbox

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type conformerLayer struct {
	q, k, v, pos, out, one, two nn.LinearLayer
	attnNorm, ffnNorm           *nn.LayerNorm
	biasU, biasV                *mlx.Array
}
type conformerEncoder struct {
	input, upInput               nn.LinearLayer
	inputNorm, upNorm, finalNorm *nn.LayerNorm
	lookOne, lookTwo, upsample   conv
	layers, upLayers             []conformerLayer
	positions, upPositions       *mlx.Array
}

func loadConformer(w *weights) *conformerEncoder {
	p := "s3gen.flow.encoder"
	e := &conformerEncoder{input: w.linear(p + ".embed.linear"), upInput: w.linear(p + ".up_embed.linear"), inputNorm: w.norm(p+".embed.norm", 1e-5), upNorm: w.norm(p+".up_embed.norm", 1e-5), finalNorm: w.norm(p+".after_norm", 1e-5), lookOne: w.conv(p+".pre_lookahead_layer.conv1", 1, 0, 1, 1), lookTwo: w.conv(p+".pre_lookahead_layer.conv2", 1, 0, 1, 1), upsample: w.conv(p+".up_layer.conv", 1, 0, 1, 1)}
	load := func(name string, count int) []conformerLayer {
		layers := make([]conformerLayer, 0, count)
		for i := range count {
			q := fmt.Sprintf("%s.%s_%d", p, name, i)
			a := q + ".self_attn"
			layers = append(layers, conformerLayer{q: w.linear(a + ".linear_q"), k: w.linear(a + ".linear_k"), v: w.linear(a + ".linear_v"), pos: w.linear(a + ".linear_pos"), out: w.linear(a + ".linear_out"), one: w.linear(q + ".feed_forward.w_1"), two: w.linear(q + ".feed_forward.w_2"), attnNorm: w.norm(q+".norm_mha", 1e-12), ffnNorm: w.norm(q+".norm_ff", 1e-12), biasU: w.get(a + ".pos_bias_u"), biasV: w.get(a + ".pos_bias_v")})
		}
		return layers
	}
	e.layers = load("encoders", 6)
	e.upLayers = load("up_encoders", 4)
	return e
}

func conformerPosition(length int) *mlx.Array {
	inv := mlx.Exp(mlx.Mul(mlx.Arange(0, 512, 2, mlx.DTypeFloat32), mlx.FromValue(float32(-math.Log(10000)/512))))
	positions := make([]float32, 2*length-1)
	for i := range positions {
		positions[i] = float32(length - 1 - i)
	}
	angles := mlx.Mul(mlx.ExpandDims(mlx.FromValues(positions, len(positions)), 1), mlx.ExpandDims(inv, 0))
	return mlx.Reshape(mlx.Stack([]*mlx.Array{mlx.Sin(angles), mlx.Cos(angles)}, -1), 1, int32(2*length-1), 512)
}

func conformerEmbed(x *mlx.Array, linear nn.LinearLayer, norm *nn.LayerNorm) *mlx.Array {
	return mlx.Mul(norm.Forward(linear.Forward(x)), mlx.FromValue(float32(math.Sqrt(512))))
}

func (l conformerLayer) forward(x, pos *mlx.Array) *mlx.Array {
	length := x.Dim(1)
	n := l.attnNorm.Forward(x)
	q := mlx.Reshape(l.q.Forward(n), 1, int32(length), 8, 64)
	k := mlx.Transpose(mlx.Reshape(l.k.Forward(n), 1, int32(length), 8, 64), 0, 2, 1, 3)
	v := mlx.Transpose(mlx.Reshape(l.v.Forward(n), 1, int32(length), 8, 64), 0, 2, 1, 3)
	p := mlx.Transpose(mlx.Reshape(l.pos.Forward(pos), 1, int32(2*length-1), 8, 64), 0, 2, 1, 3)
	ac := mlx.Matmul(mlx.Transpose(mlx.Add(q, l.biasU), 0, 2, 1, 3), mlx.Transpose(k, 0, 1, 3, 2))
	bd := mlx.Matmul(mlx.Transpose(mlx.Add(q, l.biasV), 0, 2, 1, 3), mlx.Transpose(p, 0, 1, 3, 2))
	if length > 1 {
		padded := mlx.Concatenate([]*mlx.Array{mlx.Zeros(mlx.DTypeFloat32, 1, 8, length, 1), bd}, -1)
		padded = mlx.Reshape(padded, 1, 8, int32(2*length), int32(length))
		padded = padded.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(1, 2*length), mlx.Slice())
		bd = mlx.Reshape(padded, 1, 8, int32(length), int32(2*length-1)).Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, length))
	}
	scores := mlx.Div(mlx.Add(ac, bd), mlx.FromValue(float32(8)))
	attention := mlx.SoftmaxAxis(scores, -1, false)
	a := mlx.Matmul(attention, v)
	a = mlx.Reshape(mlx.Transpose(a, 0, 2, 1, 3), 1, int32(length), 512)
	x = mlx.Add(x, l.out.Forward(a))
	return mlx.Add(x, l.two.Forward(mlx.SiLU(l.one.Forward(l.ffnNorm.Forward(x)))))
}

func (e *conformerEncoder) lookahead(x *mlx.Array) *mlx.Array {
	y := e.lookOne.forward(mlx.PadConstant(x, []int{1}, []int{0}, []int{3}))
	y = mlx.Maximum(y, mlx.Mul(y, mlx.FromValue(float32(.01))))
	y = e.lookTwo.forward(mlx.PadConstant(y, []int{1}, []int{2}, []int{0}))
	return mlx.Add(y, x)
}

func (e *conformerEncoder) beforeUp(x, pos *mlx.Array) *mlx.Array {
	for _, l := range e.layers {
		x = l.forward(x, pos)
	}
	return x
}

func (e *conformerEncoder) up(x *mlx.Array) *mlx.Array {
	length := x.Dim(1)
	x = mlx.Transpose(x, 0, 2, 1)
	x = mlx.Reshape(mlx.BroadcastTo(mlx.ExpandDims(x, -1), 1, 512, int32(length), 2), 1, 512, int32(length*2))
	x = mlx.PadConstant(x, []int{2}, []int{4}, []int{0})
	return mlx.Transpose(e.upsample.forward(mlx.Transpose(x, 0, 2, 1)), 0, 2, 1)
}

func (e *conformerEncoder) afterUp(x, pos *mlx.Array) *mlx.Array {
	for _, l := range e.upLayers {
		x = l.forward(x, pos)
	}
	return e.finalNorm.Forward(x)
}

func (e *conformerEncoder) forward(x *mlx.Array) *mlx.Array {
	position := conformerPositions(x.Dim(1), e.positions)
	x = e.beforeUp(e.lookahead(conformerEmbed(x, e.input, e.inputNorm)), position)
	x = mlx.Transpose(e.up(x), 0, 2, 1)
	position = conformerPositions(x.Dim(1), e.upPositions)
	return e.afterUp(conformerEmbed(x, e.upInput, e.upNorm), position)
}

func conformerPositions(length int, cached *mlx.Array) *mlx.Array {
	if cached == nil {
		return conformerPosition(length)
	}
	center := cached.Dim(1) / 2
	return cached.Slice(mlx.Slice(), mlx.Slice(center-length+1, center+length), mlx.Slice())
}
