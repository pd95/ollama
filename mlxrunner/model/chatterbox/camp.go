package chatterbox

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
)

type batchNorm struct{ mean, variance, weight, bias *mlx.Array }

func (w *weights) batchNorm(p string) batchNorm {
	return batchNorm{w.get(p + ".running_mean"), w.get(p + ".running_var"), w.tensors[p+".weight"], w.tensors[p+".bias"]}
}

func (b batchNorm) forward(x *mlx.Array) *mlx.Array {
	x = mlx.Mul(mlx.Sub(x, b.mean), mlx.RSqrt(mlx.Add(b.variance, mlx.FromValue(float32(1e-5)).AsType(b.variance.DType()))))
	if b.weight != nil {
		x = mlx.Add(mlx.Mul(b.weight, x), b.bias)
	}
	return x
}

type campResidual struct {
	one, two, shortcut       *mlx.Array
	bnOne, bnTwo, bnShortcut batchNorm
	stride                   int32
}

func (b campResidual) forward(x *mlx.Array) *mlx.Array {
	y := mlx.ReLU(b.bnOne.forward(mlx.Conv2d(x, b.one, b.stride, 1, 1, 1, 1, 1, 1)))
	y = b.bnTwo.forward(mlx.Conv2d(y, b.two, 1, 1, 1, 1, 1, 1, 1))
	if b.shortcut != nil {
		x = b.bnShortcut.forward(mlx.Conv2d(x, b.shortcut, b.stride, 1, 0, 0, 1, 1, 1))
	}
	return mlx.ReLU(mlx.Add(y, x))
}

type campDense struct {
	before, after          batchNorm
	input, local, one, two conv
}

func (b campDense) forward(x *mlx.Array) *mlx.Array {
	x = mlx.Transpose(x, 0, 2, 1)
	x = mlx.ReLU(b.before.forward(x))
	x = mlx.ReLU(b.after.forward(b.input.forward(x)))
	x = mlx.Transpose(x, 0, 2, 1)
	y := mlx.Transpose(b.local.forward(mlx.Transpose(x, 0, 2, 1)), 0, 2, 1)
	length, channels := x.Dim(2), x.Dim(1)
	segments := (length + 99) / 100
	padded := x
	if segments*100 > length {
		padded = mlx.Concatenate([]*mlx.Array{x, mlx.Zeros(mlx.DTypeFloat32, 1, channels, segments*100-length)}, 2)
	}
	segment := mlx.Mean(mlx.Reshape(padded, 1, int32(channels), int32(segments), 100), -1, false)
	segment = mlx.Reshape(mlx.BroadcastTo(mlx.ExpandDims(segment, -1), 1, int32(channels), int32(segments), 100), 1, int32(channels), -1)
	segment = segment.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, length))
	context := mlx.Add(mlx.Mean(x, -1, true), segment)
	context = mlx.Transpose(b.one.forward(mlx.Transpose(context, 0, 2, 1)), 0, 2, 1)
	context = mlx.ReLU(context)
	context = mlx.Transpose(b.two.forward(mlx.Transpose(context, 0, 2, 1)), 0, 2, 1).Sigmoid()
	return mlx.Mul(y, context)
}

type campTransition struct {
	norm   batchNorm
	linear conv
}

func (b campTransition) forward(x *mlx.Array) *mlx.Array {
	return mlx.Transpose(b.linear.forward(mlx.ReLU(b.norm.forward(mlx.Transpose(x, 0, 2, 1)))), 0, 2, 1)
}

type campEncoder struct {
	first, last                                          *mlx.Array
	firstNorm, lastNorm, tdnnNorm, outputNorm, denseNorm batchNorm
	headBlocks                                           []campResidual
	tdnn, dense                                          conv
	blocks                                               [][]campDense
	transitions                                          []campTransition
	bank, window                                         *mlx.Array
	derived                                              []*mlx.Array
}

func loadCAMP(w *weights) *campEncoder {
	p := "s3gen.speaker_encoder"
	c := &campEncoder{first: w.get(p + ".head.conv1.weight"), last: w.get(p + ".head.conv2.weight"), firstNorm: w.batchNorm(p + ".head.bn1"), lastNorm: w.batchNorm(p + ".head.bn2"), tdnn: w.conv(p+".tdnn.linear", 2, 2, 1, 1), tdnnNorm: w.batchNorm(p + ".tdnn.nonlinear.0"), outputNorm: w.batchNorm(p + ".out_nonlinear.0"), dense: w.conv(p+".dense.linear", 1, 0, 1, 1), denseNorm: w.batchNorm(p + ".dense.nonlinear.0")}
	for group := range 2 {
		for i := range 2 {
			q := fmt.Sprintf("%s.head.layer%d.%d", p, group+1, i)
			b := campResidual{one: w.get(q + ".conv1.weight"), two: w.get(q + ".conv2.weight"), bnOne: w.batchNorm(q + ".bn1"), bnTwo: w.batchNorm(q + ".bn2"), stride: 1}
			if i == 0 {
				b.stride = 2
				b.shortcut = w.get(q + ".shortcut.0.weight")
				b.bnShortcut = w.batchNorm(q + ".shortcut.1")
			}
			c.headBlocks = append(c.headBlocks, b)
		}
	}
	for block, count := range []int{12, 24, 16} {
		layers := make([]campDense, 0, count)
		dilation := int32(1)
		if block > 0 {
			dilation = 2
		}
		for i := range count {
			q := fmt.Sprintf("%s.blocks.%d.layers.%d", p, block, i)
			layers = append(layers, campDense{before: w.batchNorm(q + ".nonlinear1.0"), after: w.batchNorm(q + ".nonlinear2.0"), input: w.conv(q+".linear1", 1, 0, 1, 1), local: w.conv(q+".cam_layer.linear_local", 1, dilation, dilation, 1), one: w.conv(q+".cam_layer.linear1", 1, 0, 1, 1), two: w.conv(q+".cam_layer.linear2", 1, 0, 1, 1)})
		}
		c.blocks = append(c.blocks, layers)
		q := fmt.Sprintf("%s.transits.%d", p, block)
		c.transitions = append(c.transitions, campTransition{w.batchNorm(q + ".nonlinear.0"), w.conv(q+".linear", 1, 0, 1, 1)})
	}
	c.bank = campMelBank()
	n := mlx.Arange(0, 400, 1, mlx.DTypeInt32)
	hann := mlx.Sub(mlx.FromValue(float32(.5)), mlx.Mul(mlx.FromValue(float32(.5)), mlx.Cos(mlx.Div(mlx.Mul(mlx.FromValue(float32(2*math.Pi)), n), mlx.FromValue(399)))))
	c.window = hann.Power(mlx.FromValue(float32(.85)))
	c.derived = []*mlx.Array{c.bank, c.window}
	return c
}

func campMelBank() *mlx.Array {
	minimum := 2595 * math.Log10(1+20.0/700)
	maximum := 2595 * math.Log10(1+8000.0/700)
	mels := mlx.Linspace(minimum, maximum, 82, mlx.DTypeFloat32)
	hz := mlx.Mul(mlx.FromValue(float32(700)), mlx.Sub(mlx.FromValue(float32(10)).Power(mlx.Div(mels, mlx.FromValue(float32(2595)))), mlx.FromValue(float32(1))))
	diff := mlx.Sub(hz.Slice(mlx.Slice(1, 82)), hz.Slice(mlx.Slice(0, 81)))
	slopes := mlx.Sub(mlx.ExpandDims(hz, 0), mlx.ExpandDims(mlx.Linspace(0, 8000, 257, mlx.DTypeFloat32), 1))
	down := mlx.Div(mlx.Neg(slopes.Slice(mlx.Slice(), mlx.Slice(0, 80))), diff.Slice(mlx.Slice(0, 80)))
	up := mlx.Div(slopes.Slice(mlx.Slice(), mlx.Slice(2, 82)), diff.Slice(mlx.Slice(1, 81)))
	return mlx.Transpose(mlx.Maximum(mlx.FromValue(float32(0)), mlx.Minimum(down, up)), 1, 0)
}

func (c *campEncoder) fbank(pcm *mlx.Array) *mlx.Array {
	frames := 1 + (pcm.Size()-400)/160
	x := mlx.Contiguous(pcm.AsStrided([]int{frames, 400}, []int{160, 1}, 0), false)
	x = mlx.Sub(x, mlx.Mean(x, 1, true))
	x = mlx.Concatenate([]*mlx.Array{x.Slice(mlx.Slice(), mlx.Slice(0, 1)), mlx.Sub(x.Slice(mlx.Slice(), mlx.Slice(1, 400)), mlx.Mul(mlx.FromValue(float32(.97)), x.Slice(mlx.Slice(), mlx.Slice(0, 399))))}, 1)
	x = mlx.Mul(x, mlx.ExpandDims(c.window, 0))
	x = mlx.Concatenate([]*mlx.Array{x, mlx.Zeros(mlx.DTypeFloat32, frames, 112)}, 1)
	power := mlx.RFFT(x, 512, -1).Abs().Power(mlx.FromValue(2))
	return mlx.Log(mlx.Maximum(mlx.Matmul(power, mlx.Transpose(c.bank, 1, 0)), mlx.FromValue(float32(1.1920929e-7))))
}

func (c *campEncoder) head(x *mlx.Array) *mlx.Array {
	x = mlx.ExpandDims(x, -1)
	x = mlx.ReLU(c.firstNorm.forward(mlx.Conv2d(x, c.first, 1, 1, 1, 1, 1, 1, 1)))
	for _, b := range c.headBlocks {
		x = b.forward(x)
	}
	x = mlx.ReLU(c.lastNorm.forward(mlx.Conv2d(x, c.last, 2, 1, 1, 1, 1, 1, 1)))
	return mlx.Reshape(mlx.Transpose(x, 0, 3, 1, 2), int32(x.Dim(0)), int32(x.Dim(3)*x.Dim(1)), int32(x.Dim(2)))
}

func (c *campEncoder) initial(x *mlx.Array) *mlx.Array {
	return mlx.Transpose(mlx.ReLU(c.tdnnNorm.forward(c.tdnn.forward(mlx.Transpose(x, 0, 2, 1)))), 0, 2, 1)
}

func (c *campEncoder) block(x *mlx.Array, index int) *mlx.Array {
	for _, l := range c.blocks[index] {
		x = mlx.Concatenate([]*mlx.Array{x, l.forward(x)}, 1)
	}
	return x
}

func (c *campEncoder) forward(features *mlx.Array) *mlx.Array {
	x := c.initial(c.head(mlx.Transpose(features, 0, 2, 1)))
	for i := range c.blocks {
		x = c.transitions[i].forward(c.block(x, i))
	}
	x = mlx.Transpose(mlx.ReLU(c.outputNorm.forward(mlx.Transpose(x, 0, 2, 1))), 0, 2, 1)
	mean := mlx.Mean(x, -1, false)
	std := mlx.Add(mlx.Variance(x, -1, false), mlx.FromValue(float32(1e-5))).Sqrt()
	stats := mlx.Concatenate([]*mlx.Array{mean, std}, -1)
	return mlx.Squeeze(c.denseNorm.forward(c.dense.forward(mlx.ExpandDims(stats, 1))), 1)
}
