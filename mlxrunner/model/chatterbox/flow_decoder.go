package chatterbox

import (
	"context"
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

var flowMish = mlx.Compile1("ChatterboxMish", func(x *mlx.Array) *mlx.Array {
	return mlx.Mul(x, mlx.Logaddexp(x, mlx.FromValue(float32(0)).AsType(x.DType())).Tanh())
}, mlx.Shapeless())

type flowBlock struct {
	conv conv
	norm *nn.LayerNorm
}

func loadFlowBlock(w *weights, p string) flowBlock {
	return flowBlock{w.conv(p+".conv.conv", 1, 0, 1, 1), w.norm(p+".norm", 1e-5)}
}

func causalFlowConv(c conv, x *mlx.Array) *mlx.Array {
	x = mlx.Transpose(x, 0, 2, 1)
	x = mlx.PadConstant(x, []int{1}, []int{c.weight.Dim(1) - 1}, []int{0})
	return mlx.Transpose(c.forward(x), 0, 2, 1)
}

func (b flowBlock) forward(x, mask *mlx.Array) *mlx.Array {
	x = causalFlowConv(b.conv, mlx.Mul(x, mask))
	x = mlx.Transpose(b.norm.Forward(mlx.Transpose(x, 0, 2, 1)), 0, 2, 1)
	return mlx.Mul(flowMish(x), mask)
}

type flowResidual struct {
	one, two flowBlock
	time     nn.LinearLayer
	residual conv
}

func loadFlowResidual(w *weights, p string) flowResidual {
	return flowResidual{loadFlowBlock(w, p+".block1"), loadFlowBlock(w, p+".block2"), w.linear(p + ".mlp_linear"), w.conv(p+".res_conv", 1, 0, 1, 1)}
}

func (b flowResidual) forward(x, mask, time *mlx.Array) *mlx.Array {
	h := b.one.forward(x, mask)
	h = mlx.Add(h, mlx.ExpandDims(b.time.Forward(flowMish(time)), -1))
	h = b.two.forward(h, mask)
	residual := mlx.Transpose(b.residual.forward(mlx.Transpose(mlx.Mul(x, mask), 0, 2, 1)), 0, 2, 1)
	return mlx.Add(h, residual)
}

type flowTransformer struct {
	q, k, v, out, one, two nn.LinearLayer
	norm, ffn              *nn.LayerNorm
}

func loadFlowTransformer(w *weights, p string) flowTransformer {
	return flowTransformer{q: w.linear(p + ".attn.query_proj"), k: w.linear(p + ".attn.key_proj"), v: w.linear(p + ".attn.value_proj"), out: w.linear(p + ".attn.out_proj"), one: w.linear(p + ".ff.layers.0"), two: w.linear(p + ".ff.layers.1"), norm: w.norm(p+".norm1", 1e-5), ffn: w.norm(p+".norm3", 1e-5)}
}

func (l flowTransformer) forward(x, bias *mlx.Array) *mlx.Array {
	batch, length := int32(x.Dim(0)), int32(x.Dim(1))
	n := l.norm.Forward(x)
	split := func(a *mlx.Array) *mlx.Array { return mlx.Transpose(mlx.Reshape(a, batch, length, 8, 64), 0, 2, 1, 3) }
	q, k, v := split(l.q.Forward(n)), split(l.k.Forward(n)), split(l.v.Forward(n))
	scores := mlx.Mul(mlx.Matmul(q, mlx.Transpose(k, 0, 1, 3, 2)), mlx.FromValue(float32(1.0/8)))
	scores = mlx.Add(scores, bias)
	a := mlx.Matmul(mlx.SoftmaxAxis(scores, -1, false), v)
	a = mlx.Reshape(mlx.Transpose(a, 0, 2, 1, 3), batch, length, 512)
	x = mlx.Add(x, l.out.Forward(a))
	return mlx.Add(x, l.two.Forward(exactGELU(l.one.Forward(l.ffn.Forward(x)))))
}

type flowStage struct {
	residual flowResidual
	layers   []flowTransformer
}

func loadFlowStage(w *weights, p string) flowStage {
	s := flowStage{residual: loadFlowResidual(w, p+".resnet")}
	for i := range 4 {
		s.layers = append(s.layers, loadFlowTransformer(w, fmt.Sprintf("%s.transformer_%d", p, i)))
	}
	return s
}

func (s flowStage) forward(x, mask, time, bias *mlx.Array) *mlx.Array {
	x = s.residual.forward(x, mask, time)
	x = mlx.Transpose(x, 0, 2, 1)
	for _, l := range s.layers {
		x = l.forward(x, bias)
	}
	return mlx.Transpose(x, 0, 2, 1)
}

type flowDecoder struct {
	timeOne, timeTwo            nn.LinearLayer
	down, up                    flowStage
	middle                      []flowStage
	downsample, upsample, final conv
	last                        flowBlock
}

func loadFlowDecoder(w *weights) *flowDecoder {
	p := "s3gen.flow.decoder.estimator"
	d := &flowDecoder{timeOne: w.linear(p + ".time_mlp.linear_1"), timeTwo: w.linear(p + ".time_mlp.linear_2"), down: loadFlowStage(w, p+".down_blocks_0"), up: loadFlowStage(w, p+".up_blocks_0"), downsample: w.conv(p+".down_blocks_0.downsample.conv", 1, 0, 1, 1), upsample: w.conv(p+".up_blocks_0.upsample.conv", 1, 0, 1, 1), final: w.conv(p+".final_proj", 1, 0, 1, 1), last: loadFlowBlock(w, p+".final_block")}
	for i := range 12 {
		d.middle = append(d.middle, loadFlowStage(w, fmt.Sprintf("%s.mid_blocks_%d", p, i)))
	}
	return d
}

func flowTimeEmbedding(time *mlx.Array) *mlx.Array {
	inv := mlx.Exp(mlx.Mul(mlx.Arange(0, 160, 1, mlx.DTypeFloat32), mlx.FromValue(float32(-math.Log(10000)/159))))
	angles := mlx.Mul(mlx.Mul(mlx.FromValue(float32(1000)), mlx.ExpandDims(time, 1)), mlx.ExpandDims(inv, 0))
	return mlx.Concatenate([]*mlx.Array{mlx.Sin(angles), mlx.Cos(angles)}, -1)
}

func (d *flowDecoder) forward(x, mask, mu, time, speaker, condition *mlx.Array) *mlx.Array {
	embedding := d.timeTwo.Forward(mlx.SiLU(d.timeOne.Forward(flowTimeEmbedding(time))))
	batch, length := int32(x.Dim(0)), int32(x.Dim(2))
	x = mlx.Concatenate([]*mlx.Array{x, mu}, 1)
	x = mlx.Concatenate([]*mlx.Array{x, mlx.BroadcastTo(mlx.ExpandDims(speaker, -1), batch, int32(speaker.Dim(1)), length)}, 1)
	x = mlx.Concatenate([]*mlx.Array{x, condition}, 1)
	bias := mlx.Zeros(x.DType(), int(batch), 1, int(length), int(length))
	x = d.down.forward(x, mask, embedding, bias)
	skip := x
	x = causalFlowConv(d.downsample, mlx.Mul(x, mask))
	for _, s := range d.middle {
		x = s.forward(x, mask, embedding, bias)
	}
	x = mlx.Concatenate([]*mlx.Array{x, skip}, 1)
	x = d.up.forward(x, mask, embedding, bias)
	x = causalFlowConv(d.upsample, mlx.Mul(x, mask))
	x = d.last.forward(x, mask)
	x = mlx.Transpose(d.final.forward(mlx.Transpose(mlx.Mul(x, mask), 0, 2, 1)), 0, 2, 1)
	return mlx.Mul(x, mask)
}

func flowTimes(steps int) *mlx.Array {
	t := mlx.Linspace(0, 1, steps+1, mlx.DTypeFloat32)
	t = mlx.Mul(mlx.Mul(t, mlx.FromValue(float32(.5))), mlx.FromValue(float32(math.Pi)))
	return mlx.Sub(mlx.FromValue(float32(1)), mlx.Cos(t))
}

func (d *flowDecoder) eval(x, mask, mu, time, speaker, condition *mlx.Array) *mlx.Array {
	return mlx.ScopedEval(func() []*mlx.Array { return []*mlx.Array{d.forward(x, mask, mu, time, speaker, condition)} })[0]
}

func (d *flowDecoder) solve(x, times, mu, mask, speaker, condition *mlx.Array) *mlx.Array {
	result, _ := d.solveContext(context.Background(), x, times, mu, mask, speaker, condition)
	return result
}

func (d *flowDecoder) solveContext(ctx context.Context, x, times, mu, mask, speaker, condition *mlx.Array) (*mlx.Array, error) {
	t := mlx.Reshape(times.Slice(mlx.Slice(0, 1)), 1)
	dt := mlx.Sub(times.Slice(mlx.Slice(1, 2)), t)
	for step := 1; step < times.Size(); step++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		result := mlx.ScopedEval(func() []*mlx.Array {
			batchX := mlx.Concatenate([]*mlx.Array{x, x}, 0)
			batchMask := mlx.Concatenate([]*mlx.Array{mask, mask}, 0)
			batchMu := mlx.Concatenate([]*mlx.Array{mu, mlx.Zeros(mu.DType(), mu.Dims()...)}, 0)
			batchSpeaker := mlx.Concatenate([]*mlx.Array{speaker, mlx.Zeros(speaker.DType(), speaker.Dims()...)}, 0)
			batchCondition := mlx.Concatenate([]*mlx.Array{condition, mlx.Zeros(condition.DType(), condition.Dims()...)}, 0)
			velocity := d.eval(batchX, batchMask, batchMu, mlx.Concatenate([]*mlx.Array{t, t}, 0), batchSpeaker, batchCondition)
			conditional := velocity.Slice(mlx.Slice(0, 1), mlx.Slice(), mlx.Slice())
			unconditional := velocity.Slice(mlx.Slice(1, 2), mlx.Slice(), mlx.Slice())
			velocity = mlx.Sub(mlx.Mul(mlx.FromValue(float32(1.7)), conditional), mlx.Mul(mlx.FromValue(float32(.7)), unconditional))
			x = mlx.Add(x, mlx.Mul(dt, velocity))
			t = mlx.Add(t, dt)
			if step < times.Size()-1 {
				dt = mlx.Sub(times.Slice(mlx.Slice(step+1, step+2)), t)
			}
			return []*mlx.Array{x, t, dt}
		})
		x, t, dt = result[0], result[1], result[2]
	}
	return x, ctx.Err()
}
