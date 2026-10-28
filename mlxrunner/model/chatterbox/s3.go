package chatterbox

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type s3Layer struct {
	q, k, v, out, one, two nn.LinearLayer
	attnNorm, mlpNorm      *nn.LayerNorm
	memory                 conv
}
type s3Tokenizer struct {
	first, second          conv
	layers                 []s3Layer
	projection             nn.LinearLayer
	bank, window, cos, sin *mlx.Array
	derived                []*mlx.Array
}

func loadS3Tokenizer(tensors map[string]*mlx.Array) (*s3Tokenizer, error) {
	w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, 0, 0, "", nil)}
	s := &s3Tokenizer{first: w.conv("encoder.conv1", 2, 1, 1, 1), second: w.conv("encoder.conv2", 2, 1, 1, 1), projection: w.linear("quantizer.fsq_codebook.project_down")}
	for i := range 6 {
		p := fmt.Sprintf("encoder.blocks.%d", i)
		s.layers = append(s.layers, s3Layer{q: w.linear(p + ".attn.query"), k: w.linear(p + ".attn.key"), v: w.linear(p + ".attn.value"), out: w.linear(p + ".attn.out"), one: w.linear(p + ".mlp.layers.0"), two: w.linear(p + ".mlp.layers.2"), attnNorm: w.norm(p+".attn_ln", 1e-6), mlpNorm: w.norm(p+".mlp_ln", 1e-5), memory: w.conv(p+".attn.fsmn_block", 1, 0, 1, 1280)})
	}
	if w.err != nil {
		return nil, w.err
	}
	s.bank = nn.SlaneyMelFilters(16000, 400, 128, 8000)
	window := make([]float32, 400)
	for i := range window {
		window[i] = float32(.5 * (1 - math.Cos(2*math.Pi*float64(i)/399)))
	}
	s.window = mlx.FromValues(window, 400)
	inv := mlx.Div(mlx.FromValue(float32(1)), mlx.FromValue(float32(10000)).Power(mlx.Div(mlx.Arange(0, 64, 2, mlx.DTypeInt32).AsType(mlx.DTypeFloat32), mlx.FromValue(float32(64)))))
	freq := mlx.Mul(mlx.ExpandDims(mlx.Arange(0, 2048, 1, mlx.DTypeInt32), 1), mlx.ExpandDims(inv, 0)).AsType(mlx.DTypeFloat32)
	s.cos = mlx.Concatenate([]*mlx.Array{mlx.Cos(freq), mlx.Cos(freq)}, -1)
	s.sin = mlx.Concatenate([]*mlx.Array{mlx.Sin(freq), mlx.Sin(freq)}, -1)
	s.derived = []*mlx.Array{s.bank, s.window, s.cos, s.sin}
	return s, nil
}

func (s *s3Tokenizer) mels(pcm *mlx.Array) *mlx.Array {
	padded := mlx.Pad(pcm, []int{0}, []int{200}, []int{200}, mlx.FromValue(float32(0)), "reflect")
	frames := padded.AsStrided([]int{pcm.Size() / 160, 400}, []int{160, 1}, 0)
	power := mlx.RFFT(mlx.Mul(frames, s.window), 400, -1).Abs().Power(mlx.FromValue(2))
	mel := mlx.Matmul(mlx.ExpandDims(power, 0), mlx.Transpose(s.bank, 1, 0))
	mel = mlx.Transpose(mel, 0, 2, 1)
	log := mlx.Log10(mlx.Maximum(mel, mlx.FromValue(float32(1e-10))))
	log = mlx.Maximum(log, mlx.Sub(log.MaxAxis(-1, true).MaxAxis(1, true).MaxAxis(0, true), mlx.FromValue(float32(8))))
	return mlx.Div(mlx.Add(log, mlx.FromValue(float32(4))), mlx.FromValue(float32(4)))
}

func (s *s3Tokenizer) features(mels *mlx.Array) *mlx.Array {
	x := exactGELU(s.first.forward(mlx.Transpose(mels, 0, 2, 1)))
	x = exactGELU(s.second.forward(x))
	length := x.Dim(1)
	mask := mlx.Zeros(mlx.DTypeFloat32, 1, 1, length)
	padMask := mlx.BroadcastTo(mlx.FromValue(true), 1, int32(length), 1)
	cos := mlx.Reshape(s.cos.Slice(mlx.Slice(0, length), mlx.Slice()), 1, int32(length), 1, 64)
	sin := mlx.Reshape(s.sin.Slice(mlx.Slice(0, length), mlx.Slice()), 1, int32(length), 1, 64)
	rotate := func(a *mlx.Array) *mlx.Array {
		a = mlx.Reshape(a, 1, int32(length), 20, 64)
		r := mlx.Concatenate([]*mlx.Array{mlx.Neg(a.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(32, 64))), a.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, 32))}, -1)
		return mlx.Add(mlx.Mul(a, cos), mlx.Mul(r, sin))
	}
	for _, l := range s.layers {
		n := l.attnNorm.Forward(x)
		q, k := rotate(l.q.Forward(n)), rotate(l.k.Forward(n))
		v := mlx.Reshape(l.v.Forward(n), 1, int32(length), 20, 64)
		memory := mlx.Mul(mlx.Reshape(v, 1, int32(length), 1280), padMask)
		originalMemory := memory
		memory = mlx.PadConstant(memory, []int{1}, []int{15}, []int{15})
		memory = mlx.Mul(mlx.Add(l.memory.forward(memory), originalMemory), padMask)
		q = mlx.Mul(mlx.Transpose(q, 0, 2, 1, 3), mlx.FromValue(float32(math.Pow(64, -.25))))
		k = mlx.Mul(mlx.Transpose(k, 0, 2, 1, 3), mlx.FromValue(float32(math.Pow(64, -.25))))
		a := mlx.FastScaledDotProductAttention(q, k, mlx.Transpose(v, 0, 2, 1, 3), 1, mlx.FastSDPANoWindow, "array", mask)
		a = l.out.Forward(mlx.Reshape(mlx.Transpose(a, 0, 2, 1, 3), 1, int32(length), 1280))
		x = mlx.Add(x, mlx.Add(a, memory))
		x = mlx.Add(x, l.two.Forward(exactGELU(l.one.Forward(l.mlpNorm.Forward(x)))))
	}
	return x
}

func (s *s3Tokenizer) quantize(features *mlx.Array) *mlx.Array {
	h := s.projection.Forward(mlx.Reshape(features, -1, 1280)).AsType(mlx.DTypeFloat32).Tanh()
	h = mlx.Add(mlx.Round(mlx.Mul(h, mlx.FromValue(float32(.9990000128746033)))), mlx.FromValue(float32(1)))
	powers := mlx.FromValue(float32(3)).Power(mlx.Arange(0, 8, 1, mlx.DTypeFloat32))
	return mlx.Reshape(mlx.Sum(mlx.Mul(h, mlx.ExpandDims(powers, 0)), -1, false), 1, int32(features.Dim(1))).AsType(mlx.DTypeInt32)
}
