package chatterbox

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type turboLayer struct {
	qkv, out, one, two nn.LinearLayer
	norm, ffn          *nn.LayerNorm
}
type turboT3 struct {
	text, speech, position nn.EmbeddingLayer
	speaker, head          nn.LinearLayer
	norm                   *nn.LayerNorm
	layers                 []turboLayer
}

func loadTurboT3(w *weights) *turboT3 {
	embedding := func(name string) nn.EmbeddingLayer {
		e := model.MakeEmbeddingLayer(w.tensors, name, 64, 4, "affine", nil)
		if e == nil && w.err == nil {
			w.err = fmt.Errorf("missing Turbo embedding %s", name)
		}
		return e
	}
	t := &turboT3{text: embedding("t3.text_emb"), speech: embedding("t3.speech_emb"), position: embedding("t3.tfmr.wpe"), speaker: w.linear("t3.cond_enc.spkr_enc"), head: w.linear("t3.speech_head"), norm: w.norm("t3.tfmr.ln_f", 1e-5)}
	for i := range 24 {
		p := fmt.Sprintf("t3.tfmr.h.%d", i)
		t.layers = append(t.layers, turboLayer{w.linear(p + ".attn.c_attn"), w.linear(p + ".attn.c_proj"), w.linear(p + ".mlp.c_fc"), w.linear(p + ".mlp.c_proj"), w.norm(p+".ln_1", 1e-5), w.norm(p+".ln_2", 1e-5)})
	}
	return t
}

func (t *turboT3) conditioning(speaker, codes *mlx.Array) *mlx.Array {
	return mlx.Concatenate([]*mlx.Array{mlx.ExpandDims(t.speaker.Forward(mlx.Reshape(speaker, -1, 256)), 1), t.speech.Forward(codes)}, 1)
}

func (t *turboT3) prompt(condition, text *mlx.Array) *mlx.Array {
	return mlx.Concatenate([]*mlx.Array{condition, t.text.Forward(text), t.speech.Forward(mlx.FromValues([]int32{6561}, 1, 1))}, 1)
}

func turboGELU(x *mlx.Array) *mlx.Array {
	scalar := func(v float32) *mlx.Array { return mlx.FromValue(v).AsType(x.DType()) }
	return mlx.Mul(mlx.Mul(scalar(.5), x), mlx.Add(scalar(1), mlx.Mul(mlx.FromValue(float32(2/math.Pi)).Sqrt().AsType(x.DType()), mlx.Add(x, mlx.Mul(scalar(.044715), x.Power(scalar(3))))).Tanh()))
}

func newTurboState() t3State {
	s := make(t3State, 24)
	for i := range s {
		s[i].scope = mlx.NewScope()
	}
	return s
}

func (t *turboT3) forward(x *mlx.Array, state t3State) *mlx.Array {
	length := x.Dim(1)
	offset := 0
	if state[0].k != nil {
		offset = state[0].k.Dim(2)
	}
	x = mlx.Add(x, t.position.Forward(mlx.Arange(float64(offset), float64(offset+length), 1, mlx.DTypeInt32)))
	for i, l := range t.layers {
		qkv := l.qkv.Forward(l.norm.Forward(x))
		part := func(start int) *mlx.Array {
			return mlx.Transpose(mlx.Reshape(qkv.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, start+1024)), 1, int32(length), 16, 64), 0, 2, 1, 3)
		}
		q, k, v := part(0), part(1024), part(2048)
		k, v = state[i].update(k, v)
		scores := mlx.Mul(mlx.Matmul(q, mlx.Transpose(k, 0, 1, 3, 2)), mlx.FromValue(float32(.125)).AsType(q.DType()))
		positions := mlx.ExpandDims(mlx.Arange(0, float64(k.Dim(2)), 1, mlx.DTypeInt32), 0)
		queries := mlx.ExpandDims(mlx.Arange(float64(offset), float64(offset+length), 1, mlx.DTypeInt32), 1)
		mask := mlx.Where(positions.Greater(queries), mlx.FromValue(float32(math.Inf(-1))).AsType(scores.DType()), mlx.FromValue(float32(0)).AsType(scores.DType()))
		scores = mlx.Add(scores, mask)
		a := mlx.Matmul(mlx.SoftmaxAxis(scores, -1, false), v)
		a = mlx.Reshape(mlx.Transpose(a, 0, 2, 1, 3), 1, int32(length), 1024)
		x = mlx.Add(x, l.out.Forward(a))
		x = mlx.Add(x, l.two.Forward(turboGELU(l.one.Forward(l.ffn.Forward(x)))))
	}
	return t.norm.Forward(x)
}
