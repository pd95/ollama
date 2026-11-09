package chatterbox

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type t3Layer struct {
	q, k, v, out, gate, up, down nn.LinearLayer
	norm, ffn                    *mlx.Array
}
type t3 struct {
	text, speech, textPosition, speechPosition *mlx.Array
	speaker, emotion, q, k, v, out, head       nn.LinearLayer
	query                                      *mlx.Array
	perceiverNorm                              *nn.LayerNorm
	layers                                     []t3Layer
	norm, freqs                                *mlx.Array
	derived                                    []*mlx.Array
}

func loadT3(w *weights) *t3 {
	p := "t3.cond_enc.perceiver"
	t := &t3{text: w.get("t3.text_emb.weight"), speech: w.get("t3.speech_emb.weight"), textPosition: w.get("t3.text_pos_emb.emb.weight"), speechPosition: w.get("t3.speech_pos_emb.emb.weight"), speaker: w.linear("t3.cond_enc.spkr_enc"), emotion: w.linear("t3.cond_enc.emotion_adv_fc"), query: w.get(p + ".pre_attention_query"), perceiverNorm: w.norm(p+".attn.norm", 1e-5), q: w.linear(p + ".attn.to_q"), k: w.linear(p + ".attn.to_k"), v: w.linear(p + ".attn.to_v"), out: w.linear(p + ".attn.proj_out"), head: w.linear("t3.speech_head"), norm: w.get("t3.tfmr.model.norm.weight")}
	for i := range 30 {
		p := fmt.Sprintf("t3.tfmr.model.layers.%d", i)
		t.layers = append(t.layers, t3Layer{q: w.linear(p + ".self_attn.q_proj"), k: w.linear(p + ".self_attn.k_proj"), v: w.linear(p + ".self_attn.v_proj"), out: w.linear(p + ".self_attn.o_proj"), gate: w.linear(p + ".mlp.gate_proj"), up: w.linear(p + ".mlp.up_proj"), down: w.linear(p + ".mlp.down_proj"), norm: w.get(p + ".input_layernorm.weight"), ffn: w.get(p + ".post_attention_layernorm.weight")})
	}
	t.freqs = t3RopeFreqs()
	t.derived = []*mlx.Array{t.freqs}
	return t
}

func t3RopeFreqs() *mlx.Array {
	f := mlx.FromValue(float32(500000)).Power(mlx.Div(mlx.Arange(0, 64, 2, mlx.DTypeInt32), mlx.FromValue(64)))
	wave := mlx.Mul(mlx.FromValue(float32(2*math.Pi)), f)
	f = mlx.Where(wave.Greater(mlx.FromValue(float32(8192))), mlx.Mul(f, mlx.FromValue(8)), f)
	medium := wave.Greater(mlx.FromValue(float32(2048))).BitwiseAnd(wave.Less(mlx.FromValue(float32(8192))))
	smooth := mlx.Div(mlx.Sub(mlx.Div(mlx.FromValue(float32(8192)), wave), mlx.FromValue(float32(1))), mlx.FromValue(float32(3)))
	blended := mlx.Div(f, mlx.Add(mlx.Div(mlx.Sub(mlx.FromValue(float32(1)), smooth), mlx.FromValue(float32(8))), smooth))
	return mlx.Where(medium, blended, f)
}

func (t *t3) perceive(query, context *mlx.Array) *mlx.Array {
	x, y := t.perceiverNorm.Forward(query), t.perceiverNorm.Forward(context)
	split := func(a *mlx.Array) *mlx.Array {
		return mlx.Transpose(mlx.Reshape(a, int32(a.Dim(0)), int32(a.Dim(1)), 4, 256), 0, 2, 1, 3)
	}
	a := mlx.FastScaledDotProductAttention(split(t.q.Forward(x)), split(t.k.Forward(y)), split(t.v.Forward(y)), 1.0/16, mlx.FastSDPANoWindow, "", nil)
	a = mlx.Reshape(mlx.Transpose(a, 0, 2, 1, 3), int32(query.Dim(0)), int32(query.Dim(1)), 1024)
	return mlx.Add(query, t.out.Forward(a))
}

func (t *t3) conditioning(speaker, codes, emotion *mlx.Array) *mlx.Array {
	positions := mlx.Arange(0, float64(codes.Dim(1)), 1, mlx.DTypeInt32)
	ref := mlx.Add(mlx.Take(t.speech, codes, 0), mlx.Take(t.speechPosition, positions, 0))
	query := mlx.BroadcastTo(t.query, int32(ref.Dim(0)), 32, 1024)
	ref = t.perceive(query, ref)
	ref = t.perceive(ref, ref)
	speaker = mlx.ExpandDims(t.speaker.Forward(mlx.Reshape(speaker, -1, 256)), 1)
	return mlx.Concatenate([]*mlx.Array{speaker, ref, t.emotion.Forward(emotion)}, 1)
}

func (t *t3) prompt(conditioning, textIDs *mlx.Array) *mlx.Array {
	positions := mlx.Arange(0, float64(textIDs.Dim(1)), 1, mlx.DTypeInt32)
	text := mlx.Take(t.text, textIDs, 0)
	// CFG zeros the unconditional text before adding learned positions.
	if textIDs.Dim(0) == 2 {
		text = mlx.Concatenate([]*mlx.Array{text.Slice(mlx.Slice(0, 1), mlx.Slice(), mlx.Slice()), mlx.Zeros(text.DType(), 1, text.Dim(1), 1024)}, 0)
	}
	text = mlx.Add(text, mlx.Take(t.textPosition, positions, 0))
	bos := mlx.Add(mlx.Take(t.speech, mlx.FromValues([]int32{6561}, 1, 1), 0), mlx.Take(t.speechPosition, mlx.FromValues([]int32{0}, 1, 1), 0))
	batch := int32(textIDs.Dim(0))
	return mlx.Concatenate([]*mlx.Array{mlx.BroadcastTo(conditioning, batch, int32(conditioning.Dim(1)), 1024), text, mlx.BroadcastTo(bos, batch, 1, 1024), mlx.BroadcastTo(bos, batch, 1, 1024)}, 1)
}

type t3KV struct {
	k, v  *mlx.Array
	scope *mlx.Scope
}
type t3State []t3KV

func newT3State() t3State {
	s := make(t3State, 30)
	for i := range s {
		s[i].scope = mlx.NewScope()
	}
	return s
}

func (s t3State) close() {
	for i := range s {
		s[i].scope.Close()
	}
}

func (k *t3KV) update(key, value *mlx.Array) (*mlx.Array, *mlx.Array) {
	if k.k != nil {
		key = mlx.Concatenate([]*mlx.Array{k.k, key}, 2)
		value = mlx.Concatenate([]*mlx.Array{k.v, value}, 2)
		k.scope.Discard(k.k, k.v)
	}
	k.k, k.v = key, value
	k.scope.Attach(key, value)
	return key, value
}

func (t *t3) forward(x *mlx.Array, state t3State) *mlx.Array {
	batch, length := int32(x.Dim(0)), int32(x.Dim(1))
	mode := ""
	if length > 1 {
		mode = "causal"
	}
	for i, l := range t.layers {
		n := mlx.RMSNormFn(x, l.norm, 1e-5)
		split := func(a *mlx.Array) *mlx.Array { return mlx.Transpose(mlx.Reshape(a, batch, length, 16, 64), 0, 2, 1, 3) }
		q, k, v := split(l.q.Forward(n)), split(l.k.Forward(n)), split(l.v.Forward(n))
		offset := 0
		if state[i].k != nil {
			offset = state[i].k.Dim(2)
		}
		q = mlx.RoPEWithFreqs(q, 64, false, 500000, 1, mlx.FromValue(offset), t.freqs)
		k = mlx.RoPEWithFreqs(k, 64, false, 500000, 1, mlx.FromValue(offset), t.freqs)
		k, v = state[i].update(k, v)
		a := mlx.FastScaledDotProductAttention(q, k, v, 1.0/8, mlx.FastSDPANoWindow, mode, nil)
		x = mlx.Add(x, l.out.Forward(mlx.Reshape(mlx.Transpose(a, 0, 2, 1, 3), batch, length, 1024)))
		n = mlx.RMSNormFn(x, l.ffn, 1e-5)
		x = mlx.Add(x, l.down.Forward(mlx.SwiGLU(l.gate.Forward(n), l.up.Forward(n))))
	}
	return mlx.RMSNormFn(x, t.norm, 1e-5)
}
