package voxtraltts

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type weights struct {
	tensors map[string]*mlx.Array
	factory model.LinearFactory
	cfg     *config
	meta    map[string]*model.TensorQuantInfo
	derived []*mlx.Array
	err     error
}

func (w *weights) get(name string) *mlx.Array {
	a := w.tensors[name]
	if a == nil && w.err == nil {
		w.err = fmt.Errorf("Voxtral TTS missing tensor %s", name)
	}
	return a
}

func (w *weights) linear(name string) nn.LinearLayer {
	a := w.factory.Make(name)
	if a == nil && w.err == nil {
		w.err = fmt.Errorf("Voxtral TTS missing linear %s", name)
	}
	return a
}

func (w *weights) embedding(name string) nn.EmbeddingLayer {
	a := model.MakeEmbeddingLayer(w.tensors, name, w.cfg.Quant.Group, w.cfg.Quant.Bits, w.cfg.Quant.Mode, w.meta)
	if a == nil && w.err == nil {
		w.err = fmt.Errorf("Voxtral TTS missing embedding %s", name)
	}
	return a
}

type kv struct {
	k, v  *mlx.Array
	scope *mlx.Scope
}
type state []kv

func newState(layers int) state {
	s := make(state, layers)
	for i := range s {
		s[i].scope = mlx.NewScope()
	}
	return s
}

func (s state) close() {
	for i := range s {
		s[i].scope.Close()
	}
}

func (s *kv) offset() int {
	if s.k == nil {
		return 0
	}
	return s.k.Dim(2)
}

func (s *kv) update(k, v *mlx.Array) (*mlx.Array, *mlx.Array) {
	if s.k != nil {
		k = mlx.Concatenate([]*mlx.Array{s.k, k}, 2)
		v = mlx.Concatenate([]*mlx.Array{s.v, v}, 2)
		s.scope.Discard(s.k, s.v)
	}
	s.scope.Attach(k, v)
	s.k, s.v = k, v
	return k, v
}

type layer struct {
	q, k, v, o, gate, up, down                        nn.LinearLayer
	norm, ffn, qnorm, knorm, attentionScale, ffnScale *mlx.Array
}
type transformer struct {
	dims   dimensions
	layers []layer
	norm   *mlx.Array
	kind   string
}

func loadTransformer(w *weights, p string, d dimensions, kind string) *transformer {
	t := &transformer{dims: d, kind: kind}
	if d.Norm == 0 {
		t.dims.Norm = 1e-5
	}
	if kind != "codec" {
		t.norm = w.get(p + ".norm.weight")
	}
	for i := range d.Layers {
		base := fmt.Sprintf("%s.layers.%d", p, i)
		l := layer{}
		if kind == "backbone" {
			l.q = w.linear(base + ".self_attn.q_proj")
			l.k = w.linear(base + ".self_attn.k_proj")
			l.v = w.linear(base + ".self_attn.v_proj")
			l.o = w.linear(base + ".self_attn.o_proj")
			l.gate = w.linear(base + ".mlp.gate_proj")
			l.up = w.linear(base + ".mlp.up_proj")
			l.down = w.linear(base + ".mlp.down_proj")
			l.norm = w.get(base + ".input_layernorm.weight")
			l.ffn = w.get(base + ".post_attention_layernorm.weight")
		} else {
			l.q = w.linear(base + ".attention.wq")
			l.k = w.linear(base + ".attention.wk")
			l.v = w.linear(base + ".attention.wv")
			l.o = w.linear(base + ".attention.wo")
			l.gate = w.linear(base + ".feed_forward.w1")
			l.up = w.linear(base + ".feed_forward.w3")
			l.down = w.linear(base + ".feed_forward.w2")
			l.norm = w.get(base + ".attention_norm.weight")
			l.ffn = w.get(base + ".ffn_norm.weight")
			if kind == "codec" {
				l.qnorm = w.get(base + ".attention.q_norm.weight")
				l.knorm = w.get(base + ".attention.k_norm.weight")
				l.attentionScale = w.get(base + ".attention_scale")
				l.ffnScale = w.get(base + ".ffn_scale")
			}
		}
		t.layers = append(t.layers, l)
	}
	return t
}

func repeatHeads(x *mlx.Array, repeat int) *mlx.Array {
	if repeat == 1 {
		return x
	}
	return mlx.Reshape(mlx.BroadcastTo(mlx.ExpandDims(x, 2), int32(x.Dim(0)), int32(x.Dim(1)), int32(repeat), int32(x.Dim(2)), int32(x.Dim(3))), int32(x.Dim(0)), int32(x.Dim(1)*repeat), int32(x.Dim(2)), int32(x.Dim(3)))
}

// codecALiBiSlopes follows the original Voxtral codec's geometric sequence.
// The supported eight-head codec starts at one, before successive halvings.
func codecALiBiSlopes(heads int) []float32 {
	slopes := make([]float32, heads)
	for head := range slopes {
		slopes[head] = float32(math.Exp2(-float64(head)))
	}
	return slopes
}

func (t *transformer) attention(l layer, x *mlx.Array, cache *kv, window int) *mlx.Array {
	b, length := int32(x.Dim(0)), int32(x.Dim(1))
	d := t.dims
	q, k, v := l.q.Forward(x), l.k.Forward(x), l.v.Forward(x)
	if t.kind == "codec" {
		q = mlx.RMSNormFn(q, l.qnorm, 1e-6)
		k = mlx.RMSNormFn(k, l.knorm, 1e-6)
	}
	q = mlx.Transpose(mlx.Reshape(q, b, length, int32(d.Heads), int32(d.HeadDim)), 0, 2, 1, 3)
	k = mlx.Transpose(mlx.Reshape(k, b, length, int32(d.KVHeads), int32(d.HeadDim)), 0, 2, 1, 3)
	v = mlx.Transpose(mlx.Reshape(v, b, length, int32(d.KVHeads), int32(d.HeadDim)), 0, 2, 1, 3)
	var attention *mlx.Array
	scale := float32(1 / math.Sqrt(float64(d.HeadDim)))
	if t.kind == "backbone" {
		offset := 0
		if cache != nil {
			offset = cache.offset()
		}
		position := mlx.FromValue(offset)
		q = mlx.RoPEWithBase(q, d.HeadDim, true, d.Rope, 1, position)
		k = mlx.RoPEWithBase(k, d.HeadDim, true, d.Rope, 1, position)
		if cache != nil {
			k, v = cache.update(k, v)
		}
		mode := ""
		if length > 1 {
			mode = "causal"
		}
		attention = mlx.FastScaledDotProductAttention(q, k, v, scale, mlx.FastSDPANoWindow, mode, nil)
	} else {
		k = repeatHeads(k, d.Heads/d.KVHeads)
		v = repeatHeads(v, d.Heads/d.KVHeads)
		scores := mlx.Mul(mlx.Matmul(q, mlx.Transpose(k, 0, 1, 3, 2)), mlx.FromValue(scale))
		if t.kind == "codec" {
			positions := mlx.Arange(0, float64(length), 1, mlx.DTypeInt32)
			distance := mlx.Sub(mlx.ExpandDims(positions, 0), mlx.ExpandDims(positions, 1))
			slopes := codecALiBiSlopes(d.Heads)
			alibi := mlx.Mul(mlx.FromValues(slopes, d.Heads, 1, 1), mlx.ExpandDims(distance.AsType(mlx.DTypeFloat32), 0))
			causal := mlx.Where(distance.Greater(mlx.FromValue(0)), mlx.FromValue(float32(-1e9)), mlx.FromValue(float32(0)))
			if window > 0 {
				causal = mlx.Add(causal, mlx.Where(distance.Less(mlx.FromValue(-window)), mlx.FromValue(float32(-1e9)), mlx.FromValue(float32(0))))
			}
			scores = mlx.Add(mlx.Add(scores, alibi), mlx.Reshape(causal, 1, 1, length, length))
		}
		weights := mlx.SoftmaxAxis(scores.AsType(mlx.DTypeFloat32), -1, false).AsType(x.DType())
		attention = mlx.Matmul(weights, v)
	}
	return l.o.Forward(mlx.Reshape(mlx.Transpose(attention, 0, 2, 1, 3), b, length, int32(d.Heads*d.HeadDim)))
}

func (t *transformer) forward(x *mlx.Array, caches state, window int) *mlx.Array {
	for i, l := range t.layers {
		var cache *kv
		if caches != nil {
			cache = &caches[i]
		}
		h := t.attention(l, mlx.RMSNormFn(x, l.norm, t.dims.Norm), cache, window)
		if l.attentionScale != nil {
			h = mlx.Mul(h, l.attentionScale)
		}
		x = mlx.Add(x, h)
		norm := mlx.RMSNormFn(x, l.ffn, t.dims.Norm)
		h = l.down.Forward(mlx.Mul(mlx.SiLU(l.gate.Forward(norm)), l.up.Forward(norm)))
		if l.ffnScale != nil {
			h = mlx.Mul(h, l.ffnScale)
		}
		x = mlx.Add(x, h)
	}
	if t.norm != nil {
		x = mlx.RMSNormFn(x, t.norm, t.dims.Norm)
	}
	return x
}

type acoustic struct {
	transformer                        *transformer
	input, llm, time, semantic, output nn.LinearLayer
	frequency                          *mlx.Array
}

func loadAcoustic(w *weights) *acoustic {
	p := "acoustic_transformer"
	a := &acoustic{transformer: loadTransformer(w, p, w.cfg.Multimodal.Audio.Flow.Dimensions, "flow"), input: w.linear(p + ".input_projection"), llm: w.linear(p + ".llm_projection"), time: w.linear(p + ".time_projection"), semantic: w.linear(p + ".semantic_codebook_output"), output: w.linear(p + ".acoustic_codebook_output")}
	half := w.cfg.Multimodal.Audio.Flow.Dim / 2
	a.frequency = mlx.Exp(mlx.Div(mlx.Mul(mlx.FromValue(float32(-math.Log(10000))), mlx.Arange(0, float64(half), 1, mlx.DTypeInt32).AsType(mlx.DTypeFloat32)), mlx.FromValue(half)))
	w.derived = append(w.derived, a.frequency)
	return a
}

func (a *acoustic) velocity(x, time, hidden *mlx.Array) *mlx.Array {
	angles := mlx.Mul(mlx.Reshape(time, int32(time.Size()), 1), a.frequency)
	t := a.time.Forward(mlx.Concatenate([]*mlx.Array{mlx.Cos(angles), mlx.Sin(angles)}, 1))
	stack := mlx.Stack([]*mlx.Array{a.input.Forward(x), t, a.llm.Forward(hidden)}, 1)
	out := a.transformer.forward(stack, nil, 0)
	return a.output.Forward(mlx.Squeeze(out.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice()), 1))
}

func (a *acoustic) flow(hidden, noise *mlx.Array) *mlx.Array {
	x := noise
	for step := range 7 {
		start, end := float64(step)/7, float64(step+1)/7
		times := mlx.BroadcastTo(mlx.FromValue(float32(start)), 2)
		both := a.velocity(mlx.Concatenate([]*mlx.Array{x, x}, 0), times, mlx.Concatenate([]*mlx.Array{hidden, mlx.Zeros(hidden.DType(), hidden.Dims()...)}, 0))
		cond, uncond := both.Slice(mlx.Slice(0, 1), mlx.Slice()), both.Slice(mlx.Slice(1, 2), mlx.Slice())
		v := mlx.Add(mlx.Mul(mlx.FromValue(float32(1.2)), cond), mlx.Mul(mlx.FromValue(float32(-.2)), uncond))
		x = mlx.Add(x, mlx.Mul(v, mlx.FromValue(float32(end-start))))
	}
	return x
}

func (a *acoustic) codes(hidden, noise *mlx.Array) ([]int32, error) {
	logits := mlx.Contiguous(a.semantic.Forward(hidden).AsType(mlx.DTypeFloat32), false).Floats()
	codes := make([]int32, 37)
	best := float32(math.Inf(-1))
	for i := 1; i < 8194; i++ {
		if math.IsNaN(float64(logits[i])) || math.IsInf(float64(logits[i]), 0) {
			return nil, fmt.Errorf("nonfinite Voxtral semantic logits")
		}
		if logits[i] > best {
			best, codes[0] = logits[i], int32(i)
		}
	}
	if codes[0] == 1 {
		return codes, nil
	}
	x := mlx.Clip(a.flow(hidden, noise), mlx.FromValue(float32(-1)), mlx.FromValue(float32(1)))
	x = mlx.Div(mlx.Mul(mlx.Add(x, mlx.FromValue(float32(1))), mlx.FromValue(float32(20))), mlx.FromValue(float32(2)))
	values := mlx.Contiguous(x, false).Floats()
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("nonfinite Voxtral acoustic codes")
		}
		codes[i+1] = int32(math.RoundToEven(float64(value))) + 2
	}
	return codes, nil
}

type network struct {
	text, audio nn.EmbeddingLayer
	backbone    *transformer
	acoustic    *acoustic
	derived     []*mlx.Array
}

func loadNetwork(cfg *config, tensors map[string]*mlx.Array, meta map[string]*model.TensorQuantInfo) (*network, error) {
	w := &weights{tensors: tensors, cfg: cfg, meta: meta, factory: model.NewLinearFactory(tensors, cfg.Quant.Group, cfg.Quant.Bits, cfg.Quant.Mode, meta)}
	n := &network{text: w.embedding("language_model.model.model.embed_tokens"), audio: w.embedding("audio_codebook_embeddings.embeddings"), backbone: loadTransformer(w, "language_model.model.model", cfg.Dimensions, "backbone"), acoustic: loadAcoustic(w)}
	n.derived = w.derived
	return n, w.err
}

func (n *network) condition(ids, voice *mlx.Array) *mlx.Array {
	embedding := n.text.Forward(ids)
	mask := ids.Equal(mlx.FromValue(24)).AsType(mlx.DTypeInt32)
	indices := mlx.Clip(mlx.Sub(mask.Cumsum(1, false, true), mlx.FromValue(1)), mlx.FromValue(0), mlx.FromValue(voice.Dim(0)-1))
	expanded := mlx.Take(voice, indices, 0).AsType(embedding.DType())
	mask = mlx.ExpandDims(mask.AsType(embedding.DType()), 2)
	return mlx.Add(embedding, mlx.Mul(mask, mlx.Sub(expanded, embedding)))
}

func (n *network) feedback(codes *mlx.Array) *mlx.Array {
	offsets := make([]int32, 37)
	for i := 1; i < len(offsets); i++ {
		offsets[i] = 8194 + int32(i-1)*23
	}
	return mlx.Sum(n.audio.Forward(mlx.Add(codes, mlx.FromValues(offsets, 1, 37))), 1, true)
}
