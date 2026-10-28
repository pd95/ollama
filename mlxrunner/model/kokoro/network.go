// Package kokoro implements Kokoro text, prosody and waveform networks in MLX.
package kokoro

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/kokoro/config"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type weights struct {
	tensors map[string]*mlx.Array
	factory model.LinearFactory
	packed  bool
	err     error
	derived []*mlx.Array
}

func (w *weights) get(names ...string) *mlx.Array {
	for _, name := range names {
		if a := w.tensors[name]; a != nil {
			return a
		}
	}
	if w.err == nil {
		w.err = fmt.Errorf("Kokoro missing tensor %s", names[0])
	}
	return nil
}

func (w *weights) linear(name string) nn.LinearLayer {
	a := w.factory.Make(name)
	if a == nil && w.err == nil {
		w.err = fmt.Errorf("Kokoro missing linear %s", name)
	}
	return a
}

func (w *weights) embedding(name string, cfg *config.Config, meta map[string]*model.TensorQuantInfo) nn.EmbeddingLayer {
	a := model.MakeEmbeddingLayer(w.tensors, name, cfg.Quant.Group, cfg.Quant.Bits, cfg.Quant.Mode, meta)
	if a == nil && w.err == nil {
		w.err = fmt.Errorf("Kokoro missing embedding %s", name)
	}
	return a
}

func (w *weights) layerNorm(name string, eps float32) *nn.LayerNorm {
	return &nn.LayerNorm{Weight: w.get(name+".weight", name+".gamma"), Bias: w.get(name+".bias", name+".beta"), Eps: eps}
}

var gelu = mlx.Compile1("KokoroGELU", func(x *mlx.Array) *mlx.Array {
	return mlx.Div(mlx.Mul(x, mlx.Add(mlx.FromValue(float32(1)), mlx.Erf(mlx.Div(x, mlx.FromValue(float32(math.Sqrt2)))))), mlx.FromValue(float32(2)))
}, mlx.Shapeless())

func leaky(x *mlx.Array, slope float32) *mlx.Array {
	return mlx.Maximum(x, mlx.Mul(x, mlx.FromValue(slope)))
}

type bertLayer struct {
	q, k, v, out, ffn, ffnOut nn.LinearLayer
	attentionNorm, ffnNorm    *nn.LayerNorm
}
type bert struct {
	word, position, types nn.EmbeddingLayer
	embeddingNorm         *nn.LayerNorm
	mapping               nn.LinearLayer
	shared                bertLayer
	cfg                   *config.Config
}

func loadBERT(w *weights, cfg *config.Config, meta map[string]*model.TensorQuantInfo) *bert {
	p := "bert.encoder.albert_layer_groups.0.albert_layers.0"
	a := p + ".attention"
	return &bert{word: w.embedding("bert.embeddings.word_embeddings", cfg, meta), position: w.embedding("bert.embeddings.position_embeddings", cfg, meta), types: w.embedding("bert.embeddings.token_type_embeddings", cfg, meta), embeddingNorm: w.layerNorm("bert.embeddings.LayerNorm", 1e-12), mapping: w.linear("bert.encoder.embedding_hidden_mapping_in"), shared: bertLayer{q: w.linear(a + ".query"), k: w.linear(a + ".key"), v: w.linear(a + ".value"), out: w.linear(a + ".dense"), ffn: w.linear(p + ".ffn"), ffnOut: w.linear(p + ".ffn_output"), attentionNorm: w.layerNorm(a+".LayerNorm", 1e-12), ffnNorm: w.layerNorm(p+".full_layer_layer_norm", 1e-12)}, cfg: cfg}
}

func (b *bert) embeddings(ids *mlx.Array) *mlx.Array {
	pos := mlx.Reshape(mlx.Arange(0, float64(ids.Dim(1)), 1, mlx.DTypeInt32), 1, int32(ids.Dim(1)))
	types := mlx.Zeros(mlx.DTypeInt32, ids.Dim(0), ids.Dim(1))
	return b.embeddingNorm.Forward(mlx.Add(mlx.Add(b.word.Forward(ids), b.position.Forward(pos)), b.types.Forward(types)))
}

func (b *bert) forward(ids *mlx.Array) *mlx.Array {
	x := b.mapping.Forward(b.embeddings(ids))
	batch, length, heads, dim := int32(x.Dim(0)), int32(x.Dim(1)), int32(b.cfg.BERT.Heads), int32(b.cfg.BERT.Hidden/b.cfg.BERT.Heads)
	for range b.cfg.BERT.Layers {
		input := x
		x = mlx.ScopedArrays(func() []*mlx.Array {
			l := b.shared
			shape := func(a *mlx.Array) *mlx.Array {
				return mlx.Transpose(mlx.Reshape(a, batch, length, heads, dim), 0, 2, 1, 3)
			}
			q, k, v := shape(l.q.Forward(input)), shape(l.k.Forward(input)), shape(l.v.Forward(input))
			scores := mlx.Div(mlx.Matmul(q, mlx.Transpose(k, 0, 1, 3, 2)), mlx.FromValue(float32(math.Sqrt(float64(dim)))))
			attention := mlx.Matmul(mlx.SoftmaxAxis(scores, -1, false), v)
			attention = mlx.Reshape(mlx.Transpose(attention, 0, 2, 1, 3), batch, length, heads*dim)
			h := l.attentionNorm.Forward(mlx.Add(l.out.Forward(attention), input))
			return []*mlx.Array{l.ffnNorm.Forward(mlx.Add(l.ffnOut.Forward(gelu(l.ffn.Forward(h))), h))}
		})[0]
	}
	return x
}

type (
	lstmDirection struct{ wx, wh, biasI, biasH *mlx.Array }
	lstm          struct{ forward, backward lstmDirection }
)

func loadLSTM(w *weights, p string) *lstm {
	direction := func(reverse bool) lstmDirection {
		old, newName := "", "forward"
		if reverse {
			old = "_reverse"
			newName = "backward"
		}
		return lstmDirection{wx: w.get(p+".weight_ih_l0"+old, p+".Wx_"+newName), wh: w.get(p+".weight_hh_l0"+old, p+".Wh_"+newName), biasI: w.get(p+".bias_ih_l0"+old, p+".bias_ih_"+newName), biasH: w.get(p+".bias_hh_l0"+old, p+".bias_hh_"+newName)}
	}
	return &lstm{direction(false), direction(true)}
}

func (d lstmDirection) run(x *mlx.Array, reverse bool) *mlx.Array {
	hiddenSize := d.wh.Dim(1)
	hidden := mlx.Zeros(mlx.DTypeFloat32, x.Dim(0), hiddenSize)
	cell := mlx.Zeros(mlx.DTypeFloat32, x.Dim(0), hiddenSize)
	projected := mlx.Add(d.biasI, d.biasH).Addmm(x, mlx.Transpose(d.wx, 1, 0), 1, 1)
	out := make([]*mlx.Array, x.Dim(1))
	for step := range out {
		i := step
		if reverse {
			i = len(out) - 1 - step
		}
		gates := mlx.Add(mlx.Squeeze(projected.Slice(mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice()), 1), mlx.Matmul(hidden, mlx.Transpose(d.wh, 1, 0)))
		gate := func(index int) *mlx.Array {
			return gates.Slice(mlx.Slice(), mlx.Slice(index*hiddenSize, (index+1)*hiddenSize))
		}
		in, forget, candidate, output := gate(0).Sigmoid(), gate(1).Sigmoid(), gate(2).Tanh(), gate(3).Sigmoid()
		cell = mlx.Add(mlx.Mul(forget, cell), mlx.Mul(in, candidate))
		hidden = mlx.Mul(output, cell.Tanh())
		out[i] = hidden
	}
	return mlx.Stack(out, 1)
}

func (l *lstm) run(x *mlx.Array) *mlx.Array {
	return mlx.Concatenate([]*mlx.Array{l.forward.run(x, false), l.backward.run(x, true)}, 2)
}

type convolution struct {
	weight, bias                      *mlx.Array
	stride, padding, dilation, groups int32
	transpose                         bool
}

func loadWNConv(w *weights, p string, stride, padding, dilation, groups int, transpose bool) *convolution {
	v, g := w.get(p+".weight_v"), w.get(p+".weight_g")
	if v == nil || g == nil {
		return nil
	}
	if !w.packed {
		v = mlx.Transpose(v, 0, 2, 1)
	}
	norm := mlx.SumAxes(mlx.Mul(v, v), []int{1, 2}, true).Sqrt()
	weight := mlx.Mul(mlx.Div(v, mlx.Add(norm, mlx.FromValue(float32(1e-7)))), g)
	if transpose && groups == 1 {
		weight = mlx.Transpose(weight, 2, 1, 0)
	}
	w.derived = append(w.derived, weight)
	return &convolution{weight: weight, bias: w.tensors[p+".bias"], stride: int32(stride), padding: int32(padding), dilation: int32(dilation), groups: int32(groups), transpose: transpose}
}

func (c *convolution) run(x *mlx.Array) *mlx.Array {
	if c.transpose {
		return mlx.ConvTranspose1d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, 0, c.groups)
	}
	return mlx.Conv1d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups)
}

type durationLayer struct {
	lstm  *lstm
	style nn.LinearLayer
}
type textConvolution struct {
	conv *convolution
	norm *nn.LayerNorm
}

// Encoders holds the phoneme, duration and shared prosody recurrent networks.
type Encoders struct {
	bert                           *bert
	projection, duration           nn.LinearLayer
	durationLayers                 []durationLayer
	durationLSTM, shared, textLSTM *lstm
	textEmbedding                  nn.EmbeddingLayer
	textConvs                      []textConvolution
	Derived                        []*mlx.Array
}

func LoadEncoders(cfg *config.Config, tensors map[string]*mlx.Array, meta map[string]*model.TensorQuantInfo) (*Encoders, error) {
	w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, cfg.Quant.Group, cfg.Quant.Bits, cfg.Quant.Mode, meta), packed: cfg.Quant.Bits != 0}
	e := &Encoders{bert: loadBERT(w, cfg, meta), projection: w.linear("bert_encoder"), duration: w.linear("predictor.duration_proj.linear_layer"), durationLSTM: loadLSTM(w, "predictor.lstm"), shared: loadLSTM(w, "predictor.shared"), textLSTM: loadLSTM(w, "text_encoder.lstm"), textEmbedding: w.embedding("text_encoder.embedding", cfg, meta)}
	for i := range cfg.Layers {
		e.durationLayers = append(e.durationLayers, durationLayer{lstm: loadLSTM(w, fmt.Sprintf("predictor.text_encoder.lstms.%d", 2*i)), style: w.linear(fmt.Sprintf("predictor.text_encoder.lstms.%d.fc", 2*i+1))})
		p := fmt.Sprintf("text_encoder.cnn.%d", i)
		e.textConvs = append(e.textConvs, textConvolution{conv: loadWNConv(w, p+".0", 1, cfg.TextKernel/2, 1, 1, false), norm: w.layerNorm(p+".1", 1e-5)})
	}
	e.Derived = w.derived
	return e, w.err
}

func affineLayerNorm(x, style *mlx.Array, projection nn.LinearLayer) *mlx.Array {
	channels := x.Dim(2)
	h := projection.Forward(style)
	gamma := mlx.Reshape(h.Slice(mlx.Slice(), mlx.Slice(0, channels)), int32(x.Dim(0)), 1, int32(channels))
	beta := mlx.Reshape(h.Slice(mlx.Slice(), mlx.Slice(channels, 2*channels)), int32(x.Dim(0)), 1, int32(channels))
	norm := mlx.Div(mlx.Sub(x, mlx.Mean(x, -1, true)), mlx.Add(mlx.Variance(x, -1, true), mlx.FromValue(float32(1e-5))).Sqrt())
	return mlx.Add(mlx.Mul(mlx.Add(mlx.FromValue(float32(1)), gamma), norm), beta)
}

func (e *Encoders) encodeDuration(bertEncoded, style *mlx.Array) *mlx.Array {
	s := mlx.BroadcastTo(mlx.ExpandDims(style, 1), int32(bertEncoded.Dim(0)), int32(bertEncoded.Dim(1)), int32(style.Dim(1)))
	x := mlx.Concatenate([]*mlx.Array{bertEncoded, s}, 2)
	for _, layer := range e.durationLayers {
		x = layer.lstm.run(x)
		x = affineLayerNorm(x, style, layer.style)
		x = mlx.Concatenate([]*mlx.Array{x, s}, 2)
	}
	return x
}

func (e *Encoders) encodeText(ids *mlx.Array) *mlx.Array {
	x := e.textEmbedding.Forward(ids)
	for _, layer := range e.textConvs {
		x = leaky(layer.norm.Forward(layer.conv.run(x)), .2)
	}
	return e.textLSTM.run(x)
}
