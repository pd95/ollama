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

type encoderConfig struct {
	Hidden         int     `json:"hidden_size"`
	Heads          int     `json:"num_attention_heads"`
	KVHeads        int     `json:"num_key_value_heads"`
	Head           int     `json:"head_dim"`
	Layers         int     `json:"num_hidden_layers"`
	Intermediate   int     `json:"intermediate_size"`
	Rate           int     `json:"sampling_rate"`
	Channels       int     `json:"audio_channels"`
	Filters        int     `json:"num_filters"`
	Residuals      int     `json:"num_residual_layers"`
	Ratios         []int   `json:"upsampling_ratios"`
	Kernel         int     `json:"kernel_size"`
	ResidualKernel int     `json:"residual_kernel_size"`
	LastKernel     int     `json:"last_kernel_size"`
	Dilation       int     `json:"dilation_growth_rate"`
	Compress       int     `json:"compress"`
	Causal         bool    `json:"use_causal_conv"`
	Shortcut       bool    `json:"use_conv_shortcut"`
	CodeDim        int     `json:"codebook_dim"`
	CodeSize       int     `json:"codebook_size"`
	Quantizers     int     `json:"num_quantizers"`
	Semantic       int     `json:"num_semantic_quantizers"`
	Rope           float32 `json:"rope_theta"`
	Eps            float32 `json:"norm_eps"`
}

func parseEncoderConfig(raw []byte) (encoderConfig, error) {
	var cfg struct {
		Encoder    encoderConfig `json:"encoder_config"`
		Valid      int           `json:"encoder_valid_num_quantizers"`
		InputRate  int           `json:"input_sample_rate"`
		Downsample int           `json:"encode_downsample_rate"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return encoderConfig{}, err
	}
	c := cfg.Encoder
	if cfg.Valid != 16 || cfg.InputRate != 24000 || cfg.Downsample != 1920 || c.Hidden != 512 || c.Heads != 8 || c.KVHeads != 8 || c.Head != 64 || c.Layers != 8 || c.Intermediate != 2048 || c.Rate != 24000 || c.Channels != 1 || c.Filters != 64 || c.Residuals != 1 || !slices.Equal(c.Ratios, []int{8, 6, 5, 4}) || c.Kernel != 7 || c.ResidualKernel != 3 || c.LastKernel != 3 || c.Dilation != 2 || c.Compress != 2 || !c.Causal || c.Shortcut || c.CodeDim != 256 || c.CodeSize != 2048 || c.Quantizers != 32 || c.Semantic != 1 || c.Rope != 10000 || c.Eps != 1e-5 {
		return c, fmt.Errorf("unsupported Qwen3-TTS speech encoder configuration")
	}
	return c, nil
}

type (
	encoderStage struct{ one, two, down *convolution }
	encoderLayer struct {
		qkv, output, one, two                                  nn.LinearLayer
		normOne, normTwo, biasOne, biasTwo, scaleOne, scaleTwo *mlx.Array
	}
)

type (
	encoderBook   struct{ embedding, c2 *mlx.Array }
	speechEncoder struct {
		initial, final, downsample, semantic, acoustic *convolution
		stages                                         []encoderStage
		layers                                         []encoderLayer
		books                                          []encoderBook
		derived                                        []*mlx.Array
	}
)

func loadEncoder(raw []byte, tensors map[string]*mlx.Array) (*speechEncoder, error) {
	_, err := parseEncoderConfig(raw)
	if err != nil {
		return nil, err
	}
	w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, 0, 0, "", nil)}
	c := func(index, stride int) *convolution {
		return loadConv(w, fmt.Sprintf("encoder.encoder.layers.%d.conv", index), 1, 1, stride, false)
	}
	e := &speechEncoder{initial: c(0, 1), final: c(14, 1), downsample: loadConv(w, "encoder.downsample.conv", 1, 1, 2, false)}
	for i, ratio := range []int{4, 5, 6, 8} {
		p := fmt.Sprintf("encoder.encoder.layers.%d.block", 1+i*3)
		e.stages = append(e.stages, encoderStage{loadConv(w, p+".1.conv", 1, 1, 1, false), loadConv(w, p+".3.conv", 1, 1, 1, false), c(3+i*3, ratio)})
	}
	for i := range 8 {
		p := fmt.Sprintf("encoder.encoder_transformer.layers.%d", i)
		q, k, v := w.get(p+".self_attn.q_proj.weight"), w.get(p+".self_attn.k_proj.weight"), w.get(p+".self_attn.v_proj.weight")
		if q == nil || k == nil || v == nil {
			return nil, w.err
		}
		qkv := mlx.Concatenate([]*mlx.Array{q, k, v}, 0)
		w.derived = append(w.derived, qkv)
		e.layers = append(e.layers, encoderLayer{qkv: &nn.Linear{Weight: qkv}, output: w.linear(p + ".self_attn.o_proj"), one: w.linear(p + ".mlp.fc1"), two: w.linear(p + ".mlp.fc2"), normOne: w.get(p + ".input_layernorm.weight"), biasOne: w.get(p + ".input_layernorm.bias"), normTwo: w.get(p + ".post_attention_layernorm.weight"), biasTwo: w.get(p + ".post_attention_layernorm.bias"), scaleOne: w.layerScale(p + ".self_attn_layer_scale"), scaleTwo: w.layerScale(p + ".mlp_layer_scale")})
	}
	e.semantic = loadConv(w, "encoder.quantizer.semantic_residual_vector_quantizer.input_proj", 1, 1, 1, false)
	e.acoustic = loadConv(w, "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj", 1, 1, 1, false)
	for i := range 16 {
		bank, index := "acoustic", i-1
		if i == 0 {
			bank, index = "semantic", 0
		}
		p := fmt.Sprintf("encoder.quantizer.%s_residual_vector_quantizer.layers.%d.codebook", bank, index)
		sum, usage := w.get(p+".embed_sum"), w.get(p+".cluster_usage")
		if sum == nil || usage == nil {
			return nil, w.err
		}
		emb := mlx.Div(sum, mlx.ExpandDims(mlx.Maximum(usage, mlx.FromValue(float32(1e-5))), 1))
		c2 := mlx.Div(mlx.Sum(emb.Power(mlx.FromValue(2)), -1, false), mlx.FromValue(float32(2)))
		e.books = append(e.books, encoderBook{emb, c2})
		w.derived = append(w.derived, emb, c2)
	}
	e.derived = w.derived
	return e, w.err
}

// Preserve NCL padding before the convolution's layout conversion, as in Mimi.
func encoderConv(c *convolution, x *mlx.Array, edge bool) *mlx.Array {
	kernel, stride := c.weight.Dim(1), c.stride
	padding := kernel - stride
	extra := (stride - x.Dim(2)%stride) % stride
	mode := "constant"
	if edge {
		mode = "edge"
	}
	x = mlx.Pad(x, []int{2}, []int{padding}, []int{extra}, mlx.FromValue(float32(0)), mode)
	x = mlx.Conv1d(mlx.Transpose(x, 0, 2, 1), c.weight, c.bias, int32(stride), 0, 1, 1)
	return mlx.Transpose(x, 0, 2, 1)
}

var encoderELU = mlx.Compile1("QwenSpeechEncoderELU", func(x *mlx.Array) *mlx.Array {
	return mlx.Where(x.Greater(mlx.FromValue(float32(0))), x, mlx.Sub(mlx.Exp(x), mlx.FromValue(float32(1))))
}, mlx.Shapeless())

var encoderGELU = mlx.Compile1("QwenSpeechEncoderGELU", func(x *mlx.Array) *mlx.Array {
	x3 := x.Power(mlx.FromValue(3))
	inner := mlx.Mul(mlx.FromValue(float32(math.Sqrt(2/math.Pi))), mlx.Add(x, mlx.Mul(mlx.FromValue(float32(.044715)), x3)))
	return mlx.Mul(mlx.Mul(mlx.FromValue(float32(.5)), x), mlx.Add(mlx.FromValue(float32(1)), inner.Tanh()))
}, mlx.Shapeless())

func (e *speechEncoder) seanet(x *mlx.Array) *mlx.Array {
	x = encoderConv(e.initial, x, false)
	for _, s := range e.stages {
		r := encoderConv(s.one, encoderELU(x), false)
		r = encoderConv(s.two, encoderELU(r), false)
		x = encoderConv(s.down, encoderELU(mlx.Add(r, x)), false)
	}
	return encoderConv(e.final, encoderELU(x), false)
}

func (e *speechEncoder) transform(x *mlx.Array) *mlx.Array {
	x = mlx.Transpose(x, 0, 2, 1)
	length := x.Dim(1)
	mask := make([]float32, length*length)
	for row := range length {
		for col := row + 1; col < length; col++ {
			mask[row*length+col] = float32(math.Inf(-1))
		}
	}
	m := mlx.FromValues(mask, 1, 1, length, length)
	for _, l := range e.layers {
		n := mlx.LayerNormFn(x, l.normOne, l.biasOne, 1e-5)
		qkv := mlx.Reshape(l.qkv.Forward(n), 1, int32(length), 3, 8, 64)
		part := func(i int) *mlx.Array {
			return mlx.Transpose(mlx.Squeeze(qkv.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice()), 2), 0, 2, 1, 3)
		}
		q, k, v := part(0), part(1), part(2)
		q = mlx.RoPEWithBase(q, 64, false, 10000, 1, mlx.FromValue(0))
		k = mlx.RoPEWithBase(k, 64, false, 10000, 1, mlx.FromValue(0))
		a := mlx.FastScaledDotProductAttention(q, k, v, 1.0/8, mlx.FastSDPANoWindow, "array", m)
		a = l.output.Forward(mlx.Reshape(mlx.Transpose(a, 0, 2, 1, 3), 1, int32(length), 512))
		x = mlx.Add(x, mlx.Mul(a, l.scaleOne))
		ff := l.two.Forward(encoderGELU(l.one.Forward(mlx.LayerNormFn(x, l.normTwo, l.biasTwo, 1e-5))))
		x = mlx.Add(x, mlx.Mul(ff, l.scaleTwo))
	}
	return mlx.Transpose(x, 0, 2, 1)
}

func (e *speechEncoder) quantize(x *mlx.Array) *mlx.Array {
	semantic := encoderConv(e.semantic, x, false)
	residual := encoderConv(e.acoustic, x, false)
	codes := make([]*mlx.Array, 0, 16)
	for i, b := range e.books {
		input := residual
		if i == 0 {
			input = semantic
		}
		input = mlx.Transpose(input, 0, 2, 1)
		flat := mlx.Reshape(input, -1, 256).AsType(mlx.DTypeFloat32)
		dot := mlx.Matmul(flat, mlx.Transpose(b.embedding, 1, 0))
		// argmax(dot-c2) is argmin(c2-dot), including first-index ties.
		indices := mlx.Reshape(mlx.Sub(dot, b.c2).Argmax(-1, false), 1, int32(x.Dim(2)))
		codes = append(codes, mlx.ExpandDims(indices, 1))
		if i > 0 {
			residual = mlx.Sub(residual, mlx.Transpose(mlx.Take(b.embedding, indices, 0), 0, 2, 1))
		}
	}
	return mlx.Concatenate(codes, 1)
}

func (e *speechEncoder) forward(audio *mlx.Array) *mlx.Array {
	x := e.seanet(audio)
	x = e.transform(x)
	x = encoderConv(e.downsample, x, true)
	return e.quantize(x)
}
