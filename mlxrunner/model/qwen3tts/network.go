// Package qwen3tts implements the Qwen3-TTS talker and speech decoder in MLX.
package qwen3tts

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type stackConfig struct {
	Hidden       int     `json:"hidden_size"`
	Layers       int     `json:"num_hidden_layers"`
	Heads        int     `json:"num_attention_heads"`
	KVHeads      int     `json:"num_key_value_heads"`
	Head         int     `json:"head_dim"`
	Intermediate int     `json:"intermediate_size"`
	Eps          float32 `json:"rms_norm_eps"`
	Rope         float32 `json:"rope_theta"`
}

type talkerConfig struct {
	stackConfig
	Groups     int                        `json:"num_code_groups"`
	Vocab      int                        `json:"vocab_size"`
	TextVocab  int                        `json:"text_vocab_size"`
	TextHidden int                        `json:"text_hidden_size"`
	BOS        int32                      `json:"codec_bos_id"`
	EOS        int32                      `json:"codec_eos_token_id"`
	Think      int32                      `json:"codec_think_id"`
	NoThink    int32                      `json:"codec_nothink_id"`
	ThinkBOS   int32                      `json:"codec_think_bos_id"`
	ThinkEOS   int32                      `json:"codec_think_eos_id"`
	Pad        int32                      `json:"codec_pad_id"`
	Languages  map[string]int32           `json:"codec_language_id"`
	Speakers   map[string]int32           `json:"spk_id"`
	Dialects   map[string]json.RawMessage `json:"spk_is_dialect"`
	Predictor  stackConfig                `json:"code_predictor_config"`
}

type config struct {
	Type    string       `json:"tts_model_type"`
	Size    string       `json:"tts_model_size"`
	Talker  talkerConfig `json:"talker_config"`
	TextBOS int32        `json:"tts_bos_token_id"`
	TextEOS int32        `json:"tts_eos_token_id"`
	TextPad int32        `json:"tts_pad_token_id"`
	Quant   struct {
		Group int    `json:"group_size"`
		Bits  int    `json:"bits"`
		Mode  string `json:"mode"`
	} `json:"quantization"`
}

func parseConfig(raw []byte) (*config, error) {
	var cfg config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	if cfg.Type != "custom_voice" || (cfg.Size != "0b6" && cfg.Size != "1b7") {
		return nil, fmt.Errorf("Qwen3-TTS requires 0.6B or 1.7B CustomVoice, got %s %s", cfg.Size, cfg.Type)
	}
	for _, s := range []stackConfig{cfg.Talker.stackConfig, cfg.Talker.Predictor} {
		if s.Hidden <= 0 || s.Layers <= 0 || s.Heads <= 0 || s.KVHeads <= 0 || s.Head <= 0 || s.Head%2 != 0 || s.Heads%s.KVHeads != 0 || s.Eps <= 0 || s.Rope <= 0 {
			return nil, fmt.Errorf("invalid Qwen3-TTS transformer configuration")
		}
	}
	if cfg.Talker.Groups != 16 || cfg.Talker.Vocab != 3072 || cfg.Talker.TextHidden <= 0 || cfg.Talker.TextVocab <= int(cfg.TextEOS) || len(cfg.Talker.Speakers) == 0 {
		return nil, fmt.Errorf("unsupported Qwen3-TTS conditioning layout")
	}
	if cfg.Quant.Bits != 0 && ((cfg.Quant.Bits != 8 && cfg.Quant.Bits != 4) || cfg.Quant.Group != 64 || cfg.Quant.Mode != "affine") {
		return nil, fmt.Errorf("Qwen3-TTS currently requires affine 4/8-bit weights or floating-point weights")
	}
	return &cfg, nil
}

type weights struct {
	tensors map[string]*mlx.Array
	factory model.LinearFactory
	err     error
	derived []*mlx.Array
}

func (w *weights) get(name string) *mlx.Array {
	a := w.tensors[name]
	if a == nil && w.err == nil {
		w.err = fmt.Errorf("Qwen3-TTS missing tensor %s", name)
	}
	return a
}

// The runner normalizes every manifest .scale suffix to _scale, including
// the codec's floating-point layer scales. Raw reference checkpoints keep dots.
func (w *weights) layerScale(name string) *mlx.Array {
	if a := w.tensors[name+".scale"]; a != nil {
		return a
	}
	return w.get(name + "_scale")
}

func (w *weights) linear(name string) nn.LinearLayer {
	a := w.factory.Make(name)
	if a == nil && w.err == nil {
		w.err = fmt.Errorf("Qwen3-TTS missing linear %s", name)
	}
	return a
}

func (w *weights) embedding(name string, group, bits int, mode string, meta map[string]*model.TensorQuantInfo) nn.EmbeddingLayer {
	a := model.MakeEmbeddingLayer(w.tensors, name, group, bits, mode, meta)
	if a == nil && w.err == nil {
		w.err = fmt.Errorf("Qwen3-TTS missing embedding %s", name)
	}
	return a
}

type kv struct {
	k, v   *mlx.Array
	offset int
	scope  *mlx.Scope
}
type state []kv

func newState(n int) state {
	s := make(state, n)
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

func (k *kv) update(key, value *mlx.Array) (*mlx.Array, *mlx.Array) {
	if k.k != nil {
		key = mlx.Concatenate([]*mlx.Array{k.k, key}, 2)
		value = mlx.Concatenate([]*mlx.Array{k.v, value}, 2)
		k.scope.Discard(k.k, k.v)
	}
	k.scope.Attach(key, value)
	k.k, k.v = key, value
	k.offset = key.Dim(2)
	return key, value
}

type layer struct {
	q, k, v, o, gate, up, down                   nn.LinearLayer
	norm, ffn, qnorm, knorm, attnScale, mlpScale *mlx.Array
}
type stack struct {
	cfg     stackConfig
	layers  []layer
	norm    *mlx.Array
	decoder bool
	mrope   bool
}

func loadStack(w *weights, prefix string, cfg stackConfig, headNorm, scales bool) *stack {
	s := &stack{cfg: cfg, decoder: scales, norm: w.get(prefix + ".norm.weight")}
	for i := range cfg.Layers {
		p := fmt.Sprintf("%s.layers.%d", prefix, i)
		l := layer{q: w.linear(p + ".self_attn.q_proj"), k: w.linear(p + ".self_attn.k_proj"), v: w.linear(p + ".self_attn.v_proj"), o: w.linear(p + ".self_attn.o_proj"), gate: w.linear(p + ".mlp.gate_proj"), up: w.linear(p + ".mlp.up_proj"), down: w.linear(p + ".mlp.down_proj"), norm: w.get(p + ".input_layernorm.weight"), ffn: w.get(p + ".post_attention_layernorm.weight")}
		if headNorm {
			l.qnorm = w.get(p + ".self_attn.q_norm.weight")
			l.knorm = w.get(p + ".self_attn.k_norm.weight")
		}
		if scales {
			l.attnScale = w.layerScale(p + ".self_attn_layer_scale")
			l.mlpScale = w.layerScale(p + ".mlp_layer_scale")
		}
		s.layers = append(s.layers, l)
	}
	return s
}

// Speech positions share their temporal/height/width indices, so the talker's
// interleaved MRoPE reduces to the same rotary frequencies on all three axes.
func rotary(x *mlx.Array, offset int, base float32, mrope bool) *mlx.Array {
	d := x.Dim(3)
	n := x.Dim(2)
	cos, sin := rotaryFrequencies(d, n, offset, base, x.DType(), mrope)
	return applyRotary(x, cos, sin)
}

func rotaryFrequencies(d, n, offset int, base float32, dtype mlx.DType, mrope bool) (*mlx.Array, *mlx.Array) {
	inv := mlx.Div(mlx.FromValue(float32(1)), mlx.FromValue(base).Power(mlx.Div(mlx.Arange(0, float64(d), 2, mlx.DTypeFloat32), mlx.FromValue(float32(d)))))
	freq := mlx.Mul(mlx.Reshape(inv, 1, 1, 1, int32(d/2)), mlx.Reshape(mlx.Arange(float64(offset), float64(offset+n), 1, mlx.DTypeFloat32), 1, 1, int32(n), 1))
	if mrope {
		inv = mlx.BroadcastTo(mlx.Reshape(inv, 1, 1, int32(d/2), 1), 3, 1, int32(d/2), 1)
		pos := mlx.BroadcastTo(mlx.Reshape(mlx.Arange(float64(offset), float64(offset+n), 1, mlx.DTypeFloat32), 1, 1, 1, int32(n)), 3, 1, 1, int32(n))
		freq = mlx.Transpose(mlx.Matmul(inv, pos), 0, 1, 3, 2).Slice(mlx.Slice(0, 1), mlx.Slice(), mlx.Slice(), mlx.Slice())
	}
	freq = mlx.Concatenate([]*mlx.Array{freq, freq}, 3)
	cos, sin := mlx.Cos(freq).AsType(dtype), mlx.Sin(freq).AsType(dtype)
	return cos, sin
}

var applyRotary = mlx.Compile3("Qwen3TTSRotary", func(x, cos, sin *mlx.Array) *mlx.Array {
	d := x.Dim(3)
	a := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, d/2))
	b := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(d/2, mlx.End))
	rotated := mlx.Concatenate([]*mlx.Array{b.Negative(), a}, 3)
	return mlx.Add(mlx.Mul(x, cos), mlx.Mul(rotated, sin))
})

func (s *stack) rms(x, weight *mlx.Array) *mlx.Array {
	if !s.decoder {
		return mlx.RMSNormFn(x, weight, s.cfg.Eps)
	}
	f := x.AsType(mlx.DTypeFloat32)
	variance := mlx.Mean(f.Power(mlx.FromValue(2)), -1, true)
	normed := mlx.Mul(f, mlx.RSqrt(mlx.Add(variance, mlx.FromValue(s.cfg.Eps))))
	return mlx.Mul(weight, normed).AsType(x.DType())
}

func (s *stack) forward(x *mlx.Array, caches state) *mlx.Array {
	return s.rms(s.layersForward(x, caches), s.norm)
}

func (s *stack) layersForward(x *mlx.Array, caches state) *mlx.Array {
	offset := 0
	if caches != nil {
		offset = caches[0].offset
	}
	for i, l := range s.layers {
		input := x
		x = mlx.ScopedArrays(func() []*mlx.Array {
			h := s.rms(input, l.norm)
			B, L := int32(h.Dim(0)), int32(h.Dim(1))
			D := int32(s.cfg.Head)
			q := mlx.Reshape(l.q.Forward(h), B, L, int32(s.cfg.Heads), D)
			k := mlx.Reshape(l.k.Forward(h), B, L, int32(s.cfg.KVHeads), D)
			v := mlx.Reshape(l.v.Forward(h), B, L, int32(s.cfg.KVHeads), D)
			if l.qnorm != nil {
				q = mlx.RMSNormFn(q, l.qnorm, s.cfg.Eps)
				k = mlx.RMSNormFn(k, l.knorm, s.cfg.Eps)
			}
			q = rotary(mlx.Transpose(q, 0, 2, 1, 3), offset, s.cfg.Rope, s.mrope)
			k = rotary(mlx.Transpose(k, 0, 2, 1, 3), offset, s.cfg.Rope, s.mrope)
			v = mlx.Transpose(v, 0, 2, 1, 3)
			if caches != nil {
				k, v = caches[i].update(k, v)
			}
			mode := ""
			var mask *mlx.Array
			if L > 1 {
				values := make([]float32, int(L)*k.Dim(2))
				for row := range int(L) {
					for col := offset + row + 1; col < k.Dim(2); col++ {
						values[row*k.Dim(2)+col] = -math.MaxFloat32
					}
				}
				mask = mlx.FromValues(values, int(L), k.Dim(2)).AsType(q.DType())
				mode = "array"
			}
			out := mlx.FastScaledDotProductAttention(q, k, v, float32(1/math.Sqrt(float64(D))), mlx.FastSDPANoWindow, mode, mask)
			out = l.o.Forward(mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, int32(s.cfg.Heads)*D))
			if l.attnScale != nil {
				out = mlx.Mul(out, l.attnScale)
			}
			h = mlx.Add(input, out)
			normed := s.rms(h, l.ffn)
			ffn := l.down.Forward(mlx.SwiGLU(l.gate.Forward(normed), l.up.Forward(normed)))
			if l.mlpScale != nil {
				ffn = mlx.Mul(ffn, l.mlpScale)
			}
			return []*mlx.Array{mlx.Add(h, ffn)}
		})[0]
	}
	return x
}

type talker struct {
	body *stack
	head nn.LinearLayer
}

func (t *talker) forward(x *mlx.Array, s state) (*mlx.Array, *mlx.Array) {
	h := t.body.forward(x, s)
	return t.head.Forward(h), h
}

type network struct {
	cfg         *config
	text, codec nn.EmbeddingLayer
	fc1, fc2    nn.LinearLayer
	talker      *talker
	predictor   *stack
	small       nn.LinearLayer
	codeEmb     []nn.EmbeddingLayer
	codeHead    []nn.LinearLayer
}

func loadNetwork(cfg *config, tensors map[string]*mlx.Array, meta map[string]*model.TensorQuantInfo) (*network, error) {
	w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, cfg.Quant.Group, cfg.Quant.Bits, cfg.Quant.Mode, meta)}
	n := &network{cfg: cfg, text: w.embedding("talker.model.text_embedding", cfg.Quant.Group, cfg.Quant.Bits, cfg.Quant.Mode, meta), codec: w.embedding("talker.model.codec_embedding", cfg.Quant.Group, cfg.Quant.Bits, cfg.Quant.Mode, meta), fc1: w.linear("talker.text_projection.linear_fc1"), fc2: w.linear("talker.text_projection.linear_fc2")}
	body := loadStack(w, "talker.model", cfg.Talker.stackConfig, true, false)
	body.mrope = true
	n.talker = &talker{body: body, head: w.linear("talker.codec_head")}
	n.predictor = loadStack(w, "talker.code_predictor.model", cfg.Talker.Predictor, true, false)
	if cfg.Talker.Predictor.Hidden != cfg.Talker.Hidden {
		n.small = w.linear("talker.code_predictor.small_to_mtp_projection")
	}
	for i := range cfg.Talker.Groups - 1 {
		n.codeEmb = append(n.codeEmb, w.embedding(fmt.Sprintf("talker.code_predictor.model.codec_embedding.%d", i), cfg.Quant.Group, cfg.Quant.Bits, cfg.Quant.Mode, meta))
		n.codeHead = append(n.codeHead, w.linear(fmt.Sprintf("talker.code_predictor.lm_head.%d", i)))
	}
	return n, w.err
}

func (n *network) project(ids []int32) *mlx.Array {
	return n.fc2.Forward(mlx.SiLU(n.fc1.Forward(n.text.Forward(mlx.FromValues(ids, 1, len(ids))))))
}

func timeSlice(a *mlx.Array, start, stop int) *mlx.Array {
	return a.Slice(mlx.Slice(), mlx.Slice(start, stop), mlx.Slice())
}

func (n *network) prepare(ids []int32, language, speaker string) (*mlx.Array, *mlx.Array, error) {
	if len(ids) < 8 {
		return nil, nil, fmt.Errorf("Qwen3-TTS prompt has no text tokens")
	}
	speaker = strings.ToLower(speaker)
	spk, ok := n.cfg.Talker.Speakers[speaker]
	if !ok {
		return nil, nil, fmt.Errorf("unknown Qwen3-TTS voice %q", speaker)
	}
	language = languageName(language)
	if language == "auto" || language == "chinese" {
		var dialect string
		if json.Unmarshal(n.cfg.Talker.Dialects[speaker], &dialect) == nil && dialect != "" {
			language = dialect
		}
	}
	c := n.cfg.Talker
	prefix := []int32{c.NoThink, c.ThinkBOS, c.ThinkEOS}
	if language != "auto" {
		id, ok := c.Languages[language]
		if !ok {
			return nil, nil, fmt.Errorf("unsupported Qwen3-TTS language %q", language)
		}
		prefix = []int32{c.Think, c.ThinkBOS, id, c.ThinkEOS}
	}
	text := n.project(ids)
	special := n.project([]int32{n.cfg.TextBOS, n.cfg.TextEOS, n.cfg.TextPad})
	bos, eos, pad := timeSlice(special, 0, 1), timeSlice(special, 1, 2), timeSlice(special, 2, 3)
	codecIDs := append(prefix, spk, c.Pad, c.BOS)
	codec := n.codec.Forward(mlx.FromValues(codecIDs, 1, len(codecIDs)))
	combined := mlx.Add(mlx.Concatenate([]*mlx.Array{mlx.BroadcastTo(pad, 1, int32(len(codecIDs)-2), int32(c.Hidden)), bos}, 1), timeSlice(codec, 0, len(codecIDs)-1))
	full := mlx.Concatenate([]*mlx.Array{timeSlice(text, 3, len(ids)-5), eos}, 1)
	result := mlx.Concatenate([]*mlx.Array{timeSlice(text, 0, 3), combined, mlx.Add(full, timeSlice(codec, len(codecIDs)-2, len(codecIDs)-1)), mlx.Add(pad, timeSlice(codec, len(codecIDs)-1, len(codecIDs)))}, 1)
	return result, pad, nil
}

func languageName(language string) string {
	language = strings.ToLower(language)
	aliases := map[string]string{"en": "english", "de": "german", "zh": "chinese", "fr": "french", "es": "spanish", "it": "italian", "pt": "portuguese", "ja": "japanese", "ko": "korean", "ru": "russian", "": "auto"}
	if alias, ok := aliases[language]; ok {
		language = alias
	}
	return language
}
