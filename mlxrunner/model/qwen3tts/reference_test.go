package qwen3tts

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestConfig(t *testing.T) {
	raw, err := os.ReadFile("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := parseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Talker.Hidden != 1024 || cfg.Talker.Groups != 16 {
		t.Fatal("wrong configuration")
	}
	var values map[string]any
	if err := json.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}
	values["tts_model_type"] = "base"
	bad, err := json.Marshal(values)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := parseConfig(bad); err == nil {
		t.Fatal("Base requires a separate reference-audio execution path")
	}
}

func compare(t *mlxtest.T, label string, got, want *mlx.Array, atol, rtol float64) {
	t.Helper()
	if got == nil || want == nil || !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("%s: shape mismatch", label)
	}
	g := mlx.Contiguous(got.AsType(mlx.DTypeFloat32), false).Floats()
	w := mlx.Contiguous(want.AsType(mlx.DTypeFloat32), false).Floats()
	maxError := 0.0
	bad := 0
	for i, v := range g {
		err := math.Abs(float64(v - w[i]))
		maxError = math.Max(maxError, err)
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || err > atol+rtol*math.Abs(float64(w[i])) {
			bad++
		}
	}
	t.Logf("%s (%v/%v): max absolute error %.7g; %d/%d outside tolerance", label, got.DType(), want.DType(), maxError, bad, len(g))
	if bad != 0 {
		t.Errorf("%s: numerical mismatch", label)
	}
}

func sourceTensors(t *mlxtest.T, path string) map[string]*mlx.Array {
	t.Helper()
	file, err := mlx.LoadSafetensorsNative(path)
	if err != nil {
		t.Fatal(err)
	}
	file.Free()
	tensors := map[string]*mlx.Array{}
	for name, value := range mlx.Load(path) {
		if strings.HasSuffix(name, ".scales") {
			name = strings.TrimSuffix(name, ".scales") + ".weight_scale"
		}
		if strings.HasSuffix(name, ".biases") {
			name = strings.TrimSuffix(name, ".biases") + ".weight_qbias"
		}
		if strings.HasSuffix(name, ".scale") {
			name = strings.TrimSuffix(name, ".scale") + "_scale"
		}
		tensors[name] = value
	}
	return tensors
}

func TestReference(t *testing.T) {
	source, reference := os.Getenv("OLLAMA_QWEN3_TTS_SOURCE"), os.Getenv("OLLAMA_SPEECH_REFERENCE_DIR")
	if source == "" || reference == "" {
		t.Skip("set source and reference directories for native parity checks")
	}
	if err := mlx.CheckInit(); err != nil {
		t.Fatal(err)
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		path, _ := mlx.LoadedLibraryPath()
		t.Logf("backend %s", path)
		raw, err := os.ReadFile(filepath.Join(source, "config.json"))
		if err != nil {
			t.Fatal(err)
		}
		cfg, err := parseConfig(raw)
		if err != nil {
			t.Fatal(err)
		}
		network, err := loadNetwork(cfg, sourceTensors(t, filepath.Join(source, "model.safetensors")), nil)
		if err != nil {
			t.Fatal(err)
		}
		codecRaw, err := os.ReadFile(filepath.Join(source, "speech_tokenizer/config.json"))
		if err != nil {
			t.Fatal(err)
		}
		decoder, err := loadDecoder(codecRaw, sourceTensors(t, filepath.Join(source, "speech_tokenizer/model.safetensors")))
		if err != nil {
			t.Fatal(err)
		}
		for _, language := range []string{"en", "de"} {
			want := sourceTensors(t, filepath.Join(reference, "hello-"+language+".safetensors"))
			if want["text_embedding"] != nil {
				embed := network.text.Forward(mlx.Reshape(want["text_ids"], 1, int32(want["text_ids"].Size())))
				compare(t, language+" embedding", embed, want["text_embedding"], 0, 0)
				one := network.fc1.Forward(embed)
				compare(t, language+" projection fc1", one, want["projection_fc1"], 1e-4, 1e-3)
				activated := mlx.SiLU(one)
				compare(t, language+" projection silu", activated, want["projection_silu"], 1e-4, 1e-3)
				compare(t, language+" projection fc2", network.fc2.Forward(activated), want["projection_fc2"], 1e-4, 1e-3)
				norm := mlx.RMSNormFn(want["input_embeds"], network.talker.body.layers[0].norm, cfg.Talker.Eps)
				compare(t, language+" first norm", norm, want["layer0_norm"], 1e-4, 1e-3)
				query := network.talker.body.layers[0].q.Forward(norm)
				compare(t, language+" first q projection", query, want["layer0_q_proj"], 1e-4, 1e-3)
				query = mlx.RMSNormFn(mlx.Reshape(query, 1, int32(query.Dim(1)), int32(cfg.Talker.Heads), int32(cfg.Talker.Head)), network.talker.body.layers[0].qnorm, cfg.Talker.Eps)
				compare(t, language+" first q norm", query, want["layer0_q_norm"], 1e-4, 1e-3)
			}
			if want["layer0_q_rotary"] != nil {
				l := network.talker.body.layers[0]
				x := want["input_embeds"]
				length := int32(x.Dim(1))
				norm := mlx.RMSNormFn(x, l.norm, cfg.Talker.Eps)
				q := mlx.RMSNormFn(mlx.Reshape(l.q.Forward(norm), 1, length, 16, 128), l.qnorm, cfg.Talker.Eps)
				k := mlx.RMSNormFn(mlx.Reshape(l.k.Forward(norm), 1, length, 8, 128), l.knorm, cfg.Talker.Eps)
				v := mlx.Transpose(mlx.Reshape(l.v.Forward(norm), 1, length, 8, 128), 0, 2, 1, 3)
				cos, sin := rotaryFrequencies(128, int(length), 0, cfg.Talker.Rope, x.DType(), true)
				compare(t, language+" cos", mlx.Squeeze(cos, 1), want["rotary_cos"], 0, 0)
				compare(t, language+" sin", mlx.Squeeze(sin, 1), want["rotary_sin"], 0, 0)
				compare(t, language+" rotary exact cos", applyRotary(mlx.Transpose(q, 0, 2, 1, 3), mlx.ExpandDims(want["rotary_cos"], 1), mlx.ExpandDims(want["rotary_sin"], 1)), want["layer0_q_rotary"], 1e-4, 1e-3)
				q = rotary(mlx.Transpose(q, 0, 2, 1, 3), 0, cfg.Talker.Rope, true)
				k = rotary(mlx.Transpose(k, 0, 2, 1, 3), 0, cfg.Talker.Rope, true)
				compare(t, language+" query rotary", q, want["layer0_q_rotary"], 1e-4, 1e-3)
				compare(t, language+" key rotary", k, want["layer0_k_rotary"], 1e-4, 1e-3)
				mask := make([]float32, length*length)
				for row := range int(length) {
					for col := row + 1; col < int(length); col++ {
						mask[row*int(length)+col] = -math.MaxFloat32
					}
				}
				attn := mlx.FastScaledDotProductAttention(q, k, v, float32(1/math.Sqrt(128)), mlx.FastSDPANoWindow, "array", mlx.FromValues(mask, int(length), int(length)).AsType(x.DType()))
				compare(t, language+" attention", attn, want["layer0_attention"], 1e-4, 1e-3)
				out := l.o.Forward(mlx.Reshape(mlx.Transpose(attn, 0, 2, 1, 3), 1, length, 2048))
				compare(t, language+" o projection", out, want["layer0_o_proj"], 1e-4, 1e-3)
				residual := mlx.Add(x, out)
				compare(t, language+" residual", residual, want["layer0_residual"], 1e-4, 1e-3)
				norm = mlx.RMSNormFn(residual, l.ffn, cfg.Talker.Eps)
				compare(t, language+" ffn norm", norm, want["layer0_ffnorm"], 1e-4, 1e-3)
				ff := mlx.SwiGLU(l.gate.Forward(norm), l.up.Forward(norm))
				compare(t, language+" swiglu", ff, want["layer0_swiglu"], 1e-4, 1e-3)
				compare(t, language+" down", l.down.Forward(ff), want["layer0_down"], 1e-4, 1e-3)
			}
			if want["layer0_output"] != nil {
				first := *network.talker.body
				first.layers = first.layers[:1]
				compare(t, language+" layer0 output", first.layersForward(want["input_embeds"], nil), want["layer0_output"], 1e-4, 1e-3)
			}
			prepared, padding, err := network.prepare(want["text_ids"].Ints(), language, "Ryan")
			if err != nil {
				t.Fatal(err)
			}
			if want["instruction_ids"] != nil {
				prepared = mlx.Concatenate([]*mlx.Array{network.project(want["instruction_ids"].Ints()), prepared}, 1)
			}
			compare(t, language+" input", prepared, want["input_embeds"], 1e-4, 1e-3)
			compare(t, language+" pad", padding, want["tts_pad_embed"], 1e-4, 1e-3)
			state := newState(len(network.talker.body.layers))
			logits, hidden := network.talker.forward(want["input_embeds"], state)
			compare(t, language+" talker logits", logits, want["talker_logits"], 1e-4, 1e-3)
			compare(t, language+" talker hidden", hidden, want["talker_hidden"], 1e-4, 1e-3)
			if want["predictor_logits_0"] != nil {
				pred := newState(len(network.predictor.layers))
				known := want["speech_codes"].AsType(mlx.DTypeInt32).Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice()).Ints()
				for i := range 15 {
					var x *mlx.Array
					if i == 0 {
						x = mlx.Concatenate([]*mlx.Array{timeSlice(hidden, hidden.Dim(1)-1, hidden.Dim(1)), network.codec.Forward(mlx.FromValues(known[:1], 1, 1))}, 1)
					} else {
						x = network.codeEmb[i-1].Forward(mlx.FromValues(known[i:i+1], 1, 1))
					}
					if network.small != nil {
						x = network.small.Forward(x)
					}
					out := network.codeHead[i].Forward(network.predictor.forward(x, pred))
					compare(t, fmt.Sprintf("%s predictor %d", language, i), out, want[fmt.Sprintf("predictor_logits_%d", i)], 0, 0)
				}
				pred.close()
				next := network.codec.Forward(mlx.FromValues(known[:1], 1, 1))
				for i := range 15 {
					next = mlx.Add(next, network.codeEmb[i].Forward(mlx.FromValues(known[i+1:i+2], 1, 1)))
				}
				next = mlx.Add(next, padding)
				compare(t, language+" next input", next, want["next_input_embeds"], 0, 0)
				nextLogits, nextHidden := network.talker.forward(next, state)
				compare(t, language+" cached talker logits", nextLogits, want["talker_next_logits"], 0, 0)
				compare(t, language+" cached talker hidden", nextHidden, want["talker_next_hidden"], 0, 0)
			}
			state.close()
			if want["decode_quantizer"] != nil {
				codes := want["speech_codes"]
				code := func(i int) *mlx.Array {
					return mlx.Squeeze(codes.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(i, i+1)), 2)
				}
				sem := mlx.Take(decoder.books[0], code(0), 0)
				ac := mlx.Zeros(mlx.DTypeFloat32, codes.Dim(0), codes.Dim(1), decoder.books[1].Dim(1))
				for i := 1; i < len(decoder.books); i++ {
					ac = mlx.Add(ac, mlx.Take(decoder.books[i], code(i), 0))
				}
				h := mlx.Add(decoder.semantic.forward(sem), decoder.acoustic.forward(ac))
				compare(t, language+" decoder quantizer", h, want["decode_quantizer"], 1e-5, 1e-4)
				h = decoder.pre.forward(h)
				compare(t, language+" decoder pre", h, want["decode_pre"], 1e-5, 1e-4)
				h = decoder.out.Forward(decoder.transformer.forward(decoder.in.Forward(h), nil))
				compare(t, language+" decoder transformer", h, want["decode_transformer"], 1e-5, 1e-4)
				for i, up := range decoder.ups {
					h = up.block.forward(up.up.forward(h))
					compare(t, language+" decoder up", h, want[fmt.Sprintf("decode_up%d", i)], 1e-5, 1e-4)
				}
				h = decoder.initial.forward(h)
				compare(t, language+" decoder initial", h, want["decode_block0"], 1e-5, 1e-4)
				for i, b := range decoder.blocks {
					h = b.forward(h)
					if i == 0 {
						compare(t, language+" decoder block1", h, want["decode_block1"], 1e-5, 1e-4)
					}
				}
				h = decoder.final.forward(decoder.last.forward(h))
				compare(t, language+" decoder final", h, want["decode_block6"], 1e-5, 1e-4)
			}
			audio := decoder.decode(want["speech_codes"])
			if want["fixed_code_decode_eager"] != nil {
				compare(t, language+" eager decoder", audio, want["fixed_code_decode_eager"], 1e-4, 1e-3)
			}
			compare(t, language+" fixed-code decoder", audio, want["fixed_code_decode"], 1e-4, 1e-3)
			prefixCodes := want["speech_codes"].Slice(mlx.Slice(), mlx.Slice(0, 25), mlx.Slice())
			prefixAudio := decoder.decode(prefixCodes)
			compare(t, language+" causal codec prefix", prefixAudio, audio.Slice(mlx.Slice(), mlx.Slice(0, 25*1920)), 1e-4, 1e-3)
		}
	})
}
