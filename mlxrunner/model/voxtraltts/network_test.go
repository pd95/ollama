package voxtraltts

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
)

func loadTensors(t *mlxtest.T, path string) map[string]*mlx.Array {
	t.Helper()
	checked, err := mlx.LoadSafetensorsNative(path)
	if err != nil {
		t.Fatal(err)
	}
	checked.Free()
	result := map[string]*mlx.Array{}
	for name, a := range mlx.Load(path) {
		if strings.HasSuffix(name, ".scales") {
			name = strings.TrimSuffix(name, ".scales") + ".weight_scale"
		}
		if strings.HasSuffix(name, ".biases") {
			name = strings.TrimSuffix(name, ".biases") + ".weight_qbias"
		}
		result[name] = a
	}
	return result
}

func compare(t *mlxtest.T, label string, got, want *mlx.Array) {
	t.Helper()
	if got == nil || want == nil || !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("%s shapes %v / %v", label, got, want)
	}
	g := mlx.Contiguous(got.AsType(mlx.DTypeFloat32), false).Floats()
	w := mlx.Contiguous(want.AsType(mlx.DTypeFloat32), false).Floats()
	worst := 0.
	bad := 0
	for i, value := range g {
		difference := math.Abs(float64(value - w[i]))
		worst = math.Max(worst, difference)
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || difference > 1e-5+1e-4*math.Abs(float64(w[i])) {
			bad++
		}
	}
	t.Logf("%s max error %.7g, %d/%d outside tolerance", label, worst, bad, len(g))
	if bad > 0 {
		t.Errorf("%s numerical mismatch", label)
	}
}

func TestReferenceNetwork(t *testing.T) {
	source, reference := os.Getenv("OLLAMA_VOXTRAL_TTS_SOURCE"), os.Getenv("OLLAMA_SPEECH_REFERENCE_DIR")
	if source == "" || reference == "" {
		t.Skip("set source/reference for paired Voxtral checks")
	}
	if err := mlx.CheckInit(); err != nil {
		t.Fatal(err)
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			mlx.EnableCompile()
			raw, err := os.ReadFile(filepath.Join(source, "config.json"))
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := parseConfig(raw)
			if err != nil {
				t.Fatal(err)
			}
			tensors := map[string]*mlx.Array{}
			files, err := filepath.Glob(filepath.Join(source, "model*.safetensors"))
			if err != nil || len(files) == 0 {
				t.Fatal("missing weights", err)
			}
			for _, file := range files {
				for name, a := range loadTensors(t, file) {
					tensors[name] = a
				}
			}
			w := &weights{tensors: tensors, cfg: cfg, factory: model.NewLinearFactory(tensors, cfg.Quant.Group, cfg.Quant.Bits, cfg.Quant.Mode, nil)}
			decoder := loadCodec(w)
			if w.err != nil {
				t.Fatal(w.err)
			}
			n, err := loadNetwork(cfg, tensors, nil)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"hello-en", "hello-de"} {
				mlx.Scoped(func() {
					want := loadTensors(t, filepath.Join(reference, name+".safetensors"))
					state := newState(cfg.Layers)
					defer state.close()
					voice := "casual_male"
					if name == "hello-de" {
						voice = "de_male"
					}
					preset := loadTensors(t, filepath.Join(source, "voice_embedding", voice+".safetensors"))["embedding"]
					input := n.condition(want["input_ids"], preset)
					compare(t, name+" codec features", decoder.features(want["speech_codes"]), want["codec_features"])
					compare(t, name+" waveform", decoder.decode(want["speech_codes"]), want["audio"])
					prefix := want["speech_codes"].Slice(mlx.Slice(), mlx.Slice(0, min(10, want["speech_codes"].Dim(1))), mlx.Slice())
					compare(t, name+" prefix waveform", decoder.decode(prefix), want["prefix_audio"])
					if want["long_codes"] != nil {
						original := mlx.Contiguous(want["long_codes"].AsType(mlx.DTypeInt32), false).Ints()
						long := make([]int32, 64*37)
						for i := range long {
							long[i] = original[i%len(original)]
						}
						whole := decoder.decode(mlx.FromValues(long, 1, 64, 37))
						var parts []*mlx.Array
						for emitted := 0; emitted < 64; emitted += 25 {
							end, start := min(emitted+25, 64), max(0, emitted-32)
							wave := decoder.decode(mlx.FromValues(long[start*37:end*37], 1, end-start, 37))
							parts = append(parts, wave.Slice(mlx.Slice(), mlx.Slice((emitted-start)*1920, mlx.End)))
						}
						compare(t, "long fixed-code waveform", whole, want["long_audio"])
						compare(t, "streaming context and sample continuity", mlx.Concatenate(parts, 1), want["streamed_audio"])
						full := mlx.Contiguous(want["long_audio"], false).Floats()
						streamed := mlx.Contiguous(want["streamed_audio"], false).Floats()
						maxDelta, squared := 0.0, 0.0
						for i, value := range full {
							delta := float64(value - streamed[i])
							maxDelta = math.Max(maxDelta, math.Abs(delta))
							squared += delta * delta
						}
						rms := math.Sqrt(squared / float64(len(full)))
						t.Logf("cross-shape streaming rounding: max=%g RMS=%g", maxDelta, rms)
						if maxDelta > 4.0/32768 || rms > .5/32768 {
							t.Errorf("streaming boundary differs beyond PCM rounding: max=%g RMS=%g", maxDelta, rms)
						}
					}
					compare(t, name+" conditioning", input, want["input_embeddings"])
					hidden := n.backbone.forward(input, state, 0)
					compare(t, name+" backbone prefill", hidden, want["prefill_hidden"])
					last := mlx.Squeeze(hidden.Slice(mlx.Slice(), mlx.Slice(hidden.Dim(1)-1, hidden.Dim(1)), mlx.Slice()), 1)
					compare(t, name+" semantic logits", n.acoustic.semantic.Forward(last), want["semantic_logits"])
					velocity := n.acoustic.velocity(mlx.Concatenate([]*mlx.Array{want["flow_noise"], want["flow_noise"]}, 0), mlx.Zeros(mlx.DTypeFloat32, 2), mlx.Concatenate([]*mlx.Array{last, mlx.Zeros(last.DType(), last.Dims()...)}, 0))
					compare(t, name+" velocity zero", velocity, want["velocity_zero"])
					codes, err := n.acoustic.codes(last, want["flow_noise"])
					if err != nil {
						t.Fatal(err)
					}
					compare(t, name+" fixed-noise frame", mlx.FromValues(codes, 1, 37), want["first_codes"])
					feedback := n.feedback(want["first_codes"])
					compare(t, name+" feedback", feedback, want["next_embedding"])
					compare(t, name+" cached continuation", n.backbone.forward(feedback, state, 0), want["cached_hidden"])
				})
			}
		})
	})
}
