package kokoro

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model/kokoro/config"
)

func compare(t *mlxtest.T, label string, got, want *mlx.Array) {
	t.Helper()
	if !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("%s shape %v, want %v", label, got.Dims(), want.Dims())
	}
	g := mlx.Contiguous(got.AsType(mlx.DTypeFloat32), false).Floats()
	w := mlx.Contiguous(want.AsType(mlx.DTypeFloat32), false).Floats()
	maxError := 0.0
	bad := 0
	for i, v := range g {
		err := math.Abs(float64(v - w[i]))
		maxError = math.Max(maxError, err)
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || err > 1e-5+1e-4*math.Abs(float64(w[i])) {
			bad++
		}
	}
	t.Logf("%s max error %.7g, %d/%d outside tolerance", label, maxError, bad, len(g))
	if bad != 0 {
		t.Errorf("%s numerical mismatch", label)
	}
}

func loadTensors(t *mlxtest.T, path string) map[string]*mlx.Array {
	t.Helper()
	checked, err := mlx.LoadSafetensorsNative(path)
	if err != nil {
		t.Fatal(err)
	}
	checked.Free()
	out := map[string]*mlx.Array{}
	for name, a := range mlx.Load(path) {
		if strings.HasSuffix(name, ".scales") {
			name = strings.TrimSuffix(name, ".scales") + ".weight_scale"
		}
		if strings.HasSuffix(name, ".biases") {
			name = strings.TrimSuffix(name, ".biases") + ".weight_qbias"
		}
		out[name] = a
	}
	return out
}

func TestReferenceEncoders(t *testing.T) {
	source, reference := os.Getenv("OLLAMA_KOKORO_SOURCE"), os.Getenv("OLLAMA_KOKORO_REFERENCE_DIR")
	if source == "" || reference == "" {
		t.Skip("set Kokoro source and reference directories for native checks")
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
			cfg, err := config.Parse(raw)
			if err != nil {
				t.Fatal(err)
			}
			weights := loadTensors(t, filepath.Join(source, "kokoro-v1_0.safetensors"))
			e, err := LoadEncoders(cfg, weights, nil)
			if err != nil {
				t.Fatal(err)
			}
			fixture := filepath.Join(reference, "core.safetensors")
			if _, err := os.Stat(fixture); os.IsNotExist(err) {
				fixture = filepath.Join(reference, "hello-phonemes.safetensors")
			}
			want := loadTensors(t, fixture)
			prosody, err := LoadProsody(cfg, weights, nil)
			if err != nil {
				t.Fatal(err)
			}
			ids, style := want["input_ids"], want["style"].Slice(mlx.Slice(), mlx.Slice(128, 256))
			compare(t, "BERT embeddings", e.bert.embeddings(ids), want["bert_embeddings"])
			hidden := e.bert.forward(ids)
			compare(t, "BERT final", hidden, want["bert"])
			encoded := e.projection.Forward(hidden)
			compare(t, "BERT projection", mlx.Transpose(encoded, 0, 2, 1), want["bert_encoded"])
			duration := e.encodeDuration(encoded, style)
			compare(t, "duration encoder", duration, want["duration_encoded"])
			h := e.durationLSTM.run(duration)
			compare(t, "duration LSTM", h, want["duration_hidden"])
			compare(t, "duration logits", e.duration.Forward(h), want["duration_logits"])
			compare(t, "text encoder", mlx.Transpose(e.encodeText(ids), 0, 2, 1), want["text_encoded"])
			if want["shared_prosody"] != nil {
				shared := e.shared.run(mlx.Transpose(want["aligned_duration"], 0, 2, 1))
				compare(t, "shared prosody", shared, want["shared_prosody"])
				f0, noise := prosody.run(shared, style)
				compare(t, "pitch", f0, want["f0"])
				compare(t, "noise curve", noise, want["noise_curve"])
			}
			decoder, err := LoadDecoder(cfg, weights, nil)
			if err != nil {
				t.Fatal(err)
			}
			waveStyle := want["style"].Slice(mlx.Slice(), mlx.Slice(0, 128))
			asr := mlx.Transpose(want["asr"], 0, 2, 1)
			if want["decoder_features"] != nil {
				pitchDown := decoder.pitch.run(mlx.ExpandDims(want["f0"], 2))
				noiseDown := decoder.noise.run(mlx.ExpandDims(want["noise_curve"], 2))
				h := decoder.encode.run(concatChannels(asr, pitchDown, noiseDown), waveStyle)
				residual := decoder.asr.run(asr)
				if want["decoder_encode"] != nil {
					compare(t, "decoder pitch down", mlx.Transpose(pitchDown, 0, 2, 1), want["decoder_pitch"])
					compare(t, "decoder noise down", mlx.Transpose(noiseDown, 0, 2, 1), want["decoder_noise"])
					compare(t, "decoder encode", mlx.Transpose(h, 0, 2, 1), want["decoder_encode"])
					compare(t, "decoder ASR", mlx.Transpose(residual, 0, 2, 1), want["decoder_asr"])
					for i, block := range decoder.blocks {
						h = block.run(concatChannels(h, residual, pitchDown, noiseDown), waveStyle)
						compare(t, fmt.Sprintf("decoder block %d", i), mlx.Transpose(h, 0, 2, 1), want[fmt.Sprintf("decoder_block_%d", i)])
					}
				}
				compare(t, "decoder features", mlx.Transpose(decoder.features(asr, want["f0"], want["noise_curve"], waveStyle), 0, 2, 1), want["decoder_features"])
				mag, phase := sourceSpectrum(want["excitation"], decoder.generator.window)
				compare(t, "source magnitude", mlx.ExpandDims(mlx.Transpose(mag, 1, 0), 0), want["source_magnitude"])
				compare(t, "source phase", mlx.ExpandDims(mlx.Transpose(phase, 1, 0), 0), want["source_phase"])
				if want["generator_post"] != nil {
					g := decoder.generator
					x := mlx.Transpose(want["decoder_features"], 0, 2, 1)
					har := concatChannels(mlx.ExpandDims(mag, 0), mlx.ExpandDims(phase, 0))
					for i := range g.ups {
						source := g.noiseConvs[i].run(har)
						compare(t, fmt.Sprintf("generator source conv %d", i), source, want[fmt.Sprintf("generator_noise_convs_%d", i)])
						source = g.noise[i].run(source, waveStyle)
						compare(t, fmt.Sprintf("generator source residual %d", i), mlx.Transpose(source, 0, 2, 1), want[fmt.Sprintf("generator_noise_res_%d", i)])
						x = g.ups[i].run(leaky(x, .1))
						compare(t, fmt.Sprintf("generator up %d", i), x, want[fmt.Sprintf("generator_ups_%d", i)])
						if i == 1 {
							x = padGenerator(x)
						}
						x = addChannels(x, source)
						var sum *mlx.Array
						for j := range 3 {
							h := g.residual[i][j].run(x, waveStyle)
							compare(t, fmt.Sprintf("generator residual %d/%d", i, j), mlx.Transpose(h, 0, 2, 1), want[fmt.Sprintf("generator_resblocks_%d", i*3+j)])
							if sum == nil {
								sum = h
							} else {
								sum = addChannels(sum, h)
							}
						}
						x = mlx.Div(sum, mlx.FromValue(float32(3)))
					}
					compare(t, "generator post", g.post.run(leaky(x, .01)), want["generator_post"])
				}
				mag, phase = decoder.generator.spectrum(mlx.Transpose(want["decoder_features"], 0, 2, 1), waveStyle, want["excitation"])
				compare(t, "output magnitude", mlx.ExpandDims(mlx.Transpose(mag, 1, 0), 0), want["output_magnitude"])
				compare(t, "output phase", mlx.ExpandDims(mlx.Transpose(phase, 1, 0), 0), want["output_phase"])
				compare(t, "inverse STFT", inverseSpectrum(mlx.Squeeze(mlx.Transpose(want["output_magnitude"], 0, 2, 1), 0), mlx.Squeeze(mlx.Transpose(want["output_phase"], 0, 2, 1), 0), decoder.generator.window), mlx.Reshape(want["audio"], -1))
			}
			audio := decoder.run(asr, want["f0"], want["noise_curve"], waveStyle, want["excitation"])
			compare(t, "fixed excitation waveform", audio, mlx.Reshape(want["audio"], -1))
		})
	})
}
