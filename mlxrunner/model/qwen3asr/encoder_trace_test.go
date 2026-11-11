package qwen3asr

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
)

// TestQwen3ASREncoderTrace is a bounded opt-in diagnosis at two independent
// provider prefixes. It records drift without redefining fidelity tolerances.
func TestQwen3ASREncoderTrace(t *testing.T) {
	reference, output := os.Getenv("OLLAMA_QWEN3_ASR_REFERENCE"), os.Getenv("OLLAMA_QWEN3_ASR_TRACE_OUTPUT")
	if reference == "" || output == "" {
		t.Skip("set original-provider trace and owned output paths")
	}
	data, err := os.ReadFile(filepath.Join(reference, "trace.json"))
	if err != nil {
		t.Fatal(err)
	}
	var receipt struct {
		Cases []string `json:"cases"`
	}
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatal(err)
	}
	if len(receipt.Cases) != 2 {
		t.Fatal("expected two provider trace cases")
	}
	if err := mlx.CheckInit(); err != nil {
		t.Fatal(err)
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		root, err := model.Open("qwen3-asr:component")
		if err != nil {
			t.Fatal(err)
		}
		loaded, err := newModel(root)
		if err != nil {
			t.Fatal(err)
		}
		m := loaded.(*Model)
		mlx.ScopedArrays(func() []*mlx.Array {
			weights := map[string]*mlx.Array{}
			for _, layer := range root.Manifest.TensorLayers() {
				path, err := manifest.BlobsPath(layer.Digest)
				if err != nil {
					t.Fatal(err)
				}
				for n, a := range mlx.Load(path) {
					weights[n] = a
				}
			}
			if err := m.LoadWeights(weights); err != nil {
				t.Fatal(err)
			}
			return mlx.Collect(m)
		})
		var rows []map[string]any
		for _, name := range receipt.Cases {
			if filepath.Base(name) != name {
				t.Fatal("invalid trace basename")
			}
			mlx.Scoped(func() {
				refs := map[string]*mlx.Array{}
				for key, a := range mlx.Load(filepath.Join(reference, name)) {
					refs[key] = a
				}
				actual := map[string]*mlx.Array{"position": mlx.FromValues(encoderPositions(13, m.cfg.Audio.Width), 13, m.cfg.Audio.Width)}
				x := m.encoder.convolve(refs["features"])
				actual["encoder.input"] = x
				for i := range m.cfg.Audio.Layers {
					x = m.encoder.layer(x, i)
					key := "encoder.layer" + strconv.Itoa(i)
					actual[key] = x
					input := refs["encoder.input"]
					// Each layer receives its own original-provider input.
					if i > 0 {
						input = refs["encoder.layer"+strconv.Itoa(i-1)]
					}
					actual[key+".identical-input"] = m.encoder.layer(input, i)
				}
				actual["encoder.final"] = m.encoder.linear(encoderGELU(m.encoder.linear(m.encoder.norm(x, "ln_post"), "proj1")), "proj2")
				metrics := map[string]any{}
				for key, want := range refs {
					if got := actual[key]; got != nil {
						metrics[key] = traceError(t, got, want)
						if same := actual[key+".identical-input"]; same != nil {
							metrics[key+".identical-input"] = traceError(t, same, want)
						}
					}
				}
				if err := mlx.SaveSafetensors(filepath.Join(output, name), actual); err != nil {
					t.Fatal(err)
				}
				rows = append(rows, map[string]any{"case": name, "metrics": metrics})
			})
			mlx.DefaultStream().Synchronize()
			mlx.ClearCache()
		}
		encoded, err := json.MarshalIndent(rows, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(output, "trace.json"), encoded, 0o600); err != nil {
			t.Fatal(err)
		}
	})
}

func traceError(t *mlxtest.T, got, want *mlx.Array) map[string]float64 {
	t.Helper()
	if !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("trace shape mismatch %v != %v", got.Dims(), want.Dims())
	}
	g, w := got.Floats(), want.Floats()
	maximum, sum := 0.0, 0.0
	for i, v := range g {
		d := math.Abs(float64(v) - float64(w[i]))
		if math.IsNaN(d) || math.IsInf(d, 0) {
			t.Fatal("non-finite trace")
		}
		maximum = max(maximum, d)
		sum += d * d
	}
	return map[string]float64{"max": maximum, "rms": math.Sqrt(sum / float64(len(g)))}
}
