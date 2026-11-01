package qwen3asr

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
)

func compareReference(t *mlxtest.T, name string, got, want *mlx.Array, atol, rtol float64) {
	t.Helper()
	if !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("%s shapes %v != %v", name, got.Dims(), want.Dims())
	}
	g := mlx.Contiguous(got.AsType(mlx.DTypeFloat32), false).Floats()
	w := mlx.Contiguous(want.AsType(mlx.DTypeFloat32), false).Floats()
	maximum, sum, bad := 0.0, 0.0, 0
	for i, v := range g {
		d := math.Abs(float64(v) - float64(w[i]))
		maximum = max(maximum, d)
		sum += d * d
		if math.IsNaN(d) || math.IsInf(d, 0) || d > atol+rtol*math.Abs(float64(w[i])) {
			bad++
		}
	}
	t.Logf("%s max=%g rms=%g outside=%d/%d atol=%g rtol=%g", name, maximum, math.Sqrt(sum/float64(len(g))), bad, len(g), atol, rtol)
	if bad > 0 {
		t.Errorf("%s provider mismatch", name)
	}
}

func TestReference(t *testing.T) {
	source, reference := os.Getenv("OLLAMA_QWEN3_ASR_SOURCE"), os.Getenv("OLLAMA_QWEN3_ASR_REFERENCE")
	if source == "" || reference == "" {
		t.Skip("set Qwen3-ASR source and provider reference directories")
	}
	if err := mlx.CheckInit(); err != nil {
		t.Fatal(err)
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		raw, err := os.ReadFile(filepath.Join(source, "config.json"))
		if err != nil {
			t.Fatal(err)
		}
		cfg, err := parseConfig(raw)
		if err != nil {
			t.Fatal(err)
		}
		meta, err := os.ReadFile(filepath.Join(reference, "receipt.json"))
		if err != nil {
			t.Fatal(err)
		}
		var receipt struct {
			Precision string `json:"precision"`
		}
		if err := json.Unmarshal(meta, &receipt); err != nil {
			t.Fatal(err)
		}
		dtype := mlx.DTypeFloat32
		neuralAtol, neuralRtol := 1e-4, 1e-4
		layerAtol, prefillAtol, logitsAtol := 1e-3, 5e-3, 3e-4
		if receipt.Precision != "float32" {
			t.Fatal("only the F32 original-checkpoint baseline is qualified")
		}
		tensors := map[string]*mlx.Array{}
		files, err := filepath.Glob(filepath.Join(source, "*.safetensors"))
		if err != nil || len(files) == 0 {
			t.Fatalf("checkpoint shards: %v", err)
		}
		for _, file := range files {
			for n, a := range mlx.Load(file) {
				tensors[n] = a.AsType(dtype)
			}
		}
		m := &Model{cfg: cfg, raw: raw}
		if imported := os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED"); imported != "" {
			root, err := model.Open(imported)
			if err != nil {
				t.Fatal(err)
			}
			constructed, err := newModel(root)
			if err != nil {
				t.Fatal(err)
			}
			m = constructed.(*Model)
			tensors = map[string]*mlx.Array{}
			for _, layer := range root.Manifest.TensorLayers() {
				path, err := manifest.BlobsPath(layer.Digest)
				if err != nil {
					t.Fatal(err)
				}
				for name, value := range mlx.Load(path) {
					tensors[name] = value
				}
			}
			t.Logf("loaded %d tensors through imported manifest", len(tensors))
		}
		owned := mlx.ScopedArrays(func() []*mlx.Array {
			if err := m.LoadWeights(tensors); err != nil {
				t.Fatal(err)
			}
			return mlx.Collect(m)
		})
		shapes, err := TensorShapes(raw)
		if err != nil {
			t.Fatal(err)
		}
		if len(owned) != len(shapes) {
			t.Fatalf("runner weight collector retained %d arrays, want %d", len(owned), len(shapes))
		}
		refs := map[string]*mlx.Array{}
		for n, a := range mlx.Load(filepath.Join(reference, "reference.safetensors")) {
			refs[n] = a
		}
		f, err := extractFeatures(refs["waveform"].Floats())
		if err != nil {
			t.Fatal(err)
		}
		features := mlx.FromValues(f.Data, 128, f.Frames)
		// Provider NumPy is the high-precision CPU frontend control. Its Torch
		// FFT path rounds the Hann product and reductions differently.
		compareReference(t, "features.torch", features, refs["features"].Reshape(128, f.Frames), 1e-4, 1e-5)
		if numpy := refs["features_numpy"]; numpy != nil {
			compareReference(t, "features.numpy", features, numpy.Reshape(128, f.Frames), 1e-5, 1e-5)
		}
		first := m.encoder.layer(m.encoder.convolve(features.AsType(dtype)), 0)
		compareReference(t, "encoder.layer0", first, refs["encoder.layer0"], neuralAtol, neuralRtol)
		encoded := m.encoder.forward(features)
		compareReference(t, "encoder.final", encoded, refs["encoder.final"], neuralAtol, neuralRtol)
		ids := refs["ids"]
		n := ids.Dim(1)
		values := ids.Ints()
		start := -1
		for i, v := range values {
			if v == audioPad {
				start = i
				break
			}
		}
		if start < 0 {
			t.Fatal("reference has no audio placeholders")
		}
		for i := range encoded.Dim(0) {
			if values[start+i] != audioPad {
				t.Fatal("audio expansion differs")
			}
		}
		b := &batch.Batch{InputIDs: ids, SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(n)}, Media: []batch.MediaItem{{Pos: start, Features: encoded}}}
		embeddings := m.decoder.EmbedTokens.Forward(ids).SliceUpdate(mlx.ExpandDims(encoded, 0), mlx.Slice(), mlx.Slice(start, start+encoded.Dim(0)), mlx.Slice())
		layer0 := m.decoder.Layers[0].Forward(embeddings, b, nil, mlx.FromValues([]int32{0}, 1), 1, int32(n), m.decoder.Config)
		compareReference(t, "decoder.layer0", layer0, refs["decoder.layer0"], neuralAtol, neuralRtol)
		if input := refs["decoder.embeddings"]; input != nil {
			for i := range m.decoder.Layers {
				name := fmt.Sprintf("decoder.layer%d", i)
				out := m.decoder.Layers[i].Forward(input, b, nil, mlx.FromValues([]int32{0}, 1), 1, int32(n), m.decoder.Config)
				compareReference(t, name+".identical-input", out, refs[name], layerAtol, neuralRtol)
				input = refs[name]
			}
			compareReference(t, "decoder.norm.identical-input", m.decoder.Norm.Forward(input.AsType(dtype), m.cfg.Text.RMSNormEps), refs["decoder.prefill"], neuralAtol, neuralRtol)
		}
		caches := m.NewCaches()
		defer func() {
			for _, c := range caches {
				c.Free()
			}
		}()
		hidden, _ := m.Forward(b, caches)
		// Deep decoder accumulation has a separate envelope from identical-
		// input layers. Provider eager vs SDPA itself differs by ~9e-4 here.
		compareReference(t, "decoder.prefill", hidden, refs["decoder.prefill"], prefillAtol, neuralRtol)
		logits := m.Unembed(hidden.Slice(mlx.Slice(), mlx.Slice(n-1, n), mlx.Slice())).Reshape(1, 151936)
		compareReference(t, "logits", logits, refs["logits"], logitsAtol, neuralRtol)
		generated := refs["generated"].Ints()
		for i, want := range generated {
			got := logits.Argmax(-1, false).AsType(mlx.DTypeInt32).Ints()[0]
			if got != want {
				t.Fatalf("greedy token %d: %d != %d", i, got, want)
			}
			if i+1 < len(generated) {
				next := &batch.Batch{InputIDs: mlx.FromValues([]int32{got}, 1, 1), SeqOffsets: []int32{int32(n + i)}, SeqQueryLens: []int32{1}}
				h, _ := m.Forward(next, caches)
				logits = m.Unembed(h).Reshape(1, 151936)
			}
		}
	})
}
