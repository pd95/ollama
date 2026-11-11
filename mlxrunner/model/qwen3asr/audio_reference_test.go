package qwen3asr

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
)

func TestQwen3ASRAudioReference(t *testing.T) {
	source, reference := os.Getenv("OLLAMA_QWEN3_ASR_SOURCE"), os.Getenv("OLLAMA_QWEN3_ASR_AUDIO_REFERENCE")
	if source == "" || reference == "" {
		t.Skip("set source and original-provider real-audio fixture paths")
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
		m := &Model{cfg: cfg, raw: raw}
		mlx.ScopedArrays(func() []*mlx.Array {
			tensors := map[string]*mlx.Array{}
			files, err := filepath.Glob(filepath.Join(source, "*.safetensors"))
			if err != nil || len(files) == 0 {
				t.Fatalf("checkpoint shards: %v", err)
			}
			for _, file := range files {
				for n, a := range mlx.Load(file) {
					tensors[n] = a
				}
			}
			if err := m.LoadWeights(tensors); err != nil {
				t.Fatal(err)
			}
			return mlx.Collect(m)
		})
		data, err := os.ReadFile(filepath.Join(reference, "results.json"))
		if err != nil {
			t.Fatal(err)
		}
		var cases []struct {
			Case string `json:"case"`
		}
		if err := json.Unmarshal(data, &cases); err != nil {
			t.Fatal(err)
		}
		for _, testcase := range cases {
			mlx.Scoped(func() {
				refs := map[string]*mlx.Array{}
				for n, a := range mlx.Load(filepath.Join(reference, testcase.Case+".safetensors")) {
					refs[n] = a
				}
				f, err := extractFeatures(refs["waveform"].Floats())
				if err != nil {
					t.Fatal(err)
				}
				input := mlx.FromValues(f.Data, 128, f.Frames)
				compareReference(t, testcase.Case+".features", input, refs["features"].Reshape(128, f.Frames), 1e-4, 1e-5)
				encoded := m.encoder.forward(input)
				compareReference(t, testcase.Case+".encoder", encoded, refs["encoder.final"], 1e-4, 1e-4)
				ids := refs["ids"]
				start := -1
				for i, id := range ids.Ints() {
					if id == audioPad {
						start = i
						break
					}
				}
				if start < 0 {
					t.Fatal("missing audio placeholder")
				}
				n := ids.Dim(1)
				b := &batch.Batch{InputIDs: ids, SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(n)}, Media: []batch.MediaItem{{Pos: start, Features: encoded}}}
				caches := m.NewCaches()
				defer func() {
					for _, c := range caches {
						c.Free()
					}
				}()
				h, _ := m.Forward(b, caches)
				logits := m.Unembed(h.Slice(mlx.Slice(), mlx.Slice(n-1, n), mlx.Slice())).Reshape(1, 151936)
				compareReference(t, testcase.Case+".prefill-logits", logits, refs["logits"], 3e-4, 1e-4)
				generated := refs["generated"].Ints()
				topIDs := refs["step_top_ids"].Ints()
				topScores := refs["step_top_logits"].Floats()
				for step, want := range generated {
					got := logits.Argmax(-1, false).AsType(mlx.DTypeInt32).Ints()[0]
					if got != want {
						if cfg.Text.HiddenSize == 2048 {
							t.Fatalf("1.7B exact greedy mismatch at step %d: %d != %d", step, got, want)
						}
						found := -1
						for j, id := range topIDs[step*4 : step*4+4] {
							if id == got {
								found = j
								break
							}
						}
						if found < 0 {
							t.Fatalf("%s token %d: native %d not in provider top four (expected %d)", testcase.Case, step, got, want)
						}
						gap := topScores[step*4] - topScores[step*4+found]
						t.Logf("%s token %d differs at provider logit margin %g (native=%d provider=%d)", testcase.Case, step, gap, got, want)
						if gap > 1e-3 {
							t.Errorf("%s unexplained greedy token mismatch at step %d", testcase.Case, step)
						}
					}
					if step+1 < len(generated) {
						b = &batch.Batch{InputIDs: mlx.FromValues([]int32{want}, 1, 1), SeqOffsets: []int32{int32(n + step)}, SeqQueryLens: []int32{1}}
						h, _ = m.Forward(b, caches)
						logits = m.Unembed(h).Reshape(1, 151936)
					}
				}
			})
		}
	})
}
