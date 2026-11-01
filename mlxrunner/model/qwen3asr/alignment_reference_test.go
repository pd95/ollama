package qwen3asr

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
)

func TestForcedAlignerReference(t *testing.T) {
	ref, imported := os.Getenv("OLLAMA_QWEN3_ASR_REFERENCE"), os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED")
	if ref == "" || imported == "" {
		t.Skip("requires supervised imported aligner and independent provider fixture")
	}
	root, err := model.Open(imported)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		t.Fatal(err)
	}
	if !IsForcedAligner(raw) {
		t.Skip("reference is for ASR, not alignment")
	}
	data, err := os.ReadFile(filepath.Join(ref, "receipt.json"))
	if err != nil {
		t.Fatal(err)
	}
	var receipt struct {
		Precision string   `json:"precision"`
		Language  string   `json:"language"`
		Text      string   `json:"text"`
		Prompt    string   `json:"prompt"`
		Words     []string `json:"words"`
		Classes   []int32  `json:"raw_classes"`
		Fixed     []int    `json:"fixed_ms"`
		Seconds   float64  `json:"seconds"`
		Prepared  string   `json:"prepared"`
		SHA       string   `json:"prepared_sha256"`
	}
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Precision != "float32" {
		t.Fatal("aligner reference must use F32")
	}
	units, prompt, err := AlignmentPrompt(receipt.Text, receipt.Language)
	if err != nil || prompt != receipt.Prompt {
		t.Fatalf("provider prompt differs: %v", err)
	}
	for i, u := range units {
		if u.Text != receipt.Words[i] {
			t.Fatal("normalized words differ")
		}
	}
	controls, err := os.ReadFile(filepath.Join(ref, "postprocessing.json"))
	if err != nil {
		t.Fatal(err)
	}
	var post struct {
		Repair        []struct{ Raw, Fixed []int } `json:"repair"`
		Normalization []struct {
			Text, Prompt string
			Words        []string
		} `json:"normalization"`
	}
	if err := json.Unmarshal(controls, &post); err != nil {
		t.Fatal(err)
	}
	for _, c := range post.Repair {
		if got := repairTimestamps(c.Raw); !slices.Equal(got, c.Fixed) {
			t.Fatalf("provider repair: %v -> %v != %v", c.Raw, got, c.Fixed)
		}
	}
	for _, c := range post.Normalization {
		u, p, err := AlignmentPrompt(c.Text, receipt.Language)
		if err != nil || p != c.Prompt {
			t.Fatalf("provider normalization differs: %v", err)
		}
		for i, w := range c.Words {
			if u[i].Text != w {
				t.Fatal("provider word differs")
			}
		}
	}
	if err := mlx.CheckInit(); err != nil {
		t.Fatal(err)
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		constructed, err := newModel(root)
		if err != nil {
			t.Fatal(err)
		}
		m := constructed.(*Model)
		owned := mlx.ScopedArrays(func() []*mlx.Array {
			tensors := map[string]*mlx.Array{}
			for _, l := range root.Manifest.TensorLayers() {
				p, err := manifest.BlobsPath(l.Digest)
				if err != nil {
					t.Fatal(err)
				}
				for n, a := range mlx.Load(p) {
					tensors[n] = a
				}
			}
			if err := m.LoadWeights(tensors); err != nil {
				t.Fatal(err)
			}
			return mlx.Collect(m)
		})
		if len(owned) != 708 {
			t.Fatalf("retained %d tensors", len(owned))
		}
		refs := map[string]*mlx.Array{}
		for n, a := range mlx.Load(filepath.Join(ref, "reference.safetensors")) {
			refs[n] = a
		}
		f, err := extractFeatures(refs["waveform"].Floats())
		if err != nil {
			t.Fatal(err)
		}
		features := mlx.FromValues(f.Data, 128, f.Frames)
		compareReference(t, "aligner.features", features, refs["features"].Reshape(128, f.Frames), 1e-4, 1e-5)
		compareReference(t, "aligner.encoder.layer0", m.encoder.layer(m.encoder.convolve(features), 0), refs["encoder.layer0"], 1e-4, 1e-4)
		encoded := m.encoder.forward(features)
		compareReference(t, "aligner.encoder.final", encoded, refs["encoder.final"], 1e-4, 1e-4)
		ids := refs["ids"]
		values := ids.Ints()
		start := slices.Index(values, audioPad)
		if start < 0 {
			t.Fatal("missing expanded audio")
		}
		var slots []int32
		for i, id := range values {
			if id == 151705 {
				slots = append(slots, int32(i))
			}
		}
		b := &batch.Batch{InputIDs: ids, SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(len(values))}, Media: []batch.MediaItem{{Pos: start, Features: encoded}}}
		input := refs["decoder.embeddings"]
		for i, layer := range m.decoder.Layers {
			name := fmt.Sprintf("decoder.layer%d", i)
			out := layer.Forward(input, b, nil, mlx.FromValues([]int32{0}, 1), 1, int32(len(values)), m.decoder.Config)
			compareReference(t, "aligner."+name+".identical-input", out, refs[name], 1e-3, 1e-4)
			input = refs[name]
		}
		h, _ := m.Forward(b, nil)
		compareReference(t, "aligner.prefill", h, refs["decoder.prefill"], 5e-3, 1e-4)
		selected := mlx.Take(h, mlx.FromValues(slots, len(slots)), 1)
		logits := m.decoder.Unembed(selected).Reshape(len(slots), 5000)
		compareReference(t, "aligner.classification-logits", logits, refs["logits"], 3e-4, 1e-4)
		classes := logits.Argmax(-1, false).AsType(mlx.DTypeInt32).Ints()
		if !slices.Equal(classes, receipt.Classes) {
			t.Fatal("raw timestamp predictions differ")
		}
		result, err := alignmentResult(receipt.Text, units, classes, receipt.Seconds)
		if err != nil {
			t.Fatal(err)
		}
		for i, item := range result.Items {
			if item.Start != float64(receipt.Fixed[2*i])/1000 || item.End != float64(receipt.Fixed[2*i+1])/1000 {
				t.Fatal("provider repaired timestamps differ")
			}
		}
		// Actual task entrypoint, repeated A/A and cancellation. It owns no
		// persistent request tensors or KV state.
		wav, err := os.ReadFile(receipt.Prepared)
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(wav)
		if hex.EncodeToString(digest[:]) != receipt.SHA {
			t.Fatal("paired audio identity mismatch")
		}
		first, err := m.AlignContext(context.Background(), wav, receipt.Text, receipt.Language)
		if err != nil || !reflect.DeepEqual(first, result) {
			t.Fatalf("native task mismatch: %v", err)
		}
		second, err := m.AlignContext(context.Background(), wav, receipt.Text, receipt.Language)
		if err != nil || !reflect.DeepEqual(second, first) {
			t.Fatalf("request isolation: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := m.AlignContext(ctx, wav, receipt.Text, receipt.Language); !errors.Is(err, context.Canceled) {
			t.Fatal("cancelled alignment admitted")
		}
		t.Logf("PASS: %d exact timestamp classes; %d provider-repaired words; UTF8 spans; repeated task/cancellation", len(classes), len(units))
	})
}
