package chatterbox

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func compare(t *mlxtest.T, label string, got, want *mlx.Array, atol, rtol float64) {
	t.Helper()
	if got == nil || want == nil || !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("%s: incompatible shapes", label)
	}
	g := mlx.Contiguous(got.AsType(mlx.DTypeFloat32), false).Floats()
	w := mlx.Contiguous(want.AsType(mlx.DTypeFloat32), false).Floats()
	maxErr, bad := float64(0), 0
	for i, v := range g {
		err := math.Abs(float64(v - w[i]))
		maxErr = math.Max(maxErr, err)
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || err > atol+rtol*math.Abs(float64(w[i])) {
			bad++
		}
	}
	t.Logf("%s %v: max absolute error %.7g; %d/%d outside tolerance", label, got.Dims(), maxErr, bad, len(g))
	if bad > 0 {
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
	out := map[string]*mlx.Array{}
	for name, value := range mlx.Load(path) {
		out[name] = value
	}
	return out
}

func TestReference(t *testing.T) {
	source, reference := os.Getenv("OLLAMA_S3_SOURCE"), os.Getenv("OLLAMA_SPEECH_REFERENCE_DIR")
	if source == "" || reference == "" {
		t.Skip("set pinned source and reference directories")
	}
	if err := mlx.CheckInit(); err != nil {
		t.Fatal(err)
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		want := sourceTensors(t, filepath.Join(reference, "core.safetensors"))
		resampled := audio.ResampleKaiserBest(want["reference_pcm"].Floats(), 24000, 16000)
		compare(t, "reference polyphase resampling", mlx.FromValues(resampled, len(resampled)), want["reference_16k"], 1e-7, 1e-7)
		s3, err := loadS3Tokenizer(sourceTensors(t, filepath.Join(source, "model.safetensors")))
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "S3 mel", s3.mels(want["reference_16k"]), want["s3_mels"], 1e-6, 1e-6)
		features := s3.features(want["s3_mels"])
		compare(t, "S3 features", features, want["s3_hidden"], 1e-5, 1e-4)
		compare(t, "S3 fixed features codes", s3.quantize(want["s3_hidden"]), want["s3_tokens"], 0, 0)
		compare(t, "S3 end-to-end codes", s3.quantize(s3.features(s3.mels(want["reference_16k"]))), want["s3_tokens"], 0, 0)
		root := os.Getenv("OLLAMA_CHATTERBOX_SOURCE")
		if root == "" {
			t.Fatal("set pinned Chatterbox source")
		}
		tensors := sourceTensors(t, filepath.Join(root, "model.safetensors"))
		w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, 0, 0, "", nil)}
		voice := loadVoiceEncoder(w)
		if w.err != nil {
			t.Fatal(w.err)
		}
		trimmed := trimVoiceReference(resampled)
		compare(t, "VE trimmed PCM", mlx.FromValues(trimmed, len(trimmed)), want["ve_trimmed_pcm"], 0, 0)
		compare(t, "VE mel", voice.mels(want["ve_trimmed_pcm"]), want["ve_mels"], 1e-6, 1e-6)
		compare(t, "VE partial windows", voice.partials(want["ve_mels"]), want["ve_partials"], 0, 0)
		x := want["ve_partials"]
		for i, l := range voice.layers {
			x = l.forward(x)
			compare(t, "VE LSTM "+string(rune('0'+i)), x, want["ve_layer_"+string(rune('0'+i))], 1e-6, 1e-6)
		}
		compare(t, "VE partial embeddings", voice.forward(want["ve_partials"]), want["ve_partial_embeddings"], 1e-6, 1e-6)
		compare(t, "VE speaker embedding", voice.embedding(want["ve_mels"]), want["t3_speaker"], 1e-6, 1e-6)
		generator := loadT3(w)
		if w.err != nil {
			t.Fatal(w.err)
		}
		conditioning := generator.conditioning(want["t3_speaker"], want["t3_reference_tokens"], want["t3_emotion"])
		compare(t, "T3 conditioning", conditioning, want["t3_conditioning"], 1e-6, 1e-6)
		raw, err := os.ReadFile(filepath.Join(root, "tokenizer.json"))
		if err != nil {
			t.Fatal(err)
		}
		tok, err := tokenizer.LoadFromBytes(raw)
		if err != nil {
			t.Fatal(err)
		}
		casesRaw, err := os.ReadFile(filepath.Join(reference, "text-cases.json"))
		if err != nil {
			t.Fatal(err)
		}
		var cases []struct {
			Text, Language string
			IDs            []int32
		}
		if err := json.Unmarshal(casesRaw, &cases); err != nil {
			t.Fatal(err)
		}
		for _, c := range cases {
			if ids := tok.Encode(normalizedV3Text(c.Text, c.Language), false); !slices.Equal(ids, c.IDs) {
				t.Errorf("text case %q: got %v, want %v", c.Text, ids, c.IDs)
			}
		}
		for _, language := range []string{"en", "de"} {
			v := sourceTensors(t, filepath.Join(reference, "hello-"+language+".safetensors"))
			text := "Hello."
			if language == "de" {
				text = "Guten Tag."
			}
			ids := tok.Encode(normalizedV3Text(text, language), false)
			compare(t, language+" text IDs", mlx.FromValues(ids, 1, len(ids)), v["text_tokens"], 0, 0)
			prefix := generator.prompt(conditioning, v["padded_text_tokens"])
			compare(t, language+" T3 prompt", prefix, v["input_embeds"], 1e-6, 1e-6)
			state := newT3State()
			hidden := generator.forward(prefix, state)
			compare(t, language+" T3 hidden", hidden, v["t3_hidden"], 1e-5, 1e-4)
			compare(t, language+" T3 logits", generator.head.Forward(hidden.Slice(mlx.Slice(), mlx.Slice(hidden.Dim(1)-1, hidden.Dim(1)), mlx.Slice())), v["t3_logits"], 1e-5, 1e-4)
			first := v["speech_tokens"].AsType(mlx.DTypeInt32).Slice(mlx.Slice(), mlx.Slice(0, 1))
			next := mlx.Add(mlx.Take(generator.speech, first, 0), mlx.Take(generator.speechPosition, mlx.FromValues([]int32{1}, 1, 1), 0))
			next = mlx.Concatenate([]*mlx.Array{next, next}, 0)
			compare(t, language+" next embedding", next, v["next_embedding"], 0, 0)
			nextHidden := generator.forward(next, state)
			compare(t, language+" cached T3 hidden", nextHidden, v["next_hidden"], 1e-5, 1e-4)
			compare(t, language+" cached T3 logits", generator.head.Forward(nextHidden), v["next_logits"], 1e-5, 1e-4)
			state.close()
		}
	})
}
