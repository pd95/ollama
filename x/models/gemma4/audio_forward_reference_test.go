package gemma4

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	runnermodel "github.com/ollama/ollama/x/mlxrunner/model"
	gemma4metadata "github.com/ollama/ollama/x/models/gemma4/metadata"
)

func TestAudioForwardReference(t *testing.T) {
	modelDir := os.Getenv("GEMMA4_AUDIO_MODEL_DIR")
	refDir := os.Getenv("GEMMA4_AUDIO_REF_DIR")
	wavPath := os.Getenv("GEMMA4_AUDIO_WAV")
	if modelDir == "" || refDir == "" || wavPath == "" {
		t.Skip("set GEMMA4_AUDIO_MODEL_DIR, GEMMA4_AUDIO_REF_DIR, and GEMMA4_AUDIO_WAV for audio parity")
	}
	skipIfNoMLX(t)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
	}

	configData, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	audioConfig, err := parseAudioConfig(configData)
	if err != nil || audioConfig == nil {
		t.Fatalf("parse audio config: %v", err)
	}
	textConfig, err := parseTextConfig(configData)
	if err != nil {
		t.Fatal(err)
	}
	processorData, err := os.ReadFile(filepath.Join(modelDir, "processor_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	processorConfig, err := parseAudioProcessorConfig(processorData)
	if err != nil {
		t.Fatal(err)
	}
	wav, err := os.ReadFile(wavPath)
	if err != nil {
		t.Fatal(err)
	}
	input, err := preprocessGemma4Audio(context.Background(), wav, processorConfig)
	if err != nil {
		t.Fatal(err)
	}

	source, err := mlx.LoadSafetensorsNative(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Free()
	reference, err := mlx.LoadSafetensorsNative(filepath.Join(refDir, "audio-checkpoints.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer reference.Free()
	if wantFrames := reference.Get("input_features").Dim(1); wantFrames > input.Frames {
		padding := wantFrames - input.Frames
		input.Features = append(input.Features, make([]float32, padding*128)...)
		input.FeatureMask = append(input.FeatureMask, make([]bool, padding)...)
		input.Frames = wantFrames
	}

	required, err := gemma4metadata.RequiredAudioTensorShapes(audioMetadataConfig(audioConfig, textConfig.HiddenSize))
	if err != nil {
		t.Fatal(err)
	}
	tensors := make(map[string]*mlx.Array, len(required))
	for name := range required {
		tensors[name] = source.Get(name)
		if tensors[name] == nil {
			t.Fatalf("source is missing %s", name)
		}
	}
	audioModel, err := loadAudioModel(tensors, audioConfig, textConfig.HiddenSize, 0, 0, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	embedAudio, err := loadMultimodalEmbedder(tensors, "embed_audio", audioConfig.RMSNormEps, 0, 0, "", nil)
	if err != nil {
		t.Fatal(err)
	}

	features := mlx.FromValues(input.Features, 1, input.Frames, 128)
	compareAudioReference(t, "input_features", features, reference.Get("input_features"), 2e-5, 1e-5)
	compareAudioMaskReference(t, "input_features_mask", input.FeatureMask, reference.Get("input_features_mask"), false)
	valid := append([]bool(nil), input.FeatureMask...)
	x := mlx.ExpandDims(features, -1)
	x, valid = audioModel.Conv0.Forward(x, valid)
	x, valid = audioModel.Conv1.Forward(x, valid)
	compareAudioMaskReference(t, "audio_forward_mask", valid, reference.Get("audio_forward_mask"), true)
	validOutputs := 0
	for _, value := range valid {
		if value {
			validOutputs++
		}
	}
	if validOutputs != input.SoftTokens {
		t.Fatalf("valid audio outputs = %d, want %d", validOutputs, input.SoftTokens)
	}
	x = mlx.Reshape(x, 1, int32(x.Dim(1)), int32(x.Dim(2)*x.Dim(3)))
	x = audioModel.InputProj.Forward(x)
	compareAudioReference(t, "subsample", x, reference.Get("subsample"), 0.5, 0.05)
	for index, layer := range audioModel.Layers {
		x = layer.Forward(x, valid)
		if index == 0 || index == len(audioModel.Layers)-1 {
			name := "layer_0"
			if index != 0 {
				name = "layer_11"
			}
			compareAudioReference(t, name, x, reference.Get(name), 0.5, 0.05)
		}
	}
	x = audioModel.OutputProj.Forward(x)
	compareAudioReference(t, "output_projection", x, reference.Get("output_projection"), 0.5, 0.05)
	x = embedAudio.Forward(x)
	compareAudioReference(t, "multimodal_projection", x, reference.Get("multimodal_projection"), 0.5, 0.05)

	forward := audioModel.Forward(input)
	compareAudioReference(t, "audio_physical_output", forward, reference.Get("audio_physical_output"), 0.5, 0.05)
	projectedForward := embedAudio.Forward(forward)
	inputIDs := reference.Get("input_ids")
	embedWeight := source.Get("model.language_model.embed_tokens.weight")
	if inputIDs == nil || embedWeight == nil {
		t.Fatal("reference input IDs or source token embeddings are missing")
	}
	inputIDs = inputIDs.AsType(mlx.DTypeInt32)
	tokenEmbeddings := mlx.MulScalar(mlx.Take(embedWeight, inputIDs, 0), textConfig.EmbedScale)
	mlx.Eval(inputIDs)
	start, end, err := mediaTokenSpan(intsToInt32(inputIDs.Ints()), textConfig.AudioTokenIDValue)
	if err != nil {
		t.Fatal(err)
	}
	if end-start != input.SoftTokens {
		t.Fatalf("audio token span = %d, want %d", end-start, input.SoftTokens)
	}
	parts := make([]*mlx.Array, 0, 3)
	if start > 0 {
		parts = append(parts, mlx.SliceStartStop(tokenEmbeddings, []int32{0, 0, 0}, []int32{1, int32(start), textConfig.HiddenSize}))
	}
	parts = append(parts, projectedForward)
	if end < inputIDs.Dim(1) {
		parts = append(parts, mlx.SliceStartStop(tokenEmbeddings, []int32{0, int32(end), 0}, []int32{1, int32(inputIDs.Dim(1)), textConfig.HiddenSize}))
	}
	finalEmbeddings := mlx.Concatenate(parts, 1)
	compareAudioReference(t, "final_input_embeddings", finalEmbeddings, reference.Get("final_input_embeddings"), 0.5, 0.05)

	if importedModel := os.Getenv("GEMMA4_AUDIO_MODEL_NAME"); importedModel != "" {
		fullModel := loadImportedGemma4ReferenceModel(t, importedModel)
		pleIDs := intsToInt32(inputIDs.Ints())
		for i := start; i < end; i++ {
			pleIDs[i] = 0
		}
		modelBatch := &batch.Batch{
			InputIDs:        inputIDs,
			InputEmbeddings: finalEmbeddings,
			PLEInputIDs:     mlx.FromValues(pleIDs, 1, len(pleIDs)),
			SeqOffsets:      []int32{0},
			SeqQueryLens:    []int32{int32(len(pleIDs))},
		}
		hidden := fullModel.Forward(modelBatch, nil)
		logits := fullModel.Unembed(hidden)
		last := mlx.SliceStartStop(logits, []int32{0, int32(len(pleIDs) - 1), 0}, []int32{1, int32(len(pleIDs)), textConfig.VocabSize})
		wantLogits := reference.Get("prefill_logits")
		compareAudioReference(t, "prefill_logits", last, wantLogits, 0.5, 0.05)
		gotID := last.Argmax(-1, false).AsType(mlx.DTypeInt32)
		wantID := wantLogits.Argmax(-1, false).AsType(mlx.DTypeInt32)
		mlx.Eval(gotID, wantID)
		if got, want := gotID.Int(), wantID.Int(); got != want {
			t.Fatalf("prefill argmax = %d, want %d", got, want)
		}
	}
}

func loadImportedGemma4ReferenceModel(t *testing.T, name string) *Model {
	t.Helper()
	root, err := runnermodel.Open(name)
	if err != nil {
		t.Fatalf("open imported model %q: %v", name, err)
	}
	defer root.Close()
	bm, err := newModel(root)
	if err != nil {
		t.Fatalf("construct imported model: %v", err)
	}
	tensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	for _, layer := range root.Manifest.GetTensorLayers("") {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		for tensorName, array := range mlx.Load(root.Manifest.BlobPath(layer.Digest)) {
			tensors[tensorName] = array
		}
	}
	for tensorName, array := range tensors {
		if strings.HasSuffix(tensorName, ".scale") {
			tensors[strings.TrimSuffix(tensorName, ".scale")+"_scale"] = array
		}
	}
	if err := bm.LoadWeights(tensors); err != nil {
		t.Fatalf("load imported model weights: %v", err)
	}
	collected := mlx.Collect(bm)
	for _, array := range collected {
		mlx.Pin(array)
	}
	mlx.Eval(collected...)
	m, ok := bm.(*Model)
	if !ok {
		t.Fatalf("imported model type = %T, want *gemma4.Model", bm)
	}
	return m
}

func compareAudioMaskReference(t *testing.T, name string, got []bool, want *mlx.Array, invertWant bool) {
	t.Helper()
	if want == nil {
		t.Fatalf("%s tensor is missing", name)
	}
	want = want.AsType(mlx.DTypeInt32)
	mlx.Eval(want)
	values := want.Ints()
	if len(got) != len(values) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(values))
	}
	for i, value := range values {
		expected := value != 0
		if invertWant {
			expected = !expected
		}
		if got[i] != expected {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], expected)
		}
	}
}

func intsToInt32(values []int) []int32 {
	out := make([]int32, len(values))
	for i, value := range values {
		out[i] = int32(value)
	}
	return out
}

func compareAudioReference(t *testing.T, name string, got, want *mlx.Array, atol, rtol float64) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatalf("%s tensor is missing", name)
	}
	if !equalIntShape(got.Dims(), want.Dims()) {
		t.Fatalf("%s shape = %v, want %v", name, got.Dims(), want.Dims())
	}
	got = got.AsType(mlx.DTypeFloat32)
	want = want.AsType(mlx.DTypeFloat32)
	mlx.Eval(got, want)
	gotValues, wantValues := got.Floats(), want.Floats()
	var maxDiff float64
	for i := range wantValues {
		diff := math.Abs(float64(gotValues[i] - wantValues[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		tolerance := atol + rtol*math.Abs(float64(wantValues[i]))
		if diff > tolerance {
			t.Fatalf("%s[%d] = %g, want %g (diff %g, tolerance %g)", name, i, gotValues[i], wantValues[i], diff, tolerance)
		}
	}
	t.Logf("%s matched %d values; max absolute difference %g", name, len(wantValues), maxDiff)
}
