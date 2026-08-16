package gemma4

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
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
	compareAudioReference(t, "input_features", features, reference.Get("input_features"), 1e-5, 1e-5)
	valid := append([]bool(nil), input.FeatureMask...)
	x := mlx.ExpandDims(features, -1)
	x, valid = audioModel.Conv0.Forward(x, valid)
	x, valid = audioModel.Conv1.Forward(x, valid)
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
