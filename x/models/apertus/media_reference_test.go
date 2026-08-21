package apertus

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestMediaCodeReference(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	path := os.Getenv("APERTUS_MEDIA_REF_PATH")
	if path == "" {
		t.Skip("set APERTUS_MEDIA_REF_PATH to the pinned media-code reference")
	}
	var reference struct {
		ImageCodes []int `json:"image_codes"`
		AudioCodes []int `json:"audio_codes"`
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}

	thread, err := mlxthread.Start("apertus-media-reference", func() error {
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		return nil
	})
	if err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	defer func() {
		if err := thread.Stop(context.Background(), func() { mlx.Sweep(); mlx.ClearCache() }); err != nil {
			t.Fatal(err)
		}
	}()

	if err := thread.Do(context.Background(), func() error {
		m := loadImportedModel(t, firstNonEmpty(os.Getenv("APERTUS_MODEL_NAME"), "apertus-mlx"))
		imageInput, err := preprocessApertusImage(context.Background(), mediaReferencePNG(t))
		if err != nil {
			return err
		}
		imageData := mlx.FromValues(imageInput.pixels, len(imageInput.pixels))
		imageCodes, err := m.Vision.encode(imageData, imageInput.width, imageInput.height)
		if err != nil {
			return err
		}
		mlx.Eval(imageCodes)
		compareCodeIDs(t, "vision", imageCodes.Ints(), reference.ImageCodes)

		samples := make([]float32, 24017)
		targetPeak := math.Pow(10, -3.0/20.0)
		for i := range samples {
			samples[i] = float32(targetPeak * math.Sin(2*math.Pi*440*float64(i)/24000))
		}
		audioData := mlx.FromValues(samples, len(samples))
		audioCodes, err := m.Audio.encode(audioData, len(samples))
		if err != nil {
			return err
		}
		mlx.Eval(audioCodes)
		compareCodeIDs(t, "audio", audioCodes.Ints(), reference.AudioCodes)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func mediaReferencePNG(t *testing.T) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, 256, 256))
	for y := range 256 {
		for x := range 256 {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x*3 + y), G: uint8(x + y*5), B: uint8(x*7 + y*11), A: 255,
			})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func compareCodeIDs(t *testing.T, name string, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s code count = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s code %d = %d, want %d", name, i, got[i], want[i])
		}
	}
	t.Logf("%s codes exactly match pinned reference (%d IDs)", name, len(got))
}
