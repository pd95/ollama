package apertus

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
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
		ImageCodes        []int  `json:"image_codes"`
		ResizedImageCodes []int  `json:"resized_image_codes"`
		AudioCodes        []int  `json:"audio_codes"`
		ShortAudioCodes   []int  `json:"short_audio_codes"`
		ResizedPixels     string `json:"resized_pixels_base64"`
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	fixedImage := mediaReferencePNG(t, 256, 256)
	resizedImage := mediaReferencePNG(t, 320, 180)

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
		imageInput, err := preprocessApertusImage(context.Background(), fixedImage)
		if err != nil {
			return err
		}
		imageData := mlx.FromValues(imageInput.pixels, len(imageInput.pixels))
		imageCodes, err := m.Vision.encode(imageData, imageInput.width, imageInput.height)
		if err != nil {
			return err
		}
		mlx.Eval(imageCodes)
		if err := compareCodeIDs("vision", imageCodes.Ints(), reference.ImageCodes); err != nil {
			return err
		}
		t.Logf("vision codes exactly match pinned reference (%d IDs)", len(reference.ImageCodes))
		resizedInput, err := preprocessApertusImage(context.Background(), resizedImage)
		if err != nil {
			return err
		}
		resizedPixels, err := base64.StdEncoding.DecodeString(reference.ResizedPixels)
		if err != nil {
			return fmt.Errorf("decode resized pixel reference: %w", err)
		}
		if len(resizedPixels) != len(resizedInput.pixels) {
			return fmt.Errorf("resized pixel count = %d, want %d", len(resizedInput.pixels), len(resizedPixels))
		}
		for i, want := range resizedPixels {
			got := uint8(math.Round(float64((resizedInput.pixels[i] + 1) * 127.5)))
			if got != want {
				return fmt.Errorf("resized pixel %d = %d, want %d", i, got, want)
			}
		}
		resizedData := mlx.FromValues(resizedInput.pixels, len(resizedInput.pixels))
		resizedCodes, err := m.Vision.encode(resizedData, resizedInput.width, resizedInput.height)
		if err != nil {
			return err
		}
		mlx.Eval(resizedCodes)
		if err := compareCodeIDs("resized vision", resizedCodes.Ints(), reference.ResizedImageCodes); err != nil {
			return err
		}
		t.Logf("resized vision codes exactly match pinned reference (%d IDs)", len(reference.ResizedImageCodes))

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
		if err := compareCodeIDs("audio", audioCodes.Ints(), reference.AudioCodes); err != nil {
			return err
		}
		t.Logf("audio codes exactly match pinned reference (%d IDs)", len(reference.AudioCodes))
		shortSamples := []float32{float32(targetPeak), float32(-targetPeak / 2)}
		shortData := mlx.FromValues(shortSamples, len(shortSamples))
		shortCodes, err := m.Audio.encode(shortData, len(shortSamples))
		if err != nil {
			return err
		}
		mlx.Eval(shortCodes)
		if err := compareCodeIDs("short audio", shortCodes.Ints(), reference.ShortAudioCodes); err != nil {
			return err
		}
		t.Logf("short audio codes exactly match pinned reference (%d IDs)", len(reference.ShortAudioCodes))
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func mediaReferencePNG(t *testing.T, width, height int) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
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

func compareCodeIDs(name string, got, want []int) error {
	if len(got) != len(want) {
		return fmt.Errorf("%s code count = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("%s code %d = %d, want %d", name, i, got[i], want[i])
		}
	}
	return nil
}
