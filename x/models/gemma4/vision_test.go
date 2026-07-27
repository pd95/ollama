package gemma4

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"
)

func TestParseVisionConfigDefaults(t *testing.T) {
	cfg, err := parseVisionConfig([]byte(`{"vision_config":{}}`))
	if err != nil {
		t.Fatalf("parseVisionConfig() error = %v", err)
	}
	if cfg == nil {
		t.Fatal("parseVisionConfig() = nil, want config")
	}
	if cfg.HiddenSize != 768 || cfg.IntermediateSize != 3072 || cfg.NumHiddenLayers != 16 {
		t.Fatalf("unexpected default dimensions: hidden=%d intermediate=%d layers=%d", cfg.HiddenSize, cfg.IntermediateSize, cfg.NumHiddenLayers)
	}
	if cfg.PatchSize != 16 || cfg.PoolingKernelSize != 3 || cfg.DefaultOutputLength != 280 {
		t.Fatalf("unexpected image defaults: patch=%d pooling=%d output=%d", cfg.PatchSize, cfg.PoolingKernelSize, cfg.DefaultOutputLength)
	}
	if cfg.RopeParameters.RopeTheta != 100 {
		t.Fatalf("vision rope theta = %v, want 100", cfg.RopeParameters.RopeTheta)
	}
}

func TestGemma4ResizeDimensions(t *testing.T) {
	gotW, gotH, err := gemma4ResizeDimensions(1024, 768, 16, 280*9, 3)
	if err != nil {
		t.Fatalf("gemma4ResizeDimensions() error = %v", err)
	}
	if gotW != 912 || gotH != 672 {
		t.Fatalf("gemma4ResizeDimensions() = %dx%d, want 912x672", gotW, gotH)
	}
	if gotW%48 != 0 || gotH%48 != 0 {
		t.Fatalf("resize dimensions must be multiples of pooled patch size, got %dx%d", gotW, gotH)
	}
}

func TestImageToCHWFloat32UsesBoundsAndChannelOrder(t *testing.T) {
	img := image.NewRGBA(image.Rect(10, 20, 12, 22))
	img.SetRGBA(10, 20, color.RGBA{R: 255, A: 255})
	img.SetRGBA(11, 20, color.RGBA{G: 128, A: 255})
	img.SetRGBA(10, 21, color.RGBA{B: 64, A: 255})
	img.SetRGBA(11, 21, color.RGBA{R: 32, G: 16, B: 8, A: 255})

	got := imageToCHWFloat32(img)
	want := []float32{
		1, 0, 0, 32.0 / 255.0,
		0, 128.0 / 255.0, 0, 16.0 / 255.0,
		0, 0, 64.0 / 255.0, 8.0 / 255.0,
	}
	if len(got) != len(want) {
		t.Fatalf("len(imageToCHWFloat32()) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("pixel[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestImageTokenSpan(t *testing.T) {
	start, end, err := imageTokenSpan([]int32{1, 258880, 258880, 2}, 258880)
	if err != nil {
		t.Fatalf("imageTokenSpan() error = %v", err)
	}
	if start != 1 || end != 3 {
		t.Fatalf("imageTokenSpan() = %d, %d; want 1, 3", start, end)
	}

	if _, _, err := imageTokenSpan([]int32{258880, 2, 258880}, 258880); err == nil {
		t.Fatal("imageTokenSpan() error = nil, want split-token error")
	}
	if _, _, err := imageTokenSpan([]int32{1, 2, 3}, 258880); err == nil {
		t.Fatal("imageTokenSpan() error = nil, want missing-token error")
	}
}

func TestPreprocessGemma4ImageSoftTokenBudget(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 8, 4))
	src.SetRGBA(0, 0, color.RGBA{R: 255, A: 255})

	var buf bytes.Buffer
	if err := png.Encode(&buf, src); err != nil {
		t.Fatalf("png.Encode() error = %v", err)
	}

	img, err := preprocessGemma4Image(buf.Bytes(), &VisionConfig{PatchSize: 16, PoolingKernelSize: 3, DefaultOutputLength: 1}, 1)
	if err != nil {
		t.Fatalf("preprocessGemma4Image() error = %v", err)
	}
	if img.Width != 48 || img.Height != 48 {
		t.Fatalf("preprocessed dimensions = %dx%d, want 48x48", img.Width, img.Height)
	}
	if img.PatchWidth != 3 || img.PatchHeight != 3 || img.SoftTokens != 1 {
		t.Fatalf("patch layout = %dx%d soft=%d, want 3x3 soft=1", img.PatchWidth, img.PatchHeight, img.SoftTokens)
	}
	if len(img.Pixels) != 3*48*48 {
		t.Fatalf("pixel count = %d, want %d", len(img.Pixels), 3*48*48)
	}
}
