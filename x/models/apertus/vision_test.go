package apertus

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"
)

func TestVisionEncodeCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := (&VisionTokenizer{}).encode(ctx, []float32{0, 0, 0}, 1, 1); err != context.Canceled {
		t.Fatalf("encode error = %v, want context.Canceled", err)
	}
}

func TestApertusImageSize(t *testing.T) {
	tests := []struct {
		name          string
		width, height int
		wantW, wantH  int
		wantTokens    int
	}{
		{name: "small COCO landscape", width: 640, height: 432, wantW: 640, wantH: 432, wantTokens: 1080},
		{name: "small COCO four by three", width: 640, height: 480, wantW: 640, wantH: 480, wantTokens: 1200},
		{name: "desktop screenshot", width: 1602, height: 968, wantW: 1600, wantH: 976, wantTokens: 6100},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotW, gotH := apertusImageSize(tt.width, tt.height)
			if gotW != tt.wantW || gotH != tt.wantH {
				t.Fatalf("apertusImageSize(%d, %d) = %dx%d, want %dx%d", tt.width, tt.height, gotW, gotH, tt.wantW, tt.wantH)
			}
			if got := gotW / 16 * (gotH / 16); got != tt.wantTokens {
				t.Fatalf("visual tokens = %d, want %d", got, tt.wantTokens)
			}
		})
	}
}

func TestApertusAdaptiveImageMemoryTiers(t *testing.T) {
	const gib = uint64(1 << 30)
	limits := []uint64{12 * gib, 18 * gib, 24 * gib, 36 * gib, 48 * gib, 96 * gib}
	previousPixels := 0
	for _, limit := range limits {
		input := testApertusImageInput(1602, 968)
		model := Model{mediaMemoryLimit: limit, mediaResident: 8 * gib}
		if err := model.fitApertusImages([]apertusMediaItem{{image: input}}); err != nil {
			t.Fatalf("limit %d: %v", limit, err)
		}
		pixels := input.width * input.height
		if pixels < minApertusImageArea {
			t.Fatalf("limit %d produced %d pixels below floor", limit, pixels)
		}
		if pixels < previousPixels {
			t.Fatalf("limit %d produced %d pixels after %d", limit, pixels, previousPixels)
		}
		if got := estimateApertusImagePeak(model.mediaResident, []*apertusImageInput{input}); got > limit {
			t.Fatalf("limit %d admitted estimated peak %d", limit, got)
		}
		previousPixels = pixels
		if limit <= 24*gib && input.width == input.canonicalWidth && input.height == input.canonicalHeight {
			t.Fatalf("limit %d retained unsafe canonical geometry", limit)
		}
		if limit >= 36*gib && (input.width != 1600 || input.height != 976) {
			t.Fatalf("safe high-memory tier geometry = %dx%d, want 1600x976", input.width, input.height)
		}
	}
}

func TestApertusAdaptiveImageMemoryUsesCommonCap(t *testing.T) {
	const gib = uint64(1 << 30)
	large := testApertusImageInput(1602, 968)
	small := testApertusImageInput(320, 240)
	model := Model{mediaMemoryLimit: 18 * gib, mediaResident: 8 * gib}
	if err := model.fitApertusImages([]apertusMediaItem{{image: large}, {image: small}}); err != nil {
		t.Fatal(err)
	}
	if large.width == large.canonicalWidth && large.height == large.canonicalHeight {
		t.Fatal("large image was not reduced")
	}
	if small.width != small.canonicalWidth || small.height != small.canonicalHeight {
		t.Fatalf("small image changed from %dx%d to %dx%d", small.canonicalWidth, small.canonicalHeight, small.width, small.height)
	}
}

func TestApertusAdaptiveImageMemoryRejectsBelowFloor(t *testing.T) {
	const gib = uint64(1 << 30)
	input := testApertusImageInput(1602, 968)
	model := Model{mediaMemoryLimit: 18 * gib, mediaResident: 19 * gib}
	err := model.fitApertusImages([]apertusMediaItem{{image: input}})
	if err == nil || !strings.Contains(err.Error(), "minimum resolution") {
		t.Fatalf("fitApertusImages() error = %v, want minimum-resolution rejection", err)
	}
}

func TestInspectApertusImageDefersPixels(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 320, 180))
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, img); err != nil {
		t.Fatal(err)
	}
	input, err := inspectApertusImage(context.Background(), encoded.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if input.pixels != nil {
		t.Fatal("metadata inspection allocated normalized pixels")
	}
	pixels, err := materializeApertusImage(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(pixels), input.width*input.height*3; got != want {
		t.Fatalf("pixel count = %d, want %d", got, want)
	}
}

func testApertusImageInput(width, height int) *apertusImageInput {
	canonicalW, canonicalH := apertusImageSize(width, height)
	return &apertusImageInput{
		originalWidth: width, originalHeight: height,
		canonicalWidth: canonicalW, canonicalHeight: canonicalH,
		width: canonicalW, height: canonicalH, gridWidth: canonicalW / 16, gridHeight: canonicalH / 16,
	}
}

func TestResizeApertusImageMatchesTorchvisionBicubic(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 320, 180))
	for y := range 180 {
		for x := range 320 {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x*3 + y), G: uint8(x + y*5), B: uint8(x*7 + y*11), A: 255,
			})
		}
	}
	pixels := resizeApertusImage(img, img.Bounds(), 336, 192)
	raw := make([]byte, len(pixels))
	for i, value := range pixels {
		raw[i] = byte(value)
	}
	if got, want := fmt.Sprintf("%x", sha256.Sum256(raw)), "ac5dd8f4674a41f51f32affc2a2f341f3d383c578be670e49ad4cb40ded8ed49"; got != want {
		t.Errorf("resized pixel SHA-256 = %s, want Torchvision BICUBIC %s", got, want)
	}
	want := map[[2]int][3]float32{
		{0, 0}:     {0, 0, 0},
		{1, 0}:     {3, 1, 6},
		{6, 0}:     {17, 6, 40},
		{100, 50}:  {77, 73, 157},
		{160, 90}:  {29, 62, 202},
		{335, 191}: {112, 190, 106},
	}
	for point, expected := range want {
		i := (point[1]*336 + point[0]) * 3
		got := [3]float32{pixels[i], pixels[i+1], pixels[i+2]}
		if got != expected {
			t.Errorf("pixel %v = %v, want Torchvision BICUBIC %v", point, got, expected)
		}
	}
}
