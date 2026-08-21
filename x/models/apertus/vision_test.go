package apertus

import (
	"crypto/sha256"
	"fmt"
	"image"
	"image/color"
	"testing"
)

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
