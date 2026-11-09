package voxtraltts

import (
	"slices"
	"testing"
)

func TestCodecALiBiSlopesMatchOriginalVoxtral(t *testing.T) {
	// Original eight-head codec slopes are 2^(-8*head/heads), starting at one.
	// The previous implementation started at one half, changing every head.
	want := []float32{1, .5, .25, .125, .0625, .03125, .015625, .0078125}
	if got := codecALiBiSlopes(8); !slices.Equal(got, want) {
		t.Fatalf("codec slopes %v, want original-provider sequence %v", got, want)
	}
}
