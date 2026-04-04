package gptossmxfp4

import (
	"slices"
	"testing"
)

func TestStableAndExperimentalMXFP4Match(t *testing.T) {
	blockShape := []int32{2, 3, 2, 16}
	scaleShape := []int32{2, 3, 2}

	blocks := make([]byte, 2*3*2*16)
	for i := range blocks {
		blocks[i] = byte(i)
	}
	scales := make([]byte, 2*3*2)
	for i := range scales {
		scales[i] = byte(17 + i)
	}

	comp, err := Compare(blocks, blockShape, scales, scaleShape)
	if err != nil {
		t.Fatalf("Compare failed: %v", err)
	}

	if !slices.Equal(comp.StableShape, []int32{2, 3, 64}) {
		t.Fatalf("stable shape = %v, want %v", comp.StableShape, []int32{2, 3, 64})
	}
	if !slices.Equal(comp.ExperimentalShape, comp.StableShape) {
		t.Fatalf("experimental shape = %v, want %v", comp.ExperimentalShape, comp.StableShape)
	}
	if comp.StableSHA256 != comp.ExperimentalSHA256 {
		t.Fatalf("stable bytes sha = %s, experimental bytes sha = %s", comp.StableSHA256, comp.ExperimentalSHA256)
	}
}
