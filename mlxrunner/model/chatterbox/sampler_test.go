package chatterbox

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestSampler(t *testing.T) {
	a, b := rand.New(rand.NewPCG(42, 43)), rand.New(rand.NewPCG(42, 43))
	logits := []float32{10, 9, 1, -3}
	for range 100 {
		x, e := sampleT3(logits, []int32{0, 0}, a)
		y, f := sampleT3(logits, []int32{0}, b)
		if e != nil || f != nil || x != y || x > 1 {
			t.Fatal(x, y, e, f)
		}
	}
	for _, bad := range [][]float32{nil, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, e := sampleT3(bad, nil, a); e == nil {
			t.Fatal("accepted nonfinite/empty logits")
		}
	}
	if _, e := sampleT3(logits, []int32{99}, a); e == nil {
		t.Fatal("accepted invalid history")
	}
}
