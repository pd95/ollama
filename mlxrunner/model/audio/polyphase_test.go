package audio

import (
	"math"
	"testing"
)

func TestKaiserBestResample(t *testing.T) {
	for _, rates := range [][2]int{{24000, 16000}, {16000, 24000}, {48000, 16000}, {44100, 24000}} {
		input := make([]float32, rates[0]/10)
		for i := range input {
			input[i] = .25
		}
		out := ResampleKaiserBest(input, rates[0], rates[1])
		wantLength := (len(input)*rates[1] + rates[0] - 1) / rates[0]
		if len(out) != wantLength {
			t.Fatal("incorrect resampled length")
		}
		for _, v := range out {
			if math.Abs(float64(v-.25)) > 1e-6 {
				t.Fatalf("DC/edge response %g at %v", v, rates)
			}
		}
	}
	// Reject aliases above the destination Nyquist without attenuating speech
	// frequencies in the pass band. Ignore the finite clip's edge transients.
	for _, frequency := range []float64{1000, 11000} {
		input := make([]float32, 24000)
		for i := range input {
			input[i] = float32(math.Sin(2 * math.Pi * frequency * float64(i) / 24000))
		}
		out := ResampleKaiserBest(input, 24000, 16000)
		var energy float64
		for _, v := range out[256 : len(out)-256] {
			energy += float64(v) * float64(v)
		}
		rms := math.Sqrt(energy / float64(len(out)-512))
		if frequency == 1000 && math.Abs(rms-1/math.Sqrt2) > .002 || frequency == 11000 && rms > 1e-5 {
			t.Fatalf("unexpected %g Hz RMS %g", frequency, rms)
		}
	}
}
