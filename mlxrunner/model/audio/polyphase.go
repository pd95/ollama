package audio

import "math"

// ResampleKaiserBest matches the whole-buffer, edge-padded polyphase FIR used
// by MLX-Audio reference conditioning. Keep Resample's existing profile for
// models already qualified with it; the two filters have different bandwidths.
func ResampleKaiserBest(samples []float32, from, to int) []float32 {
	if from == to || len(samples) == 0 {
		return samples
	}
	if from <= 0 || to <= 0 {
		return nil
	}
	a, b := from, to
	for b != 0 {
		a, b = b, a%b
	}
	up, down := to/a, from/a
	const rolloff = 0.9475937167399596
	const beta = 14.769656459379492
	half := 64 * max(up, down)
	cutoff := rolloff / float64(max(up, down))
	filter := make([]float64, 2*half+1)
	denom := besselI0(beta)
	var normalization float64
	for i := range filter {
		distance := float64(i - half)
		u := distance / float64(half)
		filter[i] = cutoff * sinc(cutoff*distance) * besselI0(beta*math.Sqrt(max(0, 1-u*u))) / denom
		normalization += filter[i]
	}
	for i := range filter {
		filter[i] *= float64(up) / normalization
	}
	length := (int64(len(samples))*int64(up) + int64(down) - 1) / int64(down)
	out := make([]float32, int(length))
	for i := range out {
		center := i * down
		lo := int(math.Ceil(float64(center-half) / float64(up)))
		hi := (center + half) / up
		var sum float64
		for j := lo; j <= hi; j++ {
			sum += float64(samples[max(0, min(j, len(samples)-1))]) * filter[half+center-j*up]
		}
		out[i] = float32(sum)
	}
	return out
}
