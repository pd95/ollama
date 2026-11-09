package chatterbox

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// T3 uses temperature .8, min-p .05 and a 1.2 penalty over all generated
// history, including its initial speech BOS. Sampling is request-local.
func sampleT3(values []float32, history []int32, rng *rand.Rand) (int32, error) {
	logits := append([]float32(nil), values...)
	for _, id := range history {
		if id < 0 || int(id) >= len(logits) {
			return 0, fmt.Errorf("invalid Chatterbox history token")
		}
	}
	seen := make(map[int32]bool, len(history))
	for _, id := range history {
		if !seen[id] {
			if logits[id] < 0 {
				logits[id] *= 1.2
			} else {
				logits[id] /= 1.2
			}
			seen[id] = true
		}
	}
	maximum := math.Inf(-1)
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, fmt.Errorf("nonfinite Chatterbox logits")
		}
		maximum = math.Max(maximum, float64(v))
	}
	if len(logits) == 0 {
		return 0, fmt.Errorf("empty Chatterbox logits")
	}
	probabilities := make([]float64, len(logits))
	total := float64(0)
	for i, v := range logits {
		p := math.Exp((float64(v) - maximum) / .8)
		if p >= .05 {
			probabilities[i] = p
			total += p
		}
	}
	chosen := rng.Float64() * total
	for i, p := range probabilities {
		if p > 0 {
			chosen -= p
			if chosen <= 0 {
				return int32(i), nil
			}
		}
	}
	return 0, fmt.Errorf("invalid Chatterbox sampling distribution")
}
