package kokoro

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestSourceSpectrumOriginalRealEndpoints(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		for _, alternating := range []bool{false, true} {
			input := make([]float32, 60)
			for i := range input {
				input[i] = -1
				if alternating && i%2 != 0 {
					input[i] = 1
				}
			}
			_, phase := sourceSpectrum(mlx.FromValues(input, 60), periodicHann())
			values := phase.Floats()
			for frame := range phase.Dim(0) {
				for _, bin := range []int{0, 10} {
					value := float64(values[frame*11+bin])
					if math.Signbit(value) || (value != 0 && math.Abs(value-math.Pi) > 1e-6) {
						t.Errorf("alternating=%v frame=%d bin=%d phase=%g, want 0 or +pi", alternating, frame, bin, value)
					}
				}
			}
		}
	})
}
