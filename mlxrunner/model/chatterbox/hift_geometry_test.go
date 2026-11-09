package chatterbox

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestHiFTSourceSTFTOriginalCenteredGeometry(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		input := make([]float32, 32)
		for i := range input {
			input[i] = float32(i)
		}
		window := make([]float32, 16)
		for i := range window {
			window[i] = 1
		}
		for _, turbo := range []bool{false, true} {
			h := hiFT{turbo: turbo, window: mlx.FromValues(window, 16)}
			real, imag := h.sourceSTFT(mlx.FromValues(input, 1, 1, 32))
			// Original centered n_fft=16, hop=4 gives 1+32/4 frames.
			if !slices.Equal(real.Dims(), []int{1, 9, 9}) || !slices.Equal(imag.Dims(), []int{1, 9, 9}) {
				t.Fatalf("turbo=%v STFT shapes %v/%v, want [1 9 9]", turbo, real.Dims(), imag.Dims())
			}
			// First reflection-padded frame is [8..1,0..7], summing to 64.
			if dc := real.Floats()[0]; math.Abs(float64(dc)-64) > 1e-5 {
				t.Errorf("turbo=%v first DC=%g, want 64", turbo, dc)
			}
		}
	})
}
