package qwen3tts

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestSpeechEncoderGELUMatchesOriginal(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		input := []float32{-3, -1, 0, 1, 3}
		// Original erf GELU reference values; tanh approximation differs enough
		// at +/-3 to fail this gate.
		want := []float64{-.00404969409489031, -.15865525393145707, 0, .8413447460685429, 2.99595030590511}
		got := encoderGELU(mlx.FromValues(input, len(input))).Floats()
		for i, value := range got {
			if math.Abs(float64(value)-want[i]) > 2e-6 {
				t.Errorf("GELU(%g) = %g, want %g", input[i], value, want[i])
			}
		}
	})
}
