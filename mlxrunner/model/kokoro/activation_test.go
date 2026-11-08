package kokoro

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestALBERTGELUMatchesOriginal(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// Fixed original gelu_new values distinguish it from erf GELU.
		input := []float32{-3, -1, 0, 1, 3}
		want := []float64{-.0036373920817729943, -.15880800939172324, 0, .8411919906082768, 2.996362607918227}
		got := gelu(mlx.FromValues(input, len(input))).Floats()
		for i, value := range got {
			if math.Abs(float64(value)-want[i]) > 2e-6 {
				t.Errorf("gelu_new(%g) = %g, want %g", input[i], value, want[i])
			}
		}
	})
}
