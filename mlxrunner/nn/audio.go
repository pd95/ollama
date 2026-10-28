package nn

import (
	"math"

	"github.com/ollama/ollama/mlx"
)

// SlaneyMelFilters builds normalized triangular filters with float32 MLX
// arithmetic, matching MLX-Audio's default DSP construction (zero minimum Hz).
func SlaneyMelFilters(sampleRate, nFFT, nMels int, maxHz float64) *mlx.Array {
	logStep := math.Log(6.4) / 27
	maxMel := maxHz / (200.0 / 3)
	if maxHz >= 1000 {
		maxMel = 15 + math.Log(maxHz/1000)/logStep
	}
	mels := mlx.Linspace(0, maxMel, nMels+2, mlx.DTypeFloat32)
	hz := mlx.Where(mlx.FromValue(float32(15)).LessEqual(mels), mlx.Mul(mlx.FromValue(float32(1000)), mlx.Exp(mlx.Mul(mlx.FromValue(float32(logStep)), mlx.Sub(mels, mlx.FromValue(float32(15)))))), mlx.Mul(mlx.FromValue(float32(200.0/3)), mels))
	diff := mlx.Sub(hz.Slice(mlx.Slice(1, nMels+2)), hz.Slice(mlx.Slice(0, nMels+1)))
	slopes := mlx.Sub(mlx.ExpandDims(hz, 0), mlx.ExpandDims(mlx.Linspace(0, float64(sampleRate/2), nFFT/2+1, mlx.DTypeFloat32), 1))
	down := mlx.Div(mlx.Neg(slopes.Slice(mlx.Slice(), mlx.Slice(0, nMels))), diff.Slice(mlx.Slice(0, nMels)))
	up := mlx.Div(slopes.Slice(mlx.Slice(), mlx.Slice(2, nMels+2)), diff.Slice(mlx.Slice(1, nMels+1)))
	bank := mlx.Maximum(mlx.FromValue(float32(0)), mlx.Minimum(down, up))
	norm := mlx.Div(mlx.FromValue(float32(2)), mlx.Sub(hz.Slice(mlx.Slice(2, nMels+2)), hz.Slice(mlx.Slice(0, nMels))))
	return mlx.Transpose(mlx.Mul(bank, mlx.ExpandDims(norm, 0)), 1, 0)
}
