package kokoro

import (
	"math/rand/v2"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func linearTime(x *mlx.Array, length int) *mlx.Array {
	x = mlx.Transpose(x, 0, 2, 1)
	width := x.Dim(2)
	if width == 1 {
		return mlx.Transpose(mlx.BroadcastTo(x, int32(x.Dim(0)), int32(x.Dim(1)), int32(length)), 0, 2, 1)
	}
	var positions *mlx.Array
	if length == 1 {
		positions = mlx.FromValues([]float32{0}, 1)
	} else {
		scale := mlx.FromValue(float32(float64(width) / float64(length)))
		positions = mlx.Mul(mlx.Arange(0, float64(length), 1, mlx.DTypeInt32), scale)
		positions = mlx.Maximum(mlx.Sub(mlx.Add(positions, mlx.Mul(mlx.FromValue(float32(.5)), scale)), mlx.FromValue(float32(.5))), mlx.FromValue(float32(0)))
	}
	low := positions.FloorDivide(mlx.FromValue(float32(1))).AsType(mlx.DTypeInt32)
	high := mlx.Minimum(mlx.Add(low, mlx.FromValue(1)), mlx.FromValue(width-1))
	fraction := mlx.Reshape(mlx.Sub(positions, low), 1, 1, int32(length))
	y := mlx.Add(mlx.Mul(mlx.Take(x, low, 2), mlx.Sub(mlx.FromValue(float32(1)), fraction)), mlx.Mul(mlx.Take(x, high, 2), fraction))
	return mlx.Transpose(y, 0, 2, 1)
}

// harmonicSource uses the same phase integration and interpolation as the
// pinned Kokoro NSF source. noise is additive [1,samples,9] excitation noise.
func harmonicSource(pitch, initial, noise *mlx.Array, projection nn.LinearLayer) *mlx.Array {
	f0 := repeatTime(mlx.ExpandDims(pitch, 2), 300)
	harmonics := mlx.Reshape(mlx.Arange(1, 10, 1, mlx.DTypeInt32), 1, 1, 9)
	cycles := mlx.Div(mlx.Mul(f0, harmonics), mlx.FromValue(float32(24000))).Remainder(mlx.FromValue(float32(1)))
	first := mlx.Add(cycles.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice()), initial)
	cycles = mlx.Concatenate([]*mlx.Array{first, cycles.Slice(mlx.Slice(), mlx.Slice(1, mlx.End), mlx.Slice())}, 1)
	// Original interpolate(scale_factor=1/300) floors the output size.
	// f0 has exactly 300 samples per pitch frame, so retain that frame
	// count directly rather than rounding a floating-point product up.
	coarseLength := pitch.Dim(1)
	coarse := linearTime(cycles, coarseLength)
	phase := mlx.Mul(mlx.Mul(coarse.Cumsum(1, false, true), mlx.FromValue(float32(2))), mlx.FromValue(float32(3.141592653589793)))
	phase = linearTime(mlx.Mul(phase, mlx.FromValue(300)), coarseLength*300)
	phase = phase.Slice(mlx.Slice(), mlx.Slice(0, f0.Dim(1)), mlx.Slice())
	voiced := f0.Greater(mlx.FromValue(float32(10))).AsType(mlx.DTypeFloat32)
	sine := mlx.Mul(mlx.Sin(phase), mlx.FromValue(float32(.1)))
	return projection.Forward(mlx.Add(mlx.Mul(sine, voiced), noise)).Tanh()
}

func randomSource(pitch *mlx.Array, rng *rand.Rand, projection nn.LinearLayer) *mlx.Array {
	initial := make([]float32, 9)
	for i := 1; i < len(initial); i++ {
		initial[i] = rng.Float32()
	}
	count := pitch.Dim(1) * 300
	normal := make([]float32, count*9)
	for i := range normal {
		normal[i] = float32(rng.NormFloat64())
	}
	f0 := repeatTime(mlx.ExpandDims(pitch, 2), 300)
	voiced := f0.Greater(mlx.FromValue(float32(10))).AsType(mlx.DTypeFloat32)
	amp := mlx.Add(mlx.Mul(voiced, mlx.FromValue(float32(.003))), mlx.Div(mlx.Mul(mlx.Sub(mlx.FromValue(float32(1)), voiced), mlx.FromValue(float32(.1))), mlx.FromValue(3)))
	noise := mlx.Mul(amp, mlx.FromValues(normal, 1, count, 9))
	return harmonicSource(pitch, mlx.FromValues(initial, 1, 1, 9), noise, projection)
}
