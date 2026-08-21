package nn

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestAffineQuantizedLinearEmbeddingAndTiedOutputMatchDequantized(t *testing.T) {
	skipIfNoMLX(t)

	weightVals := make([]float32, 8*64)
	for i := range weightVals {
		weightVals[i] = float32((i%19)-9) / 11
	}
	inputVals := make([]float32, 2*64)
	for i := range inputVals {
		inputVals[i] = float32((i%13)-6) / 9
	}
	weight := mlx.FromValues(weightVals, 8, 64).AsType(mlx.DTypeBFloat16)
	input := mlx.FromValues(inputVals, 2, 64).AsType(mlx.DTypeBFloat16)
	indices := mlx.FromValues([]int32{1, 6}, 2)
	mlx.Eval(weight, input, indices)

	for _, bits := range []int{2, 3, 4, 5, 6, 8} {
		quantized := NewQuantizedLinear(weight, nil, 64, bits, "affine")
		dequantized := mlx.Dequantize(quantized.Weight, quantized.Scales, quantized.QBiases, 64, bits, "affine", nil)
		mlx.Eval(dequantized)

		wantLinear := NewLinear(dequantized, nil).Forward(input).AsType(mlx.DTypeFloat32)
		gotLinear := quantized.Forward(input).AsType(mlx.DTypeFloat32)

		embedding := &QuantizedEmbedding{
			Weight: quantized.Weight, Scales: quantized.Scales, QBiases: quantized.QBiases,
			GroupSize: 64, Bits: bits, Mode: "affine",
		}
		wantEmbedding := NewEmbedding(dequantized).Forward(indices).AsType(mlx.DTypeFloat32)
		gotEmbedding := embedding.Forward(indices).AsType(mlx.DTypeFloat32)
		wantTied := NewLinear(dequantized, nil).Forward(input).AsType(mlx.DTypeFloat32)
		gotTied := embedding.AsLinear().Forward(input).AsType(mlx.DTypeFloat32)
		mlx.Eval(wantLinear, gotLinear, wantEmbedding, gotEmbedding, wantTied, gotTied)

		for label, pair := range map[string][2][]float32{
			"linear":    {gotLinear.Floats(), wantLinear.Floats()},
			"embedding": {gotEmbedding.Floats(), wantEmbedding.Floats()},
			"tied":      {gotTied.Floats(), wantTied.Floats()},
		} {
			if len(pair[0]) != len(pair[1]) {
				t.Fatalf("%d-bit %s output length = %d, want %d", bits, label, len(pair[0]), len(pair[1]))
			}
			for i := range pair[0] {
				if !approxEqual(pair[0][i], pair[1][i], 2e-2) {
					t.Fatalf("%d-bit %s output[%d] = %.6f, want %.6f", bits, label, i, pair[0][i], pair[1][i])
				}
			}
		}
	}
}
