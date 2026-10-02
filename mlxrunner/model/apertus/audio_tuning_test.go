package apertus

import (
	"fmt"
	"math/rand"
	"os"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func tuningEagerLSTM(l *apertureLSTM, input *mlx.Array, materialize func(...*mlx.Array)) (*mlx.Array, error) {
	residual := input
	x := input
	for layer := range l.inputWeights {
		if x == nil || x.Dim(1) == 0 {
			return nil, fmt.Errorf("Apertus audio LSTM layer %d received an empty sequence", layer)
		}
		h := mlx.Zeros(mlx.DTypeFloat32, 1, int(l.hidden))
		c := mlx.Zeros(mlx.DTypeFloat32, 1, int(l.hidden))
		steps := x.Dim(1)
		outputs := make([]*mlx.Array, 0, steps)
		for t := range steps {
			xt := x.Slice(mlx.Slice(), mlx.Slice(t, t+1), mlx.Slice()).Squeeze(1)
			gates := mlx.Add(mlx.Add(mlx.Matmul(xt, mlx.Transpose(l.inputWeights[layer], 1, 0)), l.inputBiases[layer]), mlx.Add(mlx.Matmul(h, mlx.Transpose(l.hiddenWeights[layer], 1, 0)), l.hiddenBiases[layer]))
			i := mlx.Sigmoid(gates.Slice(mlx.Slice(), mlx.Slice(0, int(l.hidden))))
			f := mlx.Sigmoid(gates.Slice(mlx.Slice(), mlx.Slice(int(l.hidden), int(2*l.hidden))))
			g := gates.Slice(mlx.Slice(), mlx.Slice(int(2*l.hidden), int(3*l.hidden))).Tanh()
			o := mlx.Sigmoid(gates.Slice(mlx.Slice(), mlx.Slice(int(3*l.hidden), int(4*l.hidden))))
			c = mlx.Add(mlx.Mul(f, c), mlx.Mul(i, g))
			h = mlx.Mul(o, c.Tanh())
			if materialize != nil {
				keep := make([]*mlx.Array, 0, len(outputs)+4)
				keep = append(keep, outputs...)
				// x is sliced again by the next recurrent step; residual is
				// reused after every LSTM layer.
				keep = append(keep, h, c, x, residual)
				materialize(keep...)
			}
			outputs = append(outputs, h.ExpandDims(1))
		}
		x = mlx.Concatenate(outputs, 1)
	}
	return mlx.Add(x, residual), nil
}

func tuningAudioLSTM(hidden int, layers int) *apertureLSTM {
	l := &apertureLSTM{hidden: int32(hidden)}
	// Avoid the highly correlated rank-two sine matrices used by linear tests:
	// at width 512 they create an unstable synthetic recurrence.
	rng := rand.New(rand.NewSource(42))
	weights := func() *mlx.Array {
		values := make([]float32, 4*hidden*hidden)
		for i := range values {
			values[i] = (rng.Float32() - 0.5) * 0.1
		}
		return mlx.FromValues(values, 4*hidden, hidden)
	}
	for range layers {
		l.inputWeights = append(l.inputWeights, weights())
		l.hiddenWeights = append(l.hiddenWeights, weights())
		l.inputBiases = append(l.inputBiases, tuningValues(4*hidden))
		l.hiddenBiases = append(l.hiddenBiases, tuningValues(4*hidden))
	}
	return l
}

func TestTuningBatchedLSTM(t *testing.T) {
	for _, steps := range []int{1, 127, 128, 129, 257} {
		t.Run(fmt.Sprint(steps), func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.EnableCompile()
				l := tuningAudioLSTM(8, 2)
				input := tuningValues(1, steps, 8)
				want, err := tuningEagerLSTM(l, input, mlx.Eval)
				if err != nil {
					t.Fatal(err)
				}
				got, err := l.forward(input, mlx.Eval)
				if err != nil {
					t.Fatal(err)
				}
				// Batched GEMM and fused gates can change float32 accumulation order.
				assertTuningClose(t, got, want, 5e-4)
				// Repeated calls must reset hidden/cell state and retain learned arrays.
				again, err := l.forward(input, mlx.Eval)
				if err != nil {
					t.Fatal(err)
				}
				assertTuningClose(t, again, want, 5e-4)
				// Production media encoding constructs a lazy graph and evaluates
				// after its scope ends, without a per-step callback.
				lazy := mlx.ScopedEval(func() []*mlx.Array {
					output, err := l.forward(input, nil)
					if err != nil {
						t.Fatal(err)
					}
					return []*mlx.Array{output}
				})[0]
				assertTuningClose(t, lazy, want, 5e-4)
				if _, err := l.forward(mlx.Zeros(mlx.DTypeFloat32, 1, 0, 8), mlx.Eval); err == nil {
					t.Fatal("empty sequence accepted")
				}
			})
		})
	}
}

func TestTuningAudioProfile(t *testing.T) {
	path := os.Getenv("PORTING_APERTUS_AUDIO_OUTPUT")
	if path == "" {
		t.Skip("set PORTING_APERTUS_AUDIO_OUTPUT for component timings")
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		results := map[string][]float64{}
		l := tuningAudioLSTM(512, 2)
		mlx.Eval(append(append(append(l.inputWeights, l.hiddenWeights...), l.inputBiases...), l.hiddenBiases...)...)
		for _, steps := range []int{129, 513} {
			x := tuningValues(1, steps, 512)
			mlx.Eval(x)
			eager := func() *mlx.Array {
				y, err := tuningEagerLSTM(l, x, nil)
				if err != nil {
					t.Fatal(err)
				}
				return y
			}
			batched := func() *mlx.Array {
				y, err := l.forward(x, nil)
				if err != nil {
					t.Fatal(err)
				}
				return y
			}
			assertTuningClose(t, batched(), eager(), 5e-4)
			tuningProfilePair(results, fmt.Sprintf("lstm-%d", steps), eager, batched)
		}
		tuningWriteProfile(t, path, results)
	})
}
