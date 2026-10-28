package nn

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

// Exercise the real GPT-OSS head geometry across window boundaries and runner
// chunk sizes. The reference retains the previous explicit-mask semantics.
func TestSDPAGPTOSSWindowBF16Parity(t *testing.T) {
	for _, tc := range []struct {
		name string
		L, K int
	}{
		{"before-window", 127, 127},
		{"at-window", 128, 128},
		{"past-window", 129, 129},
		{"chunk", 2048, 2048},
		{"continuation", 129, 256},
		{"continued-chunk", 2048, 2175},
		{"continued-tail", 9, 136},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			const Hq, Hkv, D, W = 64, 8, 64, 128
			tensor := func(heads, length int, phase float64) *mlx.Array {
				vals := make([]float32, heads*length*D)
				for i := range vals {
					vals[i] = float32(0.3 * math.Sin(float64(i)*0.17+phase))
				}
				return mlx.FromValues(vals, 1, heads, length, D).AsType(mlx.DTypeBFloat16)
			}
			q, k, v := tensor(Hq, tc.L, 0.2), tensor(Hkv, tc.K, 0.7), tensor(Hkv, tc.K, 1.1)
			sinkValues := make([]float32, Hq)
			for i := range sinkValues {
				sinkValues[i] = float32(i-Hq/2) * 0.03
			}
			sinks := mlx.FromValues(sinkValues, Hq).AsType(mlx.DTypeBFloat16)
			b := newBatch([]int32{int32(tc.K - tc.L)}, tc.L, nil)
			applied := false
			applier := recordingApplier{w: W, l: tc.L, K: tc.K, applied: &applied}
			history := NewKVHistory(k, v, applier)
			mask := CausalMask()
			for _, sink := range []*mlx.Array{sinks, nil} {
				opts := []SDPAOption{WithKVHistory(history), WithMask(mask)}
				if sink != nil {
					opts = append(opts, WithSinks(sink))
				}
				got := ScaledDotProductAttention(b, q, 0.125, opts...)
				want := mlx.FastScaledDotProductAttentionWithSinks(q, k, v, 0.125, mlx.FastSDPANoWindow,
					"array", windowRefMask(tc.L, tc.K, W).AsType(mlx.DTypeBFloat16), sink)
				mlx.Eval(got, want)
				actual, reference := got.AsType(mlx.DTypeFloat32).Floats(), want.AsType(mlx.DTypeFloat32).Floats()
				for i, w := range reference {
					if math.IsNaN(float64(actual[i])) || math.Abs(float64(actual[i]-w)) > 0.002 {
						t.Fatalf("sink=%t output[%d]=%g, reference=%g", sink != nil, i, actual[i], w)
					}
				}
			}
			key := dispatchInputs{batch: b, mask: mask, applier: applier, K: tc.K, dtype: k.DType()}
			cached, ok := b.Memo.Get(key)
			if !ok {
				t.Fatal("missing memoized dispatch")
			}
			dispatch := cached.(sdpaDispatch)
			if dispatch.window != W || dispatch.mode != "causal" || dispatch.arr != nil || applied {
				t.Fatalf("expected native causal window without a materialized mask: %+v, applied=%t", dispatch, applied)
			}
		})
	}
}
