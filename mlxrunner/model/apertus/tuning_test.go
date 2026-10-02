package apertus

import (
	"encoding/json"
	"math"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// These equivalence checks diagnose implementation errors, not answer quality.
func TestTuningXIELU(t *testing.T) {
	for _, dtype := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeFloat16, mlx.DTypeBFloat16} {
		// Compile's closure is initialized once. Establish the compiled closure
		// before exercising global disable, otherwise the cached closure is eager.
		for _, compiled := range []bool{true, false} {
			mlxtest.Run(t, func(t *mlxtest.T) {
				if compiled {
					mlx.EnableCompile()
				} else {
					mlx.DisableCompile()
				}
				t.Cleanup(mlx.EnableCompile)
				x := mlx.FromValues([]float32{-8, -1, -1e-6, 0, 1e-6, 0.5, 4}, 1, 7).AsType(dtype)
				for _, a := range []*XIELU{{AlphaP: 0.8, AlphaN: 1.3, Beta: 0.5, Eps: -1e-6}, {AlphaP: 1.1, AlphaN: 0.9, Beta: 0.2, Eps: -1e-5}} {
					a.Params = []*mlx.Array{mlx.FromValue(a.AlphaP), mlx.FromValue(a.AlphaN), mlx.FromValue(a.Beta), mlx.FromValue(a.Eps)}
					got := a.Forward(x)
					assertTuningClose(t, got, eagerTuningXIELU(a, x), 0.02)
					if got.DType() != dtype {
						t.Fatalf("output dtype %s, want %s", got.DType(), dtype)
					}
				}
			})
		}
	}
}

func eagerTuningXIELU(a *XIELU, x *mlx.Array) *mlx.Array {
	dt := x.DType()
	x = x.AsType(mlx.DTypeFloat32)
	positive := mlx.Add(mlx.Mul(mlx.FromValue(a.AlphaP), mlx.Mul(x, x)), mlx.Mul(mlx.FromValue(a.Beta), x))
	negative := mlx.Add(mlx.Mul(mlx.Sub(mlx.Sub(mlx.Exp(mlx.Minimum(x, mlx.FromValue(a.Eps))), mlx.FromValue[float32](1)), x), mlx.FromValue(a.AlphaN)), mlx.Mul(mlx.FromValue(a.Beta), x))
	return mlx.Where(x.Greater(mlx.FromValue[float32](0)), positive, negative).AsType(dt)
}

func assertTuningClose(t *mlxtest.T, got, want *mlx.Array, tolerance float64) {
	t.Helper()
	diff := mlx.Sub(got.AsType(mlx.DTypeFloat32), want.AsType(mlx.DTypeFloat32)).Abs().MaxAxis(-1, false)
	mlx.Eval(diff)
	for _, value := range diff.Floats() {
		if math.IsNaN(float64(value)) || float64(value) > tolerance {
			t.Fatalf("maximum error %g exceeds %g", value, tolerance)
		}
	}
}

func TestTuningHeadNorm(t *testing.T) {
	for _, dtype := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeFloat16, mlx.DTypeBFloat16} {
		for _, length := range []int{1, 7, 257} {
			mlxtest.Run(t, func(t *mlxtest.T) {
				x := tuningValues(1, length, 4, 8).AsType(dtype)
				norm := nn.NewRMSNorm(tuningValues(8).Abs().Add(mlx.FromValue[float32](1)), 1e-5)
				want := headRMSNorm(norm, x.Transpose(0, 2, 1, 3), 1, 4, int32(length), 8, 1e-5)
				got := norm.Forward(x, 1e-5).Transpose(0, 2, 1, 3)
				assertTuningClose(t, got, want, 1e-5)
			})
		}
	}
}

func tuningValues(dims ...int) *mlx.Array {
	n := 1
	for _, d := range dims {
		n *= d
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(math.Sin(float64(i)*0.7)) * 0.1
	}
	return mlx.FromValues(v, dims...)
}

// Opt-in synchronized native profile, runnable through mlx_porting_verify.sh.
// Its output is diagnostic; paired end-to-end API blocks decide acceptance.
func TestTuningProfile(t *testing.T) {
	path := os.Getenv("PORTING_APERTUS_TUNING_OUTPUT")
	if path == "" {
		t.Skip("set PORTING_APERTUS_TUNING_OUTPUT for native component timings")
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		library, err := mlx.LoadedLibraryPath()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("native library=%s GPU=%v Metal=%v", library, mlx.GPUIsAvailable(), mlx.MetalIsAvailable())
		results := make(map[string][]float64)
		measure := func(name string, fn func() *mlx.Array) {
			for range 3 {
				mlx.Scoped(func() { mlx.Eval(fn()) })
			}
			for range 10 {
				start := time.Now()
				mlx.Scoped(func() { mlx.Eval(fn()) })
				results[name] = append(results[name], float64(time.Since(start).Nanoseconds())/1e6)
			}
		}
		for _, length := range []int{1, 2048} {
			mlx.Scoped(func() {
				x := tuningValues(1, length, 21504).AsType(mlx.DTypeBFloat16)
				mlx.Eval(x)
				a := &XIELU{AlphaP: 0.8, AlphaN: 1.3, Beta: 0.5, Eps: -1e-6}
				a.Params = []*mlx.Array{mlx.FromValue(a.AlphaP), mlx.FromValue(a.AlphaN), mlx.FromValue(a.Beta), mlx.FromValue(a.Eps)}
				name := "xielu-decode"
				if length > 1 {
					name = "xielu-prefill"
				}
				measure(name, func() *mlx.Array { return a.Forward(x) })
				measure(name+"-eager", func() *mlx.Array { return eagerTuningXIELU(a, x) })
			})
			mlx.ClearCache()
		}
		for _, length := range []int{1, 2048} {
			mlx.Scoped(func() {
				x := tuningValues(1, length, 32, 128).AsType(mlx.DTypeBFloat16)
				norm := nn.NewRMSNorm(mlx.AddScalar(tuningValues(128).Abs(), 1), 1e-5)
				mlx.Eval(x, norm.Weight)
				name := "headnorm-decode"
				if length > 1 {
					name = "headnorm-prefill"
				}
				measure(name+"-old", func() *mlx.Array { return headRMSNorm(norm, x.Transpose(0, 2, 1, 3), 1, 32, int32(length), 128, 1e-5) })
				measure(name+"-contiguous", func() *mlx.Array { return norm.Forward(x, 1e-5).Transpose(0, 2, 1, 3) })
			})
			mlx.ClearCache()
		}
		data, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	})
}

// Paired component timings screen hypotheses before expensive API qualification.
func tuningProfilePair(results map[string][]float64, name string, before, after func() *mlx.Array) {
	for range 3 {
		mlx.Scoped(func() { mlx.Eval(before()) })
		mlx.Scoped(func() { mlx.Eval(after()) })
	}
	funcs := []func() *mlx.Array{before, after}
	labels := []string{name + "-before", name + "-after"}
	for block := range 10 {
		for j := range 2 {
			i := (j + block) % 2
			start := time.Now()
			mlx.Scoped(func() { mlx.Eval(funcs[i]()) })
			results[labels[i]] = append(results[labels[i]], float64(time.Since(start).Nanoseconds())/1e6)
		}
	}
}

func tuningWriteProfile(t *mlxtest.T, path string, results map[string][]float64) {
	t.Helper()
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
