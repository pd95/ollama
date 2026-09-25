package apertus

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
)

// Compare two independent execution paths: whole-sequence causal prefill and
// incremental attention with retained history and nonzero RoPE offsets.
func TestForwardCachedChunksMatchFullPrefill(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	base, err := model.New(minimalManifestRoot(t, "ApertusForCausalLM"))
	if err != nil {
		t.Fatal(err)
	}
	m := base.(*Model)
	tensors := tinyDenseTensors(m.Config)
	for name, tensor := range tensors {
		values := make([]float32, tensor.Size())
		for i := range values {
			values[i] = float32(math.Sin(float64(i+1)*0.7)) * 0.1
			if strings.Contains(name, "norm.weight") {
				values[i] = 1
			}
		}
		if tensor.NumDims() == 0 {
			tensors[name] = mlx.FromValue(values[0]).AsType(mlx.DTypeBFloat16)
		} else {
			tensors[name] = mlx.FromValues(values, tensor.Dims()...).AsType(mlx.DTypeBFloat16)
		}
	}
	if err := m.LoadWeights(tensors); err != nil {
		t.Fatal(err)
	}
	tokens := []int32{1, 2, 0, 1}
	makeBatch := func(start, end int) *batch.Batch {
		return &batch.Batch{InputIDs: mlx.FromValues(tokens[start:end], 1, end-start), SeqOffsets: []int32{int32(start)}, SeqQueryLens: []int32{int32(end - start)}}
	}
	full, _ := m.Forward(makeBatch(0, len(tokens)), nil)
	full = full.AsType(mlx.DTypeFloat32)
	mlx.Eval(full)
	want := full.Floats()
	for _, chunk := range []int{1, 2} {
		t.Run(fmt.Sprintf("chunk_%d", chunk), func(t *testing.T) {
			caches := m.NewCaches()
			defer func() {
				for _, c := range caches {
					c.Free()
				}
			}()
			for start := 0; start < len(tokens); start += chunk {
				end := min(start+chunk, len(tokens))
				out, _ := m.Forward(makeBatch(start, end), caches)
				out = out.AsType(mlx.DTypeFloat32)
				mlx.Eval(out)
				for i, got := range out.Floats() {
					expected := want[start*int(m.HiddenSize)+i]
					if math.IsNaN(float64(got)) || math.Abs(float64(got-expected)) > 0.03125 {
						t.Fatalf("offset=%d value[%d]=%g, full prefill=%g", start, i, got, expected)
					}
				}
				for _, c := range caches {
					if c.Offset() != end {
						t.Fatalf("cache offset=%d, want %d", c.Offset(), end)
					}
				}
			}
		})
	}
}
