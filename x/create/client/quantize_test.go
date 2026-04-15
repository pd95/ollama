package client

import (
	"io"
	"testing"

	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/safetensors"
)

func TestDecodeSourceFP8TensorAcceptsWeightScale(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX unavailable: %v", err)
	}

	weight := mlx.FromValues([]uint8{0, 1, 2, 3}, 2, 2)
	scale := mlx.FromValues([]float32{1}, 1, 1).AsType(mlx.DTypeBFloat16)
	got, err := decodeSourceFP8Tensor(weight, scale)
	if err != nil {
		t.Fatal(err)
	}
	mlx.Eval(got)
	if dims := got.Dims(); len(dims) != 2 || dims[0] != 2 || dims[1] != 2 {
		t.Fatalf("decoded dims = %v, want [2 2]", dims)
	}
}

func TestPackedTensorReaderFallsBackToRaw(t *testing.T) {
	raw := safetensors.NewTensorDataFromBytes("blocks.0.experts.gate_proj.weight", "BF16", []int32{2, 4}, []byte{1, 2, 3, 4})
	reader := packedTensorReader(create.PackedTensorInput{
		Name: raw.Name,
		Raw:  raw,
	})
	if reader == nil {
		t.Fatal("packedTensorReader() = nil")
	}

	data, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if len(data) <= len(raw.Shape) {
		t.Fatalf("reader produced %d bytes, want safetensors payload", len(data))
	}
}
