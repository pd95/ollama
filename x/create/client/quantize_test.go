package client

import (
	"io"
	"testing"

	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/safetensors"
)

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

