package create

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
)

func originalChatterboxTestFloat(name string, shape []int32, values ...float32) *safetensors.TensorData {
	raw := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	return safetensors.NewTensorDataFromBytes(name, "F32", shape, raw)
}

func TestOriginalChatterboxTensorOperations(t *testing.T) {
	for _, test := range []struct {
		name string
		item originalChatterboxTensor
		want []float32
	}{
		{"square transpose", originalChatterboxTensor{dtype: "F32", source: originalChatterboxTestFloat("w", []int32{2, 2}, 1, 2, 3, 4), axes: []int{1, 0}}, []float32{1, 3, 2, 4}},
		{"conv transpose", originalChatterboxTensor{dtype: "F32", source: originalChatterboxTestFloat("w", []int32{2, 1, 2}, 1, 2, 3, 4), axes: []int{1, 2, 0}}, []float32{1, 3, 2, 4}},
		{"bias sum", originalChatterboxTensor{dtype: "F32", source: originalChatterboxTestFloat("a", []int32{2}, 1, -2), other: originalChatterboxTestFloat("b", []int32{2}, 3, 5), operation: "add"}, []float32{4, 3}},
		{"weight norm", originalChatterboxTensor{dtype: "F32", source: originalChatterboxTestFloat("v", []int32{2, 1, 4}, 3, 4, 0, 0, 0, 0, 6, 8), other: originalChatterboxTestFloat("g", []int32{2, 1, 1}, 10, 5), operation: "weight_norm"}, []float32{6, 8, 0, 0, 0, 0, 3, 4}},
	} {
		t.Run(test.name, func(t *testing.T) {
			raw, err := originalChatterboxTensorBytes(t.Context(), test.item)
			if err != nil {
				t.Fatal(err)
			}
			want := originalChatterboxTestFloat("want", nil, test.want...)
			data, err := originalChatterboxRead(t.Context(), want)
			if err != nil || !bytes.Equal(data, raw) {
				t.Fatalf("got %x want %x: %v", raw, data, err)
			}
		})
	}
}

func TestOriginalChatterboxTensorRejections(t *testing.T) {
	for _, axes := range [][]int{{1, 1}, {0, 2}, {-1, 1}, {0}} {
		if _, err := originalChatterboxPermute(t.Context(), make([]byte, 16), []int32{2, 2}, axes); err == nil {
			t.Errorf("accepted axes %v", axes)
		}
	}
	if _, err := originalChatterboxPermute(t.Context(), make([]byte, 16), []int32{2, 0}, []int{0, 1}); err == nil {
		t.Fatal("accepted invalid extent")
	}
	bad := originalChatterboxTensor{dtype: "F32", source: originalChatterboxTestFloat("bad", []int32{1}, float32(math.Inf(1)))}
	if _, err := originalChatterboxTensorBytes(t.Context(), bad); err == nil {
		t.Fatal("accepted infinite source")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := originalChatterboxTensorBytes(ctx, bad); err != context.Canceled {
		t.Fatalf("canceled tensor: %v", err)
	}
	for _, n := range []int64{-1, math.MaxInt32 + 1} {
		data := binary.LittleEndian.AppendUint64(nil, uint64(n))
		item := originalChatterboxTensor{dtype: "I32", source: safetensors.NewTensorDataFromBytes("token", "I64", []int32{1}, data), operation: "narrow"}
		if _, err := originalChatterboxTensorBytes(t.Context(), item); err == nil {
			t.Errorf("accepted token %d", n)
		}
	}
}

func TestOriginalTurboSourceRouting(t *testing.T) {
	dir := t.TempDir()
	for _, name := range append(slices.Clone(originalTurboWeightFiles), originalTurboConfigFiles...) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("fixture"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if !IsSafetensorsModelDir(dir) {
		t.Fatal("original layout was not recognized")
	}
	weights, err := SafetensorsWeightFiles(dir)
	if err != nil || !slices.Equal(weights, originalTurboWeightFiles) {
		t.Fatalf("source transfer %v %v", weights, err)
	}
	configs, err := SafetensorsConfigFiles(dir)
	if err != nil || !slices.Equal(configs, originalTurboConfigFiles) {
		t.Fatalf("metadata transfer %v %v", configs, err)
	}
	cfg, raw, err := readSourceModelConfig(dir)
	if err != nil || cfg.Architecture() != chatterboxTurboArchitecture {
		t.Fatalf("synthesized architecture: %v, %v", cfg, err)
	}
	var fields map[string]any
	if err := json.Unmarshal(raw, &fields); err != nil || fields["quantization"] != nil || fields["quantization_config"] != nil {
		t.Fatalf("original source advertised quantization: %s, %v", raw, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if IsSafetensorsModelDir(dir) || validateOriginalTurboFiles(dir) == nil {
		t.Fatal("accepted ambiguous original/native layout")
	}
}

func TestOriginalTurboGeometryAndTables(t *testing.T) {
	r, err := gzip.NewReader(bytes.NewReader(originalTurboShapes))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	var shapes map[string]map[string][]int32
	if err := json.NewDecoder(r).Decode(&shapes); err != nil {
		t.Fatal(err)
	}
	if len(shapes["main"]) != 1684 || len(shapes["camp"]) != 815 || len(shapes["s3"]) != 104 {
		t.Fatalf("unexpected complete geometry: %d/%d/%d", len(shapes["main"]), len(shapes["camp"]), len(shapes["s3"]))
	}
	tables := originalTurboTables()
	for _, table := range tables {
		if !slices.Equal(table.shape, shapes["main"][table.name]) || int64(len(table.derived)) != table.size() {
			t.Fatalf("invalid derived table %s", table.name)
		}
	}
	position := tables[0].derived
	for i := range 256 {
		zero := binary.LittleEndian.Uint32(position[(4999*512+2*i)*4:])
		one := binary.LittleEndian.Uint32(position[(4999*512+2*i+1)*4:])
		if zero != 0 || one != math.Float32bits(1) {
			t.Fatal("incorrect center position")
		}
	}
	file := filepath.Join(t.TempDir(), "tables.safetensors")
	if err := writeOriginalChatterboxFile(t.Context(), file, tables); err != nil {
		t.Fatal(err)
	}
	reader, err := safetensors.OpenForExtraction(file)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	if len(reader.ListTensors()) != 4 {
		t.Fatal("incomplete written tables")
	}
}
