package create

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
	schema "github.com/ollama/ollama/mlxrunner/model/chatterbox/config"
)

type chatterboxHeader struct {
	Dtype string  `json:"dtype"`
	Shape []int32 `json:"shape"`
}

// The complete pinned original headers provide realistic metadata without
// trained values. Any attempt to read weights in the planner fails the test.
type chatterboxHeaderSource struct {
	headers map[string]chatterboxHeader
	reads   *int
}

func (s chatterboxHeaderSource) Close() error { return nil }

func (s chatterboxHeaderSource) ListTensors() []string { return slices.Sorted(maps.Keys(s.headers)) }

func (s chatterboxHeaderSource) ReadAt([]byte, int64) (int, error) {
	*s.reads++
	return 0, fmt.Errorf("metadata planner read a tensor")
}

func (s chatterboxHeaderSource) GetTensor(name string) (*safetensors.TensorData, error) {
	h, ok := s.headers[name]
	if !ok {
		return nil, fmt.Errorf("tensor not found: %s", name)
	}
	size := int64(4)
	if h.Dtype == "I64" {
		size = 8
	}
	for _, dim := range h.Shape {
		size *= int64(dim)
	}
	return safetensors.NewTensorDataFromReaderAt(name, h.Dtype, h.Shape, s, size), nil
}

func originalV3HeaderSources(t *testing.T) (*sourceFiles, *int) {
	t.Helper()
	f, err := os.Open("testdata/chatterbox-v3-original-headers.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	var fixture struct {
		ModelID string                                 `json:"model_id"`
		Rev     string                                 `json:"revision"`
		Headers map[string]map[string]chatterboxHeader `json:"tensor_headers"`
	}
	if err := json.NewDecoder(io.LimitReader(r, 1<<20)).Decode(&fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.ModelID != "ResembleAI/chatterbox" || fixture.Rev != "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18" || len(fixture.Headers) != 3 {
		t.Fatal("unexpected original v3 fixture identity")
	}
	reads := new(int)
	sources := &sourceFiles{cache: map[string]tensorSource{}}
	for file, headers := range fixture.Headers {
		sources.cache[file] = chatterboxHeaderSource{headers: headers, reads: reads}
	}
	return sources, reads
}

func TestOriginalV3CompleteMetadataPlan(t *testing.T) {
	sources, reads := originalV3HeaderSources(t)
	v := originalChatterboxVariant{weights: originalV3WeightFiles, shapes: originalV3Shapes}
	groups, err := planOriginalChatterbox(t.Context(), sources, v)
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 || len(groups["model.safetensors"]) != 2487 || len(groups["s3_tokenizer/model.safetensors"]) != 102 || *reads != 0 {
		t.Fatalf("incomplete or non-metadata plan: %d/%d reads %d", len(groups["model.safetensors"]), len(groups["s3_tokenizer/model.safetensors"]), *reads)
	}
	normalized, summed := 0, 0
	for _, item := range groups["model.safetensors"] {
		if item.operation == "weight_norm" {
			normalized++
		}
		if item.operation == "add" {
			summed++
		}
	}
	if normalized != 82 || summed != 3 {
		t.Fatalf("parameter pairs: %d norms and %d bias sums", normalized, summed)
	}
}

func TestOriginalV3RejectsIncompleteOrUnknownInventory(t *testing.T) {
	for _, mode := range []string{"missing pair", "missing learned tensor", "unknown learned tensor"} {
		t.Run(mode, func(t *testing.T) {
			sources, reads := originalV3HeaderSources(t)
			file := "t3_mtl23ls_v3.safetensors"
			if mode == "missing pair" {
				file = "s3gen_v3.safetensors"
			}
			headers := sources.cache[file].(chatterboxHeaderSource).headers
			switch mode {
			case "missing pair":
				for name := range headers {
					if strings.HasSuffix(name, ".parametrizations.weight.original0") {
						delete(headers, name)
						break
					}
				}
			case "missing learned tensor":
				delete(headers, "speech_head.weight")
			case "unknown learned tensor":
				headers["unknown.weight"] = chatterboxHeader{Dtype: "F32", Shape: []int32{1}}
			}
			v := originalChatterboxVariant{weights: originalV3WeightFiles, shapes: originalV3Shapes}
			if _, err := planOriginalChatterbox(t.Context(), sources, v); err == nil {
				t.Fatal("accepted unsupported inventory")
			}
			if *reads != 0 {
				t.Fatal("rejected inventory read trained data")
			}
		})
	}
}

func TestOriginalV3SourceProfileAndTransfer(t *testing.T) {
	dir := t.TempDir()
	for _, name := range append(slices.Clone(originalV3WeightFiles), originalV3ConfigFiles...) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("fixture"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if !IsSafetensorsModelDir(dir) {
		t.Fatal("original v3 layout not recognized")
	}
	cfg, raw, err := readSourceModelConfig(dir)
	if err != nil || cfg.Architecture() != chatterboxArchitecture {
		t.Fatalf("original v3 config: %+v, %v", cfg, err)
	}
	slope, err := schema.VocoderFinalSlope(raw)
	if err != nil || slope != .01 {
		t.Fatalf("original provider activation was not selected automatically: %v, %v", slope, err)
	}
	weights, err := SafetensorsWeightFiles(dir)
	if err != nil || !slices.Equal(weights, originalV3WeightFiles) {
		t.Fatalf("source transfer: %v, %v", weights, err)
	}
	assets, err := SafetensorsConfigFiles(dir)
	if err != nil || !slices.Equal(assets, originalV3ConfigFiles) {
		t.Fatalf("asset transfer: %v, %v", assets, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "t3_turbo_v1.safetensors"), []byte("fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	if IsSafetensorsModelDir(dir) {
		t.Fatal("accepted ambiguous original v3/Turbo source")
	}
}
