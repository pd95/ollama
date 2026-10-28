package create

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/types/model"
)

type blobStoreFunc func(io.Reader, string, string) (LayerInfo, error)

func (f blobStoreFunc) WriteBlob(r io.Reader, mediaType, name string) (LayerInfo, error) {
	return f(r, mediaType, name)
}

func TestCreatePipeline(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"],"dtype":"bfloat16"}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{8, 8}, make([]byte, 8*8*2)),
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 8*2)),
	})

	store := newCaptureStore()
	var gotName string
	var gotConfig LayerInfo
	var gotModelConfig model.ConfigV2
	var gotLayers []LayerInfo
	var gotClass Classification
	writeManifest := func(_ context.Context, name string, info ManifestInfo) error {
		gotName, gotConfig, gotLayers = name, info.ConfigLayer, info.Layers
		gotModelConfig = info.ModelConfig
		gotClass = info.Class
		return nil
	}

	opts := testPipelineOptions()
	opts.Requires = "v0.20.0"
	if err := Create(context.Background(), "mymodel", dir, opts, store, writeManifest, func(string) {}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	if gotName != "mymodel" {
		t.Errorf("manifest name = %q, want mymodel", gotName)
	}
	if gotConfig.Name != "config.json" {
		t.Errorf("config layer = %q, want config.json", gotConfig.Name)
	}
	if gotModelConfig.ModelFormat != "safetensors" || !slices.Equal(gotModelConfig.Capabilities, []string{"completion"}) {
		t.Errorf("model config = %#v, want safetensors completion model", gotModelConfig)
	}
	if gotModelConfig.Requires != "0.20.0" {
		t.Errorf("requires = %q, want 0.20.0", gotModelConfig.Requires)
	}
	if gotModelConfig.FileType != "BF16" {
		t.Errorf("file type = %q, want BF16 from the stored weights", gotModelConfig.FileType)
	}
	if gotClass.Kind != SourceFloat || gotClass.Quantize != "" {
		t.Errorf("classification = {%s %q}, want {float %q}", gotClass.Kind, gotClass.Quantize, "")
	}
	if len(gotLayers) != 3 {
		t.Fatalf("manifest layers = %d, want 3 (2 tensors + config.json)", len(gotLayers))
	}
	for _, n := range []string{"model.embed_tokens.weight", "model.norm.weight", "config.json"} {
		if _, ok := store.blobs[n]; !ok {
			t.Errorf("missing written blob %q (have %v)", n, store.names())
		}
	}
}

func TestCreatePipelinePhononRequires(t *testing.T) {
	dir := t.TempDir()
	writeGradiumPhononSource(t, dir, []int32{1, 512, 2})
	// Remote creates send the generic safetensors minimum; it must not win.
	for requires, want := range map[string]string{"": phononMinOllamaVersion, "0.19.0": phononMinOllamaVersion, "0.41.0": "0.41.0"} {
		var got string
		opts := testPipelineOptions()
		opts.Requires = requires
		err := Create(context.Background(), "phonon", dir, opts, newCaptureStore(), func(_ context.Context, _ string, info ManifestInfo) error {
			got = info.ModelConfig.Requires
			return nil
		}, func(string) {})
		if err != nil || got != want {
			t.Errorf("requires %q: got %q, %v; want %q", requires, got, err, want)
		}
	}
}

func TestCreatePipelineRejectsUnsupportedArchitectureBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["UnsupportedForCausalLM"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
	})

	store := newCaptureStore()
	manifestCalled := false
	err := Create(context.Background(), "mymodel", dir, PipelineOptions{}, store, func(context.Context, string, ManifestInfo) error {
		manifestCalled = true
		return nil
	}, func(string) {})
	if !errors.Is(err, ErrUnsupportedMLXArchitecture) {
		t.Fatalf("Create() error = %v, want ErrUnsupportedMLXArchitecture", err)
	}
	if len(store.blobs) != 0 {
		t.Fatalf("Create() wrote blobs before validation: %v", store.names())
	}
	if manifestCalled {
		t.Fatal("Create() published a manifest before validation")
	}
}

func TestCreatePipelineRejectsInvalidRequiresBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3ForCausalLM"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
	})

	store := newCaptureStore()
	opts := testPipelineOptions()
	opts.Requires = "not-semver"
	err := Create(context.Background(), "mymodel", dir, opts, store, func(context.Context, string, ManifestInfo) error {
		t.Fatal("Create() published a manifest with invalid requires")
		return nil
	}, func(string) {})
	if !errors.Is(err, ErrInvalidRequires) {
		t.Fatalf("Create() error = %v, want ErrInvalidRequires", err)
	}
	if len(store.blobs) != 0 {
		t.Fatalf("Create() wrote blobs before requires validation: %v", store.names())
	}
}

func TestCreatePipelineIncludesDraftLayers(t *testing.T) {
	modelDir := t.TempDir()
	writeConfigJSON(t, modelDir, `{"architectures":["Qwen3ForCausalLM"]}`)
	createTestSafetensors(t, filepath.Join(modelDir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
	})
	draftDir := t.TempDir()
	writeConfigJSON(t, draftDir, `{"architectures":["DFlashDraftModel"]}`)
	createTestSafetensors(t, filepath.Join(draftDir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
	})
	tokenizers := map[string][]byte{
		modelDir: []byte(`{"model":{"type":"BPE","vocab":{"target":0}},"added_tokens":[{"id":1000,"content":"target-added"}]}`),
		draftDir: []byte(`{"model":{"type":"BPE","vocab":{"draft":0}},"added_tokens":[{"id":2000,"content":"draft-added"}]}`),
	}
	for dir, data := range tokenizers {
		if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}

	store := newCaptureStore()
	var layers []LayerInfo
	opts := testPipelineOptions()
	opts.DraftDir = draftDir
	err := Create(context.Background(), "mymodel", modelDir, opts, store, func(_ context.Context, _ string, info ManifestInfo) error {
		layers = info.Layers
		return nil
	}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"model.norm.weight", "draft.model.norm.weight", "config.json", "draft/config.json", "tokenizer.json", "draft/tokenizer.json"} {
		if !slices.ContainsFunc(layers, func(layer LayerInfo) bool { return layer.Name == name }) {
			t.Errorf("manifest layers are missing %q: %#v", name, layers)
		}
	}
	if !bytes.Equal(store.blobs["tokenizer.json"], tokenizers[modelDir]) || !bytes.Equal(store.blobs["draft/tokenizer.json"], tokenizers[draftDir]) {
		t.Fatal("combined import lost target or draft tokenizer bytes")
	}
}

func TestCreatePipelineRejectsUnsafeTargetOrDraftTokenizerBeforeWriting(t *testing.T) {
	for _, location := range []string{"target", "draft"} {
		for _, force := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/force=%t", location, force), func(t *testing.T) {
				if !force && runtime.GOOS != "darwin" {
					t.Skip("non-forced import requires native Darwin MLX validation")
				}
				modelDir, draftDir := t.TempDir(), t.TempDir()
				for dir, arch := range map[string]string{modelDir: "Qwen3ForCausalLM", draftDir: "DFlashDraftModel"} {
					writeConfigJSON(t, dir, fmt.Sprintf(`{"architectures":[%q]}`, arch))
					createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
						st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
					})
					if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"type":"BPE","vocab":{"base":0}},"added_tokens":[{"id":1000,"content":"added"}]}`), 0o600); err != nil {
						t.Fatal(err)
					}
				}
				invalidDir := modelDir
				if location == "draft" {
					invalidDir = draftDir
				}
				if err := os.WriteFile(filepath.Join(invalidDir, "tokenizer.json"), []byte(`{"model":{"type":"BPE","vocab":{"base":0}},"added_tokens":[{"id":0,"content":"collision"}]}`), 0o600); err != nil {
					t.Fatal(err)
				}
				opts := testPipelineOptions()
				opts.Validation.Force = force
				opts.DraftDir = draftDir
				store := newCaptureStore()
				manifestCalled := false
				err := Create(t.Context(), "unsafe-tokenizer", modelDir, opts, store, func(context.Context, string, ManifestInfo) error {
					manifestCalled = true
					return nil
				}, func(string) {})
				if !errors.Is(err, ErrInvalidTokenizer) {
					t.Fatalf("Create() error = %v, want invalid %s tokenizer", err, location)
				}
				if len(store.blobs) != 0 || manifestCalled {
					t.Fatalf("invalid %s tokenizer wrote blobs %v or manifest=%v", location, store.names(), manifestCalled)
				}
			})
		}
	}
}

func TestCreatePipelineReportsPrequantizedFileType(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"],"dtype":"bfloat16"}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("linear.weight", "U8", []int32{16, 8}, make([]byte, 16*8)),
		st.NewTensorDataFromBytes("linear.weight_scale", "F8_E4M3", []int32{16, 1}, make([]byte, 16)),
		st.NewTensorDataFromBytes("linear.weight_scale_2", "F32", []int32{}, f32le(1)),
	})

	store := newCaptureStore()
	var got Classification
	var gotModelConfig model.ConfigV2
	writeManifest := func(_ context.Context, _ string, info ManifestInfo) error {
		got = info.Class
		gotModelConfig = info.ModelConfig
		return nil
	}
	if err := Create(context.Background(), "mymodel", dir, testPipelineOptions(), store, writeManifest, func(string) {}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if got.Kind != SourcePrequantized || got.Quantize != "nvfp4" {
		t.Errorf("classification = {%s %q}, want {prequantized nvfp4}", got.Kind, got.Quantize)
	}
	if gotModelConfig.FileType != "" {
		t.Errorf("file type = %q, want empty before manifest writer selects NVFP4", gotModelConfig.FileType)
	}
}

func TestCreatePipelineRejectsMalformedMetadata(t *testing.T) {
	tests := []struct {
		name  string
		file  string
		value string
	}{
		{name: "config", file: "config.json", value: "{"},
		{name: "tokenizer config", file: "tokenizer_config.json", value: "{"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			writeConfigJSON(t, dir, `{"architectures":["Qwen3ForCausalLM"]}`)
			createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
				st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
			})
			if err := os.WriteFile(filepath.Join(dir, tt.file), []byte(tt.value), 0o600); err != nil {
				t.Fatal(err)
			}

			err := Create(context.Background(), "mymodel", dir, testPipelineOptions(), newCaptureStore(), func(context.Context, string, ManifestInfo) error {
				return nil
			}, func(string) {})
			if err == nil || !strings.Contains(err.Error(), "parse") {
				t.Fatalf("Create() error = %v, want parse error", err)
			}
		})
	}
}

func TestCreatePipelineReturnsCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := Create(ctx, "mymodel", t.TempDir(), testPipelineOptions(), newCaptureStore(), func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Create() error = %v, want context.Canceled", err)
	}
}

func TestCreatePipelineDoesNotPublishAfterCancellation(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{8, 8}, make([]byte, 8*8*2)),
	})
	if err := os.WriteFile(filepath.Join(dir, "z.json"), []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	store := newCaptureStore()
	cancelingStore := blobStoreFunc(func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		layer, err := store.WriteBlob(r, mediaType, name)
		if name == "z.json" {
			cancel()
		}
		return layer, err
	})
	manifestCalled := false
	err := Create(ctx, "mymodel", dir, testPipelineOptions(), cancelingStore, func(context.Context, string, ManifestInfo) error {
		manifestCalled = true
		return nil
	}, func(string) {})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Create() error = %v, want context.Canceled", err)
	}
	if manifestCalled {
		t.Fatal("Create() published a manifest after cancellation")
	}
}

func TestCreatePipelineStopsBlobReadAfterCancellation(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{8, 8}, make([]byte, 8*8*2)),
	})

	ctx, cancel := context.WithCancel(context.Background())
	var streamErr error
	store := blobStoreFunc(func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		buf := make([]byte, 1)
		if _, err := r.Read(buf); err != nil {
			return LayerInfo{}, err
		}
		cancel()
		_, streamErr = io.Copy(io.Discard, r)
		return LayerInfo{}, streamErr
	})

	err := Create(ctx, "mymodel", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Create() error = %v, want context.Canceled", err)
	}
	if !errors.Is(streamErr, context.Canceled) {
		t.Fatalf("blob stream error = %v, want context.Canceled", streamErr)
	}
}

func TestCreatePipelineRejectsNilContext(t *testing.T) {
	var ctx context.Context
	err := Create(ctx, "mymodel", t.TempDir(), testPipelineOptions(), newCaptureStore(), func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if err == nil || err.Error() != "nil context" {
		t.Fatalf("Create() error = %v, want nil context", err)
	}
}

func testPipelineOptions() PipelineOptions {
	return PipelineOptions{Validation: MLXValidationOptions{
		Force:   true,
		Warning: func(string) {},
	}}
}

func TestCreatePipelineRejectsUnsafeTokenizerBeforeWriting(t *testing.T) {
	for _, tt := range []struct {
		name string
		data string
	}{
		{"collision", `{"model":{"type":"BPE","vocab":{"base":0}},"added_tokens":[{"id":0,"content":"other"}]}`},
		{"negative", `{"model":{"type":"BPE","vocab":{"base":0}},"added_tokens":[{"id":-1,"content":"bad"}]}`},
		{"oversized", `{"model":{"type":"BPE","vocab":{"base":0}},"added_tokens":[{"id":1048576,"content":"bad"}]}`},
		{"malformed JSON", `{"model":`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			writeConfigJSON(t, dir, `{"architectures":["Qwen3ForCausalLM"]}`)
			createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
				st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
			})
			if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tt.data), 0o600); err != nil {
				t.Fatal(err)
			}
			store := newCaptureStore()
			manifestCalled := false
			err := Create(context.Background(), "mymodel", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error {
				manifestCalled = true
				return nil
			}, func(string) {})
			if !errors.Is(err, ErrInvalidTokenizer) || !strings.Contains(err.Error(), "tokenizer.json") {
				t.Fatalf("Create() error = %v, want tokenizer.json rejection", err)
			}
			if len(store.blobs) != 0 || manifestCalled {
				t.Fatalf("invalid tokenizer wrote blobs %v or manifest=%v", store.names(), manifestCalled)
			}
		})
	}
}

func TestCreatePipelinePreservesValidatedSparseTokenizerBytes(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3ForCausalLM"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
	})
	data := []byte("{\n  \"model\": {\"type\": \"BPE\", \"vocab\": {\"base\": 0}},\n  \"added_tokens\": [{\"id\": 1000, \"content\": \"added\"}]\n}\n")
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	store := newCaptureStore()
	if err := Create(context.Background(), "mymodel", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {}); err != nil {
		t.Fatal(err)
	}
	if got := store.blobs["tokenizer.json"]; !bytes.Equal(got, data) {
		t.Fatalf("imported tokenizer bytes differ: got %q, want %q", got, data)
	}
}

func TestCreatePipelineImportsPreflightTokenizerBytesIfSourceChanges(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3ForCausalLM"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
	})
	path := filepath.Join(dir, "tokenizer.json")
	validated := []byte(`{"model":{"type":"BPE","vocab":{"base":0}},"added_tokens":[{"id":1000,"content":"added"}]}`)
	if err := os.WriteFile(path, validated, 0o600); err != nil {
		t.Fatal(err)
	}
	capture := newCaptureStore()
	store := blobStoreFunc(func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		layer, err := capture.WriteBlob(r, mediaType, name)
		if err == nil && name == "model.norm.weight" {
			if writeErr := os.WriteFile(path, []byte(`{"model":{"type":"BPE","vocab":{"base":0}},"added_tokens":[{"id":0,"content":"bad"}]}`), 0o600); writeErr != nil {
				return LayerInfo{}, writeErr
			}
		}
		return layer, err
	})
	if err := Create(context.Background(), "mymodel", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {}); err != nil {
		t.Fatal(err)
	}
	if got := capture.blobs["tokenizer.json"]; !bytes.Equal(got, validated) {
		t.Fatalf("imported tokenizer bytes differ from preflight: got %q, want %q", got, validated)
	}
}

func TestImportConfigBlobsCombinesValidatedTokenizerAndConfigOverride(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	data := []byte("{\n\"model\":{\"type\":\"BPE\",\"vocab\":{\"base\":0}},\"added_tokens\":[{\"id\":1000,\"content\":\"added\"}]\n}\n")
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	snapshot, err := validateTokenizerSource(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"model":{"vocab":{"bad":-1}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	override := []byte(`{"architectures":["TestModel"],"normalized":true}`)
	store := newCaptureStore()
	if _, _, err := importConfigBlobs(context.Background(), dir, "", override, snapshot, store, func(string) {}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(store.blobs["config.json"], override) || !bytes.Equal(store.blobs["tokenizer.json"], data) {
		t.Fatal("config override or tokenizer snapshot lost")
	}
}

func TestTokenizerPreflightHandlesSourceRaces(t *testing.T) {
	for _, draft := range []bool{false, true} {
		for _, change := range []string{"replaced", "appeared", "disappeared"} {
			t.Run(fmt.Sprintf("draft=%v/%s", draft, change), func(t *testing.T) {
				dir := t.TempDir()
				arch := "Qwen3ForCausalLM"
				if draft {
					arch = "DFlashDraftModel"
				}
				writeConfigJSON(t, dir, fmt.Sprintf(`{"architectures":[%q]}`, arch))
				createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16))})
				path := filepath.Join(dir, "tokenizer.json")
				valid := []byte(`{"model":{"type":"BPE","vocab":{"base":0}},"added_tokens":[{"id":1000,"content":"added"}]}`)
				if change != "appeared" {
					if err := os.WriteFile(path, valid, 0o600); err != nil {
						t.Fatal(err)
					}
				}
				capture := newCaptureStore()
				mutated := false
				store := blobStoreFunc(func(r io.Reader, mediaType, name string) (LayerInfo, error) {
					layer, err := capture.WriteBlob(r, mediaType, name)
					if err == nil && strings.HasSuffix(name, "model.norm.weight") {
						mutated = true
						if change == "disappeared" {
							err = os.Remove(path)
						} else {
							err = os.WriteFile(path, []byte(`{"model":{"type":"BPE","vocab":{"bad":-1}}}`), 0o600)
						}
					}
					return layer, err
				})
				manifestCalled := false
				var err error
				if draft {
					_, err = CreateDraftLayers(context.Background(), dir, "draft.", "draft/", "", testPipelineOptions().Validation, store, func(string) {})
				} else {
					err = Create(context.Background(), "test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { manifestCalled = true; return nil }, func(string) {})
				}
				if !mutated {
					t.Fatal("race hook did not execute")
				}
				if change == "replaced" {
					if err != nil {
						t.Fatal(err)
					}
					name := "tokenizer.json"
					if draft {
						name = "draft/" + name
					}
					if !bytes.Equal(capture.blobs[name], valid) {
						t.Fatal("preflight tokenizer bytes changed")
					}
				} else {
					if !errors.Is(err, ErrInvalidTokenizer) {
						t.Fatalf("error = %v, want invalid tokenizer", err)
					}
					if manifestCalled {
						t.Fatal("published manifest after tokenizer presence changed")
					}
				}
			})
		}
	}
}
