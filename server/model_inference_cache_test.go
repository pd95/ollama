package server

import (
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/types/model"
)

func TestInferenceModelCache(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_GO_TEMPLATE", "")

	_, completionDigest := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
	}, nil)
	writeTestModelManifest(t, "inference-cache", completionDigest, "{{ .Prompt }}")

	cache := newInferenceModelCache()
	loadCount := 0
	cache.loadModel = func(name string) (*Model, error) {
		loadCount++
		return GetModel(name)
	}

	first, err := cache.Get("inference-cache")
	if err != nil {
		t.Fatal(err)
	}
	if loadCount != 1 {
		t.Fatalf("load count = %d, want 1", loadCount)
	}
	if !first.capabilitiesCached {
		t.Fatal("capabilities were not cached")
	}
	if got := first.Capabilities(); !slices.Contains(got, model.CapabilityCompletion) {
		t.Fatalf("capabilities = %v, want completion", got)
	}

	// Returned models are request-local clones. Mutating one must not alter the
	// cached model or a later request.
	first.Config.Parser = "mutated"
	first.Config.ModelFamilies[0] = "mutated"
	first.capabilities[0] = model.CapabilityImage

	second, err := cache.Get("inference-cache")
	if err != nil {
		t.Fatal(err)
	}
	if loadCount != 1 {
		t.Fatalf("cache hit load count = %d, want 1", loadCount)
	}
	if second.Config.Parser == "mutated" || slices.Contains(second.Config.ModelFamilies, "mutated") {
		t.Fatalf("cached model was mutated: %#v", second.Config)
	}
	if got := second.Capabilities(); !slices.Contains(got, model.CapabilityCompletion) || slices.Contains(got, model.CapabilityImage) {
		t.Fatalf("cached capabilities = %v, want completion only", got)
	}

	// Recreating the manifest changes its digest and invalidates the entry.
	_, embeddingDigest := createBinFile(t, ggml.KV{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, nil)
	writeTestModelManifest(t, "inference-cache", embeddingDigest, "{{ .Prompt }}")

	third, err := cache.Get("inference-cache")
	if err != nil {
		t.Fatal(err)
	}
	if loadCount != 2 {
		t.Fatalf("invalidated load count = %d, want 2", loadCount)
	}
	if got := third.Capabilities(); !slices.Contains(got, model.CapabilityEmbedding) || slices.Contains(got, model.CapabilityCompletion) {
		t.Fatalf("refreshed capabilities = %v, want embedding only", got)
	}
}

func TestInferenceModelCacheConcurrentMiss(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_GO_TEMPLATE", "")

	_, digest := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
	}, nil)
	writeTestModelManifest(t, "inference-cache-concurrent", digest, "{{ .Prompt }}")

	cache := newInferenceModelCache()
	var loadCount atomic.Int32
	cache.loadModel = func(name string) (*Model, error) {
		loadCount.Add(1)
		time.Sleep(10 * time.Millisecond)
		return GetModel(name)
	}

	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := cache.Get("inference-cache-concurrent")
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := loadCount.Load(); got != 1 {
		t.Fatalf("load count = %d, want 1", got)
	}
}

func TestInferenceModelCacheGemma4VisionTensorCapabilities(t *testing.T) {
	setTestHome(t, t.TempDir())

	cfg := model.ConfigV2{
		ModelFormat:  "safetensors",
		Renderer:     gemma4RendererLarge,
		Capabilities: []string{"completion", "vision"},
	}
	createSafetensorsTestModel(t, "gemma4-cache", cfg, gemma4VisionManifestLayers(t))

	cache := newInferenceModelCache()
	loadCount := 0
	cache.loadModel = func(name string) (*Model, error) {
		loadCount++
		return GetModel(name)
	}

	first, err := cache.Get("gemma4-cache")
	if err != nil {
		t.Fatal(err)
	}
	if !first.capabilitiesCached || !slices.Contains(first.Capabilities(), model.CapabilityVision) {
		t.Fatalf("cold capabilities = %v, want cached vision", first.Capabilities())
	}
	if len(first.TensorLayerNames) == 0 {
		t.Fatal("cold model did not retain Gemma 4 tensor layer names")
	}
	first.TensorLayerNames[0] = "mutated"

	second, err := cache.Get("gemma4-cache")
	if err != nil {
		t.Fatal(err)
	}
	if loadCount != 1 {
		t.Fatalf("cache hit load count = %d, want 1", loadCount)
	}
	if !slices.Contains(second.Capabilities(), model.CapabilityVision) {
		t.Fatalf("cached capabilities = %v, want vision", second.Capabilities())
	}
	if slices.Contains(second.TensorLayerNames, "mutated") {
		t.Fatalf("cached tensor layer names were mutated: %v", second.TensorLayerNames)
	}

	createSafetensorsTestModel(t, "gemma4-cache", cfg, nil)
	third, err := cache.Get("gemma4-cache")
	if err != nil {
		t.Fatal(err)
	}
	if loadCount != 2 {
		t.Fatalf("invalidated load count = %d, want 2", loadCount)
	}
	if slices.Contains(third.Capabilities(), model.CapabilityVision) {
		t.Fatalf("refreshed capabilities = %v, did not expect vision", third.Capabilities())
	}
}
