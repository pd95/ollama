package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	fsgguf "github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestModelListCacheHydratesSummary(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-cache", map[string]any{
		"test.context_length":   uint32(4096),
		"test.embedding_length": uint32(384),
	}, "{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")

	cache := newModelListCache()
	if err := cache.hydrate(context.Background()); err != nil {
		t.Fatalf("hydrate failed: %v", err)
	}

	summary, ok := cache.Get(model.ParseName("list-cache"))
	if !ok {
		t.Fatal("list summary missing")
	}

	if summary.Model != "list-cache:latest" || summary.Name != "list-cache:latest" {
		t.Fatalf("summary model/name = %q/%q, want list-cache:latest", summary.Model, summary.Name)
	}
	if summary.Digest == "" {
		t.Fatal("summary digest is empty")
	}
	if summary.Size == 0 {
		t.Fatal("summary size is zero")
	}
	if summary.Details.Family != "test" || summary.Details.Format != "gguf" {
		t.Fatalf("summary details = %+v, want gguf/test", summary.Details)
	}
	if summary.Details.ContextLength != 4096 {
		t.Fatalf("context length = %d, want 4096", summary.Details.ContextLength)
	}
	if summary.Details.EmbeddingLength != 384 {
		t.Fatalf("embedding length = %d, want 384", summary.Details.EmbeddingLength)
	}

	for _, capability := range []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityInsert} {
		if !slices.Contains(summary.Capabilities, capability) {
			t.Fatalf("capabilities = %v, want %s", summary.Capabilities, capability)
		}
	}

	listModel := summary.ListModelResponse()
	if !slices.Contains(listModel.Capabilities, model.CapabilityTools) ||
		listModel.Details.ContextLength != 4096 ||
		listModel.Details.EmbeddingLength != 384 {
		t.Fatalf("list response = %+v, want capabilities/context/embedding", listModel)
	}
}

func TestModelListCacheSuppressesNemotronSafetensorsMedia(t *testing.T) {
	caps := []model.Capability{
		model.CapabilityCompletion,
		model.CapabilityTools,
		model.CapabilityThinking,
		model.CapabilityVision,
		model.CapabilityAudio,
	}
	got := filterUnsupportedModelListCapabilities(caps, model.ConfigV2{
		ModelFormat: "safetensors",
		Renderer:    "nemotron-3-nano",
		Parser:      "nemotron-3-nano",
	})

	for _, capability := range []model.Capability{
		model.CapabilityCompletion,
		model.CapabilityTools,
		model.CapabilityThinking,
	} {
		if !slices.Contains(got, capability) {
			t.Fatalf("capabilities = %v, want %s", got, capability)
		}
	}
	for _, capability := range []model.Capability{model.CapabilityVision, model.CapabilityAudio} {
		if slices.Contains(got, capability) {
			t.Fatalf("capabilities = %v, did not expect %s", got, capability)
		}
	}
}

func TestModelListCacheRefreshUpdatesEntry(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-refresh", map[string]any{"test.context_length": uint32(1024)}, "")

	cache := newModelListCache()
	if err := cache.hydrate(context.Background()); err != nil {
		t.Fatalf("hydrate failed: %v", err)
	}

	name := model.ParseName("list-refresh")
	first, ok := cache.Get(name)
	if !ok {
		t.Fatal("list summary missing")
	}

	changeShowCacheManifest(t, "list-refresh")
	if err := cache.RefreshModel(name); err != nil {
		t.Fatalf("refresh failed: %v", err)
	}

	refreshed, ok := cache.Get(name)
	if !ok {
		t.Fatal("refreshed list summary missing")
	}
	if refreshed.Digest == first.Digest {
		t.Fatalf("digest did not change after refresh: %s", refreshed.Digest)
	}
	if cache.Len() != 1 {
		t.Fatalf("cache entries = %d, want 1", cache.Len())
	}
}

func TestModelListSummaryGemma4SafetensorsVisionRequiresTensorLayers(t *testing.T) {
	setTestHome(t, t.TempDir())

	cfg := model.ConfigV2{
		ModelFormat:  "safetensors",
		Renderer:     gemma4RendererLarge,
		Capabilities: []string{"completion", "vision", "audio"},
	}

	createSafetensorsTestModel(t, "list-gemma4-text-only", cfg, nil)
	mf, err := manifest.ParseNamedManifest(model.ParseName("list-gemma4-text-only"))
	if err != nil {
		t.Fatal(err)
	}
	summary, err := buildModelListSummary(model.ParseName("list-gemma4-text-only"), mf)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(summary.Capabilities, model.CapabilityCompletion) {
		t.Fatalf("capabilities = %v, want completion", summary.Capabilities)
	}
	if slices.Contains(summary.Capabilities, model.CapabilityVision) {
		t.Fatalf("capabilities = %v, did not expect vision", summary.Capabilities)
	}
	if slices.Contains(summary.Capabilities, model.CapabilityAudio) {
		t.Fatalf("capabilities = %v, did not expect audio", summary.Capabilities)
	}

	createSafetensorsTestModel(t, "list-gemma4-vision", cfg, gemma4VisionManifestLayers(t))
	mf, err = manifest.ParseNamedManifest(model.ParseName("list-gemma4-vision"))
	if err != nil {
		t.Fatal(err)
	}
	summary, err = buildModelListSummary(model.ParseName("list-gemma4-vision"), mf)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(summary.Capabilities, model.CapabilityVision) {
		t.Fatalf("capabilities = %v, want vision", summary.Capabilities)
	}
	if slices.Contains(summary.Capabilities, model.CapabilityAudio) {
		t.Fatalf("capabilities = %v, did not expect audio", summary.Capabilities)
	}
}

func TestModelListSummaryGemma4UnifiedVisionRequiresTensorLayers(t *testing.T) {
	setTestHome(t, t.TempDir())
	cfg := model.ConfigV2{
		ModelFormat:  "safetensors",
		Renderer:     gemma4RendererLarge,
		Capabilities: []string{"completion", "vision", "audio", "tools", "thinking"},
	}
	for _, tt := range []struct {
		name       string
		layers     []manifest.Layer
		wantVision bool
	}{
		{"complete", gemma4UnifiedVisionManifestLayers(t, true), true},
		{"partial", gemma4UnifiedVisionManifestLayers(t, false), false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			modelName := "list-gemma4-unified-vision-" + tt.name
			createSafetensorsTestModel(t, modelName, cfg, tt.layers)
			mf, err := manifest.ParseNamedManifest(model.ParseName(modelName))
			if err != nil {
				t.Fatal(err)
			}
			summary, err := buildModelListSummary(model.ParseName(modelName), mf)
			if err != nil {
				t.Fatal(err)
			}
			if got := slices.Contains(summary.Capabilities, model.CapabilityVision); got != tt.wantVision {
				t.Fatalf("vision capability = %v, want %v (%v)", got, tt.wantVision, summary.Capabilities)
			}
			for _, capability := range []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityThinking} {
				if !slices.Contains(summary.Capabilities, capability) {
					t.Fatalf("capabilities = %v, want preserved %s", summary.Capabilities, capability)
				}
			}
		})
	}
}

func TestModelListSummaryGemma4AudioRequiresRuntimeMetadataAndTensors(t *testing.T) {
	setTestHome(t, t.TempDir())
	cfg := model.ConfigV2{ModelFormat: "safetensors", Renderer: gemma4RendererSmall, Capabilities: []string{"completion", "audio"}}
	for _, tt := range []struct {
		name      string
		layers    []manifest.Layer
		wantAudio bool
	}{
		{"complete", gemma4AudioManifestLayers(t), true},
		{"partial", slices.DeleteFunc(gemma4AudioManifestLayers(t), func(layer manifest.Layer) bool {
			return layer.Name == "model.audio_tower.layers.0.self_attn.q_proj.input_max"
		}), false},
		{"unified-complete", gemma4UnifiedAudioManifestLayers(t, true), true},
		{"unified-partial", gemma4UnifiedAudioManifestLayers(t, false), false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			modelName := "list-gemma4-audio-" + tt.name
			createSafetensorsTestModel(t, modelName, cfg, tt.layers)
			mf, err := manifest.ParseNamedManifest(model.ParseName(modelName))
			if err != nil {
				t.Fatal(err)
			}
			summary, err := buildModelListSummary(model.ParseName(modelName), mf)
			if err != nil {
				t.Fatal(err)
			}
			if got := slices.Contains(summary.Capabilities, model.CapabilityAudio); got != tt.wantAudio {
				t.Fatalf("audio capability = %v, want %v (%v)", got, tt.wantAudio, summary.Capabilities)
			}
		})
	}
}

func TestModelListCacheMutationHooks(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	cache := newModelListCache()
	s := Server{modelCaches: &modelCaches{modelList: cache}}

	_, digest := createBinFile(t, map[string]any{"test.context_length": uint32(2048)}, nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:  "list-hooks",
		Files:  map[string]string{"model.gguf": digest},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("create model status = %d, want 200: %s", w.Code, w.Body.String())
	}

	if _, ok := cache.Get(model.ParseName("list-hooks")); !ok {
		t.Fatal("create did not refresh model list cache")
	}

	w = createRequest(t, s.CopyHandler, api.CopyRequest{
		Source:      "list-hooks",
		Destination: "list-hooks-copy",
	})
	if w.Code != http.StatusOK {
		t.Fatalf("copy model status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if _, ok := cache.Get(model.ParseName("list-hooks-copy")); !ok {
		t.Fatal("copy did not refresh model list cache")
	}

	w = createRequest(t, s.DeleteHandler, api.DeleteRequest{Model: "list-hooks-copy"})
	if w.Code != http.StatusOK {
		t.Fatalf("delete model status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if _, ok := cache.Get(model.ParseName("list-hooks-copy")); ok {
		t.Fatal("delete did not remove model list cache entry")
	}
}

func TestModelListCacheSyncsManifestChanges(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-sync-a", map[string]any{"test.context_length": uint32(1024)}, "")

	cache := newModelListCache()
	cache.Start(context.Background())
	if err := cache.Wait(context.Background()); err != nil {
		t.Fatalf("wait failed: %v", err)
	}

	createListCacheModel(t, "list-sync-b", map[string]any{"test.context_length": uint32(2048)}, "")
	models, err := cache.List(context.Background())
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}

	names := make([]string, 0, len(models))
	for _, m := range models {
		names = append(names, m.Name)
	}
	for _, want := range []string{"list-sync-a:latest", "list-sync-b:latest"} {
		if !slices.Contains(names, want) {
			t.Fatalf("names = %v, want %s", names, want)
		}
	}

	var other Server
	w := createRequest(t, other.DeleteHandler, api.DeleteRequest{Model: "list-sync-a"})
	if w.Code != http.StatusOK {
		t.Fatalf("delete model status = %d, want 200: %s", w.Code, w.Body.String())
	}

	models, err = cache.List(context.Background())
	if err != nil {
		t.Fatalf("list after delete failed: %v", err)
	}
	names = names[:0]
	for _, m := range models {
		names = append(names, m.Name)
	}
	if slices.Contains(names, "list-sync-a:latest") || !slices.Contains(names, "list-sync-b:latest") {
		t.Fatalf("names after delete = %v, want only list-sync-b", names)
	}
}

func TestModelListCacheSyncDropsStaleEntryOnRefreshFailure(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-stale", map[string]any{"test.context_length": uint32(1024)}, "")

	cache := newModelListCache()
	cache.Start(context.Background())
	if err := cache.Wait(context.Background()); err != nil {
		t.Fatalf("wait failed: %v", err)
	}

	name := model.ParseName("list-stale")
	if _, ok := cache.Get(name); !ok {
		t.Fatal("list summary missing")
	}

	changeShowCacheManifest(t, "list-stale")
	cache.build = func(model.Name, *manifest.Manifest) (modelListSummary, error) {
		return modelListSummary{}, errors.New("refresh failed")
	}

	models, err := cache.List(context.Background())
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}
	if len(models) != 0 {
		t.Fatalf("models = %+v, want stale entry removed", models)
	}
	if _, ok := cache.Get(name); ok {
		t.Fatal("stale entry remained in cache after refresh failure")
	}
}

func TestReadModelListGGUFRejectsMalformedMetadata(t *testing.T) {
	cases := []struct {
		name string
		data []byte
		want string
	}{
		{
			name: "oversized key string",
			data: modelListGGUFTestFile(func(b *bytes.Buffer) {
				writeModelListGGUFHeader(t, b, 1)
				writeModelListGGUFUint64(t, b, fsgguf.MaxStringLength+1)
			}),
			want: "string",
		},
		{
			name: "oversized skipped string",
			data: modelListGGUFTestFile(func(b *bytes.Buffer) {
				writeModelListGGUFHeader(t, b, 1)
				writeModelListGGUFString(t, b, "unused")
				writeModelListGGUFUint32(t, b, modelListGGUFTypeString)
				writeModelListGGUFUint64(t, b, fsgguf.MaxStringLength+1)
			}),
			want: "string",
		},
		{
			name: "oversized skipped array",
			data: modelListGGUFTestFile(func(b *bytes.Buffer) {
				writeModelListGGUFHeader(t, b, 1)
				writeModelListGGUFString(t, b, "unused")
				writeModelListGGUFUint32(t, b, modelListGGUFTypeArray)
				writeModelListGGUFUint32(t, b, modelListGGUFTypeUint8)
				writeModelListGGUFUint64(t, b, fsgguf.MaxArraySize+1)
			}),
			want: "array size",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("readModelListGGUF panicked: %v", r)
				}
			}()

			path := t.TempDir() + "/model.gguf"
			if err := os.WriteFile(path, tt.data, 0o600); err != nil {
				t.Fatal(err)
			}

			_, err := readModelListGGUF(path)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func createListCacheModel(t *testing.T, name string, kv map[string]any, tmpl string) {
	t.Helper()
	_, digest := createBinFile(t, kv, nil)

	req := api.CreateRequest{
		Model:  name,
		Files:  map[string]string{"model.gguf": digest},
		Stream: &stream,
	}
	if tmpl != "" {
		req.Template = tmpl
	}

	var s Server
	w := createRequest(t, s.CreateHandler, req)
	if w.Code != http.StatusOK {
		t.Fatalf("create model status = %d, want 200: %s", w.Code, w.Body.String())
	}
}

func modelListGGUFTestFile(fn func(*bytes.Buffer)) []byte {
	var b bytes.Buffer
	fn(&b)
	return b.Bytes()
}

func writeModelListGGUFHeader(t *testing.T, b *bytes.Buffer, numKV uint64) {
	t.Helper()
	writeModelListGGUFUint32(t, b, modelListGGUFMagicLE)
	writeModelListGGUFUint32(t, b, 3)
	writeModelListGGUFUint64(t, b, 0)
	writeModelListGGUFUint64(t, b, numKV)
}

func writeModelListGGUFString(t *testing.T, b *bytes.Buffer, s string) {
	t.Helper()
	writeModelListGGUFUint64(t, b, uint64(len(s)))
	if _, err := b.WriteString(s); err != nil {
		t.Fatal(err)
	}
}

func writeModelListGGUFUint32(t *testing.T, b *bytes.Buffer, v uint32) {
	t.Helper()
	if err := binary.Write(b, binary.LittleEndian, v); err != nil {
		t.Fatal(err)
	}
}

func writeModelListGGUFUint64(t *testing.T, b *bytes.Buffer, v uint64) {
	t.Helper()
	if err := binary.Write(b, binary.LittleEndian, v); err != nil {
		t.Fatal(err)
	}
}
