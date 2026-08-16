package server

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
	gemma4metadata "github.com/ollama/ollama/x/models/gemma4/metadata"
	"github.com/ollama/ollama/x/safetensors"
)

func TestPruneLayersSkipsRecentOrphans(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	recentDigest := "sha256:0000000000000000000000000000000000000000000000000000000000000001"
	oldDigest := "sha256:0000000000000000000000000000000000000000000000000000000000000002"

	for _, digest := range []string{recentDigest, oldDigest} {
		p, err := manifest.BlobsPath(digest)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	oldPath, err := manifest.BlobsPath(oldDigest)
	if err != nil {
		t.Fatal(err)
	}
	oldTime := time.Now().Add(-layerPruneGracePeriod - time.Hour)
	if err := os.Chtimes(oldPath, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}

	if err := PruneLayers(); err != nil {
		t.Fatal(err)
	}

	recentPath, err := manifest.BlobsPath(recentDigest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(recentPath); err != nil {
		t.Fatalf("recent orphan was pruned: %v", err)
	}
	if _, err := os.Stat(oldPath); !os.IsNotExist(err) {
		t.Fatalf("old orphan still exists: %v", err)
	}
}

func TestGetModelTemplateMetadata(t *testing.T) {
	customTemplate := "CUSTOM {{ .Prompt }}"

	t.Run("records chat template and Go TEMPLATE layer", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture":    "llama",
			"tokenizer.chat_template": "{{ bos_token }}{{ messages[0]['content'] }}",
		}, nil)
		writeTestModelManifest(t, "template-disabled", digest, customTemplate)

		m, err := GetModel("template-disabled")
		if err != nil {
			t.Fatal(err)
		}
		if !m.HasChatTemplate {
			t.Fatal("expected GGUF chat template to be detected")
		}
		if !m.HasGoTemplate {
			t.Fatal("expected Go TEMPLATE layer to be detected")
		}
		if got := m.Template.String(); got != customTemplate {
			t.Fatalf("template = %q, want %q", got, customTemplate)
		}
	})

	t.Run("prefers chat template when Go TEMPLATE has fewer capabilities", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture":    "llama",
			"tokenizer.chat_template": "{% if tools %}{{ tools }}{% endif %}{{ messages[0]['content'] }}",
		}, nil)
		writeTestModelManifest(t, "chat-template-tools", digest, customTemplate)

		m, err := GetModel("chat-template-tools")
		if err != nil {
			t.Fatal(err)
		}
		if !m.PreferChatTemplate {
			t.Fatal("expected chat template to be preferred")
		}
		if got := m.CheckCapabilities(model.CapabilityTools); got != nil {
			t.Fatalf("expected tools capability, got %v", got)
		}
	})

	t.Run("prefers Qwen chat template with tools and inferred thinking", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture":    "llama",
			"tokenizer.chat_template": "{% if tools %}{{ tools }}{% endif %}{% set content = (content.split('</think>')|last) %}",
		}, nil)
		writeTestModelManifest(t, "chat-template-tools-thinking", digest, "{{ range .Messages }}{{ if .Thinking }}<think>{{ .Thinking }}</think>{{ end }}{{ .Content }}{{ end }}")

		m, err := GetModel("chat-template-tools-thinking")
		if err != nil {
			t.Fatal(err)
		}
		if !m.PreferChatTemplate {
			t.Fatal("expected chat template to be preferred")
		}
		if got := m.CheckCapabilities(model.CapabilityTools); got != nil {
			t.Fatalf("expected tools capability, got %v", got)
		}
		if got := m.CheckCapabilities(model.CapabilityThinking); got != nil {
			t.Fatalf("expected thinking capability, got %v", got)
		}
	})

	t.Run("prefers chat template with stronger tool round trip", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture": "llama",
			"tokenizer.chat_template": `{% if tools %}{{ tools }}{% endif %}
{% for message in messages %}
{% if message.tool_calls %}
{% for tool_call in message.tool_calls %}{{ tool_call.function.name }}{% endfor %}
{% endif %}
{% if message.role == 'tool' %}tool_response {{ message.content }}{% endif %}
{% endfor %}`,
		}, nil)
		writeTestModelManifest(t, "chat-template-tool-round-trip", digest, `{{ if .Tools }}tools{{ end }}
{{ range .Messages }}
{{ range .ToolCalls }}{{ .Function.Name }}{{ end }}
{{ end }}`)

		m, err := GetModel("chat-template-tool-round-trip")
		if err != nil {
			t.Fatal(err)
		}
		if !m.PreferChatTemplate {
			t.Fatal("expected chat template to be preferred")
		}
		if got := m.CheckCapabilities(model.CapabilityTools); got != nil {
			t.Fatalf("expected tools capability, got %v", got)
		}
	})

	t.Run("keeps Go TEMPLATE when chat template has weaker tool support", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture": "llama",
			"tokenizer.chat_template": `{%- if tools and not available_tools -%}
{{- set available_tools = tools -}}
{%- endif -%}
{%- if available_tools -%}
{{ '<|start_of_role|>available_tools<|end_of_role|>' }}{{ available_tools | tojson }}{{ '<|end_of_text|>' }}
{%- endif -%}
{%- if thinking -%}<think></think><response></response>{%- endif -%}
{%- for message in messages -%}
{{ '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>' }}
{%- endfor -%}`,
		}, nil)
		writeTestModelManifest(t, "chat-template-weaker-tools", digest, `{{ if .Tools }}tools{{ end }}
{{ range .Messages }}
{{ if eq .Role "tool" }}tool_response{{ else }}{{ .Role }}{{ end }}
{{ if .ToolCalls }}<|tool_call|>{{ range .ToolCalls }}{{ .Function.Name }}{{ end }}{{ else }}{{ .Content }}{{ end }}
{{ end }}`)

		m, err := GetModel("chat-template-weaker-tools")
		if err != nil {
			t.Fatal(err)
		}
		if m.PreferChatTemplate {
			t.Fatal("expected Go TEMPLATE to be preferred")
		}
		if got := m.CheckCapabilities(model.CapabilityTools); got != nil {
			t.Fatalf("expected tools capability, got %v", got)
		}
		if got := m.CheckCapabilities(model.CapabilityThinking); got == nil {
			t.Fatal("expected thinking capability to remain unavailable on Go TEMPLATE path")
		}
	})

	t.Run("respects explicit Go TEMPLATE enablement", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "1")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture":    "llama",
			"tokenizer.chat_template": "{% if tools %}{{ tools }}{% endif %}{{ messages[0]['content'] }}",
		}, nil)
		writeTestModelManifest(t, "go-template-forced", digest, customTemplate)

		m, err := GetModel("go-template-forced")
		if err != nil {
			t.Fatal(err)
		}
		if m.PreferChatTemplate {
			t.Fatal("expected explicit Go TEMPLATE setting to suppress chat_template preference")
		}
		if got := m.CheckCapabilities(model.CapabilityTools); got == nil {
			t.Fatal("expected tools capability to be unavailable when Go TEMPLATE is explicitly enabled")
		}
	})

	t.Run("respects explicit Go TEMPLATE disablement", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "0")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture":    "llama",
			"tokenizer.chat_template": "{% if tools %}{{ tools }}{% endif %}{{ messages[0]['content'] }}",
		}, nil)
		writeTestModelManifest(t, "go-template-disabled", digest, customTemplate)

		m, err := GetModel("go-template-disabled")
		if err != nil {
			t.Fatal(err)
		}
		if m.PreferChatTemplate {
			t.Fatal("expected explicit Go TEMPLATE setting to suppress chat_template preference")
		}
		if got := m.CheckCapabilities(model.CapabilityTools); got != nil {
			t.Fatalf("expected tools capability from GGUF chat_template, got %v", got)
		}
	})

	t.Run("records missing chat template", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture": "llama",
		}, nil)
		writeTestModelManifest(t, "missing-chat-template", digest, customTemplate)

		m, err := GetModel("missing-chat-template")
		if err != nil {
			t.Fatal(err)
		}
		if m.HasChatTemplate {
			t.Fatal("expected missing GGUF chat template")
		}
		if !m.HasGoTemplate {
			t.Fatal("expected Go TEMPLATE layer to be detected")
		}
	})
}

func writeTestModelManifest(t *testing.T, name, digest, tmpl string) {
	t.Helper()

	modelLayer, err := manifest.NewLayerFromLayer(digest, "application/vnd.ollama.image.model", "")
	if err != nil {
		t.Fatal(err)
	}
	templateLayer, err := manifest.NewLayer(strings.NewReader(tmpl), "application/vnd.ollama.image.template")
	if err != nil {
		t.Fatal(err)
	}

	layers := []manifest.Layer{modelLayer, templateLayer}
	configLayer, err := createConfigLayer(model.ConfigV2{
		ModelFormat:   "gguf",
		ModelFamily:   "llama",
		ModelFamilies: []string{"llama"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifest(model.ParseName(name), *configLayer, layers); err != nil {
		t.Fatal(err)
	}
}

func TestModelCapabilities(t *testing.T) {
	// Create completion model (llama architecture without vision)
	completionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
	}, []*ggml.Tensor{})

	ggufToolTemplateModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":    "llama",
		"tokenizer.chat_template": `{% if tools %}<tool_call>{{ tools }}</tool_call>{% endif %}<think>{{ messages[0]['content'] }}</think>`,
	}, []*ggml.Tensor{})

	// Create vision model (llama architecture with vision block count)
	visionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	// Create embedding model (bert architecture with pooling type)
	embeddingModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, []*ggml.Tensor{})

	audioProjectorPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":    "clip",
		"clip.has_audio_encoder":  true,
		"vision.projector_type":   "pixtral",
		"clip.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	nemotronOmniModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":                 "nemotron_h_omni",
		"nemotron_h_omni.vision.block_count":   uint32(1),
		"nemotron_h_omni.audio.block_count":    uint32(1),
		"nemotron_h_omni.embedding_length":     uint32(1),
		"nemotron_h_omni.attention.head_count": uint32(1),
	}, []*ggml.Tensor{})

	suppressedAudioProjectorPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":    "clip",
		"clip.has_audio_encoder":  true,
		"vision.projector_type":   "gemma4v",
		"clip.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	toolsInsertTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	chatTemplate, err := template.Parse("{{ .prompt }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	toolsTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	testModels := []struct {
		name         string
		model        Model
		expectedCaps []model.Capability
	}{
		{
			name: "model with image generation capability via config",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image"},
				},
			},
			expectedCaps: []model.Capability{model.CapabilityImage},
		},
		{
			name: "model with image and vision capability (image editing)",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image", "vision"},
				},
			},
			expectedCaps: []model.Capability{model.CapabilityImage, model.CapabilityVision},
		},
		{
			name: "model with completion capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion},
		},

		{
			name: "model with completion, tools, and insert capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsInsertTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model with tools capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools},
		},
		{
			name: "model with GGUF chat_template tools and thinking",
			model: Model{
				ModelPath: ggufToolTemplateModelPath,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityThinking},
		},
		{
			name: "model with Go TEMPLATE ignores GGUF chat_template capabilities",
			model: Model{
				ModelPath:       ggufToolTemplateModelPath,
				Template:        chatTemplate,
				HasGoTemplate:   true,
				HasChatTemplate: true,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion},
		},
		{
			name: "model with tools capability from config and parser",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"completion", "tools"},
					Parser:       "qwen3-coder",
				},
				Template: chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools},
		},
		{
			name: "model with vision capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision},
		},
		{
			name: "model with vision, tools, and insert capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  toolsInsertTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model with embedding capability",
			model: Model{
				ModelPath: embeddingModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityEmbedding},
		},
		{
			name: "model with audio projector capability",
			model: Model{
				ModelPath:      completionModelPath,
				ProjectorPaths: []string{audioProjectorPath},
				Template:       chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityAudio},
		},
		{
			name: "model with parser and projector capabilities without template",
			model: Model{
				ModelPath:      completionModelPath,
				ProjectorPaths: []string{audioProjectorPath},
				Config: model.ConfigV2{
					Parser: "functiongemma",
				},
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityAudio, model.CapabilityTools},
		},
		{
			name: "gemma4 projector exposes audio capability",
			model: Model{
				ModelPath:      completionModelPath,
				ProjectorPaths: []string{suppressedAudioProjectorPath},
				Template:       chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityAudio},
		},
		{
			name: "gemma4 gguf exposes audio capability",
			model: Model{
				ModelPath:      completionModelPath,
				ProjectorPaths: []string{audioProjectorPath},
				Config: model.ConfigV2{
					Renderer:     gemma4RendererSmall,
					Capabilities: []string{"audio"},
				},
				Template: chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityAudio, model.CapabilityCompletion, model.CapabilityVision},
		},
		{
			name: "nemotron3 gguf suppresses audio capability",
			model: Model{
				ModelPath: nemotronOmniModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision},
		},
		{
			name: "nemotron3 projector suppresses audio capability",
			model: Model{
				ModelPath:      completionModelPath,
				ProjectorPaths: []string{audioProjectorPath},
				Config: model.ConfigV2{
					ModelFamily: "nemotron_h_omni",
				},
				Template: chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision},
		},
		{
			name: "nemotron3 safetensors suppresses vision and audio but keeps thinking",
			model: Model{
				Config: model.ConfigV2{
					ModelFormat:  "safetensors",
					Parser:       "nemotron-3-nano",
					Renderer:     "nemotron-3-nano",
					Capabilities: []string{"completion", "vision", "audio"},
				},
				Template: chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityThinking},
		},
		{
			name: "gemma4 small safetensors exposes vision and suppresses audio",
			model: Model{
				Config: model.ConfigV2{
					ModelFormat:  "safetensors",
					Renderer:     gemma4RendererSmall,
					Capabilities: []string{"vision", "audio"},
				},
				TensorLayerNames:    gemma4VisionTensorNames(2),
				Gemma4VisionConfig:  gemma4VisionConfig(2),
				Gemma4VisionTensors: testGemma4VisionTensorDescriptors(2),
				Template:            chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityVision},
		},
		{
			name: "gemma4 large safetensors exposes vision and suppresses audio",
			model: Model{
				Config: model.ConfigV2{
					ModelFormat:  "safetensors",
					Renderer:     gemma4RendererLarge,
					Capabilities: []string{"vision", "audio"},
				},
				TensorLayerNames:    gemma4VisionTensorNames(2),
				Gemma4VisionConfig:  gemma4VisionConfig(2),
				Gemma4VisionTensors: testGemma4VisionTensorDescriptors(2),
				Template:            chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityVision},
		},
		{
			name: "default gemma4 safetensors exposes vision and suppresses audio",
			model: Model{
				Config: model.ConfigV2{
					ModelFormat:  "safetensors",
					Renderer:     gemma4RendererLegacy,
					Capabilities: []string{"vision", "audio"},
				},
				TensorLayerNames:    gemma4VisionTensorNames(2),
				Gemma4VisionConfig:  gemma4VisionConfig(2),
				Gemma4VisionTensors: testGemma4VisionTensorDescriptors(2),
				Template:            chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityVision},
		},
		{
			name: "gemma4 safetensors exposes complete audio",
			model: Model{
				Config: model.ConfigV2{
					ModelFormat:  "safetensors",
					Renderer:     gemma4RendererSmall,
					Capabilities: []string{"audio"},
				},
				TensorLayerNames:   gemma4AudioTensorNames(1),
				Gemma4AudioConfig:  gemma4AudioConfig(1),
				Gemma4AudioTensors: testGemma4AudioTensorDescriptors(1),
				Gemma4AudioReady:   true,
				Template:           chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityAudio},
		},
	}

	// compare two slices of model.Capability regardless of order
	compareCapabilities := func(a, b []model.Capability) bool {
		if len(a) != len(b) {
			return false
		}

		aCount := make(map[model.Capability]int)
		for _, cap := range a {
			aCount[cap]++
		}

		bCount := make(map[model.Capability]int)
		for _, cap := range b {
			bCount[cap]++
		}

		for cap, count := range aCount {
			if bCount[cap] != count {
				return false
			}
		}

		return true
	}

	for _, tt := range testModels {
		t.Run(tt.name, func(t *testing.T) {
			// Test Capabilities method
			caps := tt.model.Capabilities()
			if !compareCapabilities(caps, tt.expectedCaps) {
				t.Errorf("Expected capabilities %v, got %v", tt.expectedCaps, caps)
			}
		})
	}
}

func TestGemma4SafetensorsVisionCapabilityRequiresTensorLayers(t *testing.T) {
	setTestHome(t, t.TempDir())

	cfg := model.ConfigV2{
		ModelFormat:  "safetensors",
		Renderer:     gemma4RendererLarge,
		Capabilities: []string{"completion", "vision", "audio"},
	}

	createSafetensorsTestModel(t, "gemma4-text-only", cfg, nil)
	m, err := GetModel("gemma4-text-only")
	if err != nil {
		t.Fatal(err)
	}
	caps := m.Capabilities()
	if !slices.Contains(caps, model.CapabilityCompletion) {
		t.Fatalf("capabilities = %v, want completion", caps)
	}
	if slices.Contains(caps, model.CapabilityVision) {
		t.Fatalf("capabilities = %v, did not expect vision", caps)
	}
	if slices.Contains(caps, model.CapabilityAudio) {
		t.Fatalf("capabilities = %v, did not expect audio", caps)
	}

	err = m.CheckCapabilities(model.CapabilityVision)
	if err == nil || !strings.Contains(err.Error(), "includes Gemma 4 vision tensor layers") {
		t.Fatalf("CheckCapabilities(vision) error = %v, want Gemma 4 vision tensor hint", err)
	}

	createSafetensorsTestModel(t, "gemma4-vision", cfg, gemma4VisionManifestLayers(t))
	m, err = GetModel("gemma4-vision")
	if err != nil {
		t.Fatal(err)
	}
	caps = m.Capabilities()
	if !slices.Contains(caps, model.CapabilityVision) {
		t.Fatalf("capabilities = %v, want vision", caps)
	}
	if slices.Contains(caps, model.CapabilityAudio) {
		t.Fatalf("capabilities = %v, did not expect audio", caps)
	}

	partialLayers := slices.DeleteFunc(gemma4VisionManifestLayers(t), func(layer manifest.Layer) bool {
		return layer.MediaType == manifest.MediaTypeImageTensor &&
			layer.Name != "model.vision_tower.patch_embedder.input_proj.weight" &&
			layer.Name != "model.embed_vision.embedding_projection.weight"
	})
	createSafetensorsTestModel(t, "gemma4-partial-vision", cfg, partialLayers)
	m, err = GetModel("gemma4-partial-vision")
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(m.Capabilities(), model.CapabilityVision) {
		t.Fatalf("partial vision capabilities = %v, did not expect vision", m.Capabilities())
	}
}

func TestGemma4SafetensorsAudioCapabilityRequiresCompleteInventory(t *testing.T) {
	complete := Model{
		Config: model.ConfigV2{
			ModelFormat: "safetensors", Renderer: gemma4RendererLarge,
			Capabilities: []string{"completion", "audio"},
		},
		TensorLayerNames:   gemma4AudioTensorNames(1),
		Gemma4AudioConfig:  gemma4AudioConfig(1),
		Gemma4AudioTensors: testGemma4AudioTensorDescriptors(1),
		Gemma4AudioReady:   true,
	}
	if !slices.Contains(complete.Capabilities(), model.CapabilityAudio) {
		t.Fatal("complete audio inventory did not expose audio")
	}

	partial := complete
	partial.TensorLayerNames = slices.Clone(complete.TensorLayerNames)
	partial.Gemma4AudioTensors = maps.Clone(complete.Gemma4AudioTensors)
	missing := "model.audio_tower.layers.0.self_attn.q_proj.input_max"
	partial.TensorLayerNames = slices.DeleteFunc(partial.TensorLayerNames, func(name string) bool { return name == missing })
	delete(partial.Gemma4AudioTensors, missing)
	if slices.Contains(partial.Capabilities(), model.CapabilityAudio) {
		t.Fatal("partial audio inventory exposed audio")
	}
	if err := partial.CheckCapabilities(model.CapabilityAudio); err == nil || !strings.Contains(err.Error(), "includes Gemma 4 audio tensor layers") {
		t.Fatalf("CheckCapabilities(audio) error = %v, want Gemma 4 audio hint", err)
	}

	nearMatch := partial
	nearMatch.TensorLayerNames = append(slices.Clone(partial.TensorLayerNames), missing+".extra")
	nearMatch.Gemma4AudioTensors = maps.Clone(partial.Gemma4AudioTensors)
	nearMatch.Gemma4AudioTensors[missing+".extra"] = gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: []int32{}}
	if slices.Contains(nearMatch.Capabilities(), model.CapabilityAudio) {
		t.Fatal("near-match audio tensor exposed audio")
	}

	malformed := complete
	malformed.Gemma4AudioConfig = gemma4AudioConfig(1)
	malformed.Gemma4AudioConfig.AudioConfig.NumAttentionHeads = 3
	if slices.Contains(malformed.Capabilities(), model.CapabilityAudio) {
		t.Fatal("malformed audio config exposed audio")
	}

	remoteCfg := complete.Config
	remoteCfg.RemoteHost = "https://example.invalid"
	remoteCaps := filterUnsupportedModelListCapabilities([]model.Capability{model.CapabilityCompletion, model.CapabilityAudio}, remoteCfg)
	if slices.Contains(remoteCaps, model.CapabilityAudio) {
		t.Fatal("remote Gemma 4 list capability exposed unsupported local audio")
	}
	otherCfg := remoteCfg
	otherCfg.Renderer = "other"
	otherCaps := filterUnsupportedModelListCapabilities([]model.Capability{model.CapabilityCompletion, model.CapabilityAudio}, otherCfg)
	if !slices.Contains(otherCaps, model.CapabilityAudio) {
		t.Fatal("remote non-Gemma audio capability was suppressed")
	}
}

func TestGemma4InstalledAudioCapabilityDescriptorAndPayloadMatrix(t *testing.T) {
	const target = "model.audio_tower.output_proj.weight"
	tests := []struct {
		name string
		edit func(*testing.T, []manifest.Layer) []manifest.Layer
		want bool
	}{
		{name: "complete", want: true},
		{name: "partial", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			return slices.DeleteFunc(layers, func(layer manifest.Layer) bool { return layer.Name == target })
		}},
		{name: "missing processor metadata", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			return slices.DeleteFunc(layers, func(layer manifest.Layer) bool { return layer.Name == "processor_config.json" })
		}},
		{name: "missing tokenizer metadata", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			return slices.DeleteFunc(layers, func(layer manifest.Layer) bool { return layer.Name == "tokenizer_config.json" })
		}},
		{name: "missing tokenizer vocabulary", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			return slices.DeleteFunc(layers, func(layer manifest.Layer) bool { return layer.Name == "tokenizer.json" })
		}},
		{name: "invalid runtime scalar", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			cfg := gemma4AudioConfig(1)
			cfg.AudioConfig.ResidualWeight = 0
			return replaceGemma4AudioConfigLayer(t, layers, cfg)
		}},
		{name: "float32 overflow runtime scalar", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			valid, err := json.Marshal(gemma4AudioConfig(1))
			if err != nil {
				t.Fatal(err)
			}
			overflow := []byte(strings.Replace(string(valid), `"gradient_clipping":10000000000`, `"gradient_clipping":1e39`, 1))
			if string(overflow) == string(valid) {
				t.Fatal("failed to construct float32-overflow audio config")
			}
			return replaceGemma4ManifestJSON(t, layers, "config.json", overflow)
		}},
		{name: "wrong audio token id", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			return replaceGemma4ManifestJSON(t, layers, "tokenizer.json", []byte(`{"model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[{"id":5,"content":"<|audio>","special":true},{"id":8,"content":"<|audio|>","special":true},{"id":6,"content":"<audio|>","special":true}]}`))
		}},
		{name: "vocab-only markers are not singleton encodings", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			return replaceGemma4ManifestJSON(t, layers, "tokenizer.json", []byte(`{"model":{"type":"BPE","vocab":{"<|audio>":5,"<|audio|>":7,"<audio|>":6},"merges":[]},"added_tokens":[]}`))
		}},
		{name: "near-match internal name", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			return replaceGemma4AudioFixtureLayer(t, layers, target, target+".extra", gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: []int32{3, 4}}, nil)
		}},
		{name: "wrong shape", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			return replaceGemma4AudioFixtureLayer(t, layers, target, target, gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: []int32{4, 3}}, nil)
		}},
		{name: "wrong dtype", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			return replaceGemma4AudioFixtureLayer(t, layers, target, target, gemma4metadata.TensorDescriptor{Dtype: "U8", Shape: []int32{3, 4}}, nil)
		}},
		{name: "duplicate descriptor", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			return append(layers, gemma4AudioFixtureLayer(t, "model.audio_tower.duplicate.weight", target, gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: []int32{3, 4}}, nil))
		}},
		{name: "truncated payload", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			return replaceGemma4AudioFixtureLayer(t, layers, target, target, gemma4metadata.TensorDescriptor{}, []byte{1, 2, 3, 4})
		}},
		{name: "invalid payload range", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			header, err := json.Marshal(map[string]any{target: map[string]any{"dtype": "F32", "shape": []int{3, 4}, "data_offsets": []int{0, 4}}})
			if err != nil {
				t.Fatal(err)
			}
			data := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
			data = append(data, header...)
			data = append(data, make([]byte, 4)...)
			return replaceGemma4AudioFixtureLayer(t, layers, target, target, gemma4metadata.TensorDescriptor{}, data)
		}},
		{name: "malformed header", edit: func(t *testing.T, layers []manifest.Layer) []manifest.Layer {
			data := binary.LittleEndian.AppendUint64(nil, 5)
			data = append(data, []byte("nope!")...)
			return replaceGemma4AudioFixtureLayer(t, layers, target, target, gemma4metadata.TensorDescriptor{}, data)
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setTestHome(t, t.TempDir())
			layers := gemma4AudioManifestLayers(t)
			if tt.edit != nil {
				layers = tt.edit(t, layers)
			}
			cfg := model.ConfigV2{ModelFormat: "safetensors", Renderer: gemma4RendererLarge, Capabilities: []string{"completion", "audio"}}
			name := "gemma4-audio-" + strings.ReplaceAll(tt.name, " ", "-")
			createSafetensorsTestModel(t, name, cfg, layers)

			m, err := GetModel(name)
			if err != nil {
				t.Fatal(err)
			}
			showAudio := slices.Contains(m.Capabilities(), model.CapabilityAudio)
			mf, err := manifest.ParseNamedManifest(model.ParseName(name))
			if err != nil {
				t.Fatal(err)
			}
			summary, err := buildModelListSummary(model.ParseName(name), mf)
			if err != nil {
				t.Fatal(err)
			}
			listAudio := slices.Contains(summary.Capabilities, model.CapabilityAudio)
			if showAudio != tt.want || listAudio != tt.want {
				t.Fatalf("show/list audio = %t/%t, want %t", showAudio, listAudio, tt.want)
			}
		})
	}
}

func TestGemma4InstalledAudioCapabilityRejectsUnboundedConfig(t *testing.T) {
	tests := []struct {
		name string
		edit func(*gemma4metadata.ConfigFile)
	}{
		{name: "shape product overflow", edit: func(cfg *gemma4metadata.ConfigFile) {
			cfg.AudioConfig.HiddenSize = 1 << 30
			cfg.AudioConfig.NumAttentionHeads = 1
		}},
		{name: "impractical layer count", edit: func(cfg *gemma4metadata.ConfigFile) { cfg.AudioConfig.NumHiddenLayers = 1 << 30 }},
		{name: "impractical convolution channels", edit: func(cfg *gemma4metadata.ConfigFile) { cfg.AudioConfig.SubsamplingConvChannels = []int{2, 1 << 30} }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setTestHome(t, t.TempDir())
			cfg := gemma4AudioConfig(1)
			tt.edit(cfg)
			layers := replaceGemma4AudioConfigLayer(t, gemma4AudioManifestLayers(t), cfg)
			modelCfg := model.ConfigV2{ModelFormat: "safetensors", Renderer: gemma4RendererLarge, Capabilities: []string{"completion", "audio"}}
			name := "gemma4-audio-config-" + strings.ReplaceAll(tt.name, " ", "-")
			createSafetensorsTestModel(t, name, modelCfg, layers)

			m, err := GetModel(name)
			if err != nil {
				t.Fatal(err)
			}
			mf, err := manifest.ParseNamedManifest(model.ParseName(name))
			if err != nil {
				t.Fatal(err)
			}
			summary, err := buildModelListSummary(model.ParseName(name), mf)
			if err != nil {
				t.Fatal(err)
			}
			if slices.Contains(m.Capabilities(), model.CapabilityAudio) || slices.Contains(summary.Capabilities, model.CapabilityAudio) {
				t.Fatalf("unbounded config exposed show/list audio")
			}
		})
	}
}

func gemma4AudioManifestLayers(t *testing.T) []manifest.Layer {
	t.Helper()
	descriptors := testGemma4AudioTensorDescriptors(1)
	layers := make([]manifest.Layer, 0, len(descriptors)+4)
	for name, descriptor := range descriptors {
		layers = append(layers, gemma4AudioFixtureLayer(t, name, name, descriptor, nil))
	}
	config, err := json.Marshal(gemma4AudioConfig(1))
	if err != nil {
		t.Fatal(err)
	}
	digest := createTestBlob(t, config)
	layers = append(layers, manifest.Layer{MediaType: "application/vnd.ollama.image.json", Digest: digest, Size: int64(len(config)), Name: "config.json"})
	for name, data := range map[string][]byte{
		"processor_config.json": []byte(`{"audio_seq_length":750,"feature_extractor":{"feature_size":128,"fft_length":512,"frame_length":320,"hop_length":160,"input_scale_factor":1,"max_frequency":8000,"mel_floor":0.001,"padding_side":"right","sampling_rate":16000}}`),
		"tokenizer_config.json": []byte(`{"boa_token":"<|audio>","audio_token":"<|audio|>","eoa_token":"<audio|>"}`),
		"tokenizer.json":        []byte(`{"model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[{"id":5,"content":"<|audio>","special":true},{"id":7,"content":"<|audio|>","special":true},{"id":6,"content":"<audio|>","special":true}]}`),
	} {
		digest := createTestBlob(t, data)
		layers = append(layers, manifest.Layer{MediaType: "application/vnd.ollama.image.json", Digest: digest, Size: int64(len(data)), Name: name})
	}
	return layers
}

func replaceGemma4AudioConfigLayer(t *testing.T, layers []manifest.Layer, cfg *gemma4metadata.ConfigFile) []manifest.Layer {
	t.Helper()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := range layers {
		if layers[i].Name == "config.json" {
			digest := createTestBlob(t, data)
			layers[i] = manifest.Layer{MediaType: "application/vnd.ollama.image.json", Digest: digest, Size: int64(len(data)), Name: "config.json"}
			return layers
		}
	}
	t.Fatal("missing audio config fixture layer")
	return nil
}

func replaceGemma4ManifestJSON(t *testing.T, layers []manifest.Layer, name string, contents []byte) []manifest.Layer {
	t.Helper()
	digest := createTestBlob(t, contents)
	for i := range layers {
		if layers[i].Name == name {
			layers[i].Digest = digest
			layers[i].Size = int64(len(contents))
			return layers
		}
	}
	t.Fatalf("manifest config %q not found", name)
	return nil
}

func gemma4AudioFixtureLayer(t *testing.T, manifestName, internalName string, descriptor gemma4metadata.TensorDescriptor, raw []byte) manifest.Layer {
	t.Helper()
	data := raw
	if data == nil {
		shape := make([]int64, len(descriptor.Shape))
		for i, dim := range descriptor.Shape {
			shape[i] = int64(dim)
		}
		size, err := gemma4SafetensorByteSize(descriptor.Dtype, shape)
		if err != nil {
			t.Fatal(err)
		}
		built, err := io.ReadAll(safetensors.BuildPackedSafetensorsReader([]*safetensors.TensorData{
			safetensors.NewTensorDataFromBytes(internalName, descriptor.Dtype, descriptor.Shape, make([]byte, int(size))),
		}))
		if err != nil {
			t.Fatal(err)
		}
		data = built
	}
	digest := createTestBlob(t, data)
	return manifest.Layer{MediaType: manifest.MediaTypeImageTensor, Digest: digest, Size: int64(len(data)), Name: manifestName}
}

func replaceGemma4AudioFixtureLayer(t *testing.T, layers []manifest.Layer, manifestName, internalName string, descriptor gemma4metadata.TensorDescriptor, raw []byte) []manifest.Layer {
	t.Helper()
	for i := range layers {
		if layers[i].Name == manifestName {
			layers[i] = gemma4AudioFixtureLayer(t, manifestName, internalName, descriptor, raw)
			return layers
		}
	}
	t.Fatalf("missing audio fixture layer %s", manifestName)
	return nil
}

func TestGemma4VisionTensorValidationRejectsIncompleteAndNearMatches(t *testing.T) {
	bareTensorNames := func() []string {
		names := gemma4VisionTensorNames(2)
		for i := range names {
			names[i] = strings.TrimPrefix(names[i], "model.")
		}
		return names
	}
	tests := []struct {
		name  string
		names []string
		want  bool
	}{
		{name: "missing"},
		{name: "tower only", names: []string{"model.vision_tower.patch_embedder.input_proj.weight"}},
		{name: "projector only", names: []string{"model.embed_vision.embedding_projection.weight"}},
		{
			name: "near matches",
			names: []string{
				"model.vision_tower.patch_embedder.input_proj.weight.extra",
				"model.embed_visionary.embedding_projection.weight",
			},
		},
		{name: "sentinels only", names: []string{"model.vision_tower.patch_embedder.input_proj.weight", "model.embed_vision.embedding_projection.weight"}},
		{name: "complete model prefix", names: gemma4VisionTensorNames(2), want: true},
		{name: "complete bare prefix", names: bareTensorNames(), want: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := gemma4metadata.ValidateVisionTensors(*gemma4VisionConfig(2), tt.names) == nil
			if got != tt.want {
				t.Fatalf("ValidateVisionTensors(%v) success = %t, want %t", tt.names, got, tt.want)
			}
		})
	}
}

func TestGemma4InstalledVisionCapabilityRejectsMalformedDescriptors(t *testing.T) {
	valid := Model{
		Config:              model.ConfigV2{ModelFormat: "safetensors", Renderer: gemma4RendererLarge, Capabilities: []string{"vision"}},
		Gemma4VisionConfig:  gemma4VisionConfig(2),
		Gemma4VisionTensors: testGemma4VisionTensorDescriptors(2),
	}
	if !slices.Contains(valid.Capabilities(), model.CapabilityVision) {
		t.Fatal("valid installed descriptors did not expose vision")
	}
	malformed := valid
	malformed.Gemma4VisionTensors = maps.Clone(valid.Gemma4VisionTensors)
	name := "model.vision_tower.patch_embedder.input_proj.weight"
	descriptor := malformed.Gemma4VisionTensors[name]
	descriptor.Shape = []int32{4, 11}
	malformed.Gemma4VisionTensors[name] = descriptor
	if slices.Contains(malformed.Capabilities(), model.CapabilityVision) {
		t.Fatal("complete-name installed inventory with wrong shape exposed vision")
	}
	packedAlias := valid
	packedAlias.Gemma4VisionTensors = maps.Clone(valid.Gemma4VisionTensors)
	delete(packedAlias.Gemma4VisionTensors, name)
	packedAlias.Gemma4VisionTensors[strings.TrimSuffix(name, ".weight")+".weight_packed"] = gemma4metadata.TensorDescriptor{Dtype: "U8", Shape: []int32{4, 6}}
	if slices.Contains(packedAlias.Capabilities(), model.CapabilityVision) {
		t.Fatal("source-only packed alias exposed installed vision")
	}
}

func TestGemma4InstalledVisionCapabilityRejectsMissingOrZeroTextWidth(t *testing.T) {
	for _, name := range []string{"missing", "zero"} {
		t.Run(name, func(t *testing.T) {
			cfg := gemma4VisionConfig(2)
			cfg.TextConfig.HiddenSize = 0
			m := Model{
				Config:              model.ConfigV2{ModelFormat: "safetensors", Renderer: gemma4RendererLarge, Capabilities: []string{"vision"}},
				Gemma4VisionConfig:  cfg,
				Gemma4VisionTensors: testGemma4VisionTensorDescriptors(2),
			}
			if slices.Contains(m.Capabilities(), model.CapabilityVision) {
				t.Fatal("non-executable text width exposed installed vision")
			}
		})
	}
}

func gemma4VisionManifestLayers(t *testing.T) []manifest.Layer {
	t.Helper()

	layers := make([]manifest.Layer, 0, len(gemma4VisionTensorNames(2))+1)
	for name, descriptor := range testGemma4VisionTensorDescriptors(2) {
		shape := make([]int64, len(descriptor.Shape))
		for i, dim := range descriptor.Shape {
			shape[i] = int64(dim)
		}
		payloadSize, err := gemma4SafetensorByteSize(descriptor.Dtype, shape)
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(safetensors.BuildPackedSafetensorsReader([]*safetensors.TensorData{
			safetensors.NewTensorDataFromBytes(name, descriptor.Dtype, descriptor.Shape, make([]byte, int(payloadSize))),
		}))
		if err != nil {
			t.Fatal(err)
		}
		digest := createTestBlob(t, data)
		layers = append(layers, manifest.Layer{
			MediaType: manifest.MediaTypeImageTensor,
			Digest:    digest,
			Size:      int64(len(data)),
			Name:      name,
		})
	}
	config := []byte(`{"text_config":{"hidden_size":6},"vision_config":{"hidden_size":4,"intermediate_size":8,"num_hidden_layers":2,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,"default_output_length":1,"patch_size":2,"position_embedding_size":16,"pooling_kernel_size":1}}`)
	configDigest := createTestBlob(t, config)
	layers = append(layers, manifest.Layer{
		MediaType: "application/vnd.ollama.image.json",
		Digest:    configDigest,
		Size:      int64(len(config)),
		Name:      "config.json",
	})
	if descriptors, err := gemma4VisionTensorDescriptors(layers); err != nil {
		t.Fatalf("read Gemma4 descriptor fixture: %v", err)
	} else if err := gemma4metadata.ValidateVisionInstalledInventory(*gemma4VisionConfig(2), descriptors); err != nil {
		t.Fatalf("validate Gemma4 descriptor fixture: %v", err)
	}
	return layers
}

func TestOpenGemma4TensorLayerRejectsUnsafeHeaders(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{name: "truncated length", data: []byte{1, 2, 3, 4}},
		{name: "truncated header", data: append(binary.LittleEndian.AppendUint64(nil, 32), []byte("{}")...)},
		{name: "oversized header", data: binary.LittleEndian.AppendUint64(nil, maxGemma4SafetensorsHeaderSize+1)},
		{name: "uint64 overflow header", data: binary.LittleEndian.AppendUint64(nil, ^uint64(0))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "model.safetensors")
			if err := os.WriteFile(path, tt.data, 0o644); err != nil {
				t.Fatal(err)
			}
			if ext, err := openGemma4TensorLayer(path); err == nil {
				ext.Close()
				t.Fatal("unsafe safetensors header accepted")
			}
		})
	}
}

func TestOpenGemma4TensorLayerRejectsInvalidPayloadRanges(t *testing.T) {
	write := func(t *testing.T, dtype string, shape []int64, offsets [2]int64, payload int) string {
		t.Helper()
		header, err := json.Marshal(map[string]any{
			"tensor": map[string]any{"dtype": dtype, "shape": shape, "data_offsets": offsets},
		})
		if err != nil {
			t.Fatal(err)
		}
		data := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
		data = append(data, header...)
		data = append(data, make([]byte, payload)...)
		path := filepath.Join(t.TempDir(), "model.safetensors")
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}
	tests := []struct {
		name    string
		dtype   string
		shape   []int64
		offsets [2]int64
		payload int
	}{
		{name: "truncated range", dtype: "F32", shape: []int64{2}, offsets: [2]int64{0, 4}, payload: 4},
		{name: "overlong range", dtype: "F32", shape: []int64{1}, offsets: [2]int64{0, 8}, payload: 8},
		{name: "overlong payload", dtype: "F32", shape: []int64{1}, offsets: [2]int64{0, 4}, payload: 8},
		{name: "shape byte overflow", dtype: "F64", shape: []int64{math.MaxInt64, 2}, offsets: [2]int64{0, 0}},
		{name: "out of file", dtype: "U8", shape: []int64{2}, offsets: [2]int64{0, 2}, payload: 1},
		{name: "negative range", dtype: "U8", shape: []int64{1}, offsets: [2]int64{-1, 0}, payload: 1},
		{name: "reversed range", dtype: "U8", shape: []int64{1}, offsets: [2]int64{1, 0}, payload: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if ext, err := openGemma4TensorLayer(write(t, tt.dtype, tt.shape, tt.offsets, tt.payload)); err == nil {
				ext.Close()
				t.Fatal("invalid safetensors payload range accepted")
			}
		})
	}
}

func TestGemma4VisionTensorDescriptorsRejectsExcessiveInventory(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	tensors := make([]*safetensors.TensorData, 0, maxGemma4VisionDescriptors+1)
	for i := 0; i <= maxGemma4VisionDescriptors; i++ {
		tensors = append(tensors, safetensors.NewTensorDataFromBytes(fmt.Sprintf("tensor.%04d", i), "U8", []int32{1}, []byte{0}))
	}
	data, err := io.ReadAll(safetensors.BuildPackedSafetensorsReader(tensors))
	if err != nil {
		t.Fatal(err)
	}
	digest := createTestBlob(t, data)
	layers := []manifest.Layer{{
		MediaType: manifest.MediaTypeImageTensor,
		Digest:    digest,
		Size:      int64(len(data)),
		Name:      "model.vision_tower.synthetic.weight",
	}}
	if _, err := gemma4VisionTensorDescriptors(layers); err == nil || !strings.Contains(err.Error(), "descriptors") {
		t.Fatalf("excessive inventory error = %v", err)
	}
	layers[0].Name = "model.audio_tower.synthetic.weight"
	if _, err := gemma4AudioTensorDescriptors(layers); err == nil || !strings.Contains(err.Error(), "descriptors") {
		t.Fatalf("excessive audio inventory error = %v", err)
	}
}

func gemma4VisionConfig(layers int) *gemma4metadata.ConfigFile {
	return &gemma4metadata.ConfigFile{
		TextConfig:   gemma4metadata.TextConfig{HiddenSize: 6},
		VisionConfig: &gemma4metadata.VisionConfig{HiddenSize: 4, IntermediateSize: 8, NumHiddenLayers: layers, NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: 4, DefaultOutputLength: 1, PatchSize: 2, PositionEmbeddingSize: 16, PoolingKernelSize: 1},
	}
}

func gemma4AudioConfig(layers int) *gemma4metadata.ConfigFile {
	return &gemma4metadata.ConfigFile{
		TextConfig:   gemma4metadata.TextConfig{HiddenSize: 5, VocabSize: 32},
		AudioTokenID: 7,
		AudioConfig: &gemma4metadata.AudioConfig{
			AttentionChunkSize: 2, AttentionContextLeft: 2,
			AttentionInvalidLogit: -1e9, AttentionLogitCap: 50,
			ConvKernelSize: 3, HiddenSize: 4, NumAttentionHeads: 2,
			NumHiddenLayers: layers, OutputProjDims: 3,
			GradientClipping: 1e10, ResidualWeight: 0.5, RMSNormEps: 1e-6,
			SubsamplingConvChannels: []int{2, 2}, UseClippedLinears: true,
		},
	}
}

func gemma4AudioTensorNames(layers int) []string {
	shapes, err := gemma4metadata.RequiredAudioTensorShapes(*gemma4AudioConfig(layers))
	if err != nil {
		panic(err)
	}
	names := make([]string, 0, len(shapes))
	for name := range shapes {
		names = append(names, name)
	}
	return names
}

func testGemma4AudioTensorDescriptors(layers int) map[string]gemma4metadata.TensorDescriptor {
	shapes, err := gemma4metadata.RequiredAudioTensorShapes(*gemma4AudioConfig(layers))
	if err != nil {
		panic(err)
	}
	tensors := make(map[string]gemma4metadata.TensorDescriptor, len(shapes))
	for name, shape := range shapes {
		tensors[name] = gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: shape}
	}
	return tensors
}

func testGemma4VisionTensorDescriptors(layers int) map[string]gemma4metadata.TensorDescriptor {
	return testGemma4VisionTensorDescriptorsForGeometry(layers, 4, 8, 6, 2, 16, 4)
}

func testGemma4VisionTensorDescriptorsForGeometry(layers int, hidden, intermediate, textHidden, patch, positions, headDim int32) map[string]gemma4metadata.TensorDescriptor {
	descriptors := map[string]gemma4metadata.TensorDescriptor{
		"model.vision_tower.patch_embedder.input_proj.weight":        {Dtype: "F32", Shape: []int32{hidden, 3 * patch * patch}},
		"model.vision_tower.patch_embedder.position_embedding_table": {Dtype: "F32", Shape: []int32{2, positions, hidden}},
		"model.embed_vision.embedding_projection.weight":             {Dtype: "F32", Shape: []int32{textHidden, hidden}},
	}
	for i := range layers {
		layer := fmt.Sprintf("model.vision_tower.encoder.layers.%d", i)
		for _, suffix := range []string{".self_attn.q_proj.linear.weight", ".self_attn.k_proj.linear.weight", ".self_attn.v_proj.linear.weight", ".self_attn.o_proj.linear.weight"} {
			descriptors[layer+suffix] = gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: []int32{hidden, hidden}}
		}
		for _, suffix := range []string{".mlp.gate_proj.linear.weight", ".mlp.up_proj.linear.weight"} {
			descriptors[layer+suffix] = gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: []int32{intermediate, hidden}}
		}
		descriptors[layer+".mlp.down_proj.linear.weight"] = gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: []int32{hidden, intermediate}}
		for _, suffix := range []string{".self_attn.q_norm.weight", ".self_attn.k_norm.weight"} {
			descriptors[layer+suffix] = gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: []int32{headDim}}
		}
		for _, suffix := range []string{".input_layernorm.weight", ".post_attention_layernorm.weight", ".pre_feedforward_layernorm.weight", ".post_feedforward_layernorm.weight"} {
			descriptors[layer+suffix] = gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: []int32{hidden}}
		}
	}
	return descriptors
}

func TestGemma4InstalledVisionCapabilityReleasedUnequalHeadGeometry(t *testing.T) {
	cfg := &gemma4metadata.ConfigFile{
		TextConfig: gemma4metadata.TextConfig{HiddenSize: 2560},
		VisionConfig: &gemma4metadata.VisionConfig{
			HiddenSize: 768, IntermediateSize: 3072, NumHiddenLayers: 1,
			NumAttentionHeads: 12, NumKeyValueHeads: 12, HeadDim: 64,
			RMSNormEps: 1e-6, DefaultOutputLength: 280, PatchSize: 16,
			PositionEmbeddingSize: 10240, PoolingKernelSize: 3,
		},
	}
	m := Model{
		Config:              model.ConfigV2{ModelFormat: "safetensors", Renderer: gemma4RendererLarge, Capabilities: []string{"vision"}},
		Gemma4VisionConfig:  cfg,
		Gemma4VisionTensors: testGemma4VisionTensorDescriptorsForGeometry(1, 768, 3072, 2560, 16, 10240, 64),
	}
	if !slices.Contains(m.Capabilities(), model.CapabilityVision) {
		t.Fatal("released-compatible unequal hidden/head geometry did not expose vision")
	}
}

func gemma4VisionTensorNames(layers int) []string {
	names := []string{
		"model.vision_tower.patch_embedder.input_proj.weight",
		"model.vision_tower.patch_embedder.position_embedding_table",
		"model.embed_vision.embedding_projection.weight",
	}
	for i := range layers {
		layer := fmt.Sprintf("model.vision_tower.encoder.layers.%d", i)
		for _, projection := range []string{
			".self_attn.q_proj.linear.weight", ".self_attn.k_proj.linear.weight",
			".self_attn.v_proj.linear.weight", ".self_attn.o_proj.linear.weight",
			".mlp.gate_proj.linear.weight", ".mlp.up_proj.linear.weight", ".mlp.down_proj.linear.weight",
		} {
			names = append(names, layer+projection)
		}
		for _, norm := range []string{
			".self_attn.q_norm.weight", ".self_attn.k_norm.weight",
			".input_layernorm.weight", ".post_attention_layernorm.weight",
			".pre_feedforward_layernorm.weight", ".post_feedforward_layernorm.weight",
		} {
			names = append(names, layer+norm)
		}
	}
	return names
}

func TestModelCheckCapabilities(t *testing.T) {
	// Create simple model file for tests that don't depend on GGUF content
	completionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
	}, []*ggml.Tensor{})

	// Create vision model (llama architecture with vision block count)
	visionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	// Create embedding model (bert architecture with pooling type)
	embeddingModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, []*ggml.Tensor{})

	toolsInsertTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	chatTemplate, err := template.Parse("{{ .prompt }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	toolsTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	tests := []struct {
		name           string
		model          Model
		checkCaps      []model.Capability
		expectedErrMsg string
	}{
		{
			name: "completion model without tools capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityTools},
			expectedErrMsg: "does not support tools",
		},
		{
			name: "model with all needed capabilities",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsInsertTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model missing insert capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityInsert},
			expectedErrMsg: "does not support insert",
		},
		{
			name: "model missing vision capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityVision},
			expectedErrMsg: "does not support vision",
		},
		{
			name: "model with vision capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  chatTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityVision},
		},
		{
			name: "model with embedding capability",
			model: Model{
				ModelPath: embeddingModelPath,
				Template:  chatTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityEmbedding},
		},
		{
			name: "unknown capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{"unknown"},
			expectedErrMsg: "unknown capability",
		},
		{
			name: "model missing image generation capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityImage},
			expectedErrMsg: "does not support image generation",
		},
		{
			name: "model with image generation capability",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image"},
				},
			},
			checkCaps: []model.Capability{model.CapabilityImage},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test CheckCapabilities method
			err := tt.model.CheckCapabilities(tt.checkCaps...)
			if tt.expectedErrMsg == "" {
				if err != nil {
					t.Errorf("Expected no error, got: %v", err)
				}
			} else {
				if err == nil {
					t.Errorf("Expected error containing %q, got nil", tt.expectedErrMsg)
				} else if !strings.Contains(err.Error(), tt.expectedErrMsg) {
					t.Errorf("Expected error containing %q, got: %v", tt.expectedErrMsg, err)
				}
			}
		})
	}
}

func TestPullModelManifest(t *testing.T) {
	cases := []struct {
		name     string
		manifest string
	}{
		{
			name: "pretty printed",
			manifest: `{  "schemaVersion": 2,  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": { "digest": "sha256:abc", "mediaType": "application/vnd.docker.container.image.v1+json", "size": 50 },
  "layers": [{ "digest": "sha256:t1", "mediaType": "application/vnd.ollama.image.tensor", "size": 1024, "name": "model.weight" }]
}`,
		},
		{
			name:     "non-standard field order",
			manifest: `{"layers":[{"size":999,"digest":"sha256:def","mediaType":"application/vnd.ollama.image.model"}],"schemaVersion":2,"config":{"size":50,"digest":"sha256:abc","mediaType":"application/vnd.docker.container.image.v1+json"},"mediaType":"application/vnd.docker.distribution.manifest.v2+json"}`,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(tt.manifest))
			}))
			defer ts.Close()

			n := model.ParseName("test/model:latest")
			n.ProtocolScheme = "http"
			n.Host = strings.TrimPrefix(ts.URL, "http://")

			mf, data, err := pullModelManifest(t.Context(), n, &registryOptions{})
			if err != nil {
				t.Fatal(err)
			}

			// Raw bytes must be byte-for-byte identical to what the server sent
			if string(data) != tt.manifest {
				t.Fatalf("raw bytes differ from server response")
			}

			// SHA256 of returned data must match the expected registry digest
			expectedDigest := fmt.Sprintf("%x", sha256.Sum256([]byte(tt.manifest)))
			gotDigest := fmt.Sprintf("%x", sha256.Sum256(data))
			if gotDigest != expectedDigest {
				t.Fatalf("digest mismatch\ngot:  %s\nwant: %s", gotDigest, expectedDigest)
			}

			// Parsed manifest must still be usable
			if mf.SchemaVersion != 2 {
				t.Fatalf("schemaVersion = %d, want 2", mf.SchemaVersion)
			}
			if mf.Config.Digest == "" {
				t.Fatal("config digest is empty")
			}
			if len(mf.Layers) == 0 {
				t.Fatal("expected at least one layer")
			}
		})
	}
}

// TestPullModelDuplicateDigestVerifiesBlob pulls a manifest whose config and
// layer share a digest. The registry redirects blob downloads via Location to
// an "internal" path serving bytes that don't match the digest, so PullModel
// must reject the pull with errDigestMismatch.
func TestPullModelDuplicateDigestVerifiesBlob(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	const bogusDigest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

	backendURL := ""
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/manifests/"):
			w.Header().Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
			fmt.Fprintf(w, `{
				"schemaVersion": 2,
				"mediaType": "application/vnd.docker.distribution.manifest.v2+json",
				"config": {
					"mediaType": "application/vnd.ollama.image.config",
					"digest": %q,
					"size": 5
				},
				"layers": [{
					"mediaType": "application/vnd.ollama.image.model",
					"digest": %q,
					"size": 5
				}]
			}`, bogusDigest, bogusDigest)
		case strings.Contains(r.URL.Path, "/internal/blobs/"):
			w.Write([]byte("attacker-controlled-bytes"))
		case strings.Contains(r.URL.Path, "/blobs/"):
			w.Header().Set("Location", backendURL+"/internal"+r.URL.Path)
			w.WriteHeader(http.StatusOK)
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()
	backendURL = ts.URL

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	n := model.ParseName(u.Host + "/test/attack")
	n.ProtocolScheme = "http"

	err = PullModel(t.Context(), n.String(), &registryOptions{Insecure: true}, func(api.ProgressResponse) {})
	if !errors.Is(err, errDigestMismatch) {
		t.Fatalf("PullModel = %v, want errDigestMismatch (unverified blob would persist)", err)
	}
}
