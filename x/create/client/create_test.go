package client

import (
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/create"
	gemma4metadata "github.com/ollama/ollama/x/models/gemma4/metadata"
	"github.com/ollama/ollama/x/safetensors"
)

func TestModelfileConfig(t *testing.T) {
	// Test that ModelfileConfig struct works as expected
	config := &ModelfileConfig{
		Template: "{{ .Prompt }}",
		System:   "You are a helpful assistant.",
		License:  "MIT",
		Parser:   "qwen3",
		Renderer: "qwen3",
	}

	if config.Template != "{{ .Prompt }}" {
		t.Errorf("Template = %q, want %q", config.Template, "{{ .Prompt }}")
	}
	if config.System != "You are a helpful assistant." {
		t.Errorf("System = %q, want %q", config.System, "You are a helpful assistant.")
	}
	if config.License != "MIT" {
		t.Errorf("License = %q, want %q", config.License, "MIT")
	}
	if config.Parser != "qwen3" {
		t.Errorf("Parser = %q, want %q", config.Parser, "qwen3")
	}
	if config.Renderer != "qwen3" {
		t.Errorf("Renderer = %q, want %q", config.Renderer, "qwen3")
	}
}

func TestNemotronNanoOmniMetadataInference(t *testing.T) {
	dir := t.TempDir()
	config := `{
		"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"],
		"model_type": "NemotronH_Nano_Omni_Reasoning_V3",
		"vision_config": {"patch_size": 16},
		"sound_config": {"model_type": "parakeet"},
		"llm_config": {"model_type": "nemotron_h"}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, want := getParserName(dir), "nemotron-3-nano"; got != want {
		t.Fatalf("parser = %q, want %q", got, want)
	}
	if got, want := getRendererName(dir), "nemotron-3-nano"; got != want {
		t.Fatalf("renderer = %q, want %q", got, want)
	}
	caps := inferSafetensorsCapabilities(dir, getParserName(dir))
	if !slices.Equal(caps, []string{"completion", "vision", "audio", "tools", "thinking"}) {
		t.Fatalf("capabilities = %v, want completion/vision/audio/tools/thinking", caps)
	}
}

func TestNemotron35MetadataInference(t *testing.T) {
	dir := t.TempDir()
	config := `{"architectures":["NemotronHForCausalLM"],"model_type":"nemotron_h"}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte("{reasoning effort: efficient}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, want := getParserName(dir), "nemotron-3.5-nano"; got != want {
		t.Fatalf("parser = %q, want %q", got, want)
	}
	if got, want := getRendererName(dir), "nemotron-3.5-nano"; got != want {
		t.Fatalf("renderer = %q, want %q", got, want)
	}
}

func TestConfigFromModelfile(t *testing.T) {
	modelfile, err := parser.ParseFile(strings.NewReader(`
FROM ./model
DRAFT ./assistant
TEMPLATE {{ .Prompt }}
REQUIRES 0.20.0
PARAMETER temperature 0.7
PARAMETER stop USER:
PARAMETER stop ASSISTANT:
`))
	if err != nil {
		t.Fatal(err)
	}

	modelDir, mfConfig, err := ConfigFromModelfile(modelfile)
	if err != nil {
		t.Fatal(err)
	}

	if modelDir != "./model" {
		t.Fatalf("modelDir = %q, want %q", modelDir, "./model")
	}

	if mfConfig.Template != "{{ .Prompt }}" {
		t.Fatalf("Template = %q, want %q", mfConfig.Template, "{{ .Prompt }}")
	}

	if mfConfig.Draft != "./assistant" {
		t.Fatalf("Draft = %q, want %q", mfConfig.Draft, "./assistant")
	}

	if mfConfig.Requires != "0.20.0" {
		t.Fatalf("Requires = %q, want %q", mfConfig.Requires, "0.20.0")
	}

	if got := mfConfig.Parameters["temperature"]; got != float32(0.7) {
		t.Fatalf("temperature = %#v, want %v", got, float32(0.7))
	}

	if got := mfConfig.Parameters["stop"]; got == nil || len(got.([]string)) != 2 {
		t.Fatalf("unexpected stop params: %#v", got)
	}
}

func TestConfigFromModelfile_RequiresBelowMinimum(t *testing.T) {
	modelfile, err := parser.ParseFile(strings.NewReader(`
FROM ./model
REQUIRES 0.14.0
`))
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = ConfigFromModelfile(modelfile)
	if err == nil {
		t.Fatal("expected error for REQUIRES below minimum, got nil")
	}
	if !strings.Contains(err.Error(), "minimum supported version") {
		t.Fatalf("error = %v, want error mentioning minimum supported version", err)
	}
}

func TestConfigFromModelfile_RequiresInvalidSemver(t *testing.T) {
	modelfile, err := parser.ParseFile(strings.NewReader(`
FROM ./model
REQUIRES not-a-version
`))
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = ConfigFromModelfile(modelfile)
	if err == nil {
		t.Fatal("expected error for invalid semver, got nil")
	}
	if !strings.Contains(err.Error(), "valid semver") {
		t.Fatalf("error = %v, want semver error", err)
	}
}

func TestModelfileConfig_Empty(t *testing.T) {
	config := &ModelfileConfig{}

	if config.Template != "" {
		t.Errorf("Template should be empty, got %q", config.Template)
	}
	if config.System != "" {
		t.Errorf("System should be empty, got %q", config.System)
	}
	if config.License != "" {
		t.Errorf("License should be empty, got %q", config.License)
	}
	if config.Parser != "" {
		t.Errorf("Parser should be empty, got %q", config.Parser)
	}
	if config.Renderer != "" {
		t.Errorf("Renderer should be empty, got %q", config.Renderer)
	}
}

func TestModelfileConfig_PartialFields(t *testing.T) {
	// Test config with only some fields set
	config := &ModelfileConfig{
		Template: "{{ .Prompt }}",
		// System and License intentionally empty
	}

	if config.Template == "" {
		t.Error("Template should not be empty")
	}
	if config.System != "" {
		t.Error("System should be empty")
	}
	if config.License != "" {
		t.Error("License should be empty")
	}
	if config.Parser != "" {
		t.Error("Parser should be empty")
	}
	if config.Renderer != "" {
		t.Error("Renderer should be empty")
	}
}

func TestMinOllamaVersion(t *testing.T) {
	// Verify the minimum version constant is set
	if MinOllamaVersion == "" {
		t.Error("MinOllamaVersion should not be empty")
	}
	if MinOllamaVersion != "0.19.0" {
		t.Errorf("MinOllamaVersion = %q, want %q", MinOllamaVersion, "0.19.0")
	}
}

func TestCreateModel_InvalidDir(t *testing.T) {
	// Test that CreateModel returns error for invalid directory
	err := CreateModel(CreateOptions{
		ModelName: "test-model",
		ModelDir:  "/nonexistent/path",
	}, nil)
	if err == nil {
		t.Error("expected error for nonexistent directory, got nil")
	}
}

func TestCreateModel_NotSafetensorsDir(t *testing.T) {
	// Test that CreateModel returns error for directory without safetensors
	dir := t.TempDir()

	err := CreateModel(CreateOptions{
		ModelName: "test-model",
		ModelDir:  dir,
	}, nil)
	if err == nil {
		t.Error("expected error for empty directory, got nil")
	}
}

func TestCreateModel_DraftQuantizeRequiresDraft(t *testing.T) {
	err := CreateModel(CreateOptions{
		ModelName:     "test-model",
		ModelDir:      t.TempDir(),
		DraftQuantize: "mxfp8",
	}, nil)
	if err == nil || !strings.Contains(err.Error(), "--draft-quantize requires a DRAFT model") {
		t.Fatalf("error = %v, want draft-quantize requires DRAFT", err)
	}
}

func TestCreateOptions(t *testing.T) {
	opts := CreateOptions{
		ModelName:     "my-model",
		ModelDir:      "/path/to/model",
		Quantize:      "fp8",
		DraftQuantize: "mxfp8",
		Modelfile: &ModelfileConfig{
			Template: "test",
			System:   "system",
			License:  "MIT",
			Parser:   "qwen3-thinking",
			Renderer: "qwen3",
			Parameters: map[string]any{
				"temperature": float32(0.7),
			},
		},
	}

	if opts.ModelName != "my-model" {
		t.Errorf("ModelName = %q, want %q", opts.ModelName, "my-model")
	}
	if opts.ModelDir != "/path/to/model" {
		t.Errorf("ModelDir = %q, want %q", opts.ModelDir, "/path/to/model")
	}
	if opts.Quantize != "fp8" {
		t.Errorf("Quantize = %q, want %q", opts.Quantize, "fp8")
	}
	if opts.DraftQuantize != "mxfp8" {
		t.Errorf("DraftQuantize = %q, want %q", opts.DraftQuantize, "mxfp8")
	}
	if opts.Modelfile == nil {
		t.Error("Modelfile should not be nil")
	}
	if opts.Modelfile.Template != "test" {
		t.Errorf("Modelfile.Template = %q, want %q", opts.Modelfile.Template, "test")
	}
	if opts.Modelfile.Parser != "qwen3-thinking" {
		t.Errorf("Modelfile.Parser = %q, want %q", opts.Modelfile.Parser, "qwen3-thinking")
	}
	if opts.Modelfile.Renderer != "qwen3" {
		t.Errorf("Modelfile.Renderer = %q, want %q", opts.Modelfile.Renderer, "qwen3")
	}
	if opts.Modelfile.Parameters["temperature"] != float32(0.7) {
		t.Errorf("Modelfile.Parameters[temperature] = %v, want %v", opts.Modelfile.Parameters["temperature"], float32(0.7))
	}
}

func TestResolveParserName(t *testing.T) {
	tests := []struct {
		name     string
		mf       *ModelfileConfig
		inferred string
		want     string
	}{
		{
			name:     "nil modelfile uses inferred",
			mf:       nil,
			inferred: "qwen3",
			want:     "qwen3",
		},
		{
			name: "empty parser uses inferred",
			mf: &ModelfileConfig{
				Parser: "",
			},
			inferred: "qwen3",
			want:     "qwen3",
		},
		{
			name: "explicit parser overrides inferred",
			mf: &ModelfileConfig{
				Parser: "qwen3-thinking",
			},
			inferred: "qwen3",
			want:     "qwen3-thinking",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveParserName(tt.mf, tt.inferred); got != tt.want {
				t.Fatalf("resolveParserName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestResolveRendererName(t *testing.T) {
	tests := []struct {
		name     string
		mf       *ModelfileConfig
		inferred string
		want     string
	}{
		{
			name:     "nil modelfile uses inferred",
			mf:       nil,
			inferred: "qwen3-coder",
			want:     "qwen3-coder",
		},
		{
			name: "empty renderer uses inferred",
			mf: &ModelfileConfig{
				Renderer: "",
			},
			inferred: "qwen3-coder",
			want:     "qwen3-coder",
		},
		{
			name: "explicit renderer overrides inferred",
			mf: &ModelfileConfig{
				Renderer: "qwen3",
			},
			inferred: "qwen3-coder",
			want:     "qwen3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveRendererName(tt.mf, tt.inferred); got != tt.want {
				t.Fatalf("resolveRendererName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCreateOptions_Defaults(t *testing.T) {
	opts := CreateOptions{
		ModelName: "test",
		ModelDir:  "/tmp",
	}

	// Quantize should default to empty
	if opts.Quantize != "" {
		t.Errorf("Quantize should be empty by default, got %q", opts.Quantize)
	}
	if opts.DraftQuantize != "" {
		t.Errorf("DraftQuantize should be empty by default, got %q", opts.DraftQuantize)
	}

	// Modelfile should default to nil
	if opts.Modelfile != nil {
		t.Error("Modelfile should be nil by default")
	}
}

func TestInferSafetensorsCapabilities(t *testing.T) {
	tests := []struct {
		name       string
		configJSON string
		want       []string
	}{
		{
			name: "qwen3.5 text model",
			configJSON: `{
				"architectures": ["Qwen3_5ForCausalLM"],
				"model_type": "qwen3"
			}`,
			want: []string{"completion", "thinking"},
		},
		{
			name: "qwen3.5 multimodal model",
			configJSON: `{
				"architectures": ["Qwen3_5ForConditionalGeneration"],
				"model_type": "qwen3",
				"vision_config": {"hidden_size": 1024}
			}`,
			want: []string{"completion", "vision", "thinking"},
		},
		{
			name: "gemma4 with audio config and missing vision tensors",
			configJSON: `{
				"architectures": ["Gemma4ForConditionalGeneration"],
				"model_type": "gemma4",
				"vision_config": {"hidden_size": 1024},
				"audio_config": {"num_mel_bins": 128}
			}`,
			want: []string{"completion"},
		},
		{
			name: "model with audio but no vision",
			configJSON: `{
				"architectures": ["SomeAudioModel"],
				"model_type": "other",
				"audio_config": {"num_mel_bins": 128}
			}`,
			want: []string{"completion", "audio"},
		},
		{
			name: "model with sound config",
			configJSON: `{
				"architectures": ["SomeSoundModel"],
				"model_type": "other",
				"sound_config": {"model_type": "parakeet"}
			}`,
			want: []string{"completion", "audio"},
		},
		{
			name: "non-qwen conditional generation model",
			configJSON: `{
				"architectures": ["SomeOtherForConditionalGeneration"],
				"model_type": "other"
			}`,
			want: []string{"completion"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644); err != nil {
				t.Fatal(err)
			}

			if got := inferSafetensorsCapabilities(dir, ""); !slices.Equal(got, tt.want) {
				t.Fatalf("inferSafetensorsCapabilities() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestApertusMetadataInference(t *testing.T) {
	tests := []struct {
		name       string
		configJSON string
		parser     string
		renderer   string
		caps       []string
	}{
		{
			name:       "architecture wins over top-level and nested model type",
			configJSON: `{"architectures":["ApertusForCausalLM"],"model_type":"qwen3","llm_config":{"model_type":"gpt_oss"}}`,
			parser:     "apertus",
			renderer:   "apertus",
			caps:       []string{"completion", "tools"},
		},
		{
			name:       "top-level wins over nested model type",
			configJSON: `{"model_type":"apertus","llm_config":{"model_type":"qwen3"}}`,
			parser:     "apertus",
			renderer:   "apertus",
			caps:       []string{"completion", "tools"},
		},
		{
			name:       "nested model type",
			configJSON: `{"model_type":"wrapper","llm_config":{"model_type":"apertus"}}`,
			parser:     "apertus",
			renderer:   "apertus",
			caps:       []string{"completion", "tools"},
		},
		{
			name:       "Apertus 1.5 retains parser-level thinking",
			configJSON: `{"architectures":["Apertus1p5ForConditionalGeneration"],"model_type":"apertus_1_5"}`,
			parser:     "apertus",
			renderer:   "apertus1p5",
			caps:       []string{"completion", "tools", "thinking"},
		},
		{
			name:       "near matches are not Apertus",
			configJSON: `{"architectures":["ApertusForCausalLMExtra"],"model_type":"apertus-next","llm_config":{"model_type":"not_apertus"}}`,
			caps:       []string{"completion"},
		},
		{
			name:       "missing identifiers",
			configJSON: `{}`,
			caps:       []string{"completion"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644); err != nil {
				t.Fatal(err)
			}
			if got := getParserName(dir); got != tt.parser {
				t.Fatalf("parser = %q, want %q", got, tt.parser)
			}
			if got := getRendererName(dir); got != tt.renderer {
				t.Fatalf("renderer = %q, want %q", got, tt.renderer)
			}
			if got := inferSafetensorsCapabilities(dir, getParserName(dir)); !slices.Equal(got, tt.caps) {
				t.Fatalf("capabilities = %v, want %v", got, tt.caps)
			}
		})
	}

	for _, invalid := range []string{"not json", `{"architectures":"ApertusForCausalLM"}`} {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(invalid), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := getParserName(dir); got != "" {
			t.Fatalf("invalid config parser = %q, want empty", got)
		}
		if got := getRendererName(dir); got != "" {
			t.Fatalf("invalid config renderer = %q, want empty", got)
		}
	}
}

func TestInferSafetensorsCapabilitiesGemma4VisionRequiresTensors(t *testing.T) {
	configJSON := `{
		"architectures": ["Gemma4ForConditionalGeneration"],
		"model_type": "gemma4",
		"text_config": {"hidden_size": 6},
		"vision_config": {"hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 2, "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 4, "default_output_length": 1, "patch_size": 2, "position_embedding_size": 16, "pooling_kernel_size": 1}
	}`

	tests := []struct {
		name    string
		tensors []string
		want    []string
	}{
		{name: "missing vision tensors", want: []string{"completion"}},
		{
			name:    "patch only",
			tensors: []string{"model.vision_tower.patch_embedder.input_proj.weight"},
			want:    []string{"completion"},
		},
		{
			name:    "projector only",
			tensors: []string{"model.embed_vision.embedding_projection.weight"},
			want:    []string{"completion"},
		},
		{
			name: "near-match names",
			tensors: []string{
				"model.vision_tower.patch_embedder.input_proj.weight.extra",
				"model.embed_visionary.embedding_projection.weight",
			},
			want: []string{"completion"},
		},
		{
			name: "partial vision tower and projector",
			tensors: []string{
				"model.vision_tower.patch_embedder.input_proj.weight",
				"model.embed_vision.embedding_projection.weight",
			},
			want: []string{"completion"},
		},
		{
			name:    "complete vision tower and projector",
			tensors: gemma4ClientVisionTensorNames(2),
			want:    []string{"completion", "vision"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
				t.Fatal(err)
			}
			if len(tt.tensors) > 0 {
				writeClientSafetensors(t, dir, tt.tensors...)
			}

			if got := inferSafetensorsCapabilities(dir, ""); !slices.Equal(got, tt.want) {
				t.Fatalf("inferSafetensorsCapabilities() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestInferSafetensorsCapabilitiesGemma4UnifiedVision(t *testing.T) {
	const configJSON = `{
		"architectures":["Gemma4UnifiedForConditionalGeneration"],"model_type":"gemma4_unified",
		"text_config":{"hidden_size":5},
		"vision_config":{"model_type":"gemma4_unified_vision","mm_embed_dim":3,"mm_posemb_size":4,"model_patch_size":2,"num_soft_tokens":2,"patch_size":1,"pooling_kernel_size":2}
	}`
	valid := map[string]gemma4metadata.TensorDescriptor{
		"model.vision_embedder.patch_ln1.weight":         {Dtype: "F32", Shape: []int32{12}},
		"model.vision_embedder.patch_ln1.bias":           {Dtype: "F32", Shape: []int32{12}},
		"model.vision_embedder.patch_dense.weight":       {Dtype: "F32", Shape: []int32{3, 12}},
		"model.vision_embedder.patch_dense.bias":         {Dtype: "F32", Shape: []int32{3}},
		"model.vision_embedder.patch_ln2.weight":         {Dtype: "F32", Shape: []int32{3}},
		"model.vision_embedder.patch_ln2.bias":           {Dtype: "F32", Shape: []int32{3}},
		"model.vision_embedder.pos_embedding":            {Dtype: "F32", Shape: []int32{4, 2, 3}},
		"model.vision_embedder.pos_norm.weight":          {Dtype: "F32", Shape: []int32{3}},
		"model.vision_embedder.pos_norm.bias":            {Dtype: "F32", Shape: []int32{3}},
		"model.embed_vision.embedding_projection.weight": {Dtype: "F32", Shape: []int32{5, 3}},
	}
	check := func(t *testing.T, tensors map[string]gemma4metadata.TensorDescriptor, wantVision bool) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
			t.Fatal(err)
		}
		writeClientSafetensorDescriptors(t, dir, tensors)
		got := inferSafetensorsCapabilities(dir, "")
		if slices.Contains(got, "vision") != wantVision {
			t.Fatalf("capabilities = %v, want vision %t", got, wantVision)
		}
	}
	check(t, valid, true)
	partial := maps.Clone(valid)
	delete(partial, "model.vision_embedder.pos_norm.bias")
	check(t, partial, false)
	wrong := maps.Clone(valid)
	d := wrong["model.vision_embedder.patch_dense.weight"]
	d.Shape = []int32{3, 11}
	wrong["model.vision_embedder.patch_dense.weight"] = d
	check(t, wrong, false)
}

func TestInferSafetensorsCapabilitiesGemma4AudioInventory(t *testing.T) {
	identities := []struct {
		name         string
		architecture string
		modelType    string
	}{
		{name: "released", architecture: "Gemma4ForConditionalGeneration", modelType: "gemma4"},
		{name: "unified", architecture: "Gemma4UnifiedForConditionalGeneration", modelType: "gemma4_unified"},
	}
	for _, identity := range identities {
		t.Run(identity.name, func(t *testing.T) {
			cfg := gemma4metadata.ConfigFile{
				Architectures: []string{identity.architecture},
				ModelType:     identity.modelType,
				TextConfig:    gemma4metadata.TextConfig{HiddenSize: 5},
				AudioConfig: &gemma4metadata.AudioConfig{
					AttentionChunkSize: 2, AttentionContextLeft: 2,
					ConvKernelSize: 3, HiddenSize: 4, NumAttentionHeads: 2,
					NumHiddenLayers: 1, OutputProjDims: 3,
					SubsamplingConvChannels: []int{2, 2}, UseClippedLinears: true,
				},
			}
			configJSON, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			shapes, err := gemma4metadata.RequiredAudioTensorShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			valid := make(map[string]gemma4metadata.TensorDescriptor, len(shapes))
			for name, shape := range shapes {
				valid[name] = gemma4metadata.TensorDescriptor{Dtype: "BF16", Shape: shape}
			}

			check := func(t *testing.T, tensors map[string]gemma4metadata.TensorDescriptor, wantAudio bool) {
				t.Helper()
				dir := t.TempDir()
				if err := os.WriteFile(filepath.Join(dir, "config.json"), configJSON, 0o644); err != nil {
					t.Fatal(err)
				}
				writeClientSafetensorDescriptors(t, dir, tensors)
				got := inferSafetensorsCapabilities(dir, "")
				if slices.Contains(got, "audio") != wantAudio {
					t.Fatalf("capabilities = %v, want audio %t", got, wantAudio)
				}
			}

			t.Run("complete", func(t *testing.T) { check(t, valid, true) })
			t.Run("partial", func(t *testing.T) {
				partial := maps.Clone(valid)
				delete(partial, "model.audio_tower.layers.0.self_attn.q_proj.input_max")
				check(t, partial, false)
			})
			t.Run("malformed", func(t *testing.T) {
				malformed := maps.Clone(valid)
				d := malformed["model.audio_tower.output_proj.weight"]
				d.Shape = []int32{4, 3}
				malformed["model.audio_tower.output_proj.weight"] = d
				check(t, malformed, false)
			})
			t.Run("near match", func(t *testing.T) {
				near := maps.Clone(valid)
				name := "model.embed_audio.embedding_projection.weight"
				near[name+".extra"] = near[name]
				delete(near, name)
				check(t, near, false)
			})
		})
	}
}

func TestInferSafetensorsCapabilitiesGemma4AudioRejectsUnboundedConfig(t *testing.T) {
	base := gemma4metadata.ConfigFile{
		Architectures: []string{"Gemma4ForConditionalGeneration"}, ModelType: "gemma4",
		TextConfig: gemma4metadata.TextConfig{HiddenSize: 5},
		AudioConfig: &gemma4metadata.AudioConfig{
			AttentionChunkSize: 2, AttentionContextLeft: 2,
			ConvKernelSize: 3, HiddenSize: 4, NumAttentionHeads: 2,
			NumHiddenLayers: 1, OutputProjDims: 3,
			SubsamplingConvChannels: []int{2, 2}, UseClippedLinears: true,
		},
	}
	shapes, err := gemma4metadata.RequiredAudioTensorShapes(base)
	if err != nil {
		t.Fatal(err)
	}
	valid := make(map[string]gemma4metadata.TensorDescriptor, len(shapes))
	for name, shape := range shapes {
		valid[name] = gemma4metadata.TensorDescriptor{Dtype: "BF16", Shape: shape}
	}
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
			cfg := base
			audio := *base.AudioConfig
			audio.SubsamplingConvChannels = slices.Clone(base.AudioConfig.SubsamplingConvChannels)
			cfg.AudioConfig = &audio
			tt.edit(&cfg)
			configJSON, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), configJSON, 0o644); err != nil {
				t.Fatal(err)
			}
			writeClientSafetensorDescriptors(t, dir, valid)
			if got := inferSafetensorsCapabilities(dir, ""); slices.Contains(got, "audio") {
				t.Fatalf("unbounded config capabilities = %v, did not expect audio", got)
			}
		})
	}
}

func TestInferSafetensorsCapabilitiesGemma4PackedSourceRequiresProducerContract(t *testing.T) {
	const configJSON = `{
		"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4",
		"text_config":{"hidden_size":24},
		"vision_config":{"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":16,"default_output_length":1,"patch_size":4,"position_embedding_size":16,"pooling_kernel_size":1}
	}`
	tensors := make(map[string]gemma4metadata.TensorDescriptor)
	for _, name := range gemma4ClientVisionTensorNames(1) {
		dtype, shape := gemma4ClientVisionDescriptorForDimensions(name, 16, 32, 24, 4, 16, 16)
		tensors[name] = gemma4metadata.TensorDescriptor{Dtype: dtype, Shape: shape}
	}
	base := "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear"
	delete(tensors, base+".weight")
	tensors[base+".weight_packed"] = gemma4metadata.TensorDescriptor{Dtype: "U8", Shape: []int32{16, 8}}
	tensors[base+".weight_scale"] = gemma4metadata.TensorDescriptor{Dtype: "F8_E4M3", Shape: []int32{16, 1}}
	tensors[base+".weight_global_scale"] = gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: nil}

	check := func(t *testing.T, inventory map[string]gemma4metadata.TensorDescriptor, wantVision bool) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
			t.Fatal(err)
		}
		writeClientSafetensorDescriptors(t, dir, inventory)
		got := inferSafetensorsCapabilities(dir, "")
		if slices.Contains(got, "vision") != wantVision {
			t.Fatalf("capabilities = %v, wantVision %t", got, wantVision)
		}
	}
	t.Run("complete compressed tensors contract", func(t *testing.T) { check(t, tensors, true) })
	t.Run("missing required global scale", func(t *testing.T) {
		partial := maps.Clone(tensors)
		delete(partial, base+".weight_global_scale")
		check(t, partial, false)
	})
	t.Run("wrong producer dtypes reject capability and import", func(t *testing.T) {
		checkImport := func(t *testing.T, inventory map[string]gemma4metadata.TensorDescriptor, wantErr string) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
				t.Fatal(err)
			}
			writeClientSafetensorDescriptors(t, dir, inventory)
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			p := progress.NewProgress(io.Discard)
			defer p.Stop()
			err := CreateModel(CreateOptions{ModelName: "wrong-producer-dtype", ModelDir: dir}, p)
			if wantErr == "" && err != nil {
				t.Fatalf("CreateModel() error = %v, want import with vision capability suppressed", err)
			}
			if wantErr != "" && (err == nil || !strings.Contains(err.Error(), wantErr)) {
				t.Fatalf("CreateModel() error = %v, want %q", err, wantErr)
			}
		}
		for _, tc := range []struct {
			name, tensor, importErr string
		}{
			{name: "compressed scale", tensor: base + ".weight_scale"},
			{name: "compressed global", tensor: base + ".weight_global_scale", importErr: "expected F32 tensor"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				wrong := maps.Clone(tensors)
				d := wrong[tc.tensor]
				d.Dtype = "F16"
				wrong[tc.tensor] = d
				check(t, wrong, false)
				checkImport(t, wrong, tc.importErr)
			})
		}

		modelOpt := maps.Clone(tensors)
		delete(modelOpt, base+".weight_packed")
		delete(modelOpt, base+".weight_global_scale")
		modelOpt[base+".weight"] = gemma4metadata.TensorDescriptor{Dtype: "U8", Shape: []int32{16, 8}}
		modelOpt[base+".weight_scale_2"] = gemma4metadata.TensorDescriptor{Dtype: "F32", Shape: nil}
		for _, tc := range []struct {
			name, tensor, importErr string
		}{
			{name: "ModelOpt scale", tensor: base + ".weight_scale"},
			{name: "ModelOpt global", tensor: base + ".weight_scale_2", importErr: "expected F32 tensor"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				wrong := maps.Clone(modelOpt)
				d := wrong[tc.tensor]
				d.Dtype = "F16"
				wrong[tc.tensor] = d
				check(t, wrong, false)
				checkImport(t, wrong, tc.importErr)
			})
		}
	})
}

func TestInferSafetensorsCapabilitiesGemma4ReleasedUnequalHeadGeometry(t *testing.T) {
	const configJSON = `{
		"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4",
		"text_config":{"hidden_size":2560},
		"vision_config":{"hidden_size":768,"intermediate_size":3072,"num_hidden_layers":1,"num_attention_heads":12,"num_key_value_heads":12,"head_dim":64,"rms_norm_eps":1e-6,"default_output_length":280,"patch_size":16,"position_embedding_size":10240,"pooling_kernel_size":3}
	}`
	descriptors := make(map[string]gemma4metadata.TensorDescriptor)
	for _, name := range gemma4ClientVisionTensorNames(1) {
		dtype, shape := gemma4ClientVisionDescriptorForDimensions(name, 768, 3072, 2560, 16, 10240, 64)
		descriptors[name] = gemma4metadata.TensorDescriptor{Dtype: dtype, Shape: shape}
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	writeClientSafetensorDescriptors(t, dir, descriptors)
	if got := inferSafetensorsCapabilities(dir, ""); !slices.Contains(got, "vision") {
		t.Fatalf("released-compatible unequal hidden/head geometry capabilities = %v", got)
	}
}

func TestInferSafetensorsCapabilitiesGemma4RejectsMissingOrZeroTextWidth(t *testing.T) {
	descriptors := make(map[string]gemma4metadata.TensorDescriptor)
	for _, name := range gemma4ClientVisionTensorNames(1) {
		dtype, shape := gemma4ClientVisionDescriptorForDimensions(name, 16, 32, 24, 4, 16, 16)
		descriptors[name] = gemma4metadata.TensorDescriptor{Dtype: dtype, Shape: shape}
	}
	for _, tt := range []struct {
		name, textConfig string
	}{
		{name: "missing"},
		{name: "zero", textConfig: `"text_config":{"hidden_size":0},`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			config := `{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4",` + tt.textConfig + `"vision_config":{"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":16,"default_output_length":1,"patch_size":4,"position_embedding_size":16,"pooling_kernel_size":1}}`
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
				t.Fatal(err)
			}
			writeClientSafetensorDescriptors(t, dir, descriptors)
			if got := inferSafetensorsCapabilities(dir, ""); slices.Contains(got, "vision") {
				t.Fatalf("capabilities = %v, missing/zero text width exposed vision", got)
			}
		})
	}
}

func TestGemma4ModelConfigRejectsNearMatches(t *testing.T) {
	if isGemma4ModelConfig([]string{"NotGemma4ForConditionalGeneration"}, "gemma4-next") {
		t.Fatal("near-match Gemma 4 identifiers classified as Gemma 4")
	}
	if !isGemma4ModelConfig([]string{"Gemma4ForConditionalGeneration"}, "") {
		t.Fatal("released Gemma 4 architecture not classified")
	}
	if !isGemma4ModelConfig([]string{"Gemma4UnifiedForConditionalGeneration"}, "") {
		t.Fatal("unified Gemma 4 architecture not classified")
	}
}

func gemma4ClientVisionTensorNames(layers int) []string {
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

func writeClientSafetensors(t *testing.T, dir string, names ...string) {
	t.Helper()

	tensors := make([]*safetensors.TensorData, 0, len(names))
	for _, name := range names {
		dtype, shape := gemma4ClientVisionDescriptor(name)
		tensors = append(tensors, safetensors.NewTensorDataFromBytes(name, dtype, shape, []byte{0}))
	}

	data, err := io.ReadAll(safetensors.BuildPackedSafetensorsReader(tensors))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeClientSafetensorDescriptors(t *testing.T, dir string, descriptors map[string]gemma4metadata.TensorDescriptor) {
	t.Helper()
	tensors := make([]*safetensors.TensorData, 0, len(descriptors))
	for name, descriptor := range descriptors {
		raw := []byte{0}
		if strings.EqualFold(descriptor.Dtype, "F32") && (len(descriptor.Shape) == 0 || slices.Equal(descriptor.Shape, []int32{1})) {
			raw = []byte{0, 0, 128, 63}
		}
		tensors = append(tensors, safetensors.NewTensorDataFromBytes(name, descriptor.Dtype, descriptor.Shape, raw))
	}
	data, err := io.ReadAll(safetensors.BuildPackedSafetensorsReader(tensors))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func gemma4ClientVisionDescriptor(name string) (string, []int32) {
	return gemma4ClientVisionDescriptorForDimensions(name, 4, 8, 6, 2, 16, 4)
}

func gemma4ClientVisionDescriptorForDimensions(name string, hidden, intermediate, textHidden, patch, positions, headDim int32) (string, []int32) {
	switch {
	case strings.HasSuffix(name, "patch_embedder.input_proj.weight"):
		return "F32", []int32{hidden, 3 * patch * patch}
	case strings.HasSuffix(name, "position_embedding_table"):
		return "F32", []int32{2, positions, hidden}
	case strings.HasSuffix(name, "embed_vision.embedding_projection.weight"):
		return "F32", []int32{textHidden, hidden}
	case strings.Contains(name, ".mlp.gate_proj."), strings.Contains(name, ".mlp.up_proj."):
		return "F32", []int32{intermediate, hidden}
	case strings.Contains(name, ".mlp.down_proj."):
		return "F32", []int32{hidden, intermediate}
	case strings.Contains(name, ".self_attn.q_norm.weight"), strings.Contains(name, ".self_attn.k_norm.weight"):
		return "F32", []int32{headDim}
	case strings.Contains(name, "layernorm.weight"):
		return "F32", []int32{hidden}
	default:
		return "F32", []int32{hidden, hidden}
	}
}

func TestCreateModelfileLayersIncludesParameters(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	layers, err := createModelfileLayers(&ModelfileConfig{
		Parameters: map[string]any{
			"temperature": float32(0.7),
			"stop":        []string{"USER:", "ASSISTANT:"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(layers) != 1 {
		t.Fatalf("len(layers) = %d, want 1", len(layers))
	}

	if layers[0].MediaType != "application/vnd.ollama.image.params" {
		t.Fatalf("MediaType = %q, want %q", layers[0].MediaType, "application/vnd.ollama.image.params")
	}

	blobPath, err := manifest.BlobsPath(layers[0].Digest)
	if err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(blobPath)
	if err != nil {
		t.Fatal(err)
	}

	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}

	if got["temperature"] != float64(0.7) {
		t.Fatalf("temperature = %v, want %v", got["temperature"], float64(0.7))
	}
}

func TestNewManifestWriter_PopulatesFileTypeFromEffectiveQuantize(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	opts := CreateOptions{
		ModelName: "test-quantized",
		ModelDir:  t.TempDir(),
	}

	writer := newManifestWriter(opts, []string{"completion"}, "qwen3", "qwen3")
	class := create.Classification{Kind: create.SourceBlockFP8, Quantize: "mxfp8"}
	if err := writer(opts.ModelName, create.LayerInfo{}, nil, class); err != nil {
		t.Fatalf("newManifestWriter() error = %v", err)
	}

	name := model.ParseName(opts.ModelName)
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatalf("ParseNamedManifest() error = %v", err)
	}

	configPath, err := manifest.BlobsPath(mf.Config.Digest)
	if err != nil {
		t.Fatalf("BlobsPath() error = %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}

	var cfg model.ConfigV2
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}

	if cfg.FileType != "mxfp8" {
		t.Fatalf("FileType = %q, want %q", cfg.FileType, "mxfp8")
	}
}

func TestNewManifestWriter_PopulatesGPTOSSFamily(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{
		"architectures": ["GptOssForCausalLM"],
		"model_type": "gpt_oss"
	}`), 0o644); err != nil {
		t.Fatal(err)
	}

	opts := CreateOptions{ModelName: "gptoss-family-test", ModelDir: modelDir}
	writer := newManifestWriter(opts, []string{"completion", "tools", "thinking"}, "harmony", "")
	if err := writer(opts.ModelName, create.LayerInfo{}, nil, create.Classification{Kind: create.SourcePrequantized, Quantize: "mxfp4"}); err != nil {
		t.Fatalf("newManifestWriter() error = %v", err)
	}

	name := model.ParseName(opts.ModelName)
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	configPath, err := manifest.BlobsPath(mf.Config.Digest)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}

	var cfg model.ConfigV2
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.ModelFamily != "gptoss" {
		t.Fatalf("ModelFamily = %q, want gptoss", cfg.ModelFamily)
	}
	if !slices.Equal(cfg.ModelFamilies, []string{"gptoss"}) {
		t.Fatalf("ModelFamilies = %v, want [gptoss]", cfg.ModelFamilies)
	}
}

func TestNewManifestWriter_PopulatesApertusFamily(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{
		"architectures": ["ApertusForCausalLM"],
		"model_type": "apertus"
	}`), 0o644); err != nil {
		t.Fatal(err)
	}

	opts := CreateOptions{ModelName: "apertus-family-test", ModelDir: modelDir}
	writer := newManifestWriter(opts, []string{"completion", "tools", "thinking"}, "apertus", "apertus")
	if err := writer(opts.ModelName, create.LayerInfo{}, nil, create.Classification{Kind: create.SourceFloat, Quantize: "nvfp4"}); err != nil {
		t.Fatalf("newManifestWriter() error = %v", err)
	}

	name := model.ParseName(opts.ModelName)
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	configPath, err := manifest.BlobsPath(mf.Config.Digest)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}

	var cfg model.ConfigV2
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.ModelFamily != "apertus" {
		t.Fatalf("ModelFamily = %q, want apertus", cfg.ModelFamily)
	}
	if !slices.Equal(cfg.ModelFamilies, []string{"apertus"}) {
		t.Fatalf("ModelFamilies = %v, want [apertus]", cfg.ModelFamilies)
	}
}

func TestNewManifestWriter_PopulatesDraftMetadata(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	draftDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(draftDir, "config.json"), []byte(`{"architectures":["DFlashDraftModel"],"model_type":"qwen3"}`), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	opts := CreateOptions{
		ModelName: "test-draft",
		ModelDir:  t.TempDir(),
		Modelfile: &ModelfileConfig{Draft: draftDir},
	}

	writer := newManifestWriter(opts, []string{"completion"}, "gemma4", "gemma4")
	if err := writer(opts.ModelName, create.LayerInfo{}, nil, create.Classification{}); err != nil {
		t.Fatalf("newManifestWriter() error = %v", err)
	}

	name := model.ParseName(opts.ModelName)
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatalf("ParseNamedManifest() error = %v", err)
	}

	configPath, err := manifest.BlobsPath(mf.Config.Digest)
	if err != nil {
		t.Fatalf("BlobsPath() error = %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}

	var cfg model.ConfigV2
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if cfg.Draft == nil {
		t.Fatal("Draft metadata missing")
	}
	if cfg.Draft.TensorPrefix != "draft." || cfg.Draft.Config != "draft/config.json" {
		t.Fatalf("Draft = %#v, want draft prefix/config", cfg.Draft)
	}
	if cfg.Draft.Architecture != "DFlashDraftModel" {
		t.Fatalf("Draft architecture = %q, want DFlashDraftModel", cfg.Draft.Architecture)
	}
}

func TestDetectCapabilities(t *testing.T) {
	const thinkingTemplate = `{"chat_template": "{%- if '</think>' in content %}{{ content.split('</think>')[-1] }}{%- endif %}<think>\n</think>"}`
	const instructTemplate = `{"chat_template": "{{ '<|im_start|>assistant\n' }}"}`

	tests := []struct {
		name          string
		configJSON    string
		tokenizerJSON string
		want          modelCapabilities
	}{
		{
			name:          "thinking from chat template",
			configJSON:    `{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}`,
			tokenizerJSON: thinkingTemplate,
			want:          modelCapabilities{thinking: true},
		},
		{
			name:          "instruct template has no thinking",
			configJSON:    `{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}`,
			tokenizerJSON: instructTemplate,
			want:          modelCapabilities{thinking: false},
		},
		{
			name:       "plain qwen3 without template has no thinking",
			configJSON: `{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}`,
			want:       modelCapabilities{thinking: false},
		},
		{
			name:          "qwen3.5 moe always thinks without a thinking template",
			configJSON:    `{"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe"}`,
			tokenizerJSON: instructTemplate,
			want:          modelCapabilities{thinking: true},
		},
		{
			name:       "qwen3-next always thinks",
			configJSON: `{"architectures": ["Qwen3NextForCausalLM"]}`,
			want:       modelCapabilities{thinking: true},
		},
		{
			name:       "non-gemma vision config",
			configJSON: `{"architectures": ["SomeVisionModel"], "model_type": "other", "vision_config": {}}`,
			want:       modelCapabilities{vision: true},
		},
		{
			name:       "flat vision flag",
			configJSON: `{"architectures": ["MuseGlimmerForConditionalGeneration"], "model_type": "muse_glimmer", "has_vision": true}`,
			want:       modelCapabilities{vision: true},
		},
		{
			name:       "audio config",
			configJSON: `{"architectures": ["Qwen3OmniForConditionalGeneration"], "audio_config": {}}`,
			want:       modelCapabilities{audio: true},
		},
		{
			name:       "llama has no extra capabilities",
			configJSON: `{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}`,
			want:       modelCapabilities{},
		},
		{
			name:       "apertus uses parser-level thinking, not config-level detection",
			configJSON: `{"architectures": ["ApertusForCausalLM"], "model_type": "apertus"}`,
			want:       modelCapabilities{},
		},
		{
			name:       "invalid config json",
			configJSON: `not json`,
			want:       modelCapabilities{},
		},
		{
			name: "missing files",
			want: modelCapabilities{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if tt.configJSON != "" {
				if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if tt.tokenizerJSON != "" {
				if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(tt.tokenizerJSON), 0o644); err != nil {
					t.Fatal(err)
				}
			}

			if got := detectCapabilities(dir); got != tt.want {
				t.Errorf("detectCapabilities() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestInferSafetensorsCapabilitiesFromParser(t *testing.T) {
	tests := []struct {
		name       string
		parserName string
		want       []string
	}{
		{
			name:       "laguna tools and thinking",
			parserName: "laguna",
			want:       []string{"completion", "tools", "thinking"},
		},
		{
			name:       "poolside tools and thinking",
			parserName: "poolside-v1",
			want:       []string{"completion", "tools", "thinking"},
		},
		{
			name:       "functiongemma tools only",
			parserName: "functiongemma",
			want:       []string{"completion", "tools"},
		},
		{
			name:       "glimmer tools and thinking",
			parserName: "glimmer",
			want:       []string{"completion", "tools", "thinking"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644); err != nil {
				t.Fatal(err)
			}

			if got := inferSafetensorsCapabilities(dir, tt.parserName); !slices.Equal(got, tt.want) {
				t.Fatalf("inferSafetensorsCapabilities() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestInferSafetensorsCapabilitiesGlimmerPreservesVisionMetadata(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{
		"architectures": ["MuseGlimmerForConditionalGeneration"],
		"model_type": "muse_glimmer",
		"has_vision": true
	}`), 0o644); err != nil {
		t.Fatal(err)
	}

	got := inferSafetensorsCapabilities(dir, "glimmer")
	want := []string{"completion", "vision", "tools", "thinking"}
	if !slices.Equal(got, want) {
		t.Fatalf("inferSafetensorsCapabilities() = %#v, want %#v", got, want)
	}
}

func TestInferSafetensorsCapabilitiesLaguna(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures": ["LagunaForCausalLM"], "model_type": "laguna"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	got := inferSafetensorsCapabilities(dir, "laguna")
	for _, want := range []string{"completion", "tools", "thinking"} {
		if !slices.Contains(got, want) {
			t.Fatalf("capabilities %v missing %q", got, want)
		}
	}
	if slices.Contains(got, "vision") || slices.Contains(got, "audio") {
		t.Fatalf("unexpected non-text capability in %v", got)
	}
}

func TestGetParserName(t *testing.T) {
	tests := []struct {
		name       string
		configJSON string
		want       string
	}{
		{
			name:       "qwen3 model",
			configJSON: `{"architectures": ["Qwen3ForCausalLM"]}`,
			want:       "qwen3",
		},
		{
			name:       "qwen3.5 model",
			configJSON: `{"architectures": ["Qwen3_5ForConditionalGeneration"]}`,
			want:       "qwen3.5",
		},
		{
			name:       "deepseek model",
			configJSON: `{"architectures": ["DeepseekV3ForCausalLM"]}`,
			want:       "deepseek3",
		},
		{
			name:       "glm4 model",
			configJSON: `{"architectures": ["GLM4ForCausalLM"]}`,
			want:       "glm-4.7",
		},
		{
			name:       "llama model (no parser)",
			configJSON: `{"architectures": ["LlamaForCausalLM"]}`,
			want:       "",
		},
		{
			name:       "qwen3 via model_type",
			configJSON: `{"model_type": "qwen3"}`,
			want:       "qwen3",
		},
		{
			name:       "gpt-oss architecture",
			configJSON: `{"architectures":["GptOssForCausalLM"],"model_type":"gpt_oss"}`,
			want:       "harmony",
		},
		{
			name:       "gpt-oss model type",
			configJSON: `{"model_type":"gpt-oss"}`,
			want:       "harmony",
		},
		{
			name:       "gpt-oss nested llm model type",
			configJSON: `{"model_type":"wrapper","llm_config":{"model_type":"gpt_oss"}}`,
			want:       "harmony",
		},
		{
			name:       "laguna model",
			configJSON: `{"architectures": ["LagunaForCausalLM"], "model_type": "laguna"}`,
			want:       "laguna",
		},
		{
			name:       "glimmer model",
			configJSON: `{"architectures": ["MuseGlimmerForConditionalGeneration"], "model_type": "muse_glimmer"}`,
			want:       "glimmer",
		},
		{
			name:       "nemotron text architecture",
			configJSON: `{"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h"}`,
			want:       "nemotron-3-nano",
		},
		{
			name:       "nemotron omni architecture",
			configJSON: `{"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"], "model_type": "NemotronH_Nano_Omni_Reasoning_V3"}`,
			want:       "nemotron-3-nano",
		},
		{
			name:       "nemotron nested llm config",
			configJSON: `{"model_type": "nemotron_h_omni", "llm_config": {"model_type": "nemotron_h"}}`,
			want:       "nemotron-3-nano",
		},
		{
			name:       "no config",
			configJSON: `{}`,
			want:       "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644)

			if got := getParserName(dir); got != tt.want {
				t.Errorf("getParserName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestGetRendererName(t *testing.T) {
	tests := []struct {
		name           string
		configJSON     string
		chatTemplate   string
		standaloneOnly bool
		want           string
	}{
		{
			name:       "qwen3 model",
			configJSON: `{"architectures": ["Qwen3ForCausalLM"]}`,
			want:       "qwen3-coder",
		},
		{
			name:       "qwen3.5 model",
			configJSON: `{"architectures": ["Qwen3_5ForConditionalGeneration"]}`,
			want:       "qwen3.5",
		},
		{
			name:         "qwen3.8 embedded template",
			configJSON:   `{"architectures": ["Qwen3_5ForConditionalGeneration"]}`,
			chatTemplate: `{% set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}{% if preserve_thinking %}{% endif %}`,
			want:         "qwen3.8",
		},
		{
			name:           "qwen3.8 standalone template",
			configJSON:     `{"architectures": ["Qwen3_5ForConditionalGeneration"]}`,
			chatTemplate:   `{% set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}{% if preserve_thinking %}{% endif %}`,
			standaloneOnly: true,
			want:           "qwen3.8",
		},
		{
			name:       "deepseek model",
			configJSON: `{"architectures": ["DeepseekV3ForCausalLM"]}`,
			want:       "deepseek3",
		},
		{
			name:       "glm4 model",
			configJSON: `{"architectures": ["GLM4ForCausalLM"]}`,
			want:       "glm-4.7",
		},
		{
			name:       "llama model (no renderer)",
			configJSON: `{"architectures": ["LlamaForCausalLM"]}`,
			want:       "",
		},
		{
			name:       "laguna model",
			configJSON: `{"architectures": ["LagunaForCausalLM"], "model_type": "laguna"}`,
			want:       "laguna",
		},
		{
			name:       "glimmer model",
			configJSON: `{"architectures": ["MuseGlimmerForConditionalGeneration"], "model_type": "muse_glimmer"}`,
			want:       "glimmer",
		},
		{
			name:       "nemotron text architecture",
			configJSON: `{"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h"}`,
			want:       "nemotron-3-nano",
		},
		{
			name:       "nemotron omni architecture",
			configJSON: `{"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"], "model_type": "NemotronH_Nano_Omni_Reasoning_V3"}`,
			want:       "nemotron-3-nano",
		},
		{
			name:       "nemotron nested llm config",
			configJSON: `{"model_type": "nemotron_h_omni", "llm_config": {"model_type": "nemotron_h"}}`,
			want:       "nemotron-3-nano",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644); err != nil {
				t.Fatal(err)
			}
			if tt.chatTemplate != "" {
				if tt.standaloneOnly {
					if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(tt.chatTemplate), 0o644); err != nil {
						t.Fatal(err)
					}
				} else {
					data, err := json.Marshal(map[string]string{"chat_template": tt.chatTemplate})
					if err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), data, 0o644); err != nil {
						t.Fatal(err)
					}
				}
			}

			if got := getRendererName(dir); got != tt.want {
				t.Errorf("getRendererName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestGetLagunaRendererParserName(t *testing.T) {
	tests := []struct {
		name         string
		chatTemplate string
		want         string
	}{
		{
			name:         "v5",
			chatTemplate: `{#- Iteration on laguna_glm_thinking_v5/chat_template.jinja -#}`,
			want:         "laguna",
		},
		{
			name:         "v8",
			chatTemplate: `{#- Iteration on laguna_glm_thinking_v8/chat_template.jinja -#}`,
			want:         "poolside-v1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["LagunaForCausalLM"],"model_type":"laguna"}`), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"chat_template":"{% include 'chat_template.jinja' %}"}`), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(tt.chatTemplate), 0o644); err != nil {
				t.Fatal(err)
			}

			if got := getParserName(dir); got != tt.want {
				t.Errorf("getParserName() = %q, want %q", got, tt.want)
			}
			if got := getRendererName(dir); got != tt.want {
				t.Errorf("getRendererName() = %q, want %q", got, tt.want)
			}
		})
	}
}
