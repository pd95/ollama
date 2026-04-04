// Package client provides client-side model creation for safetensors-based models.
//
// This package is in x/ because the safetensors model storage format is under development.
// It also exists to break an import cycle: server imports x/create, so x/create
// cannot import server. This sub-package can import server because server doesn't
// import it.
package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

type sourceConfigMeta struct {
	Architectures []string `json:"architectures"`
	ModelType     string   `json:"model_type"`
}

var qwen35DefaultParameters = map[string]any{
	"top_k":            int32(20),
	"top_p":            float32(0.95),
	"min_p":            float32(0),
	"presence_penalty": float32(1.5),
	"repeat_penalty":   float32(1),
	"temperature":      float32(1),
}

var gptossDefaultParameters = map[string]any{
	"temperature": float32(1),
}

const gptossDefaultTemplate = `<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {{ currentDate }}
{{- if and .IsThinkSet .Think (ne .ThinkLevel "") }}

Reasoning: {{ .ThinkLevel }}
{{- else if or (not .IsThinkSet) (and .IsThinkSet .Think) }}

Reasoning: medium
{{- end }}

{{- $hasNonBuiltinTools := false }}
{{- if .Tools -}}
{{- $hasBrowserSearch := false }}
{{- $hasBrowserOpen := false }}
{{- $hasBrowserFind := false }}
{{- $hasPython := false }}
  {{- range .Tools }}
    {{- if eq .Function.Name "browser.search" -}}{{- $hasBrowserSearch = true -}}
    {{- else if eq .Function.Name "browser.open" -}}{{- $hasBrowserOpen = true -}}
    {{- else if eq .Function.Name "browser.find" -}}{{- $hasBrowserFind = true -}}
    {{- else if eq .Function.Name "python" -}}{{- $hasPython = true -}}
    {{- else }}{{ $hasNonBuiltinTools = true -}}
    {{- end }}
  {{- end }}
{{- if or $hasBrowserSearch $hasBrowserOpen $hasBrowserFind $hasPython }}

# Tools
{{- if or $hasBrowserSearch $hasBrowserOpen $hasBrowserFind }}

## browser

// Tool for browsing.
// The cursor appears in brackets before each browsing display, like [cursor].
// Cite information from the tool using the line-citation format shown by the browser output.
// Do not quote more than 10 words directly from the tool output.
// sources=web (default: web)
namespace browser {
{{- if $hasBrowserSearch }}

// Searches for information related to query and displays topn results.
type search = (_: {
query: string,
topn?: number, // default: 10
source?: string,
}) => any;
{{- end }}
{{- if $hasBrowserOpen }}

// Opens the link id from the page indicated by cursor starting at line number loc, showing num_lines lines.
// If cursor is not provided, the most recent page is implied.
// If id is a string, it is treated as a fully qualified URL associated with source.
// If loc is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.
// Use this function without id to scroll to a new location of an opened page.
type open = (_: {
id?: number | string, // default: -1
cursor?: number, // default: -1
loc?: number, // default: -1
num_lines?: number, // default: -1
view_source?: boolean, // default: false
source?: string,
}) => any;
{{- end }}
{{- if $hasBrowserFind }}

// Finds exact matches of pattern in the current page, or the page given by cursor.
type find = (_: {
pattern: string,
cursor?: number, // default: -1
}) => any;
{{- end }}

} // namespace browser
{{- end }}{{/* end if has browser tools */}}
{{- if $hasPython }}

## python

Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).

When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.
{{- end }}{{/* end if hasPython */}}
{{- end }}{{/* end if has any built-in tools */}}
{{- end }}{{/* end if .Tools */}}

# Valid channels: analysis, commentary, final. Channel must be included for every message.{{ if $hasNonBuiltinTools }}
Calls to these tools must go to the commentary channel: 'functions'.
{{- end -}}<|end|>{{/* end of system */ -}}
{{- if or $hasNonBuiltinTools .System -}}
<|start|>developer<|message|>{{- if $hasNonBuiltinTools }}# Tools

## functions

namespace functions {
{{- range .Tools }}
{{- if not (or (eq .Function.Name "browser.search") (eq .Function.Name "browser.open") (eq .Function.Name "browser.find") (eq .Function.Name "python")) }}
{{if .Function.Description }}
// {{ .Function.Description }}
{{- end }}
{{- if and .Function.Parameters.Properties (gt (len .Function.Parameters.Properties) 0) }}
type {{ .Function.Name }} = (_: {
{{- range $name, $prop := .Function.Parameters.Properties }}
{{- if $prop.Description }}
  // {{ $prop.Description }}
{{- end }}
  {{ $name }}: {{ $prop | toTypeScriptType }},
{{- end }}
}) => any;
{{- else }}
type {{ .Function.Name }} = () => any;
{{- end }}
{{- end }}{{/* end if not browser tool */}}
{{- end }}{{/* end of range .Tools */}}

} // namespace functions
{{- end }}{{/* end if hasNonBuiltinTools */}}
{{- if .System}}

# Instructions

{{ .System }}
{{- end -}}
<|end|>
{{- end -}}
{{- /* Find the index of the last user message */ -}}
{{- $lastUserIdx := -1 }}
{{- $prefillingContent := false }}
{{- $prefillingThinkingOnly := false }}
{{- range $i, $msg := .Messages }}
  {{- $last := eq (len (slice $.Messages $i)) 1 -}}
  {{- if eq $msg.Role "user" }}
    {{- $lastUserIdx = $i }}
  {{- end -}}
  {{- if and $last (eq $msg.Role "assistant") (gt (len $msg.Content) 0) }}
    {{- $prefillingContent = true }}
  {{- else if and $last (eq $msg.Role "assistant") (gt (len $msg.Thinking) 0) }}
    {{- $prefillingThinkingOnly = true }}
  {{- end }}
{{- end -}}
{{- /* Now render messages */ -}}
{{- range $i, $msg := .Messages }}
  {{- $last := eq (len (slice $.Messages $i)) 1 -}}
  {{- if (ne $msg.Role "system") -}}
    {{- if eq $msg.Role "tool" -}}
      {{- if or (eq $msg.ToolName "python") (eq $msg.ToolName "browser.search") (eq $msg.ToolName "browser.open") (eq $msg.ToolName "browser.find") -}}
        <|start|>{{ $msg.ToolName }} to=assistant<|message|>{{ $msg.Content }}<|end|>
      {{- else -}}
        <|start|>functions.{{ $msg.ToolName }} to=assistant<|message|>{{ $msg.Content }}<|end|>
      {{- end -}}
    {{- else if eq $msg.Role "assistant" -}}
      {{- if and $msg.Thinking (gt $i $lastUserIdx) -}}{{- /* Show thinking only after last user message */ -}}
      <|start|>assistant<|channel|>analysis<|message|>{{ $msg.Thinking }}{{- if not $prefillingThinkingOnly -}}<|end|>{{- end -}}
      {{- end -}}
      {{- if gt (len $msg.Content) 0 -}}
        <|start|>assistant<|channel|>final<|message|>{{ $msg.Content }}{{- if not $prefillingContent -}}<|end|>{{- end -}}
      {{- end -}}
      {{- if gt (len $msg.ToolCalls) 0 -}}
        {{- range $j, $toolCall := $msg.ToolCalls -}}
          {{- $isBuiltin := or (eq $toolCall.Function.Name "python") (eq $toolCall.Function.Name "browser.search") (eq $toolCall.Function.Name "browser.open") (eq $toolCall.Function.Name "browser.find") -}}
          <|start|>assistant<|channel|>{{ if $isBuiltin }}analysis{{ else }}commentary{{ end }} to={{ if not $isBuiltin}}functions.{{end}}{{ $toolCall.Function.Name }} <|constrain|>json<|message|>{{ $toolCall.Function.Arguments }}<|call|>
        {{- end -}}
      {{- end -}}
    {{- else if eq $msg.Role "user" -}}
      <|start|>{{ $msg.Role }}<|message|>{{ $msg.Content }}<|end|>
    {{- end }}
  {{- else }}
  {{- end }}
{{- end -}}
{{- if not (or $prefillingContent $prefillingThinkingOnly) -}}
<|start|>assistant
{{- end -}}`

// MinOllamaVersion is the minimum Ollama version required for safetensors models.
const MinOllamaVersion = "0.19.0"

// ModelfileConfig holds configuration extracted from a Modelfile.
type ModelfileConfig struct {
	Template   string
	System     string
	License    string
	Parser     string
	Renderer   string
	Parameters map[string]any
}

var ignoredModelfileParameters = []string{
	"penalize_newline",
	"low_vram",
	"f16_kv",
	"logits_all",
	"vocab_only",
	"use_mlock",
	"mirostat",
	"mirostat_tau",
	"mirostat_eta",
}

// ConfigFromModelfile extracts the model directory and x/create-specific
// Modelfile configuration from a parsed Modelfile.
func ConfigFromModelfile(modelfile *parser.Modelfile) (string, *ModelfileConfig, error) {
	var modelDir string
	mfConfig := &ModelfileConfig{}

	for _, cmd := range modelfile.Commands {
		switch cmd.Name {
		case "model":
			modelDir = cmd.Args
		case "template":
			mfConfig.Template = cmd.Args
		case "system":
			mfConfig.System = cmd.Args
		case "license":
			mfConfig.License = cmd.Args
		case "parser":
			mfConfig.Parser = cmd.Args
		case "renderer":
			mfConfig.Renderer = cmd.Args
		case "adapter", "message", "requires":
			continue
		default:
			if slices.Contains(ignoredModelfileParameters, cmd.Name) {
				continue
			}

			ps, err := api.FormatParams(map[string][]string{cmd.Name: {cmd.Args}})
			if err != nil {
				return "", nil, err
			}

			if mfConfig.Parameters == nil {
				mfConfig.Parameters = make(map[string]any)
			}

			for k, v := range ps {
				if ks, ok := mfConfig.Parameters[k].([]string); ok {
					mfConfig.Parameters[k] = append(ks, v.([]string)...)
				} else if vs, ok := v.([]string); ok {
					mfConfig.Parameters[k] = vs
				} else {
					mfConfig.Parameters[k] = v
				}
			}
		}
	}

	if modelDir == "" {
		modelDir = "."
	}

	return modelDir, mfConfig, nil
}

// CreateOptions holds all options for model creation.
type CreateOptions struct {
	ModelName string
	ModelDir  string
	Quantize  string           // "int4", "int8", "nvfp4", "mxfp4", or "mxfp8" for quantization
	Modelfile *ModelfileConfig // template/system/license/parser/renderer/parameters from Modelfile
}

// CreateModel imports a model from a local directory.
// This creates blobs and manifest directly on disk, bypassing the HTTP API.
// Automatically detects model type (safetensors LLM vs image gen) and routes accordingly.
func CreateModel(opts CreateOptions, p *progress.Progress) error {
	// Detect model type
	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)
	isImageGen := create.IsTensorModelDir(opts.ModelDir)

	if !isSafetensors && !isImageGen {
		return fmt.Errorf("%s is not a supported model directory (needs config.json + *.safetensors or model_index.json)", opts.ModelDir)
	}

	// Determine model type settings
	var modelType, spinnerKey string
	var capabilities []string
	var parserName, rendererName string
	if isSafetensors {
		modelType = "safetensors model"
		spinnerKey = "create"
		capabilities = inferSafetensorsCapabilities(opts.ModelDir)

		// Set parser and renderer name based on architecture
		parserName = getParserName(opts.ModelDir)
		rendererName = getRendererName(opts.ModelDir)
	} else {
		modelType = "image generation model"
		spinnerKey = "imagegen"
		capabilities = []string{"image"}
	}

	// Set up progress spinner
	statusMsg := "importing " + modelType
	spinner := progress.NewSpinner(statusMsg)
	p.Add(spinnerKey, spinner)

	progressFn := func(msg string) {
		spinner.Stop()
		statusMsg = msg
		spinner = progress.NewSpinner(statusMsg)
		p.Add(spinnerKey, spinner)
	}

	// Create the model using shared callbacks
	var err error
	if isSafetensors {
		err = create.CreateSafetensorsModel(
			opts.ModelName, opts.ModelDir, opts.Quantize,
			newLayerCreator(), newTensorLayerCreator(),
			newManifestWriter(opts, capabilities, parserName, rendererName),
			progressFn,
			newPackedTensorLayerCreator(),
		)
	} else {
		err = create.CreateImageGenModel(
			opts.ModelName, opts.ModelDir, opts.Quantize,
			newLayerCreator(), newTensorLayerCreator(),
			newManifestWriter(opts, capabilities, "", ""),
			progressFn,
		)
	}

	spinner.Stop()
	if err != nil {
		return err
	}

	fmt.Printf("Created %s '%s'\n", modelType, opts.ModelName)
	return nil
}

func inferSafetensorsCapabilities(modelDir string) []string {
	capabilities := []string{"completion"}

	if isGptOssModelDir(modelDir) {
		return []string{"completion", "tools", "thinking"}
	}

	// Qwen3.5 multimodal checkpoints use ConditionalGeneration architectures.
	if supportsVision(modelDir) {
		capabilities = append(capabilities, "vision")
	}

	if supportsThinking(modelDir) {
		capabilities = append(capabilities, "thinking")
	}

	return capabilities
}

func inferredModelFamily(modelDir string) string {
	cfg, ok := readSourceConfigMeta(modelDir)
	if !ok {
		return ""
	}

	if isGptOssConfig(cfg) {
		return "gptoss"
	}

	return ""
}

func isGptOssModelDir(modelDir string) bool {
	cfg, ok := readSourceConfigMeta(modelDir)
	return ok && isGptOssConfig(cfg)
}

func inferredDefaultParameters(modelDir string) map[string]any {
	cfg, ok := readSourceConfigMeta(modelDir)
	if !ok {
		return nil
	}

	if isQwen35Config(cfg) {
		out := make(map[string]any, len(qwen35DefaultParameters))
		for k, v := range qwen35DefaultParameters {
			out[k] = v
		}
		return out
	}

	if isGptOssConfig(cfg) {
		out := make(map[string]any, len(gptossDefaultParameters))
		for k, v := range gptossDefaultParameters {
			out[k] = v
		}
		return out
	}

	return nil
}

func mergedModelfileConfig(modelDir string, mf *ModelfileConfig) *ModelfileConfig {
	defaults := inferredDefaultParameters(modelDir)
	templateDefault := inferredDefaultTemplate(modelDir)
	if len(defaults) == 0 && templateDefault == "" {
		return mf
	}

	var merged ModelfileConfig
	if mf != nil {
		merged = *mf
	}

	if merged.Template == "" {
		merged.Template = templateDefault
	}

	mergedParams := make(map[string]any, len(defaults))
	for k, v := range defaults {
		mergedParams[k] = v
	}
	if mf != nil {
		for k, v := range mf.Parameters {
			mergedParams[k] = v
		}
	}
	merged.Parameters = mergedParams

	return &merged
}

func readSourceConfigMeta(modelDir string) (sourceConfigMeta, bool) {
	configPath := filepath.Join(modelDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return sourceConfigMeta{}, false
	}

	var cfg sourceConfigMeta
	if err := json.Unmarshal(data, &cfg); err != nil {
		return sourceConfigMeta{}, false
	}

	return cfg, true
}

func isQwen35Config(cfg sourceConfigMeta) bool {
	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		if strings.Contains(archLower, "qwen3_5") || strings.Contains(archLower, "qwen3next") {
			return true
		}
	}

	typeLower := strings.ToLower(cfg.ModelType)
	return strings.Contains(typeLower, "qwen3_5") || strings.Contains(typeLower, "qwen3next")
}

func isGptOssConfig(cfg sourceConfigMeta) bool {
	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		if strings.Contains(archLower, "gptoss") || strings.Contains(archLower, "gpt-oss") {
			return true
		}
	}

	typeLower := strings.ToLower(cfg.ModelType)
	return strings.Contains(typeLower, "gptoss") || strings.Contains(typeLower, "gpt-oss")
}

func inferredDefaultTemplate(modelDir string) string {
	cfg, ok := readSourceConfigMeta(modelDir)
	if !ok {
		return ""
	}

	if isGptOssConfig(cfg) {
		return gptossDefaultTemplate
	}

	return ""
}

// newLayerCreator returns a LayerCreator callback for creating config/JSON layers.
func newLayerCreator() create.LayerCreator {
	return func(r io.Reader, mediaType, name string) (create.LayerInfo, error) {
		layer, err := manifest.NewLayer(r, mediaType)
		if err != nil {
			return create.LayerInfo{}, err
		}

		return create.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		}, nil
	}
}

// newTensorLayerCreator returns a QuantizingTensorLayerCreator callback for creating tensor layers.
// When quantize is non-empty, returns multiple layers (weight + scales + optional qbias).
func newTensorLayerCreator() create.QuantizingTensorLayerCreator {
	return func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]create.LayerInfo, error) {
		if quantize != "" {
			return createQuantizedLayers(r, name, dtype, shape, quantize)
		}
		return createUnquantizedLayer(r, name)
	}
}

// createQuantizedLayers quantizes a tensor and returns a single combined layer.
// The combined blob contains data, scale, and optional bias tensors with metadata.
func createQuantizedLayers(r io.Reader, name, dtype string, shape []int32, quantize string) ([]create.LayerInfo, error) {
	if !QuantizeSupported() {
		return nil, fmt.Errorf("quantization requires MLX support")
	}

	// Quantize the tensor into a single combined blob
	blobData, err := quantizeTensor(r, name, dtype, shape, quantize)
	if err != nil {
		return nil, fmt.Errorf("failed to quantize %s: %w", name, err)
	}

	// Create single layer for the combined blob
	layer, err := manifest.NewLayer(bytes.NewReader(blobData), manifest.MediaTypeImageTensor)
	if err != nil {
		return nil, err
	}

	return []create.LayerInfo{
		{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		},
	}, nil
}

// createUnquantizedLayer creates a single tensor layer without quantization.
func createUnquantizedLayer(r io.Reader, name string) ([]create.LayerInfo, error) {
	layer, err := manifest.NewLayer(r, manifest.MediaTypeImageTensor)
	if err != nil {
		return nil, err
	}

	return []create.LayerInfo{
		{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		},
	}, nil
}

// newPackedTensorLayerCreator returns a PackedTensorLayerCreator callback for
// creating packed multi-tensor blob layers (used for expert groups).
func newPackedTensorLayerCreator() create.PackedTensorLayerCreator {
	return func(groupName string, tensors []create.PackedTensorInput) (create.LayerInfo, error) {
		// Check if any tensor in the group needs quantization
		hasQuantize := false
		for _, t := range tensors {
			if t.Quantize != "" {
				hasQuantize = true
				break
			}
		}

		var blobReader io.Reader
		if hasQuantize {
			if !QuantizeSupported() {
				return create.LayerInfo{}, fmt.Errorf("quantization requires MLX support")
			}
			blobData, err := quantizePackedGroup(groupName, tensors)
			if err != nil {
				return create.LayerInfo{}, fmt.Errorf("failed to quantize packed group %s: %w", groupName, err)
			}
			blobReader = bytes.NewReader(blobData)
		} else {
			// Build unquantized packed blob using streaming reader
			// Extract raw tensor data from safetensors-wrapped readers
			var tds []*safetensors.TensorData
			for _, t := range tensors {
				rawData, err := safetensors.ExtractRawFromSafetensors(t.Reader)
				if err != nil {
					return create.LayerInfo{}, fmt.Errorf("failed to extract tensor %s: %w", t.Name, err)
				}
				td := safetensors.NewTensorDataFromBytes(t.Name, t.Dtype, t.Shape, rawData)
				tds = append(tds, td)
			}
			blobReader = safetensors.BuildPackedSafetensorsReader(tds)
		}

		layer, err := manifest.NewLayer(blobReader, manifest.MediaTypeImageTensor)
		if err != nil {
			return create.LayerInfo{}, err
		}

		return create.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      groupName,
		}, nil
	}
}

// newManifestWriter returns a ManifestWriter callback for writing the model manifest.
func newManifestWriter(opts CreateOptions, capabilities []string, parserName, rendererName string) create.ManifestWriter {
	return func(modelName string, config create.LayerInfo, layers []create.LayerInfo) error {
		name := model.ParseName(modelName)
		if !name.IsValid() {
			return fmt.Errorf("invalid model name: %s", modelName)
		}

		// TODO: find a better way to detect image input support
		// For now, hardcode Flux2KleinPipeline as supporting vision (image input)
		caps := capabilities
		modelIndex := filepath.Join(opts.ModelDir, "model_index.json")
		if data, err := os.ReadFile(modelIndex); err == nil {
			var cfg struct {
				ClassName string `json:"_class_name"`
			}
			if json.Unmarshal(data, &cfg) == nil && cfg.ClassName == "Flux2KleinPipeline" {
				caps = append(caps, "vision")
			}
		}

		// Create config blob with version requirement
		effectiveModelfile := mergedModelfileConfig(opts.ModelDir, opts.Modelfile)

		configData := model.ConfigV2{
			ModelFormat:  "safetensors",
			ModelFamily:  inferredModelFamily(opts.ModelDir),
			FileType:     strings.ToLower(strings.TrimSpace(opts.Quantize)),
			Capabilities: caps,
			Requires:     MinOllamaVersion,
			Parser:       resolveParserName(effectiveModelfile, parserName),
			Renderer:     resolveRendererName(effectiveModelfile, rendererName),
		}
		if configData.ModelFamily != "" {
			configData.ModelFamilies = []string{configData.ModelFamily}
		}
		configJSON, err := json.Marshal(configData)
		if err != nil {
			return fmt.Errorf("failed to marshal config: %w", err)
		}

		// Create config layer blob
		configLayer, err := manifest.NewLayer(bytes.NewReader(configJSON), "application/vnd.docker.container.image.v1+json")
		if err != nil {
			return fmt.Errorf("failed to create config layer: %w", err)
		}

		// Convert LayerInfo to manifest.Layer
		manifestLayers := make([]manifest.Layer, 0, len(layers))
		for _, l := range layers {
			manifestLayers = append(manifestLayers, manifest.Layer{
				MediaType: l.MediaType,
				Digest:    l.Digest,
				Size:      l.Size,
				Name:      l.Name,
			})
		}

		// Add Modelfile layers if present
		if effectiveModelfile != nil {
			modelfileLayers, err := createModelfileLayers(effectiveModelfile)
			if err != nil {
				return err
			}
			manifestLayers = append(manifestLayers, modelfileLayers...)
		}

		return manifest.WriteManifest(name, configLayer, manifestLayers)
	}
}

func resolveParserName(mf *ModelfileConfig, inferred string) string {
	if mf != nil && mf.Parser != "" {
		return mf.Parser
	}

	return inferred
}

func resolveRendererName(mf *ModelfileConfig, inferred string) string {
	if mf != nil && mf.Renderer != "" {
		return mf.Renderer
	}

	return inferred
}

// createModelfileLayers creates layers for template, system, and license from Modelfile config.
func createModelfileLayers(mf *ModelfileConfig) ([]manifest.Layer, error) {
	var layers []manifest.Layer

	if mf.Template != "" {
		layer, err := manifest.NewLayer(bytes.NewReader([]byte(mf.Template)), "application/vnd.ollama.image.template")
		if err != nil {
			return nil, fmt.Errorf("failed to create template layer: %w", err)
		}
		layers = append(layers, layer)
	}

	if mf.System != "" {
		layer, err := manifest.NewLayer(bytes.NewReader([]byte(mf.System)), "application/vnd.ollama.image.system")
		if err != nil {
			return nil, fmt.Errorf("failed to create system layer: %w", err)
		}
		layers = append(layers, layer)
	}

	if mf.License != "" {
		layer, err := manifest.NewLayer(bytes.NewReader([]byte(mf.License)), "application/vnd.ollama.image.license")
		if err != nil {
			return nil, fmt.Errorf("failed to create license layer: %w", err)
		}
		layers = append(layers, layer)
	}

	if len(mf.Parameters) > 0 {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(mf.Parameters); err != nil {
			return nil, fmt.Errorf("failed to encode parameters: %w", err)
		}

		layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.params")
		if err != nil {
			return nil, fmt.Errorf("failed to create params layer: %w", err)
		}
		layers = append(layers, layer)
	}

	return layers, nil
}

// supportsThinking checks if the model supports thinking mode based on its architecture.
// This reads the config.json from the model directory and checks the architectures field.
func supportsThinking(modelDir string) bool {
	cfg, ok := readSourceConfigMeta(modelDir)
	if !ok {
		return false
	}

	// Check architectures that support thinking
	thinkingArchitectures := []string{
		"glm4moe",  // GLM-4 MoE models
		"deepseek", // DeepSeek models
		"qwen3",    // Qwen3 models
	}

	// Check the architecture list
	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		for _, thinkArch := range thinkingArchitectures {
			if strings.Contains(archLower, thinkArch) {
				return true
			}
		}
	}

	// Also check model_type
	if cfg.ModelType != "" {
		typeLower := strings.ToLower(cfg.ModelType)
		for _, thinkArch := range thinkingArchitectures {
			if strings.Contains(typeLower, thinkArch) {
				return true
			}
		}
	}

	return false
}

// supportsVision checks if the model supports image input based on its architecture.
// Qwen3.5 multimodal checkpoints are published as ConditionalGeneration architectures.
func supportsVision(modelDir string) bool {
	cfg, ok := readSourceConfigMeta(modelDir)
	if !ok {
		return false
	}

	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		if strings.Contains(archLower, "qwen3") && strings.Contains(archLower, "conditionalgeneration") {
			return true
		}
	}

	typeLower := strings.ToLower(cfg.ModelType)
	return strings.Contains(typeLower, "qwen3") && strings.Contains(typeLower, "conditionalgeneration")
}

// getParserName returns the parser name for a model based on its architecture.
// This reads the config.json from the model directory and determines the appropriate parser.
func getParserName(modelDir string) string {
	cfg, ok := readSourceConfigMeta(modelDir)
	if !ok {
		return ""
	}

	if isQwen35Config(cfg) {
		return "qwen3.5"
	}

	if isGptOssConfig(cfg) {
		return "harmony"
	}

	// Check architectures for known parsers
	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		if strings.Contains(archLower, "glm4") || strings.Contains(archLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(archLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(archLower, "qwen3") {
			return "qwen3"
		}
	}

	// Also check model_type
	if cfg.ModelType != "" {
		typeLower := strings.ToLower(cfg.ModelType)
		if strings.Contains(typeLower, "glm4") || strings.Contains(typeLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(typeLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(typeLower, "qwen3") {
			return "qwen3"
		}
	}

	return ""
}

// getRendererName returns the renderer name for a model based on its architecture.
// This reads the config.json from the model directory and determines the appropriate renderer.
func getRendererName(modelDir string) string {
	cfg, ok := readSourceConfigMeta(modelDir)
	if !ok {
		return ""
	}

	if isQwen35Config(cfg) {
		return "qwen3.5"
	}

	// Check architectures for known renderers
	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		if strings.Contains(archLower, "glm4") || strings.Contains(archLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(archLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(archLower, "qwen3") {
			return "qwen3-coder"
		}
	}

	// Also check model_type
	if cfg.ModelType != "" {
		typeLower := strings.ToLower(cfg.ModelType)
		if strings.Contains(typeLower, "glm4") || strings.Contains(typeLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(typeLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(typeLower, "qwen3") {
			return "qwen3-coder"
		}
	}

	return ""
}
