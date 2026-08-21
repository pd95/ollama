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

	"golang.org/x/mod/semver"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	modelparsers "github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/create"
	imagemanifest "github.com/ollama/ollama/x/imagegen/manifest"
	apertusmetadata "github.com/ollama/ollama/x/models/apertus/metadata"
	gemma4metadata "github.com/ollama/ollama/x/models/gemma4/metadata"
	"github.com/ollama/ollama/x/quant"
)

// MinOllamaVersion is the minimum Ollama version required for safetensors models.
const MinOllamaVersion = "0.19.0"

// ModelfileConfig holds configuration extracted from a Modelfile.
type ModelfileConfig struct {
	Template   string
	System     string
	License    string
	Draft      string
	Parser     string
	Renderer   string
	Requires   string
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
		case "draft":
			mfConfig.Draft = cmd.Args
		case "parser":
			mfConfig.Parser = cmd.Args
		case "renderer":
			mfConfig.Renderer = cmd.Args
		case "requires":
			requires := cmd.Args
			if !strings.HasPrefix(requires, "v") {
				requires = "v" + requires
			}
			if !semver.IsValid(requires) {
				return "", nil, fmt.Errorf("requires must be a valid semver (e.g. 0.14.0)")
			}
			minVersion := "v" + MinOllamaVersion
			if semver.Compare(requires, minVersion) < 0 {
				return "", nil, fmt.Errorf("requires %s is below the minimum supported version %s for safetensors models", strings.TrimPrefix(requires, "v"), MinOllamaVersion)
			}
			mfConfig.Requires = strings.TrimPrefix(requires, "v")
		case "adapter", "message":
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
	ModelName     string
	ModelDir      string
	Quantize      string           // "int4", "int8", "nvfp4", "mxfp4", or "mxfp8" for quantization
	DraftQuantize string           // optional quantization level for draft model tensors
	Modelfile     *ModelfileConfig // template/system/license/parser/renderer/parameters from Modelfile
	BaseConfig    *model.ConfigV2
}

// CreateModel imports a model from a local directory.
// This creates blobs and manifest directly on disk, bypassing the HTTP API.
// Automatically detects model type (safetensors LLM vs image gen) and routes accordingly.
func CreateModel(opts CreateOptions, p *progress.Progress) error {
	// Detect model type
	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)
	hasDraft := opts.Modelfile != nil && opts.Modelfile.Draft != ""
	isBaseModelWithDraft := hasDraft && !isSafetensors && create.IsSafetensorsLLMModel(opts.ModelDir)
	if opts.DraftQuantize != "" && !hasDraft {
		return fmt.Errorf("--draft-quantize requires a DRAFT model")
	}
	if opts.Quantize != "" && !quant.Creatable(opts.Quantize) {
		return fmt.Errorf("unsupported --quantize %q: supported types are int4, int8, nvfp4, mxfp4, mxfp8", opts.Quantize)
	}
	if opts.DraftQuantize != "" && !quant.Creatable(opts.DraftQuantize) {
		return fmt.Errorf("unsupported --draft-quantize %q: supported types are int4, int8, nvfp4, mxfp4, mxfp8", opts.DraftQuantize)
	}

	if !isSafetensors && !isBaseModelWithDraft {
		return fmt.Errorf("%s is not a supported safetensors model directory (needs config.json + *.safetensors)", opts.ModelDir)
	}

	if hasDraft && !create.IsSafetensorsModelDir(opts.Modelfile.Draft) {
		return fmt.Errorf("draft %s is not a supported safetensors model directory", opts.Modelfile.Draft)
	}

	modelType := "safetensors model"
	spinnerKey := "create"
	var capabilities []string
	var parserName, rendererName string
	if isSafetensors {
		apertusVariant := detectApertus1p1Variant(opts.ModelDir)
		if err := validateApertus1p1Variant(apertusVariant); err != nil {
			return err
		}
		parserName = getParserNameForApertusVariant(opts.ModelDir, apertusVariant)
		rendererName = getRendererNameForApertusVariant(opts.ModelDir, apertusVariant)
		capabilities = inferSafetensorsCapabilities(opts.ModelDir, resolveParserName(opts.Modelfile, parserName))
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

	var draftLayers []create.LayerInfo
	var err error

	if hasDraft {
		draftLayers, err = create.CreateDraftLayers(
			opts.Modelfile.Draft,
			"draft.",
			"draft/",
			opts.DraftQuantize,
			create.StoreFromLayerCreator(newLayerCreator()),
			progressFn,
		)
		if err != nil {
			spinner.Stop()
			return err
		}
	}

	if isBaseModelWithDraft {
		err = createModelFromBaseWithDraft(opts, draftLayers, progressFn)
		spinner.Stop()
		if err != nil {
			return err
		}
		fmt.Printf("Created safetensors model '%s'\n", opts.ModelName)
		return nil
	}

	// Create the model through the x/create pipeline (read → classify → plan
	// → write), supplying blob storage and manifest assembly.
	writer := newManifestWriter(opts, capabilities, parserName, rendererName)
	if len(draftLayers) > 0 {
		writer = appendLayersManifestWriter(writer, draftLayers)
	}
	err = create.Create(
		opts.ModelName, opts.ModelDir, opts.Quantize,
		create.StoreFromLayerCreator(newLayerCreator()),
		writer,
		progressFn,
	)

	spinner.Stop()
	if err != nil {
		return err
	}

	fmt.Printf("Created %s '%s'\n", modelType, opts.ModelName)
	return nil
}

func appendLayersManifestWriter(next create.ManifestWriter, extra []create.LayerInfo) create.ManifestWriter {
	return func(modelName string, config create.LayerInfo, layers []create.LayerInfo, class create.Classification) error {
		layers = append(layers, extra...)
		return next(modelName, config, layers, class)
	}
}

func draftMetadata(draftDir string) (*model.Draft, error) {
	configPath := filepath.Join(draftDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read draft config %s: %w", configPath, err)
	}

	var cfg struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse draft config %s: %w", configPath, err)
	}

	arch := ""
	if len(cfg.Architectures) > 0 {
		arch = cfg.Architectures[0]
	}
	if arch == "" {
		arch = cfg.ModelType
	}
	if arch == "" {
		return nil, fmt.Errorf("draft architecture not found in %s", configPath)
	}

	return &model.Draft{
		ModelFormat:  "safetensors",
		Architecture: arch,
		TensorPrefix: "draft.",
		Config:       "draft/config.json",
	}, nil
}

func createModelFromBaseWithDraft(opts CreateOptions, draftLayers []create.LayerInfo, progressFn func(string)) error {
	progressFn(fmt.Sprintf("loading base model %s", opts.ModelDir))
	baseManifest, err := imagemanifest.LoadManifest(opts.ModelDir)
	if err != nil {
		return err
	}

	baseConfig, err := readConfigV2(baseManifest)
	if err != nil {
		return err
	}
	opts.BaseConfig = baseConfig

	configLayer := baseManifest.GetConfigLayer("config.json")
	if configLayer == nil {
		return fmt.Errorf("base model %s does not contain config.json", opts.ModelDir)
	}

	layers := make([]create.LayerInfo, 0, len(baseManifest.Manifest.Layers)+len(draftLayers))
	for _, layer := range baseManifest.Manifest.Layers {
		layers = append(layers, create.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      layer.Name,
		})
	}
	layers = append(layers, draftLayers...)

	progressFn(fmt.Sprintf("writing manifest for %s", opts.ModelName))
	return newManifestWriter(opts, baseConfig.Capabilities, baseConfig.Parser, baseConfig.Renderer)(
		opts.ModelName,
		create.LayerInfo{
			Digest:    configLayer.Digest,
			Size:      configLayer.Size,
			MediaType: configLayer.MediaType,
			Name:      configLayer.Name,
		},
		layers,
		create.Classification{Quantize: quant.Canonical(opts.Quantize)},
	)
}

func readConfigV2(m *imagemanifest.ModelManifest) (*model.ConfigV2, error) {
	data, err := os.ReadFile(m.BlobPath(m.Manifest.Config.Digest))
	if err != nil {
		return nil, fmt.Errorf("failed to read base config: %w", err)
	}

	var cfg model.ConfigV2
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse base config: %w", err)
	}
	return &cfg, nil
}

func inferSafetensorsCapabilities(modelDir, parserName string) []string {
	capabilities := []string{"completion"}
	caps := detectCapabilities(modelDir)
	if caps.vision {
		capabilities = append(capabilities, "vision")
	}

	if caps.audio {
		capabilities = append(capabilities, "audio")
	}

	var builtinParser modelparsers.Parser
	if parserName != "" {
		builtinParser = modelparsers.ParserForName(parserName)
	}

	if builtinParser != nil && builtinParser.HasToolSupport() {
		capabilities = append(capabilities, "tools")
	}

	if caps.thinking || (builtinParser != nil && builtinParser.HasThinkingSupport()) {
		capabilities = append(capabilities, "thinking")
	}

	return capabilities
}

func detectApertus1p5MediaCapabilities(modelDir string) modelCapabilities {
	inv, err := create.ReadInventory(modelDir)
	if err != nil {
		return modelCapabilities{}
	}
	return apertus1p5MediaCapabilities(inv)
}

func apertus1p5MediaCapabilities(inv create.Inventory) modelCapabilities {
	cfg, err := apertusmetadata.ParseConfig(inv.RawConfig)
	if err != nil {
		return modelCapabilities{}
	}
	descriptors := make(map[string]apertusmetadata.TensorDescriptor, len(inv.Tensors))
	for name, tensor := range inv.Tensors {
		descriptors[name] = apertusmetadata.TensorDescriptor{Dtype: tensor.Dtype, Shape: slices.Clone(tensor.Shape)}
	}
	vision := apertusmetadata.ValidateVisionInventory(cfg, descriptors) == nil
	audio := apertusmetadata.ValidateAudioInventory(cfg, descriptors) == nil
	return modelCapabilities{vision: vision, audio: audio}
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

// newManifestWriter returns a ManifestWriter callback for writing the model manifest.
func newManifestWriter(opts CreateOptions, capabilities []string, parserName, rendererName string) create.ManifestWriter {
	return func(modelName string, config create.LayerInfo, layers []create.LayerInfo, class create.Classification) error {
		name := model.ParseName(modelName)
		if !name.IsValid() {
			return fmt.Errorf("invalid model name: %s", modelName)
		}

		// Create config blob with version requirement.
		configData := model.ConfigV2{}
		if opts.BaseConfig != nil {
			configData = *opts.BaseConfig
		}
		configData.ModelFormat = "safetensors"
		if class.Quantize != "" || configData.FileType == "" {
			configData.FileType = class.Quantize
		}
		configData.Capabilities = capabilities
		configData.Requires = MinOllamaVersion
		if opts.Modelfile != nil && opts.Modelfile.Requires != "" {
			configData.Requires = opts.Modelfile.Requires
		}
		configData.Parser = resolveParserName(opts.Modelfile, parserName)
		configData.Renderer = resolveRendererName(opts.Modelfile, rendererName)
		if opts.Modelfile != nil && opts.Modelfile.Draft != "" {
			draft, err := draftMetadata(opts.Modelfile.Draft)
			if err != nil {
				return err
			}
			configData.Draft = draft
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
		if opts.Modelfile != nil {
			modelfileLayers, err := createModelfileLayers(opts.Modelfile)
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

// modelCapabilities holds the input-modality and reasoning capabilities a model
// advertises, inferred from its source metadata.
type modelCapabilities struct {
	vision   bool
	audio    bool
	thinking bool
}

// detectCapabilities reads the model directory once and reports the vision,
// audio, and thinking capabilities it can infer.
func detectCapabilities(modelDir string) modelCapabilities {
	var cfg struct {
		Architectures []string        `json:"architectures"`
		ModelType     string          `json:"model_type"`
		VisionConfig  *map[string]any `json:"vision_config"`
		AudioConfig   *map[string]any `json:"audio_config"`
		HasVision     bool            `json:"has_vision"`
		SoundConfig   *map[string]any `json:"sound_config"`
	}
	if data, err := os.ReadFile(filepath.Join(modelDir, "config.json")); err == nil {
		_ = json.Unmarshal(data, &cfg)
	}

	vision := cfg.VisionConfig != nil || cfg.HasVision
	audio := cfg.AudioConfig != nil || cfg.SoundConfig != nil
	if isGemma4ModelConfig(cfg.Architectures, cfg.ModelType) {
		vision, audio = gemma4ModelDirMediaCapabilities(modelDir)
	} else if isApertus1p5ModelConfig(cfg.Architectures, cfg.ModelType) {
		media := detectApertus1p5MediaCapabilities(modelDir)
		vision, audio = media.vision, media.audio
	}

	return modelCapabilities{
		vision: vision,
		audio:  audio,
		thinking: chatTemplateHasThinkingSupport(readChatTemplate(modelDir)) ||
			alwaysSupportsThinking(cfg.Architectures, cfg.ModelType),
	}
}

func isApertus1p5ModelConfig(architectures []string, modelType string) bool {
	for _, value := range append(architectures, modelType) {
		value = strings.ToLower(value)
		if strings.Contains(value, "apertus1p5") || strings.Contains(value, "apertus-1.5") || strings.Contains(value, "apertus_1_5") {
			return true
		}
	}
	return false
}

func isGemma4ModelConfig(architectures []string, modelType string) bool {
	for _, arch := range architectures {
		if isGemma4ModelIdentifier(arch) {
			return true
		}
	}
	return isGemma4ModelIdentifier(modelType)
}

func isGemma4ModelIdentifier(value string) bool {
	switch strings.ToLower(value) {
	case "gemma4", "gemma4_unified",
		"gemma4forcausallm", "gemma4forconditionalgeneration",
		"gemma4unifiedforcausallm", "gemma4unifiedforconditionalgeneration":
		return true
	default:
		return false
	}
}

func gemma4ModelDirMediaCapabilities(modelDir string) (vision, audio bool) {
	inv, err := create.ReadInventory(modelDir)
	if err != nil {
		return false, false
	}
	var cfg gemma4metadata.ConfigFile
	if err := json.Unmarshal(inv.RawConfig, &cfg); err != nil {
		return false, false
	}
	tensors := make(map[string]gemma4metadata.TensorDescriptor, len(inv.Tensors))
	for name, tensor := range inv.Tensors {
		// Unified vision is encoder-free, so dispatch and readiness depend on
		// descriptor shapes rather than the released tower sentinel alone.
		tensors[name] = gemma4metadata.TensorDescriptor{Dtype: tensor.Dtype, Shape: slices.Clone(tensor.Shape)}
	}
	processorData, processorErr := os.ReadFile(filepath.Join(modelDir, "processor_config.json"))
	tokenizerConfigData, tokenizerErr := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json"))
	tokenizerData, tokenizerDataErr := os.ReadFile(filepath.Join(modelDir, "tokenizer.json"))
	audioReady := processorErr == nil && tokenizerErr == nil && tokenizerDataErr == nil &&
		gemma4metadata.ValidateAudioRuntimeMetadata(cfg, processorData, tokenizerConfigData, tokenizerData) == nil &&
		gemma4metadata.ValidateAudioSourceInventory(cfg, tensors) == nil
	return gemma4metadata.ValidateVisionSourceInventory(cfg, tensors) == nil, audioReady
}

// readChatTemplate returns the model's chat template, preferring the
// chat_template field of tokenizer_config.json and falling back to a standalone
// chat_template.jinja. It returns "" when neither is present.
func readChatTemplate(modelDir string) string {
	if data, err := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json")); err == nil {
		var cfg struct {
			ChatTemplate string `json:"chat_template"`
		}
		if json.Unmarshal(data, &cfg) == nil && cfg.ChatTemplate != "" {
			return cfg.ChatTemplate
		}
	}
	if data, err := os.ReadFile(filepath.Join(modelDir, "chat_template.jinja")); err == nil {
		return string(data)
	}
	return ""
}

// readApertus1p1ChatTemplate distinguishes an absent template (the official
// Mini base checkpoint) from a present but empty template (an incomplete or
// malformed Instruct download).
func readApertus1p1ChatTemplate(modelDir string) (string, bool) {
	if data, err := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json")); err == nil {
		var cfg map[string]json.RawMessage
		if json.Unmarshal(data, &cfg) != nil {
			return "", true
		}
		if raw, ok := cfg["chat_template"]; ok && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			var template string
			if json.Unmarshal(raw, &template) != nil {
				return "", true
			}
			return template, true
		}
	}
	if data, err := os.ReadFile(filepath.Join(modelDir, "chat_template.jinja")); err == nil {
		return string(data), true
	}
	return "", false
}

// chatTemplateHasThinkingSupport reports whether a chat template emits thinking
// blocks. Copied from server.chatTemplateHasThinkingSupport so this package need
// not depend on the server package for an eight-line string check.
func chatTemplateHasThinkingSupport(chatTemplate string) bool {
	if strings.Contains(chatTemplate, "<think>") && strings.Contains(chatTemplate, "</think>") {
		return true
	}

	// Some Qwen/DeepSeek templates strip prior reasoning by splitting assistant
	// content at </think>; llama.cpp can still extract reasoning from them.
	return (strings.Contains(chatTemplate, "content.split('</think>')") ||
		strings.Contains(chatTemplate, `content.split("</think>")`)) &&
		!strings.Contains(chatTemplate, "reasoning_content") &&
		!strings.Contains(chatTemplate, "<SPECIAL_12>")
}

func alwaysSupportsThinking(architectures []string, modelType string) bool {
	if isQwen35Family(modelType) {
		return true
	}
	for _, arch := range architectures {
		if isQwen35Family(arch) {
			return true
		}
	}
	return false
}

func isQwen35Family(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "qwen3_5") || strings.Contains(s, "qwen3next")
}

func isGPTOSSFamily(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "gptoss") || strings.Contains(s, "gpt_oss") || strings.Contains(s, "gpt-oss")
}

func isApertusFamily(s string) bool {
	switch strings.ToLower(s) {
	case "apertus", "apertusforcausallm":
		return true
	default:
		return false
	}
}

func isApertus1p5Family(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "apertus1p5") || strings.Contains(s, "apertus-1.5") || strings.Contains(s, "apertus_1_5")
}

func qwen35RendererName(modelDir string) string {
	template := readChatTemplate(modelDir)
	if strings.Contains(template, "resolved_reasoning_effort") &&
		strings.Contains(template, "preserve_thinking") {
		return "qwen3.8"
	}

	return "qwen3.5"
}

func lagunaRendererParserName(modelDir string) string {
	const poolsideV1Marker = "laguna_glm_thinking_v8"

	if strings.Contains(readChatTemplate(modelDir), poolsideV1Marker) {
		return "poolside-v1"
	}

	// Poolside's tokenizer config includes the standalone template by name
	// rather than embedding it, so inspect that file as well.
	if data, err := os.ReadFile(filepath.Join(modelDir, "chat_template.jinja")); err == nil &&
		strings.Contains(string(data), poolsideV1Marker) {
		return "poolside-v1"
	}

	return "laguna"
}

func nemotronRendererParserName(modelDir string) string {
	const v35Marker = "{reasoning effort: efficient}"

	// Nemotron 3.5 publishes its updated template as a standalone file while
	// tokenizer_config.json can retain the older template, so inspect both.
	if data, err := os.ReadFile(filepath.Join(modelDir, "chat_template.jinja")); err == nil &&
		strings.Contains(string(data), v35Marker) {
		return "nemotron-3.5-nano"
	}
	if strings.Contains(readChatTemplate(modelDir), v35Marker) {
		return "nemotron-3.5-nano"
	}

	return "nemotron-3-nano"
}

type apertus1p1Variant int

const (
	apertus1p1NotMini apertus1p1Variant = iota
	apertus1p1Base
	apertus1p1Instruct
	apertus1p1Invalid
)

func validateApertus1p1Metadata(modelDir string) error {
	return validateApertus1p1Variant(detectApertus1p1Variant(modelDir))
}

func validateApertus1p1Variant(variant apertus1p1Variant) error {
	if variant == apertus1p1Invalid {
		return fmt.Errorf("apertus v1.1 Mini metadata is incomplete: tokenizer.json and any chat template must contain <SPECIAL_61> through <SPECIAL_72>")
	}
	return nil
}

func detectApertus1p1Variant(modelDir string) apertus1p1Variant {
	data, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return apertus1p1NotMini
	}
	var cfg struct {
		Architectures         []string `json:"architectures"`
		ModelType             string   `json:"model_type"`
		MaxPositionEmbeddings int32    `json:"max_position_embeddings"`
		RopeTheta             float64  `json:"rope_theta"`
		RopeScaling           struct {
			RopeType string `json:"rope_type"`
			Type     string `json:"type"`
		} `json:"rope_scaling"`
	}
	if json.Unmarshal(data, &cfg) != nil {
		return apertus1p1NotMini
	}
	isApertus := strings.EqualFold(cfg.ModelType, "apertus")
	for _, arch := range cfg.Architectures {
		if strings.EqualFold(arch, "ApertusForCausalLM") {
			isApertus = true
			break
		}
	}
	ropeType := cfg.RopeScaling.RopeType
	if ropeType == "" {
		ropeType = cfg.RopeScaling.Type
	}
	if ropeType == "" {
		ropeType = "default"
	}
	if !isApertus || cfg.MaxPositionEmbeddings != 4096 || cfg.RopeTheta != 500000 ||
		(!strings.EqualFold(ropeType, "default") && !strings.EqualFold(ropeType, "linear")) {
		return apertus1p1NotMini
	}

	tokenizerData, err := os.ReadFile(filepath.Join(modelDir, "tokenizer.json"))
	if err != nil || !hasApertus1p1SpecialTokenSignature(string(tokenizerData)) {
		return apertus1p1Invalid
	}
	template, hasTemplate := readApertus1p1ChatTemplate(modelDir)
	if hasTemplate {
		if !hasApertus1p1SpecialTokenSignature(template) {
			return apertus1p1Invalid
		}
		return apertus1p1Instruct
	}
	return apertus1p1Base
}

func hasApertus1p1SpecialTokenSignature(value string) bool {
	for id := 61; id <= 72; id++ {
		if !strings.Contains(value, fmt.Sprintf("<SPECIAL_%d>", id)) {
			return false
		}
	}
	return true
}

// getParserName returns the parser name for a model based on its architecture.
// This reads the config.json from the model directory and determines the appropriate parser.
func getParserName(modelDir string) string {
	return getParserNameForApertusVariant(modelDir, detectApertus1p1Variant(modelDir))
}

func getParserNameForApertusVariant(modelDir string, variant apertus1p1Variant) string {
	switch variant {
	case apertus1p1Instruct:
		return "apertus1p1"
	case apertus1p1Base:
		return ""
	case apertus1p1Invalid:
		return ""
	}

	configPath := filepath.Join(modelDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}

	var cfg struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
		LLMConfig     struct {
			ModelType string `json:"model_type"`
		} `json:"llm_config"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return ""
	}

	for _, arch := range cfg.Architectures {
		if name := parserNameForIdentifier(modelDir, arch); name != "" {
			return name
		}
	}
	for _, modelType := range []string{cfg.ModelType, cfg.LLMConfig.ModelType} {
		if name := parserNameForIdentifier(modelDir, modelType); name != "" {
			return name
		}
	}

	return ""
}

func parserNameForIdentifier(modelDir, s string) string {
	s = strings.ToLower(s)
	switch {
	case isApertusFamily(s) || isApertus1p5Family(s):
		return "apertus"
	case strings.HasPrefix(s, "museglimmer") || s == "muse_glimmer":
		return "glimmer"
	case strings.Contains(s, "laguna"):
		return lagunaRendererParserName(modelDir)
	case strings.Contains(s, "cohere2moe") || strings.Contains(s, "cohere2_moe"):
		return "cohere"
	case isGPTOSSFamily(s):
		return "harmony"
	case strings.Contains(s, "glm4") || strings.Contains(s, "glm-4"):
		return "glm-4.7"
	case strings.Contains(s, "deepseek"):
		return "deepseek3"
	case strings.Contains(s, "gemma4"):
		return "gemma4"
	case isQwen35Family(s):
		return "qwen3.5"
	case strings.Contains(s, "qwen3"):
		return "qwen3"
	// Nemotron-H publishes NemotronHForCausalLM for text and
	// NemotronH_Nano_Omni_Reasoning_V3 for omni; model_type is nemotron_h,
	// nemotron_h_moe, or the omni name. The two stems cover all of them.
	case strings.Contains(s, "nemotronh") || strings.Contains(s, "nemotron_h"):
		return nemotronRendererParserName(modelDir)
	default:
		return ""
	}
}

// getRendererName returns the renderer name for a model based on its architecture.
// This reads the config.json from the model directory and determines the appropriate renderer.
func getRendererName(modelDir string) string {
	return getRendererNameForApertusVariant(modelDir, detectApertus1p1Variant(modelDir))
}

func getRendererNameForApertusVariant(modelDir string, variant apertus1p1Variant) string {
	switch variant {
	case apertus1p1Instruct:
		return "apertus1p1"
	case apertus1p1Base:
		return ""
	case apertus1p1Invalid:
		return ""
	}

	configPath := filepath.Join(modelDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}

	var cfg struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
		LLMConfig     struct {
			ModelType string `json:"model_type"`
		} `json:"llm_config"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return ""
	}

	for _, arch := range cfg.Architectures {
		if name := rendererNameForIdentifier(modelDir, arch); name != "" {
			return name
		}
	}
	for _, modelType := range []string{cfg.ModelType, cfg.LLMConfig.ModelType} {
		if name := rendererNameForIdentifier(modelDir, modelType); name != "" {
			return name
		}
	}

	return ""
}

func rendererNameForIdentifier(modelDir, s string) string {
	s = strings.ToLower(s)
	switch {
	case isApertus1p5Family(s):
		return "apertus1p5"
	case isApertusFamily(s):
		return "apertus"
	case strings.HasPrefix(s, "museglimmer") || s == "muse_glimmer":
		return "glimmer"
	case strings.Contains(s, "laguna"):
		return lagunaRendererParserName(modelDir)
	case strings.Contains(s, "cohere2moe") || strings.Contains(s, "cohere2_moe"):
		return "cohere"
	case strings.Contains(s, "gemma4"):
		return "gemma4"
	case strings.Contains(s, "glm4") || strings.Contains(s, "glm-4"):
		return "glm-4.7"
	case strings.Contains(s, "deepseek"):
		return "deepseek3"
	case isQwen35Family(s):
		return qwen35RendererName(modelDir)
	case strings.Contains(s, "qwen3"):
		return "qwen3-coder"
	// Nemotron-H publishes NemotronHForCausalLM for text and
	// NemotronH_Nano_Omni_Reasoning_V3 for omni; model_type is nemotron_h,
	// nemotron_h_moe, or the omni name. The two stems cover all of them.
	case strings.Contains(s, "nemotronh") || strings.Contains(s, "nemotron_h"):
		return nemotronRendererParserName(modelDir)
	default:
		return ""
	}
}
