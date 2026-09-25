package create

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	apertusmetadata "github.com/ollama/ollama/mlxrunner/model/apertus/metadata"
	modelparsers "github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
)

// inferSafetensorsConfig derives the manifest config shared by local and
// server-side safetensors imports. Explicit parser and renderer values take
// precedence over the inferred values.
func inferSafetensorsConfig(modelDir string, cfg sourceModelConfig, parserOverride, rendererOverride string) (model.ConfigV2, error) {
	chatTemplate, err := readChatTemplateStrict(modelDir)
	if err != nil {
		return model.ConfigV2{}, err
	}

	apertusVariant := detectApertus1p1Variant(modelDir, cfg)
	if err := validateApertus1p1Variant(apertusVariant); err != nil {
		return model.ConfigV2{}, err
	}
	var parserName string
	if apertusVariant == apertus1p1Instruct {
		parserName = "apertus1p1"
	} else if apertusVariant != apertus1p1Base {
		parserName, err = parserNameForConfig(modelDir, cfg, chatTemplate)
		if err != nil {
			return model.ConfigV2{}, err
		}
	}
	if parserOverride != "" {
		parserName = parserOverride
	}

	var rendererName string
	if apertusVariant == apertus1p1Instruct {
		rendererName = "apertus1p1"
	} else if apertusVariant != apertus1p1Base {
		rendererName, err = rendererNameForConfig(modelDir, cfg, chatTemplate)
		if err != nil {
			return model.ConfigV2{}, err
		}
	}
	if rendererOverride != "" {
		rendererName = rendererOverride
	}

	capabilities := inferSafetensorsCapabilitiesFromConfig(cfg, chatTemplate, parserName)
	if sourceConfigHasApertus1p5(cfg) {
		vision, audio := false, false
		if inv, err := ReadInventory(modelDir); err == nil {
			vision, audio = apertus1p5MediaCapabilities(inv)
		}
		capabilities = []string{"completion"}
		if vision {
			capabilities = append(capabilities, "vision")
		}
		if audio {
			capabilities = append(capabilities, "audio")
		}
		capabilities = append(capabilities, "tools", "thinking")
	}
	modelFamily := inferModelFamilyFromConfig(cfg)
	generationDefaults, err := readHFGenerationDefaults(modelDir)
	if err != nil {
		return model.ConfigV2{}, err
	}

	return model.ConfigV2{
		ModelFormat:        "safetensors",
		ModelFamily:        modelFamily,
		ModelFamilies:      modelFamilies(modelFamily),
		Parser:             parserName,
		Renderer:           rendererName,
		Capabilities:       capabilities,
		GenerationDefaults: generationDefaults,
	}, nil
}

func modelFamilies(family string) []string {
	if family == "" {
		return nil
	}
	return []string{family}
}

func inferModelFamilyFromConfig(cfg sourceModelConfig) string {
	for _, id := range sourceConfigIdentifiers(cfg) {
		if isApertusFamily(id) || isApertus1p5Family(id) {
			return "apertus"
		}
		if isGPTOSSFamily(id) {
			return "gptoss"
		}
	}
	return ""
}

func isGPTOSSFamily(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "gptoss") || strings.Contains(s, "gpt_oss") || strings.Contains(s, "gpt-oss")
}

func isApertusFamily(s string) bool {
	s = strings.ToLower(s)
	return s == "apertus" || s == "apertusforcausallm"
}

func isApertus1p0SourceConfig(cfg sourceModelConfig, parserName string) bool {
	if parserName != "apertus" {
		return false
	}
	for _, id := range sourceConfigIdentifiers(cfg) {
		if isApertusFamily(id) {
			return true
		}
	}
	return false
}

func readHFGenerationDefaults(modelDir string) (model.GenerationDefaults, error) {
	path := filepath.Join(modelDir, "generation_config.json")
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	} else if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}

	defaults, err := model.ParseHFGenerationDefaults(data)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return defaults, nil
}

func readChatTemplateStrict(modelDir string) (string, error) {
	tokenizerConfig := filepath.Join(modelDir, "tokenizer_config.json")
	if data, err := os.ReadFile(tokenizerConfig); err == nil {
		var cfg struct {
			ChatTemplate string `json:"chat_template"`
		}
		if err := json.Unmarshal(data, &cfg); err != nil {
			return "", fmt.Errorf("parse %s: %w", tokenizerConfig, err)
		}
		if cfg.ChatTemplate != "" {
			return cfg.ChatTemplate, nil
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("read %s: %w", tokenizerConfig, err)
	}

	chatTemplatePath := filepath.Join(modelDir, "chat_template.jinja")
	data, err := os.ReadFile(chatTemplatePath)
	if err == nil {
		return string(data), nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	return "", fmt.Errorf("read %s: %w", chatTemplatePath, err)
}

func inferSafetensorsCapabilitiesFromConfig(cfg sourceModelConfig, chatTemplate, parserName string) []string {
	capabilities := []string{"completion"}

	caps := detectCapabilitiesFromConfig(cfg, chatTemplate)
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
	if caps.thinking || (builtinParser != nil && builtinParser.HasThinkingSupport() && !isApertus1p0SourceConfig(cfg, parserName)) {
		capabilities = append(capabilities, "thinking")
	}

	return capabilities
}

func sourceConfigHasApertus1p5(cfg sourceModelConfig) bool {
	for _, id := range sourceConfigIdentifiers(cfg) {
		if isApertus1p5Family(id) {
			return true
		}
	}
	return false
}

func apertus1p5MediaCapabilities(inv Inventory) (vision, audio bool) {
	cfg, err := apertusmetadata.ParseConfig(inv.RawConfig)
	if err != nil {
		return false, false
	}
	descriptors := make(map[string]apertusmetadata.TensorDescriptor, len(inv.Tensors))
	for name, tensor := range inv.Tensors {
		descriptors[name] = apertusmetadata.TensorDescriptor{Dtype: tensor.Dtype, Shape: slices.Clone(tensor.Shape)}
	}
	return apertusmetadata.ValidateVisionInventory(cfg, descriptors) == nil,
		apertusmetadata.ValidateAudioInventory(cfg, descriptors) == nil
}

type modelCapabilities struct {
	vision   bool
	audio    bool
	thinking bool
}

func detectCapabilitiesFromConfig(cfg sourceModelConfig, chatTemplate string) modelCapabilities {
	return modelCapabilities{
		vision: cfg.VisionConfig != nil || cfg.HasVision,
		audio:  cfg.AudioConfig != nil || cfg.SoundConfig != nil,
		thinking: thinking.TemplateSupportsThinking(chatTemplate) ||
			alwaysSupportsThinking(cfg.Architectures, cfg.ModelType),
	}
}

func alwaysSupportsThinking(architectures []string, modelType string) bool {
	if isQwen35Family(modelType) || isQwen4Family(modelType) {
		return true
	}
	for _, arch := range architectures {
		if isQwen35Family(arch) || isQwen4Family(arch) {
			return true
		}
	}
	return false
}

func isQwen35Family(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "qwen3_5") || strings.Contains(s, "qwen3next")
}

func isQwen4Family(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "qwen4exp") || strings.Contains(s, "qwen4_exp")
}

func isApertus1p5Family(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "apertus1p5") || strings.Contains(s, "apertus-1.5") || strings.Contains(s, "apertus_1_5")
}

type apertus1p1Variant uint8

const (
	apertus1p1NotMini apertus1p1Variant = iota
	apertus1p1Base
	apertus1p1Instruct
	apertus1p1Invalid
)

func validateApertus1p1Variant(variant apertus1p1Variant) error {
	if variant == apertus1p1Invalid {
		return fmt.Errorf("apertus v1.1 Mini metadata is incomplete: tokenizer.json and any chat template must contain <SPECIAL_61> through <SPECIAL_72>")
	}
	return nil
}

func detectApertus1p1Variant(modelDir string, cfg sourceModelConfig) apertus1p1Variant {
	isApertus := false
	for _, id := range sourceConfigIdentifiers(cfg) {
		if isApertusFamily(id) {
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
	template, present := readApertus1p1ChatTemplate(modelDir)
	if !present {
		return apertus1p1Base
	}
	if !hasApertus1p1SpecialTokenSignature(template) {
		return apertus1p1Invalid
	}
	return apertus1p1Instruct
}

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

func hasApertus1p1SpecialTokenSignature(value string) bool {
	for id := 61; id <= 72; id++ {
		if !strings.Contains(value, fmt.Sprintf("<SPECIAL_%d>", id)) {
			return false
		}
	}
	return true
}

func qwen35RendererNameFromTemplate(chatTemplate string) string {
	if strings.Contains(chatTemplate, "resolved_reasoning_effort") &&
		strings.Contains(chatTemplate, "preserve_thinking") {
		return "qwen3.8"
	}
	return "qwen3.5"
}

func lagunaRendererParserNameFromTemplate(modelDir, chatTemplate string) (string, error) {
	const poolsideV1Marker = "laguna_glm_thinking_v8"

	if strings.Contains(chatTemplate, poolsideV1Marker) {
		return "poolside-v1", nil
	}

	chatTemplatePath := filepath.Join(modelDir, "chat_template.jinja")
	data, err := os.ReadFile(chatTemplatePath)
	if err == nil && strings.Contains(string(data), poolsideV1Marker) {
		return "poolside-v1", nil
	}
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("read %s: %w", chatTemplatePath, err)
	}
	return "laguna", nil
}

func nemotronRendererParserNameFromTemplate(modelDir, chatTemplate string) (string, error) {
	const v35Marker = "{reasoning effort: efficient}"

	chatTemplatePath := filepath.Join(modelDir, "chat_template.jinja")
	data, err := os.ReadFile(chatTemplatePath)
	if err == nil && strings.Contains(string(data), v35Marker) {
		return "nemotron-3.5-nano", nil
	}
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("read %s: %w", chatTemplatePath, err)
	}
	if strings.Contains(chatTemplate, v35Marker) {
		return "nemotron-3.5-nano", nil
	}
	return "nemotron-3-nano", nil
}

func sourceConfigIdentifiers(cfg sourceModelConfig) []string {
	ids := append([]string(nil), cfg.Architectures...)
	return append(ids, cfg.ModelType, cfg.LLMConfig.ModelType)
}

func parserNameForConfig(modelDir string, cfg sourceModelConfig, chatTemplate string) (string, error) {
	for _, id := range sourceConfigIdentifiers(cfg) {
		name, err := parserNameForIdentifier(modelDir, id, chatTemplate)
		if err != nil || name != "" {
			return name, err
		}
	}
	return "", nil
}

func parserNameForIdentifier(modelDir, s, chatTemplate string) (string, error) {
	s = strings.ToLower(s)
	switch {
	case isApertusFamily(s), isApertus1p5Family(s):
		return "apertus", nil
	case strings.HasPrefix(s, "museglimmer") || s == "muse_glimmer":
		return "glimmer", nil
	case strings.Contains(s, "laguna"):
		return lagunaRendererParserNameFromTemplate(modelDir, chatTemplate)
	case strings.Contains(s, "cohere2moe") || strings.Contains(s, "cohere2_moe"):
		return "cohere", nil
	case isGPTOSSFamily(s):
		return "harmony", nil
	case strings.Contains(s, "glm4") || strings.Contains(s, "glm-4"):
		return "glm-4.7", nil
	case strings.Contains(s, "deepseek"):
		return "deepseek3", nil
	case strings.Contains(s, "gemma4"):
		return "gemma4", nil
	case isQwen4Family(s), isQwen35Family(s):
		return "qwen3.5", nil
	case strings.Contains(s, "qwen3"):
		return "qwen3", nil
	case strings.Contains(s, "nemotronh") || strings.Contains(s, "nemotron_h"):
		return nemotronRendererParserNameFromTemplate(modelDir, chatTemplate)
	default:
		return "", nil
	}
}

func rendererNameForConfig(modelDir string, cfg sourceModelConfig, chatTemplate string) (string, error) {
	for _, id := range sourceConfigIdentifiers(cfg) {
		name, err := rendererNameForIdentifier(modelDir, id, chatTemplate)
		if err != nil || name != "" {
			return name, err
		}
	}
	return "", nil
}

func rendererNameForIdentifier(modelDir, s, chatTemplate string) (string, error) {
	s = strings.ToLower(s)
	switch {
	case isApertus1p5Family(s):
		return "apertus1p5", nil
	case isApertusFamily(s):
		return "apertus", nil
	case strings.HasPrefix(s, "museglimmer") || s == "muse_glimmer":
		return "glimmer", nil
	case strings.Contains(s, "laguna"):
		return lagunaRendererParserNameFromTemplate(modelDir, chatTemplate)
	case strings.Contains(s, "cohere2moe") || strings.Contains(s, "cohere2_moe"):
		return "cohere", nil
	case strings.Contains(s, "gemma4"):
		return "gemma4", nil
	case strings.Contains(s, "glm4") || strings.Contains(s, "glm-4"):
		return "glm-4.7", nil
	case strings.Contains(s, "deepseek"):
		return "deepseek3", nil
	case isQwen4Family(s):
		return "qwen3.8", nil
	case isQwen35Family(s):
		return qwen35RendererNameFromTemplate(chatTemplate), nil
	case strings.Contains(s, "qwen3"):
		return "qwen3-coder", nil
	case strings.Contains(s, "nemotronh") || strings.Contains(s, "nemotron_h"):
		return nemotronRendererParserNameFromTemplate(modelDir, chatTemplate)
	default:
		return "", nil
	}
}
