package renderers

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	apertusSystemStart     = "<|system_start|>"
	apertusSystemEnd       = "<|system_end|>"
	apertusDeveloperStart  = "<|developer_start|>"
	apertusDeveloperEnd    = "<|developer_end|>"
	apertusUserStart       = "<|user_start|>"
	apertusUserEnd         = "<|user_end|>"
	apertusAssistantStart  = "<|assistant_start|>"
	apertusAssistantEnd    = "<|assistant_end|>"
	apertusToolsPrefix     = "<|tools_prefix|>"
	apertusToolsSuffix     = "<|tools_suffix|>"
	apertusImageToken      = "<|image|>"
	apertusInnerOpenTag    = "<|inner_prefix|>"
	apertusInnerCloseTag   = "<|inner_suffix|>"
	apertusToolOutputStart = "<|tool_output_start|>"
	apertusToolOutputEnd   = "<|tool_output_end|>"
	maxApertusSchemaDepth  = 32
	maxApertusSchemaNodes  = 4096
)

type (
	ApertusRenderer    struct{}
	Apertus1p5Renderer struct{}
)

func (r *ApertusRenderer) LeadingBOS() string { return "" }

func (r *ApertusRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	return renderApertus(messages, tools, think, false)
}

func (r *Apertus1p5Renderer) LeadingBOS() string {
	return ""
}

func (r *Apertus1p5Renderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	return renderApertus(messages, tools, think, true)
}

func renderApertus(messages []api.Message, tools []api.Tool, think *api.ThinkValue, v1p5 bool) (string, error) {
	thinkingEnabled := think != nil && think.Bool()
	if err := validateApertusTools(tools); err != nil {
		return "", err
	}
	declaredTools := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		declaredTools[tool.Function.Name] = struct{}{}
	}
	for _, message := range messages {
		if err := validateApertusText(message.Content); err != nil {
			return "", err
		}
		if err := validateApertusText(message.Thinking); err != nil {
			return "", err
		}
	}
	if v1p5 && thinkingEnabled && len(tools) > 0 {
		return "", fmt.Errorf("Apertus 1.5 does not support tool calling with thinking enabled")
	}
	var sb strings.Builder
	start := 0
	if len(messages) > 0 && messages[0].Role == "system" {
		sb.WriteString(apertusSystemStart)
		sb.WriteString(renderApertusContent(messages[0]))
		sb.WriteString(apertusSystemEnd)
		start = 1
	} else {
		sb.WriteString(apertusSystemStart)
		if v1p5 {
			sb.WriteString("You are Apertus 1.5 Omni, a multimodal assistant developed by the Swiss AI Initiative. Extended from Apertus 1 via continued pretraining, you understand images and audio and respond in text.")
		} else {
			sb.WriteString("You are Apertus, a helpful assistant created by the SwissAI initiative.\nKnowledge cutoff: 2024-04\nCurrent date: ")
			sb.WriteString(time.Now().Format("2006-01-02"))
		}
		sb.WriteString(apertusSystemEnd)
	}
	sb.WriteString(apertusDeveloperStart)
	if thinkingEnabled {
		sb.WriteString("Deliberation: enabled\n")
	} else {
		sb.WriteString("Deliberation: disabled\n")
	}
	if len(tools) == 0 {
		sb.WriteString("Tool Capabilities: disabled")
	} else {
		sb.WriteString("Tool Capabilities:\n")
		renderApertusTools(&sb, tools)
	}
	sb.WriteString(apertusDeveloperEnd)
	inAssistant, inTool := false, false
	pendingToolResults := 0
	toolResultsStarted := false
	closeAssistant := func() {
		if inTool {
			if v1p5 {
				sb.WriteString(apertusToolOutputEnd)
			} else {
				sb.WriteString("]")
			}
			inTool = false
		}
		if inAssistant {
			sb.WriteString(apertusAssistantEnd)
			inAssistant = false
		}
	}
	for _, message := range messages[start:] {
		switch message.Role {
		case "user", "system":
			if pendingToolResults > 0 && toolResultsStarted {
				return "", fmt.Errorf("apertus tool results are incomplete")
			}
			closeAssistant()
			if message.Role == "user" {
				sb.WriteString(apertusUserStart)
				sb.WriteString(renderApertusContent(message))
				sb.WriteString(apertusUserEnd)
			} else {
				sb.WriteString(apertusSystemStart)
				sb.WriteString(renderApertusContent(message))
				sb.WriteString(apertusSystemEnd)
			}
			pendingToolResults = 0
			toolResultsStarted = false
		case "assistant":
			if pendingToolResults > 0 && toolResultsStarted {
				return "", fmt.Errorf("apertus tool results are incomplete")
			}
			pendingToolResults = 0
			toolResultsStarted = false
			if !inAssistant {
				sb.WriteString(apertusAssistantStart)
				inAssistant = true
			}
			if inTool {
				if v1p5 {
					sb.WriteString(apertusToolOutputEnd)
				} else {
					sb.WriteString("]")
				}
				inTool = false
			}
			if thinkingEnabled && message.Thinking != "" {
				sb.WriteString(apertusInnerOpenTag)
				sb.WriteString(message.Thinking)
				sb.WriteString(apertusInnerCloseTag)
			}
			sb.WriteString(message.Content)
			if len(message.ToolCalls) > 0 {
				if err := validateApertusHistoricalCalls(message.ToolCalls, declaredTools); err != nil {
					return "", err
				}
				if err := renderApertusToolCalls(&sb, message.ToolCalls); err != nil {
					return "", err
				}
				pendingToolResults = len(message.ToolCalls)
			}
		case "tool":
			if !inAssistant || pendingToolResults == 0 {
				return "", fmt.Errorf("apertus tool message does not follow an assistant tool call")
			}
			if !inTool {
				if v1p5 {
					sb.WriteString(apertusToolOutputStart)
				} else {
					sb.WriteString("[")
				}
				inTool = true
			} else {
				sb.WriteString(", ")
			}
			sb.WriteString(message.Content)
			toolResultsStarted = true
			pendingToolResults--
		default:
			return "", fmt.Errorf("unsupported apertus message role %q", message.Role)
		}
	}
	if inTool {
		if v1p5 {
			sb.WriteString(apertusToolOutputEnd)
		} else {
			sb.WriteString("]")
		}
	}
	if inAssistant && pendingToolResults == 0 {
		sb.WriteString(apertusAssistantEnd)
	}
	last := ""
	if len(messages) > 0 {
		last = messages[len(messages)-1].Role
	}
	toolDecisionPrompt := !v1p5 && len(tools) > 0 && last == "user"
	if last != "assistant" && !toolDecisionPrompt {
		sb.WriteString(apertusAssistantStart)
	}
	return sb.String(), nil
}

func validateApertusHistoricalCalls(calls []api.ToolCall, declared map[string]struct{}) error {
	if len(declared) == 0 {
		return fmt.Errorf("apertus assistant tool calls require declarations")
	}
	for _, call := range calls {
		name := call.Function.Name
		if _, ok := declared[name]; !ok {
			return fmt.Errorf("undeclared apertus assistant tool %q", name)
		}
	}
	return nil
}

func renderApertusContent(message api.Message) string {
	return strings.Repeat(apertusImageToken, len(message.Images)) + message.Content
}

func renderApertusTools(sb *strings.Builder, tools []api.Tool) {
	for i, tool := range tools {
		if tool.Function.Description != "" {
			sb.WriteString("// ")
			sb.WriteString(tool.Function.Description)
			sb.WriteString("\n")
		}
		sb.WriteString("type ")
		sb.WriteString(tool.Function.Name)
		if tool.Function.Parameters.Properties == nil || tool.Function.Parameters.Properties.Len() == 0 {
			sb.WriteString(" = () => any;")
		} else {
			sb.WriteString(" = (_: {\n")
			required := make(map[string]bool, len(tool.Function.Parameters.Required))
			for _, n := range tool.Function.Parameters.Required {
				required[n] = true
			}
			n := 0
			total := tool.Function.Parameters.Properties.Len()
			for name, prop := range tool.Function.Parameters.Properties.All() {
				if prop.Description != "" {
					sb.WriteString("// ")
					sb.WriteString(prop.Description)
					sb.WriteString("\n")
				}
				sb.WriteString(name)
				if !required[name] {
					sb.WriteString("?")
				}
				sb.WriteString(": ")
				sb.WriteString(apertusTypeScriptType(prop))
				n++
				if n < total {
					sb.WriteString(",\n")
				} else {
					sb.WriteString("\n")
				}
			}
			sb.WriteString("}) => any;")
		}
		if i < len(tools)-1 {
			sb.WriteString("\n")
		}
	}
}

func apertusTypeScriptType(prop api.ToolProperty) string {
	if len(prop.AnyOf) > 0 {
		parts := make([]string, 0, len(prop.AnyOf))
		for _, p := range prop.AnyOf {
			parts = append(parts, apertusTypeScriptType(p))
		}
		return strings.Join(parts, " | ")
	}
	if len(prop.Enum) > 0 {
		parts := make([]string, 0, len(prop.Enum))
		for _, v := range prop.Enum {
			if s, ok := v.(string); ok {
				b, _ := json.Marshal(s)
				parts = append(parts, string(b))
			} else {
				parts = append(parts, fmt.Sprint(v))
			}
		}
		return strings.Join(parts, " | ")
	}
	typ := "any"
	if len(prop.Type) > 0 {
		typ = prop.Type[0]
	}
	switch typ {
	case "array":
		return apertusArrayType(prop.Items)
	case "integer", "number":
		return "number"
	case "boolean":
		return "boolean"
	case "string":
		return "string"
	case "object":
		if prop.Properties != nil && prop.Properties.Len() > 0 {
			return apertusObjectType(prop.Properties, prop.Required)
		}
		return "object"
	default:
		if len(prop.Type) > 1 {
			parts := make([]string, 0, len(prop.Type))
			for _, t := range prop.Type {
				parts = append(parts, apertusTypeScriptType(api.ToolProperty{Type: api.PropertyType{t}}))
			}
			return strings.Join(parts, " | ")
		}
		return "any"
	}
}

func apertusArrayType(items any) string {
	if items == nil {
		return "any[]"
	}
	var p api.ToolProperty
	b, err := json.Marshal(items)
	if err != nil || json.Unmarshal(b, &p) != nil {
		return "any[]"
	}
	inner := apertusTypeScriptType(p)
	if inner == "object | object" || len(inner) > 50 {
		inner = "any"
	}
	return inner + "[]"
}

func apertusObjectType(properties *api.ToolPropertiesMap, required []string) string {
	set := make(map[string]bool, len(required))
	for _, n := range required {
		set[n] = true
	}
	var sb strings.Builder
	sb.WriteString("{\n")
	i := 0
	for name, prop := range properties.All() {
		sb.WriteString(name)
		if !set[name] {
			sb.WriteString("?")
		}
		sb.WriteString(": ")
		sb.WriteString(apertusTypeScriptType(prop))
		i++
		if i < properties.Len() {
			sb.WriteString(", ")
		}
	}
	sb.WriteString("}")
	return sb.String()
}

func renderApertusToolCalls(sb *strings.Builder, calls []api.ToolCall) error {
	sb.WriteString(apertusToolsPrefix)
	sb.WriteString("[")
	for i, call := range calls {
		if !apertusIdentifier(call.Function.Name) {
			return fmt.Errorf("invalid apertus tool name %q", call.Function.Name)
		}
		if i > 0 {
			sb.WriteString(", ")
		}
		args, err := json.Marshal(call.Function.Arguments)
		if err != nil {
			return err
		}
		name, err := json.Marshal(call.Function.Name)
		if err != nil {
			return err
		}
		sb.WriteString("{")
		sb.Write(name)
		sb.WriteString(": ")
		sb.Write(args)
		sb.WriteString("}")
	}
	sb.WriteString("]")
	sb.WriteString(apertusToolsSuffix)
	return nil
}

func validateApertusTools(tools []api.Tool) error {
	names := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		name := tool.Function.Name
		if !apertusIdentifier(name) {
			return fmt.Errorf("invalid apertus tool name %q", name)
		}
		if _, ok := names[name]; ok {
			return fmt.Errorf("duplicate apertus tool name %q", name)
		}
		names[name] = struct{}{}
		if err := validateApertusText(tool.Function.Description); err != nil {
			return err
		}
		budget := maxApertusSchemaNodes
		if err := validateApertusSchema(tool.Function.Parameters.Properties, tool.Function.Parameters.Required, 0, &budget); err != nil {
			return fmt.Errorf("invalid apertus tool %q schema: %w", name, err)
		}
	}
	return nil
}

func validateApertusSchema(props *api.ToolPropertiesMap, required []string, depth int, budget *int) error {
	if depth > maxApertusSchemaDepth {
		return fmt.Errorf("schema exceeds depth %d", maxApertusSchemaDepth)
	}
	declared := make(map[string]struct{})
	if props != nil {
		for name, prop := range props.All() {
			if !apertusIdentifier(name) {
				return fmt.Errorf("invalid property name %q", name)
			}
			declared[name] = struct{}{}
			if err := validateApertusProperty(prop, depth+1, budget); err != nil {
				return err
			}
		}
	}
	for _, name := range required {
		if _, ok := declared[name]; !ok {
			return fmt.Errorf("required property %q is not declared", name)
		}
	}
	return nil
}

func validateApertusProperty(prop api.ToolProperty, depth int, budget *int) error {
	if depth > maxApertusSchemaDepth {
		return fmt.Errorf("schema exceeds depth %d", maxApertusSchemaDepth)
	}
	if *budget == 0 {
		return fmt.Errorf("schema exceeds %d nodes", maxApertusSchemaNodes)
	}
	*budget--
	if err := validateApertusText(prop.Description); err != nil {
		return err
	}
	if err := validateApertusSchema(prop.Properties, prop.Required, depth+1, budget); err != nil {
		return err
	}
	for _, nested := range prop.AnyOf {
		if err := validateApertusProperty(nested, depth+1, budget); err != nil {
			return err
		}
	}
	if prop.Items != nil {
		b, err := json.Marshal(prop.Items)
		if err != nil {
			return err
		}
		var nested api.ToolProperty
		if err := json.Unmarshal(b, &nested); err == nil {
			if err := validateApertusProperty(nested, depth+1, budget); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateApertusText(s string) error {
	for _, token := range []string{apertusSystemStart, apertusSystemEnd, apertusDeveloperStart, apertusDeveloperEnd, apertusUserStart, apertusUserEnd, apertusAssistantStart, apertusAssistantEnd, apertusToolsPrefix, apertusToolsSuffix, apertusImageToken, apertusInnerOpenTag, apertusInnerCloseTag} {
		if strings.Contains(s, token) {
			return fmt.Errorf("apertus content contains reserved token %q", token)
		}
	}
	return nil
}

func apertusIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i := range len(s) {
		c := s[i]
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '$' || (i > 0 && c >= '0' && c <= '9')) {
			return false
		}
	}
	return true
}
