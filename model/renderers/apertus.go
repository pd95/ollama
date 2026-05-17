package renderers

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	apertusSystemStart    = "<|system_start|>"
	apertusSystemEnd      = "<|system_end|>"
	apertusDeveloperStart = "<|developer_start|>"
	apertusDeveloperEnd   = "<|developer_end|>"
	apertusUserStart      = "<|user_start|>"
	apertusUserEnd        = "<|user_end|>"
	apertusAssistantStart = "<|assistant_start|>"
	apertusAssistantEnd   = "<|assistant_end|>"
	apertusInnerPrefix    = "<|inner_prefix|>"
	apertusInnerSuffix    = "<|inner_suffix|>"
	apertusToolsPrefix    = "<|tools_prefix|>"
	apertusToolsSuffix    = "<|tools_suffix|>"
	apertusImageToken     = "<|image|>"
)

type ApertusRenderer struct{}

func (r *ApertusRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	thinkingEnabled := think != nil && think.Bool()
	var sb strings.Builder
	messageStart := 0
	if len(messages) > 0 && messages[0].Role == "system" {
		sb.WriteString(apertusSystemStart)
		sb.WriteString(r.renderContent(messages[0]))
		sb.WriteString(apertusSystemEnd)
		messageStart = 1
	} else {
		sb.WriteString(apertusSystemStart)
		sb.WriteString("You are Apertus, a helpful assistant created by the SwissAI initiative.\nKnowledge cutoff: 2024-04\nCurrent date: ")
		sb.WriteString(time.Now().Format("2006-01-02"))
		sb.WriteString(apertusSystemEnd)
	}

	sb.WriteString(apertusDeveloperStart)
	if thinkingEnabled {
		sb.WriteString("Deliberation: enabled\n")
	} else {
		sb.WriteString("Deliberation: disabled\n")
	}
	if len(tools) > 0 {
		sb.WriteString("Tool Capabilities:\n")
		renderApertusTools(&sb, tools)
	} else {
		sb.WriteString("Tool Capabilities: disabled")
	}
	sb.WriteString(apertusDeveloperEnd)

	inAssistant := false
	inTool := false
	waitingForToolOutputs := false

	closeToolOutputs := func() {
		if inTool {
			sb.WriteString("]")
			inTool = false
		}
	}
	closeAssistant := func() {
		closeToolOutputs()
		if inAssistant {
			sb.WriteString(apertusAssistantEnd)
			inAssistant = false
		}
	}

	for _, message := range messages[messageStart:] {
		switch message.Role {
		case "user":
			closeAssistant()
			sb.WriteString(apertusUserStart)
			sb.WriteString(r.renderContent(message))
			sb.WriteString(apertusUserEnd)
			waitingForToolOutputs = false
		case "system":
			closeAssistant()
			sb.WriteString(apertusSystemStart)
			sb.WriteString(r.renderContent(message))
			sb.WriteString(apertusSystemEnd)
			waitingForToolOutputs = false
		case "assistant":
			if !inAssistant {
				sb.WriteString(apertusAssistantStart)
				inAssistant = true
			}
			closeToolOutputs()
			if thinkingEnabled && message.Thinking != "" {
				sb.WriteString(apertusInnerPrefix)
				sb.WriteString(message.Thinking)
				sb.WriteString(apertusInnerSuffix)
			}
			sb.WriteString(message.Content)
			if len(message.ToolCalls) > 0 {
				if err := renderApertusToolCalls(&sb, message.ToolCalls); err != nil {
					return "", err
				}
				waitingForToolOutputs = true
			} else {
				waitingForToolOutputs = false
			}
		case "tool":
			if !inAssistant {
				return "", fmt.Errorf("apertus tool message outside assistant turn")
			}
			if !inTool {
				sb.WriteString("[")
				inTool = true
			} else {
				sb.WriteString(", ")
			}
			sb.WriteString(message.Content)
			waitingForToolOutputs = false
		default:
			return "", fmt.Errorf("unsupported apertus message role %q", message.Role)
		}
	}

	closeToolOutputs()
	if inAssistant && !waitingForToolOutputs {
		sb.WriteString(apertusAssistantEnd)
		inAssistant = false
	}

	lastRole := ""
	if len(messages) > 0 {
		lastRole = messages[len(messages)-1].Role
	}
	toolDecisionPrompt := len(tools) > 0 && lastRole == "user"
	if lastRole != "assistant" && !toolDecisionPrompt {
		sb.WriteString(apertusAssistantStart)
	}

	return sb.String(), nil
}

func (r *ApertusRenderer) renderContent(message api.Message) string {
	var sb strings.Builder
	for range message.Images {
		sb.WriteString(apertusImageToken)
	}
	sb.WriteString(message.Content)
	return sb.String()
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
			for _, name := range tool.Function.Parameters.Required {
				required[name] = true
			}
			propIndex := 0
			propCount := tool.Function.Parameters.Properties.Len()
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
				propIndex++
				if propIndex < propCount {
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
			switch v := v.(type) {
			case string:
				b, _ := json.Marshal(v)
				parts = append(parts, string(b))
			default:
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
	var prop api.ToolProperty
	b, err := json.Marshal(items)
	if err != nil || json.Unmarshal(b, &prop) != nil {
		return "any[]"
	}
	inner := apertusTypeScriptType(prop)
	if inner == "object | object" || len(inner) > 50 {
		inner = "any"
	}
	return inner + "[]"
}

func apertusObjectType(properties *api.ToolPropertiesMap, required []string) string {
	requiredSet := make(map[string]bool, len(required))
	for _, name := range required {
		requiredSet[name] = true
	}
	var sb strings.Builder
	sb.WriteString("{\n")
	i := 0
	for name, prop := range properties.All() {
		sb.WriteString(name)
		if !requiredSet[name] {
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

func renderApertusToolCalls(sb *strings.Builder, toolCalls []api.ToolCall) error {
	sb.WriteString(apertusToolsPrefix)
	sb.WriteString("[")
	for i, toolCall := range toolCalls {
		if i > 0 {
			sb.WriteString(", ")
		}
		args, err := json.Marshal(toolCall.Function.Arguments)
		if err != nil {
			return err
		}
		name, err := json.Marshal(toolCall.Function.Name)
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
