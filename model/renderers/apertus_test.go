package renderers

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestApertusRendererNoTools(t *testing.T) {
	got, err := (&ApertusRenderer{}).Render([]api.Message{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Hello"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{
		"<|system_start|>Be concise.<|system_end|>",
		"<|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|>",
		"<|user_start|>Hello<|user_end|>",
		"<|assistant_start|>",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered prompt missing %q:\n%s", want, got)
		}
	}
}

func TestApertusRendererToolsOmitGenerationPromptForToolDecision(t *testing.T) {
	got, err := (&ApertusRenderer{}).Render([]api.Message{
		{Role: "user", Content: "What is the weather in Zurich?"},
	}, []api.Tool{apertusWeatherTool()}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{
		"Tool Capabilities:\n// Get current weather.",
		"type get_weather = (_: {\n",
		"// City name.",
		"location: string",
		"unit?: \"celsius\" | \"fahrenheit\"",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered prompt missing %q:\n%s", want, got)
		}
	}
	if strings.HasSuffix(got, "<|assistant_start|>") {
		t.Fatalf("tool decision prompt should not append assistant generation prompt:\n%s", got)
	}
}

func TestApertusRendererAssistantToolCallAndOutput(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("location", "Zurich")

	got, err := (&ApertusRenderer{}).Render([]api.Message{
		{Role: "user", Content: "Weather?"},
		{Role: "assistant", ToolCalls: []api.ToolCall{{
			Function: api.ToolCallFunction{Name: "get_weather", Arguments: args},
		}}},
		{Role: "tool", Content: `{"temperature":22}`},
	}, []api.Tool{apertusWeatherTool()}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(got, `<|tools_prefix|>[{"get_weather": {"location":"Zurich"}}]<|tools_suffix|>`) {
		t.Fatalf("rendered prompt missing tool call:\n%s", got)
	}
	if !strings.Contains(got, `[{"temperature":22}]<|assistant_end|><|assistant_start|>`) {
		t.Fatalf("rendered prompt missing tool output response continuation:\n%s", got)
	}
}

func TestApertusRendererThinkingEnabled(t *testing.T) {
	think := &api.ThinkValue{Value: true}
	got, err := (&ApertusRenderer{}).Render([]api.Message{
		{Role: "user", Content: "Question"},
		{Role: "assistant", Thinking: "Need a short answer.", Content: "Answer."},
	}, nil, think)
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{
		"<|developer_start|>Deliberation: enabled\nTool Capabilities: disabled<|developer_end|>",
		"<|assistant_start|><|inner_prefix|>Need a short answer.<|inner_suffix|>Answer.<|assistant_end|>",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered prompt missing %q:\n%s", want, got)
		}
	}
}

func TestApertusRendererThinkingHistorySuppressedWhenDisabled(t *testing.T) {
	got, err := (&ApertusRenderer{}).Render([]api.Message{
		{Role: "user", Content: "Question"},
		{Role: "assistant", Thinking: "hidden", Content: "Visible."},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(got, "<|inner_prefix|>") || strings.Contains(got, "hidden") {
		t.Fatalf("thinking history should be omitted when thinking is disabled:\n%s", got)
	}
}

func TestApertusRendererMatchesHFChatTemplate(t *testing.T) {
	if os.Getenv("VERIFY_APERTUS_HF_TEMPLATE") == "" {
		t.Skip("set VERIFY_APERTUS_HF_TEMPLATE=1 to compare against the local Apertus HF chat template")
	}
	modelDir := os.Getenv("APERTUS_HF_MODEL_DIR")
	if modelDir == "" {
		modelDir = filepath.Join("..", "..", "..", "models", "Apertus-8B-Instruct-2509")
	}
	var err error
	modelDir, err = filepath.Abs(modelDir)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name                string
		messages            []api.Message
		tools               []api.Tool
		addGenerationPrompt bool
		enableThinking      bool
	}{
		{
			name: "plain_chat",
			messages: []api.Message{
				{Role: "system", Content: "Be concise."},
				{Role: "user", Content: "Hello"},
			},
			addGenerationPrompt: true,
		},
		{
			name: "tool_decision",
			messages: []api.Message{
				{Role: "system", Content: "Use tools when appropriate."},
				{Role: "user", Content: "What is the weather in Zurich?"},
			},
			tools:               []api.Tool{apertusWeatherTool()},
			addGenerationPrompt: false,
		},
		{
			name: "tool_output_continuation",
			messages: []api.Message{
				{Role: "system", Content: "Use tools when appropriate."},
				{Role: "user", Content: "Weather?"},
				apertusAssistantToolCall("get_weather", map[string]any{"location": "Zurich"}),
				{Role: "tool", Content: `{"temperature":22}`},
			},
			tools:               []api.Tool{apertusWeatherTool()},
			addGenerationPrompt: true,
		},
		{
			name: "thinking_developer_flag",
			messages: []api.Message{
				{Role: "system", Content: "Think if useful."},
				{Role: "user", Content: "Hello"},
			},
			addGenerationPrompt: true,
			enableThinking:      true,
		},
	}

	renderer := &ApertusRenderer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var think *api.ThinkValue
			if tt.enableThinking {
				think = &api.ThinkValue{Value: true}
			}
			got, err := renderer.Render(tt.messages, tt.tools, think)
			if err != nil {
				t.Fatal(err)
			}
			want := strings.TrimPrefix(renderApertusHFTemplate(t, modelDir, tt.messages, tt.tools, tt.addGenerationPrompt, tt.enableThinking), "<s>")
			if got != want {
				t.Fatalf("renderer mismatch\nwant: %q\n got: %q", want, got)
			}
		})
	}
}

func apertusAssistantToolCall(name string, values map[string]any) api.Message {
	args := api.NewToolCallFunctionArguments()
	for k, v := range values {
		args.Set(k, v)
	}
	return api.Message{
		Role: "assistant",
		ToolCalls: []api.ToolCall{{
			Function: api.ToolCallFunction{Name: name, Arguments: args},
		}},
	}
}

func renderApertusHFTemplate(t *testing.T, modelDir string, messages []api.Message, tools []api.Tool, addGenerationPrompt bool, enableThinking bool) string {
	t.Helper()

	messagesJSON, err := json.Marshal(apertusHFMessages(messages))
	if err != nil {
		t.Fatal(err)
	}
	toolsJSON, err := json.Marshal(apertusHFTools(tools))
	if err != nil {
		t.Fatal(err)
	}

	script := `
import json
import sys
from transformers import AutoTokenizer

model_dir = sys.argv[1]
messages = json.loads(sys.argv[2])
tools = json.loads(sys.argv[3])
add_generation_prompt = sys.argv[4] == "true"
enable_thinking = sys.argv[5] == "true"
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
kwargs = {
    "tokenize": False,
    "add_generation_prompt": add_generation_prompt,
    "enable_thinking": enable_thinking,
}
if tools:
    kwargs["tools"] = tools
print(tok.apply_chat_template(messages, **kwargs), end="")
`
	cmd := exec.Command("python3", "-c", script, modelDir, string(messagesJSON), string(toolsJSON), boolArg(addGenerationPrompt), boolArg(enableThinking))
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("HF chat template render failed: %v\nstderr: %s", err, stderr.String())
	}
	return stdout.String()
}

type apertusHFMessage struct {
	Role      string              `json:"role"`
	Content   string              `json:"content"`
	ToolCalls []apertusHFToolCall `json:"tool_calls,omitempty"`
}

type apertusHFToolCall struct {
	Type     string                `json:"type"`
	Function apertusHFToolFunction `json:"function"`
}

type apertusHFToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func apertusHFMessages(messages []api.Message) []apertusHFMessage {
	out := make([]apertusHFMessage, 0, len(messages))
	for _, message := range messages {
		hfMessage := apertusHFMessage{
			Role:    message.Role,
			Content: message.Content,
		}
		for _, toolCall := range message.ToolCalls {
			args, _ := json.Marshal(toolCall.Function.Arguments)
			hfMessage.ToolCalls = append(hfMessage.ToolCalls, apertusHFToolCall{
				Type: "function",
				Function: apertusHFToolFunction{
					Name:      toolCall.Function.Name,
					Arguments: string(args),
				},
			})
		}
		out = append(out, hfMessage)
	}
	return out
}

type apertusHFTool struct {
	Name        string                     `json:"name"`
	Description string                     `json:"description,omitempty"`
	Parameters  api.ToolFunctionParameters `json:"parameters"`
}

func apertusHFTools(tools []api.Tool) []apertusHFTool {
	out := make([]apertusHFTool, 0, len(tools))
	for _, tool := range tools {
		out = append(out, apertusHFTool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
		})
	}
	return out
}

func boolArg(v bool) string {
	if v {
		return "true"
	}
	return "false"
}

func apertusWeatherTool() api.Tool {
	properties := api.NewToolPropertiesMap()
	properties.Set("location", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "City name.",
	})
	properties.Set("unit", api.ToolProperty{
		Type: api.PropertyType{"string"},
		Enum: []any{"celsius", "fahrenheit"},
	})
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "Get current weather.",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Required:   []string{"location"},
				Properties: properties,
			},
		},
	}
}
