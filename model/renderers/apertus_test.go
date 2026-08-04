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

func TestApertus1p1RendererPlainChat(t *testing.T) {
	got, err := (&Apertus1p1Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Hello"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	want := "<SPECIAL_61><SPECIAL_62>" +
		"<SPECIAL_63>Deliberation: disabled\nTool Capabilities: disabled<SPECIAL_64>" +
		"<SPECIAL_65>Hello<SPECIAL_66><SPECIAL_67>"
	if got != want {
		t.Fatalf("rendered prompt mismatch\nwant: %q\n got: %q", want, got)
	}
}

func TestApertus1p1RendererSystemAndThinking(t *testing.T) {
	think := &api.ThinkValue{Value: true}
	got, err := (&Apertus1p1Renderer{}).Render([]api.Message{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Question"},
		{Role: "assistant", Thinking: "Reason.", Content: "Answer."},
	}, nil, think)
	if err != nil {
		t.Fatal(err)
	}

	want := "<SPECIAL_61>Be concise.<SPECIAL_62>" +
		"<SPECIAL_63>Deliberation: enabled\nTool Capabilities: disabled<SPECIAL_64>" +
		"<SPECIAL_65>Question<SPECIAL_66>" +
		"<SPECIAL_67><SPECIAL_69>Reason.<SPECIAL_70>Answer.<SPECIAL_68>"
	if got != want {
		t.Fatalf("rendered prompt mismatch\nwant: %q\n got: %q", want, got)
	}
}

func TestApertus1p1RendererToolsAndOutputs(t *testing.T) {
	think := &api.ThinkValue{Value: true}
	args := api.NewToolCallFunctionArguments()
	args.Set("location", "Zurich")
	got, err := (&Apertus1p1Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Weather?"},
		{Role: "assistant", Thinking: "I should check.", ToolCalls: []api.ToolCall{{
			Function: api.ToolCallFunction{Name: "get_weather", Arguments: args},
		}}},
		{Role: "tool", Content: `{"temperature":22}`},
		{Role: "assistant", Content: "It is 22 degrees."},
	}, []api.Tool{apertusWeatherTool()}, think)
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{
		"<SPECIAL_63>Deliberation: enabled\nTool Capabilities:\n",
		"<SPECIAL_67><SPECIAL_69>I should check.<SPECIAL_71>" +
			`[{"get_weather": {"location":"Zurich"}}]` + "<SPECIAL_72>" +
			`[{"temperature":22}] ` + "<SPECIAL_70>It is 22 degrees.<SPECIAL_68>",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered prompt missing %q:\n%s", want, got)
		}
	}
}

func TestApertus1p1RendererToolHistoryKeepsInnerOpen(t *testing.T) {
	think := &api.ThinkValue{Value: true}
	args := api.NewToolCallFunctionArguments()
	args.Set("location", "Zurich")
	got, err := (&Apertus1p1Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Weather?"},
		{Role: "assistant", Thinking: "I should check.", ToolCalls: []api.ToolCall{{
			Function: api.ToolCallFunction{Name: "get_weather", Arguments: args},
		}}},
		{Role: "tool", Content: `{"temperature":22}`},
	}, []api.Tool{apertusWeatherTool()}, think)
	if err != nil {
		t.Fatal(err)
	}
	wantSuffix := "<SPECIAL_69>I should check.<SPECIAL_71>" +
		`[{"get_weather": {"location":"Zurich"}}]` + "<SPECIAL_72>" +
		`[{"temperature":22}]` + "<SPECIAL_68><SPECIAL_67>"
	if !strings.HasSuffix(got, wantSuffix) || strings.Contains(got, "I should check.<SPECIAL_70><SPECIAL_71>") {
		t.Fatalf("tool history did not preserve the official inner span:\n%s", got)
	}
}

func TestApertus1p1RendererDisplayAnswersClosesInner(t *testing.T) {
	think := &api.ThinkValue{Value: true}
	args := api.NewToolCallFunctionArguments()
	args.Set("answer", "done")
	got, err := (&Apertus1p1Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Answer"},
		{Role: "assistant", Thinking: "Ready.", ToolCalls: []api.ToolCall{{
			Function: api.ToolCallFunction{Name: "display_answers", Arguments: args},
		}}},
	}, nil, think)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `<SPECIAL_69>Ready.<SPECIAL_70><SPECIAL_71>[{"display_answers": {"answer":"done"}}]<SPECIAL_72>`) {
		t.Fatalf("display_answers should close the inner span before its tool call:\n%s", got)
	}
	if !strings.HasSuffix(got, "<SPECIAL_72><SPECIAL_68>") {
		t.Fatalf("display_answers should complete the assistant turn:\n%s", got)
	}
}

func TestApertus1p1RendererMapsLeadingDeveloperInstructionsToSystem(t *testing.T) {
	got, err := (&Apertus1p1Renderer{}).Render([]api.Message{
		{Role: "system", Content: "System policy."},
		{Role: "developer", Content: "Developer policy."},
		{Role: "user", Content: "Hello"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got, "<SPECIAL_61>System policy.\n\nDeveloper policy.<SPECIAL_62><SPECIAL_63>") {
		t.Fatalf("developer instructions were not mapped into the system portion:\n%s", got)
	}
}

func TestApertus1p1RendererAlwaysPromptsToolDecision(t *testing.T) {
	got, err := (&Apertus1p1Renderer{}).Render(
		[]api.Message{{Role: "user", Content: "Weather?"}},
		[]api.Tool{apertusWeatherTool()},
		&api.ThinkValue{Value: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(got, "<SPECIAL_67>") {
		t.Fatalf("tool decision prompt should append assistant generation marker:\n%s", got)
	}
}

func TestApertus1p5RendererPlainChatDefaultSystem(t *testing.T) {
	got, err := (&Apertus1p5Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Hello"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	want := "<|system_start|>You are Apertus 1.5 Omni, a multimodal assistant developed by the Swiss AI Initiative. Extended from Apertus 1 via continued pretraining, you understand images and audio and respond in text.<|system_end|>" +
		"<|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|>" +
		"<|user_start|>Hello<|user_end|><|assistant_start|>"
	if got != want {
		t.Fatalf("rendered prompt mismatch\nwant: %q\n got: %q", want, got)
	}
}

func TestApertus1p5RendererPreservesStableMediaOrder(t *testing.T) {
	got, err := (&Apertus1p5Renderer{}).Render([]api.Message{
		{Role: "user", Content: "first", Images: []api.ImageData{api.ImageData("a")}},
		{Role: "assistant", Content: "seen"},
		{Role: "user", Content: "second", Images: []api.ImageData{api.ImageData("b"), api.ImageData("c")}},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"[img-0] first", "[img-1][img-2] second"} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered prompt missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, apertusImageToken) {
		t.Fatalf("renderer leaked architecture placeholder before media preparation:\n%s", got)
	}
}

func TestApertus1p5RendererToolsAppendGenerationPromptForToolDecision(t *testing.T) {
	got, err := (&Apertus1p5Renderer{}).Render([]api.Message{
		{Role: "user", Content: "What is the weather in Zurich?"},
	}, []api.Tool{apertusWeatherTool()}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(got, "Tool Capabilities:\n// Get current weather.") {
		t.Fatalf("rendered prompt missing tools:\n%s", got)
	}
	if !strings.HasSuffix(got, "<|assistant_start|>") {
		t.Fatalf("tool decision prompt should append assistant generation prompt:\n%s", got)
	}
}

func TestApertus1p5RendererAssistantToolCallAndOutput(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("location", "Zurich")

	got, err := (&Apertus1p5Renderer{}).Render([]api.Message{
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
	if !strings.Contains(got, `<|tool_output_start|>{"temperature":22}<|tool_output_end|><|assistant_end|><|assistant_start|>`) {
		t.Fatalf("rendered prompt missing 1.5 tool output response continuation:\n%s", got)
	}
}

func TestApertus1p5RendererThinkingEnabled(t *testing.T) {
	think := &api.ThinkValue{Value: true}
	got, err := (&Apertus1p5Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Question"},
		{Role: "assistant", Thinking: "Need a short answer.", Content: "Answer."},
	}, nil, think)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(got, "<|developer_start|>Deliberation: enabled\nTool Capabilities: disabled<|developer_end|>") {
		t.Fatalf("rendered prompt missing thinking developer flag:\n%s", got)
	}
	if !strings.Contains(got, "<|assistant_start|><|inner_prefix|>Need a short answer.<|inner_suffix|>Answer.<|assistant_end|>") {
		t.Fatalf("rendered prompt missing thinking history:\n%s", got)
	}
}

func TestApertus1p5RendererRejectsToolsWithThinking(t *testing.T) {
	_, err := (&Apertus1p5Renderer{}).Render(
		[]api.Message{{Role: "user", Content: "Weather?"}},
		[]api.Tool{apertusWeatherTool()},
		&api.ThinkValue{Value: true},
	)
	if err == nil || err.Error() != "Apertus 1.5 does not support tool calling with thinking enabled" {
		t.Fatalf("Render error = %v", err)
	}
}

func TestApertusRendererStillUsesLegacyToolOutputFraming(t *testing.T) {
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
	if strings.Contains(got, "<|tool_output_start|>") || !strings.Contains(got, `[{"temperature":22}]`) {
		t.Fatalf("legacy Apertus renderer should keep bracket tool output framing:\n%s", got)
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

func TestApertus1p1RendererMatchesHFChatTemplate(t *testing.T) {
	if os.Getenv("VERIFY_APERTUS11_HF_TEMPLATE") == "" {
		t.Skip("set VERIFY_APERTUS11_HF_TEMPLATE=1 to compare against a local Apertus v1.1 Instruct template")
	}
	modelDir := os.Getenv("APERTUS11_HF_MODEL_DIR")
	if modelDir == "" {
		modelDir = filepath.Join("..", "..", "..", "models", "Apertus-v1.1-0.5B-Instruct")
	}
	var err error
	modelDir, err = filepath.Abs(modelDir)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name     string
		messages []api.Message
	}{
		{
			name:     "empty_system",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
		},
		{
			name: "explicit_system",
			messages: []api.Message{
				{Role: "system", Content: "Be concise."},
				{Role: "user", Content: "Hello"},
			},
		},
		{
			name: "assistant_history",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi!"},
				{Role: "user", Content: "Again"},
			},
		},
	}

	renderer := &Apertus1p1Renderer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := renderer.Render(tt.messages, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			want := strings.TrimPrefix(renderApertusHFTemplate(t, modelDir, tt.messages, nil, true, false), "<s>")
			if got != want {
				t.Fatalf("renderer mismatch\nwant: %q\n got: %q", want, got)
			}
		})
	}

	t.Run("structured_thinking_tools_outputs_and_response", func(t *testing.T) {
		tools := []api.Tool{apertusWeatherTool()}
		think := &api.ThinkValue{Value: true}
		args := api.NewToolCallFunctionArguments()
		args.Set("location", "Zurich")
		got, err := renderer.Render([]api.Message{
			{Role: "user", Content: "Weather?"},
			{Role: "assistant", Thinking: "I should check.", ToolCalls: []api.ToolCall{{
				Function: api.ToolCallFunction{Name: "get_weather", Arguments: args},
			}}},
			{Role: "tool", Content: `{"temperature":22}`},
			{Role: "assistant", Content: "It is 22 degrees."},
		}, tools, think)
		if err != nil {
			t.Fatal(err)
		}

		var formattedTools strings.Builder
		renderApertusTools(&formattedTools, tools)
		hfMessages := []apertusHFMessage{
			{Role: "developer", Content: map[string]any{
				"has_thinking":    true,
				"formatted_tools": formattedTools.String(),
			}},
			{Role: "user", Content: "Weather?"},
			{Role: "assistant", Content: map[string]any{"blocks": []any{
				map[string]any{"type": "thoughts", "text": "I should check."},
				map[string]any{"type": "tool_calls", "calls": []any{
					map[string]any{"name": "get_weather", "arguments": `{"location":"Zurich"}`},
				}},
				map[string]any{"type": "tool_outputs", "outputs": []any{
					map[string]any{"output": `{"temperature":22}`},
				}},
				map[string]any{"type": "response", "text": "It is 22 degrees."},
			}}},
		}
		want := strings.TrimPrefix(renderApertusHFTemplateMessages(t, modelDir, hfMessages, false), "<s>")
		if got != want {
			t.Fatalf("structured renderer mismatch\nwant: %q\n got: %q", want, got)
		}
	})

	t.Run("structured_display_answers", func(t *testing.T) {
		think := &api.ThinkValue{Value: true}
		args := api.NewToolCallFunctionArguments()
		args.Set("answer", "done")
		got, err := renderer.Render([]api.Message{
			{Role: "user", Content: "Answer"},
			{Role: "assistant", Thinking: "Ready.", ToolCalls: []api.ToolCall{{
				Function: api.ToolCallFunction{Name: "display_answers", Arguments: args},
			}}},
		}, nil, think)
		if err != nil {
			t.Fatal(err)
		}
		hfMessages := []apertusHFMessage{
			{Role: "developer", Content: map[string]any{"has_thinking": true}},
			{Role: "user", Content: "Answer"},
			{Role: "assistant", Content: map[string]any{"blocks": []any{
				map[string]any{"type": "thoughts", "text": "Ready."},
				map[string]any{"type": "tool_calls", "calls": []any{
					map[string]any{"name": "display_answers", "arguments": `{"answer":"done"}`},
				}},
			}}},
		}
		want := strings.TrimPrefix(renderApertusHFTemplateMessages(t, modelDir, hfMessages, false), "<s>")
		if got != want {
			t.Fatalf("display_answers renderer mismatch\nwant: %q\n got: %q", want, got)
		}
	})
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
	return renderApertusHFTemplatePayload(t, modelDir, apertusHFMessages(messages), apertusHFTools(tools), addGenerationPrompt, enableThinking)
}

func renderApertusHFTemplateMessages(t *testing.T, modelDir string, messages []apertusHFMessage, addGenerationPrompt bool) string {
	t.Helper()
	return renderApertusHFTemplatePayload(t, modelDir, messages, nil, addGenerationPrompt, false)
}

func renderApertusHFTemplatePayload(t *testing.T, modelDir string, messages any, tools []apertusHFTool, addGenerationPrompt bool, enableThinking bool) string {
	t.Helper()

	messagesJSON, err := json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}
	toolsJSON, err := json.Marshal(tools)
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
	python := os.Getenv("PORTING_PYTHON")
	if python == "" {
		python = "python3"
	}
	cmd := exec.Command(python, "-c", script, modelDir, string(messagesJSON), string(toolsJSON), boolArg(addGenerationPrompt), boolArg(enableThinking))
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
	Content   any                 `json:"content"`
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
