package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
)

func TestApertusRendererGrammar(t *testing.T) {
	tool := apertusRendererTool("get_weather")
	got, err := (&ApertusRenderer{}).Render([]api.Message{{Role: "system", Content: "Be concise."}, {Role: "user", Content: "Hello"}}, []api.Tool{tool}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"<|system_start|>Be concise.<|system_end|>", "<|developer_start|>Deliberation: disabled\nTool Capabilities:\n", "type get_weather = (_: {\ncity: string\n}) => any;", "<|user_start|>Hello<|user_end|>"} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q in %q", want, got)
		}
	}
	if strings.HasSuffix(got, apertusAssistantStart) {
		t.Fatal("tool decision must not append assistant generation prompt")
	}
}

func TestApertusRendererThinkingModes(t *testing.T) {
	message := []api.Message{{Role: "assistant", Thinking: "reason", Content: "answer"}}
	for _, think := range []*api.ThinkValue{nil, {Value: false}, {Value: "low"}, {Value: true}} {
		got, err := (&ApertusRenderer{}).Render(message, nil, think)
		if err != nil {
			t.Fatal(err)
		}
		enabled := think != nil && think.Bool()
		if strings.Contains(got, apertusInnerOpenTag) != enabled || strings.Contains(got, "Deliberation: enabled") != enabled {
			t.Fatalf("think=%v got=%q", think, got)
		}
		if !enabled && strings.Contains(got, "reason") {
			t.Fatalf("disabled thinking leaked: %q", got)
		}
	}
	if _, err := (&ApertusRenderer{}).Render([]api.Message{{Role: "assistant", Thinking: apertusAssistantStart}}, nil, &api.ThinkValue{Value: true}); err == nil {
		t.Fatal("thinking control token accepted")
	}
}

func TestApertusRendererHistoryAndToolCalls(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("city", "Bern")
	got, err := (&ApertusRenderer{}).Render([]api.Message{{Role: "user", Content: "weather"}, {Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "get_weather", Arguments: args}}}}, {Role: "tool", Content: `{"temperature":22}`}}, []api.Tool{apertusRendererTool("get_weather")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `<|tools_prefix|>[{"get_weather": {"city":"Bern"}}]<|tools_suffix|>[{"temperature":22}]<|assistant_end|><|assistant_start|>`) {
		t.Fatalf("unexpected history %q", got)
	}
}

func TestApertusRendererToolHistoryStateAndIdentity(t *testing.T) {
	r := &ApertusRenderer{}
	args := api.NewToolCallFunctionArguments()
	args.Set("city", "Bern")
	call := api.ToolCall{Function: api.ToolCallFunction{Name: "get_weather", Arguments: args}}
	if _, err := r.Render([]api.Message{{Role: "assistant", Content: "text"}, {Role: "tool", Content: `{}`}}, []api.Tool{apertusRendererTool("get_weather")}, nil); err == nil {
		t.Fatal("tool after assistant text accepted")
	}
	if _, err := r.Render([]api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{call}}}, nil, nil); err == nil {
		t.Fatal("assistant call without declarations accepted")
	}
	undeclared := call
	undeclared.Function.Name = "other"
	if _, err := r.Render([]api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{undeclared}}}, []api.Tool{apertusRendererTool("get_weather")}, nil); err == nil {
		t.Fatal("undeclared assistant call accepted")
	}
	second := call
	second.Function.Arguments = api.NewToolCallFunctionArguments()
	second.Function.Arguments.Set("city", "Zurich")
	got, err := r.Render([]api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{call, second}}, {Role: "tool", Content: `{"weather":1}`}, {Role: "tool", Content: `{"weather":2}`}}, []api.Tool{apertusRendererTool("get_weather")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `[{"weather":1}, {"weather":2}]`) {
		t.Fatalf("ordered multi-result grouping missing: %q", got)
	}
}

func TestApertusRendererRepeatedCallRoundTripAndPendingTransitions(t *testing.T) {
	tool := apertusRendererTool("get_weather")
	p := &parsers.ApertusParser{}
	p.Init([]api.Tool{tool}, nil, nil)
	_, _, calls, err := p.Add(`<|tools_prefix|>[{"get_weather":{"city":"Bern"}},{"get_weather":{"city":"Zurich"}}]<|tools_suffix|>`, true)
	if err != nil || len(calls) != 2 || calls[0].Function.Index != 0 || calls[1].Function.Index != 1 {
		t.Fatalf("calls=%#v err=%v", calls, err)
	}
	r := &ApertusRenderer{}
	terminal := []api.Message{{Role: "assistant", ToolCalls: calls}}
	if _, err := r.Render(terminal, []api.Tool{tool}, nil); err != nil {
		t.Fatalf("terminal pending call rejected: %v", err)
	}
	partial := append(append([]api.Message{}, terminal...), api.Message{Role: "tool", Content: `{}`})
	for _, next := range []api.Message{{Role: "user", Content: "next"}, {Role: "system", Content: "next"}, {Role: "assistant", Content: "next"}} {
		if _, err := r.Render(append(append([]api.Message{}, partial...), next), []api.Tool{tool}, nil); err == nil {
			t.Fatalf("partial tool results followed by %s accepted", next.Role)
		}
	}
	if _, err := r.Render(append(partial, api.Message{Role: "tool", Content: `{}`}, api.Message{Role: "tool", Content: `{}`}), []api.Tool{tool}, nil); err == nil {
		t.Fatal("extra tool result accepted")
	}
	if _, err := r.Render(append(partial, api.Message{Role: "tool", Content: `{}`}, api.Message{Role: "user", Content: "next"}), []api.Tool{tool}, nil); err != nil {
		t.Fatalf("complete ordered results rejected: %v", err)
	}
}

func TestApertusRendererRejectsAmbiguousOrUnsafeSchema(t *testing.T) {
	r := &ApertusRenderer{}
	if _, err := r.Render(nil, []api.Tool{apertusRendererTool("same"), apertusRendererTool("same")}, nil); err == nil {
		t.Fatal("duplicate tools accepted")
	}
	if _, err := r.Render(nil, []api.Tool{apertusRendererTool("bad.name")}, nil); err == nil {
		t.Fatal("separator-bearing tool accepted")
	}
	missing := apertusRendererTool("missing")
	missing.Function.Parameters.Required = []string{"not_declared"}
	if _, err := r.Render(nil, []api.Tool{missing}, nil); err == nil {
		t.Fatal("required/property mismatch accepted")
	}
	deep := api.ToolProperty{Type: api.PropertyType{"object"}}
	for range maxApertusSchemaDepth + 2 {
		props := api.NewToolPropertiesMap()
		props.Set("next", deep)
		deep = api.ToolProperty{Type: api.PropertyType{"object"}, Properties: props}
	}
	tool := apertusRendererTool("deep")
	tool.Function.Parameters.Properties.Set("root", deep)
	if _, err := r.Render(nil, []api.Tool{tool}, nil); err == nil {
		t.Fatal("deep schema accepted")
	}
	wide := apertusRendererTool("wide")
	for i := range maxApertusSchemaNodes {
		wide.Function.Parameters.Properties.Set("field"+strings.Repeat("x", i+1), api.ToolProperty{Type: api.PropertyType{"string"}})
	}
	if _, err := r.Render(nil, []api.Tool{wide}, nil); err == nil {
		t.Fatal("oversized schema accepted")
	}
	if _, err := r.Render([]api.Message{{Role: "user", Content: "x" + apertusAssistantStart}}, nil, nil); err == nil {
		t.Fatal("control token accepted")
	}
	if _, err := r.Render([]api.Message{{Role: "tool", Content: "orphan"}}, nil, nil); err == nil {
		t.Fatal("orphan tool accepted")
	}
}

func TestApertus1p5RendererContract(t *testing.T) {
	got, err := (&Apertus1p5Renderer{}).Render([]api.Message{{Role: "user", Content: "Hello"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "You are Apertus 1.5 Omni") || !strings.HasSuffix(got, apertusAssistantStart) {
		t.Fatalf("unexpected 1.5 plain prompt: %q", got)
	}
	if _, err := (&Apertus1p5Renderer{}).Render([]api.Message{{Role: "user", Content: "Weather?"}}, []api.Tool{apertusRendererTool("get_weather")}, &api.ThinkValue{Value: true}); err == nil {
		t.Fatal("1.5 tools with thinking accepted")
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
	}, []api.Tool{apertusRendererTool("get_weather")}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(got, "Tool Capabilities:\ntype get_weather") {
		t.Fatalf("rendered prompt missing tools:\n%s", got)
	}
	if !strings.HasSuffix(got, "<|assistant_start|>") {
		t.Fatalf("tool decision prompt should append assistant generation prompt:\n%s", got)
	}
}

func TestApertus1p5RendererAssistantToolCallAndOutput(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("city", "Zurich")
	history, err := (&Apertus1p5Renderer{}).Render([]api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "get_weather", Arguments: args}}}}, {Role: "tool", Content: `{"temperature":22}`}}, []api.Tool{apertusRendererTool("get_weather")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(history, `<|tool_output_start|>{"temperature":22}<|tool_output_end|>`) {
		t.Fatalf("missing 1.5 output framing: %q", history)
	}
}

func apertusRendererTool(name string) api.Tool {
	props := api.NewToolPropertiesMap()
	props.Set("city", api.ToolProperty{Type: api.PropertyType{"string"}})
	return api.Tool{Type: "function", Function: api.ToolFunction{Name: name, Parameters: api.ToolFunctionParameters{Type: "object", Required: []string{"city"}, Properties: props}}}
}
