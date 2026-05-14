package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestApertusParserSingleToolCall(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{parserWeatherTool()}, nil, nil)

	content, thinking, calls, err := parser.Add(`<|tools_prefix|>[{"get_weather": {"location":"Zurich","unit":"celsius"}}]<|tools_suffix|>`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "" {
		t.Fatalf("content=%q thinking=%q, want empty", content, thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("len(calls) = %d, want 1", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("tool name = %q", calls[0].Function.Name)
	}
	if got, _ := calls[0].Function.Arguments.Get("location"); got != "Zurich" {
		t.Fatalf("location = %#v, want Zurich", got)
	}
}

func TestApertusParserStreamingSplitTags(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{parserWeatherTool()}, nil, nil)

	content, _, calls, err := parser.Add("Let me check. <|tools_", false)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Let me check." || len(calls) != 0 {
		t.Fatalf("first chunk content=%q calls=%d", content, len(calls))
	}

	content, _, calls, err = parser.Add(`prefix|>[{"get_weather": {"location":"Bern"}}]<|tools_`, false)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || len(calls) != 0 {
		t.Fatalf("second chunk content=%q calls=%d", content, len(calls))
	}

	content, _, calls, err = parser.Add("suffix|>", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || len(calls) != 1 {
		t.Fatalf("final chunk content=%q calls=%d", content, len(calls))
	}
}

func TestApertusParserStripsAssistantStartBeforeToolCall(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{parserWeatherTool()}, nil, nil)

	content, _, calls, err := parser.Add(`<|assistant_start|><|tools_prefix|>[{"get_weather": {"location":"Zurich"}}]<|tools_suffix|>`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" {
		t.Fatalf("content = %q, want empty", content)
	}
	if len(calls) != 1 {
		t.Fatalf("len(calls) = %d, want 1", len(calls))
	}
}

func TestApertusParserToolCallWithStopStrippedSuffix(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{parserWeatherTool()}, nil, nil)

	content, _, calls, err := parser.Add(`<|tools_prefix|>[{"get_weather": {"location":"Zurich"}}]`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" {
		t.Fatalf("content = %q, want empty", content)
	}
	if len(calls) != 1 {
		t.Fatalf("len(calls) = %d, want 1", len(calls))
	}
}

func TestApertusParserBareFinalToolCallArray(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{parserWeatherTool()}, nil, nil)

	content, _, calls, err := parser.Add(`[{"get_weather": {"location":"Zurich", "unit":"celsius"}}]`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" {
		t.Fatalf("content = %q, want empty", content)
	}
	if len(calls) != 1 {
		t.Fatalf("len(calls) = %d, want 1", len(calls))
	}
}

func TestApertusParserBareToolCallArrayAcrossChunks(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{parserWeatherTool()}, nil, nil)

	content, _, calls, err := parser.Add(`[{"get_weather":`, false)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || len(calls) != 0 {
		t.Fatalf("first chunk content=%q calls=%d", content, len(calls))
	}

	content, _, calls, err = parser.Add(` {"location":"Zurich", "unit":"celsius"}}]`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" {
		t.Fatalf("content = %q, want empty", content)
	}
	if len(calls) != 1 {
		t.Fatalf("len(calls) = %d, want 1", len(calls))
	}
}

func TestApertusParserBareFinalJSONContentWithoutTools(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init(nil, nil, nil)

	content, _, calls, err := parser.Add(`[{"not_a_tool": true}]`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != `[{"not_a_tool": true}]` {
		t.Fatalf("content = %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("len(calls) = %d, want 0", len(calls))
	}
}

func TestApertusParserStripsAssistantTagsFromFinalText(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init(nil, nil, nil)

	content, thinking, calls, err := parser.Add(`<|assistant_start|>hello<|assistant_end|>`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "hello" || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d, want cleaned text only", content, thinking, len(calls))
	}
}

func TestApertusParserStripsAssistantEndAfterToolCall(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{parserWeatherTool()}, nil, nil)

	content, _, calls, err := parser.Add(`<|tools_prefix|>[{"get_weather": {"location":"Zurich"}}]<|tools_suffix|><|assistant_end|>`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" {
		t.Fatalf("content = %q, want empty", content)
	}
	if len(calls) != 1 {
		t.Fatalf("len(calls) = %d, want 1", len(calls))
	}
}

func TestApertusParserMultipleToolCalls(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{parserWeatherTool(), parserTimeTool()}, nil, nil)

	_, _, calls, err := parser.Add(`<|tools_prefix|>[{"get_weather": {"location":"Zurich"}}, {"get_time": {"location":"Bern"}}]<|tools_suffix|>`, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 {
		t.Fatalf("len(calls) = %d, want 2", len(calls))
	}
	if calls[0].Function.Index != 0 || calls[1].Function.Index != 1 {
		t.Fatalf("call indexes = %d, %d", calls[0].Function.Index, calls[1].Function.Index)
	}
}

func TestApertusParserMalformedToolCallFallsBackToContent(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{parserWeatherTool()}, nil, nil)

	content, thinking, calls, err := parser.Add(`<|tools_prefix|>[{"get_weather": ]<|tools_suffix|>`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != `[{"get_weather": ]` || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d, want malformed JSON as content", content, thinking, len(calls))
	}
}

func TestApertusParserUnterminatedMalformedToolCallFallsBackToContent(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{parserWeatherTool()}, nil, nil)

	content, thinking, calls, err := parser.Add(`<|tools_prefix|>not json`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != `not json` || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d, want malformed JSON as content", content, thinking, len(calls))
	}
}

func TestApertusParserUnknownTool(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{parserWeatherTool()}, nil, nil)

	if _, _, _, err := parser.Add(`<|tools_prefix|>[{"other": {}}]<|tools_suffix|>`, true); err == nil {
		t.Fatal("expected unknown tool error")
	}
}

func TestApertusParserCapabilities(t *testing.T) {
	parser := &ApertusParser{}
	if !parser.HasToolSupport() {
		t.Fatal("expected tool support")
	}
	if parser.HasThinkingSupport() {
		t.Fatal("did not expect thinking support")
	}
}

func parserWeatherTool() api.Tool {
	properties := api.NewToolPropertiesMap()
	properties.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
	properties.Set("unit", api.ToolProperty{Type: api.PropertyType{"string"}})
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name: "get_weather",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Required:   []string{"location"},
				Properties: properties,
			},
		},
	}
}

func parserTimeTool() api.Tool {
	properties := api.NewToolPropertiesMap()
	properties.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name: "get_time",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Required:   []string{"location"},
				Properties: properties,
			},
		},
	}
}
