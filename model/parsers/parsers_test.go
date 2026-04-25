package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

type mockParser struct {
	name string
}

func (m *mockParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	return tools
}

func (m *mockParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	return "mock:" + s, "", nil, nil
}

func (m *mockParser) HasToolSupport() bool {
	return false
}

func (m *mockParser) HasThinkingSupport() bool {
	return false
}

func TestRegisterCustomParser(t *testing.T) {
	// Register a custom parser
	Register("custom-parser", func() Parser {
		return &mockParser{name: "custom"}
	})

	// Retrieve it
	parser := ParserForName("custom-parser")
	if parser == nil {
		t.Fatal("expected parser to be registered")
	}

	// Test it works
	content, _, _, err := parser.Add("test", false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "mock:test" {
		t.Errorf("expected 'mock:test', got %q", content)
	}
}

func TestBuiltInParsersStillWork(t *testing.T) {
	tests := []struct {
		name string
	}{
		{"passthrough"},
		{"qwen3"},
		{"qwen3-thinking"},
		{"qwen3-coder"},
		{"lfm2"},
		{"lfm2-thinking"},
		{"qwen3.5"},
		{"harmony"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := ParserForName(tt.name)
			if parser == nil {
				t.Fatalf("expected built-in parser %q to exist", tt.name)
			}
		})
	}
}

func TestHarmonyParserCapabilitiesAndToolFlow(t *testing.T) {
	parser := ParserForName("harmony")
	if parser == nil {
		t.Fatal("ParserForName(\"harmony\") = nil")
	}
	if !parser.HasThinkingSupport() {
		t.Fatal("harmony parser HasThinkingSupport() = false, want true")
	}
	if !parser.HasToolSupport() {
		t.Fatal("harmony parser HasToolSupport() = false, want true")
	}

	tools := parser.Init([]api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name: "get-weather",
		},
	}}, nil, &api.ThinkValue{Value: true})

	if len(tools) != 1 {
		t.Fatalf("len(processed tools) = %d, want 1", len(tools))
	}
	if tools[0].Function.Name != "get_weather" {
		t.Fatalf("processed tool name = %q, want %q", tools[0].Function.Name, "get_weather")
	}

	content, thinking, calls, err := parser.Add(
		`<|channel|>analysis<|message|>Need current weather.<|end|>`+
			`<|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>{"city":"Paris"}`,
		true,
	)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if content != "" {
		t.Fatalf("content = %q, want empty", content)
	}
	if thinking != "Need current weather." {
		t.Fatalf("thinking = %q, want %q", thinking, "Need current weather.")
	}
	if len(calls) != 1 {
		t.Fatalf("len(calls) = %d, want 1", len(calls))
	}
	if calls[0].Function.Name != "get-weather" {
		t.Fatalf("call name = %q, want %q", calls[0].Function.Name, "get-weather")
	}
	if got := calls[0].Function.Arguments.String(); got != `{"city":"Paris"}` {
		t.Fatalf("call arguments = %s, want %s", got, `{"city":"Paris"}`)
	}
}

func TestOverrideBuiltInParser(t *testing.T) {
	// Override a built-in parser
	Register("passthrough", func() Parser {
		return &mockParser{name: "override"}
	})

	// Should get the override
	parser := ParserForName("passthrough")
	if parser == nil {
		t.Fatal("expected parser to exist")
	}

	// Test it's the override
	content, _, _, err := parser.Add("test", false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "mock:test" {
		t.Errorf("expected 'mock:test' from override, got %q", content)
	}
}

func TestUnknownParserReturnsNil(t *testing.T) {
	parser := ParserForName("nonexistent-parser")
	if parser != nil {
		t.Error("expected nil for unknown parser")
	}
}

func TestSplitAtTag(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		tag        string
		trimAfter  bool
		wantBefore string
		wantAfter  string
		wantSB     string // expected content of strings.Builder after operation
	}{
		{
			name:       "basic split with trimAfter true",
			input:      "hello <!-- split --> world",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "hello",
			wantAfter:  "world",
			wantSB:     "world",
		},
		{
			name:       "basic split with trimAfter false",
			input:      "hello <!-- split -->   world",
			tag:        "<!-- split -->",
			trimAfter:  false,
			wantBefore: "hello",
			wantAfter:  "   world",
			wantSB:     "   world",
		},
		{
			name:       "tag at beginning with trimAfter true",
			input:      "<!-- split -->world",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "",
			wantAfter:  "world",
			wantSB:     "world",
		},
		{
			name:       "tag at beginning with trimAfter false",
			input:      "<!-- split -->   world",
			tag:        "<!-- split -->",
			trimAfter:  false,
			wantBefore: "",
			wantAfter:  "   world",
			wantSB:     "   world",
		},
		{
			name:       "tag at end with trimAfter true",
			input:      "hello <!-- split -->",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "hello",
			wantAfter:  "",
			wantSB:     "",
		},
		{
			name:       "tag at end with trimAfter false",
			input:      "hello <!-- split -->",
			tag:        "<!-- split -->",
			trimAfter:  false,
			wantBefore: "hello",
			wantAfter:  "",
			wantSB:     "",
		},
		{
			name:       "multiple tags splits at first occurrence",
			input:      "hello <!-- split --> world <!-- split --> end",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "hello",
			wantAfter:  "world <!-- split --> end",
			wantSB:     "world <!-- split --> end",
		},
		{
			name:       "tag not present",
			input:      "hello world",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "hello world",
			wantAfter:  "",
			wantSB:     "",
		},
		{
			name:       "empty input",
			input:      "",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "",
			wantAfter:  "",
			wantSB:     "",
		},
		{
			name:       "only whitespace before tag",
			input:      "   \t\n<!-- split -->world",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "",
			wantAfter:  "world",
			wantSB:     "world",
		},
		{
			name:       "only whitespace after tag with trimAfter true",
			input:      "hello<!-- split -->   \t\n",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "hello",
			wantAfter:  "",
			wantSB:     "",
		},
		{
			name:       "only whitespace after tag with trimAfter false",
			input:      "hello<!-- split -->   \t\n",
			tag:        "<!-- split -->",
			trimAfter:  false,
			wantBefore: "hello",
			wantAfter:  "   \t\n",
			wantSB:     "   \t\n",
		},
		{
			name:       "complex whitespace trimming",
			input:      "  hello \t\n <!-- split --> \n\t world  ",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "  hello",
			wantAfter:  "world  ",
			wantSB:     "world  ",
		},
		{
			name:       "tag with special characters",
			input:      "text <tag attr=\"value\"> more text",
			tag:        "<tag attr=\"value\">",
			trimAfter:  true,
			wantBefore: "text",
			wantAfter:  "more text",
			wantSB:     "more text",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sb := &strings.Builder{}
			sb.WriteString(tt.input)

			before, after := splitAtTag(sb, tt.tag, tt.trimAfter)

			// Check return values
			if before != tt.wantBefore {
				t.Errorf("splitAtTag() before = %q, want %q", before, tt.wantBefore)
			}
			if after != tt.wantAfter {
				t.Errorf("splitAtTag() after = %q, want %q", after, tt.wantAfter)
			}

			// Check strings.Builder state
			if sb.String() != tt.wantSB {
				t.Errorf("strings.Builder after split = %q, want %q", sb.String(), tt.wantSB)
			}
		})
	}
}
