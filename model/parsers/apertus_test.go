package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestApertusParserGrammarAndStreamingBoundaries(t *testing.T) {
	input := `prefix <|tools_prefix|>[{"get_weather":{"city":"Bern"}}]<|tools_suffix|><|assistant_end|>`
	for i := range len(input) + 1 {
		p := &ApertusParser{}
		p.Init([]api.Tool{apertusParserTool("get_weather")}, nil, nil)
		content, _, calls, err := p.Add(input[:min(i, len(input))], false)
		if err != nil {
			t.Fatalf("split %d: %v", i, err)
		}
		content2, _, calls2, err := p.Add(input[min(i, len(input)):], true)
		if err != nil {
			t.Fatalf("split %d: %v", i, err)
		}
		if content+content2 != "prefix" || len(calls)+len(calls2) != 1 {
			t.Fatalf("split %d: content=%q calls=%d", i, content+content2, len(calls)+len(calls2))
		}
	}
}

func TestApertusParserMultipleUnknownMalformedAndDuplicateDeclarations(t *testing.T) {
	p := &ApertusParser{}
	p.Init([]api.Tool{apertusParserTool("one"), apertusParserTool("two")}, nil, nil)
	_, _, calls, err := p.Add(`<|tools_prefix|>[{"one":{}},{"two":{"x":1}}]<|tools_suffix|>`, true)
	if err != nil || len(calls) != 2 || calls[0].Function.Index != 0 || calls[1].Function.Index != 1 {
		t.Fatalf("calls=%#v err=%v", calls, err)
	}
	p = &ApertusParser{}
	p.Init([]api.Tool{apertusParserTool("one")}, nil, nil)
	if _, _, _, err := p.Add(`<|tools_prefix|>[{"other":{}}]<|tools_suffix|>`, true); err == nil {
		t.Fatal("unknown call accepted")
	}
	p = &ApertusParser{}
	p.Init([]api.Tool{apertusParserTool("one")}, nil, nil)
	content, _, calls, err := p.Add(`<|tools_prefix|>[{"one":}]<|tools_suffix|>`, true)
	if err != nil || content != `[{"one":}]` || len(calls) != 0 {
		t.Fatalf("malformed content=%q calls=%d err=%v", content, len(calls), err)
	}
	p = &ApertusParser{}
	p.Init([]api.Tool{apertusParserTool("same"), apertusParserTool("same")}, nil, nil)
	if _, _, _, err := p.Add("anything", true); err == nil {
		t.Fatal("duplicate declaration accepted")
	}
	p = &ApertusParser{}
	p.Init([]api.Tool{apertusParserTool("bad.name")}, nil, nil)
	if _, _, _, err := p.Add("anything", true); err == nil {
		t.Fatal("separator-bearing declaration accepted")
	}
}

func TestApertusParserBoundsToolPayload(t *testing.T) {
	p := &ApertusParser{}
	p.Init([]api.Tool{apertusParserTool("one")}, nil, nil)
	if _, _, _, err := p.Add(apertusToolOpenTag+strings.Repeat("x", maxApertusToolCallBytes+1), false); err == nil {
		t.Fatal("oversized tool call accepted")
	}
}

func TestApertusParserBoundsBareToolPayload(t *testing.T) {
	for _, prefix := range []string{"[", "{"} {
		t.Run(prefix, func(t *testing.T) {
			p := &ApertusParser{}
			p.Init([]api.Tool{apertusParserTool("one")}, nil, nil)
			if _, _, _, err := p.Add(prefix, false); err != nil {
				t.Fatal(err)
			}
			if _, _, _, err := p.Add(strings.Repeat("x", maxApertusToolCallBytes-len(prefix)), true); err != nil {
				t.Fatalf("exact limit rejected: %v", err)
			}
			p = &ApertusParser{}
			p.Init([]api.Tool{apertusParserTool("one")}, nil, nil)
			if _, _, _, err := p.Add(prefix, false); err != nil {
				t.Fatal(err)
			}
			if _, _, _, err := p.Add(strings.Repeat("x", maxApertusToolCallBytes-len(prefix)+1), false); err == nil {
				t.Fatal("first over-limit byte accepted")
			}
		})
	}
}

func TestApertusParserBoundsRetainedPrefixesBeforeAppend(t *testing.T) {
	tests := []struct {
		name   string
		prefix string
		body   string
	}{
		{"whitespace", " ", "[{"},
		{"assistant-token", "<|assistant_", "end|>[{"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &ApertusParser{}
			p.Init([]api.Tool{apertusParserTool("one")}, nil, nil)
			if _, _, _, err := p.Add(tt.prefix, false); err != nil {
				t.Fatal(err)
			}
			exact := tt.body + strings.Repeat("x", maxApertusToolCallBytes-p.acc.Len()-len(tt.body))
			if _, _, _, err := p.Add(exact, true); err != nil {
				t.Fatalf("exact limit rejected: %v", err)
			}
			p = &ApertusParser{}
			p.Init([]api.Tool{apertusParserTool("one")}, nil, nil)
			if _, _, _, err := p.Add(tt.prefix, false); err != nil {
				t.Fatal(err)
			}
			retained := p.acc.Len()
			over := tt.body + strings.Repeat("x", maxApertusToolCallBytes-retained-len(tt.body)+1)
			if _, _, _, err := p.Add(over, false); err == nil {
				t.Fatal("first over-limit byte accepted")
			}
			if p.acc.Len() != retained {
				t.Fatalf("retained %d bytes after rejection, want %d", p.acc.Len(), retained)
			}
			if _, _, _, err := p.Add("", true); err != nil {
				t.Fatalf("retained prefix did not recover: %v", err)
			}
		})
	}
}

func TestApertusParserTaggedPayloadBoundaryBeforeAppend(t *testing.T) {
	tests := []struct {
		name   string
		prefix string
		body   string
	}{
		{"complete", "", apertusToolOpenTag},
	}
	for i := 1; i < len(apertusToolOpenTag); i++ {
		tests = append(tests, struct {
			name   string
			prefix string
			body   string
		}{name: "split", prefix: apertusToolOpenTag[:i], body: apertusToolOpenTag[i:]})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &ApertusParser{}
			p.Init([]api.Tool{apertusParserTool("one")}, nil, nil)
			if tt.prefix != "" {
				if _, _, _, err := p.Add(tt.prefix, false); err != nil {
					t.Fatal(err)
				}
			}
			exact := tt.body + strings.Repeat("x", maxApertusToolCallBytes)
			if _, _, _, err := p.Add(exact, true); err != nil {
				t.Fatalf("exact payload limit rejected: %v", err)
			}
			p = &ApertusParser{}
			p.Init([]api.Tool{apertusParserTool("one")}, nil, nil)
			if tt.prefix != "" {
				if _, _, _, err := p.Add(tt.prefix, false); err != nil {
					t.Fatal(err)
				}
			}
			retained := p.acc.Len()
			over := tt.body + strings.Repeat("x", maxApertusToolCallBytes+1)
			if _, _, _, err := p.Add(over, false); err == nil {
				t.Fatal("first payload byte over limit accepted")
			}
			if p.acc.Len() != retained {
				t.Fatalf("retained %d bytes after rejection, want %d", p.acc.Len(), retained)
			}
		})
	}
}

func TestApertusParserBareAndTaggedHardErrorsMatch(t *testing.T) {
	cases := []string{
		`[{"other":{}}]`,
		`[{"one":{},"two":{}}]`,
	}
	for _, raw := range cases {
		t.Run(raw, func(t *testing.T) {
			for _, framed := range []string{raw, apertusToolOpenTag + raw + apertusToolCloseTag} {
				p := &ApertusParser{}
				p.Init([]api.Tool{apertusParserTool("one"), apertusParserTool("two")}, nil, nil)
				if _, _, _, err := p.Add(framed, true); err == nil {
					t.Fatalf("hard-invalid %q accepted", framed)
				}
			}
		})
	}
	for _, framed := range []string{`[{"one":}]`, apertusToolOpenTag + `[{"one":}]` + apertusToolCloseTag} {
		p := &ApertusParser{}
		p.Init([]api.Tool{apertusParserTool("one")}, nil, nil)
		content, _, calls, err := p.Add(framed, true)
		if err != nil || content != `[{"one":}]` || len(calls) != 0 {
			t.Fatalf("malformed %q content=%q calls=%d err=%v", framed, content, len(calls), err)
		}
	}
}

func TestApertusParserThinkingAndTools(t *testing.T) {
	for _, think := range []*api.ThinkValue{nil, {Value: false}, {Value: "low"}, {Value: true}} {
		p := &ApertusParser{}
		p.Init([]api.Tool{apertusParserTool("one")}, nil, think)
		content, thinking, calls, err := p.Add(`<|inner_prefix|>reason<|inner_suffix|><|tools_prefix|>[{"one":{}}]<|tools_suffix|>`, true)
		if err != nil || len(calls) != 1 {
			t.Fatalf("think=%v calls=%d err=%v", think, len(calls), err)
		}
		if think != nil && think.Bool() {
			if thinking != "reason" || content != "" {
				t.Fatalf("thinking=%q content=%q", thinking, content)
			}
		} else if content != "reason" || thinking != "" {
			t.Fatalf("content=%q thinking=%q", content, thinking)
		}
	}
}

func TestApertusParserSplitThinkingTags(t *testing.T) {
	p := &ApertusParser{}
	p.Init(nil, nil, &api.ThinkValue{Value: true})

	var content, thinking string
	for _, chunk := range []struct {
		text string
		done bool
	}{
		{"<|inner_pre", false},
		{"fix|>reason<|inner_suf", false},
		{"fix|>answer", true},
	} {
		gotContent, gotThinking, _, err := p.Add(chunk.text, chunk.done)
		if err != nil {
			t.Fatal(err)
		}
		content += gotContent
		thinking += gotThinking
	}
	if thinking != "reason" || content != "answer" {
		t.Fatalf("thinking=%q content=%q", thinking, content)
	}
}

func TestApertusParserStripsApertus1p5ContinuationFraming(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init(nil, nil, nil)

	content, thinking, calls, err := parser.Add(`<|tool_output_start|>{"temperature":22}<|tool_output_end|><|assistant_start|>It is mild.<|assistant_end|>`, true)
	if err != nil {
		t.Fatal(err)
	}
	if content != `{"temperature":22}It is mild.` || thinking != "" || len(calls) != 0 {
		t.Fatalf("content=%q thinking=%q calls=%d, want cleaned continuation text", content, thinking, len(calls))
	}
}

func TestApertusParserStripsAssistantEndAfterToolCall(t *testing.T) {
	parser := &ApertusParser{}
	parser.Init([]api.Tool{apertusParserTool("get_weather")}, nil, nil)

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

func apertusParserTool(name string) api.Tool {
	return api.Tool{Type: "function", Function: api.ToolFunction{Name: name}}
}

func TestApertus1p1ParserThinkingAndToolCall(t *testing.T) {
	parser := newApertus1p1Parser()
	parser.Init([]api.Tool{parserWeatherTool()}, nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := parser.Add(
		`<SPECIAL_67><SPECIAL_69>Need weather.<SPECIAL_71>[{"get_weather":{"location":"Zurich"}}]<SPECIAL_72><SPECIAL_68>`,
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || thinking != "Need weather." || len(calls) != 1 {
		t.Fatalf("content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("tool name = %q, want get_weather", calls[0].Function.Name)
	}
}

func TestApertus1p1ParserCanonicalToolCallThenResponse(t *testing.T) {
	parser := newApertus1p1Parser()
	parser.Init([]api.Tool{parserWeatherTool()}, nil, &api.ThinkValue{Value: true})

	var content, thinking strings.Builder
	var calls []api.ToolCall
	for i, chunk := range []string{
		`<SPECIAL_67><SPECIAL_69>Need weather.<SPECIAL_7`,
		`1>[{"get_weather":{"location":"Zurich"}}]<SPECIAL_72><SPEC`,
		`IAL_70>It is sunny.<SPECIAL_68>`,
	} {
		gotContent, gotThinking, gotCalls, err := parser.Add(chunk, i == 2)
		if err != nil {
			t.Fatal(err)
		}
		content.WriteString(gotContent)
		thinking.WriteString(gotThinking)
		calls = append(calls, gotCalls...)
	}
	if content.String() != "It is sunny." || thinking.String() != "Need weather." || len(calls) != 1 {
		t.Fatalf("content=%q thinking=%q calls=%d", content.String(), thinking.String(), len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("tool name = %q, want get_weather", calls[0].Function.Name)
	}
}

func TestApertus1p1ParserStreamingSplitTags(t *testing.T) {
	parser := newApertus1p1Parser()
	parser.Init([]api.Tool{parserWeatherTool()}, nil, nil)

	content, _, calls, err := parser.Add("Let me check.<SPEC", false)
	if err != nil {
		t.Fatal(err)
	}
	if content != "Let me check." || len(calls) != 0 {
		t.Fatalf("first chunk content=%q calls=%d", content, len(calls))
	}

	content, _, calls, err = parser.Add(`IAL_71>[{"get_weather":{"location":"Bern"}}]<SPECIAL_`, false)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || len(calls) != 0 {
		t.Fatalf("second chunk content=%q calls=%d", content, len(calls))
	}

	content, _, calls, err = parser.Add("72>", true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "" || len(calls) != 1 {
		t.Fatalf("final chunk content=%q calls=%d", content, len(calls))
	}
}

func TestApertus1p1ParserRegistrationAndTokens(t *testing.T) {
	parser := ParserForName("apertus1p1")
	if parser == nil || !parser.HasToolSupport() || !parser.HasThinkingSupport() {
		t.Fatalf("ParserForName(apertus1p1) = %T with incomplete capabilities", parser)
	}
	want := []string{"<SPECIAL_71>", "<SPECIAL_72>", "<SPECIAL_67>", "<SPECIAL_68>", "<SPECIAL_69>", "<SPECIAL_70>"}
	got := parser.PreservedTokens()
	if len(got) != len(want) {
		t.Fatalf("PreservedTokens() = %#v, want %#v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("PreservedTokens()[%d] = %q, want %q", i, got[i], want[i])
		}
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
