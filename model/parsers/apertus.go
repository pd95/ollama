package parsers

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

const (
	apertusToolOpenTag        = "<|tools_prefix|>"
	apertusToolCloseTag       = "<|tools_suffix|>"
	apertusAssistantOpenTag   = "<|assistant_start|>"
	apertusAssistantCloseTag  = "<|assistant_end|>"
	apertusInnerOpenTag       = "<|inner_prefix|>"
	apertusInnerCloseTag      = "<|inner_suffix|>"
	apertusToolOutputOpenTag  = "<|tool_output_start|>"
	apertusToolOutputCloseTag = "<|tool_output_end|>"
)

type apertusGrammar struct {
	toolOpen        string
	toolClose       string
	assistantOpen   string
	assistantClose  string
	innerOpen       string
	innerClose      string
	toolOutputOpen  string
	toolOutputClose string
}

var (
	apertusLegacyGrammar = apertusGrammar{
		toolOpen:        apertusToolOpenTag,
		toolClose:       apertusToolCloseTag,
		assistantOpen:   apertusAssistantOpenTag,
		assistantClose:  apertusAssistantCloseTag,
		innerOpen:       apertusInnerOpenTag,
		innerClose:      apertusInnerCloseTag,
		toolOutputOpen:  apertusToolOutputOpenTag,
		toolOutputClose: apertusToolOutputCloseTag,
	}
	apertus1p1Grammar = apertusGrammar{
		toolOpen:       "<SPECIAL_71>",
		toolClose:      "<SPECIAL_72>",
		assistantOpen:  "<SPECIAL_67>",
		assistantClose: "<SPECIAL_68>",
		innerOpen:      "<SPECIAL_69>",
		innerClose:     "<SPECIAL_70>",
	}
)

type apertusParserState int

const (
	apertusParserStateContent apertusParserState = iota
	apertusParserStateThinking
	apertusParserStateToolCalls
)

type ApertusParser struct {
	state       apertusParserState
	returnState apertusParserState
	acc         strings.Builder
	allowedTool map[string]bool
	callIndex   int
	thinking    bool
	grammar     apertusGrammar
}

func newApertus1p1Parser() *ApertusParser {
	return &ApertusParser{grammar: apertus1p1Grammar}
}

func (p *ApertusParser) parserGrammar() apertusGrammar {
	if p.grammar.toolOpen == "" {
		return apertusLegacyGrammar
	}
	return p.grammar
}

func (p *ApertusParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.state = apertusParserStateContent
	p.returnState = apertusParserStateContent
	p.acc.Reset()
	p.allowedTool = make(map[string]bool, len(tools))
	for _, tool := range tools {
		p.allowedTool[tool.Function.Name] = true
	}
	p.callIndex = 0
	p.thinking = thinkValue != nil && thinkValue.Bool()
	return tools
}

func (p *ApertusParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.acc.WriteString(s)
	tags := p.parserGrammar()

	var contentSB strings.Builder
	var thinkingSB strings.Builder
	for {
		switch p.state {
		case apertusParserStateContent:
			current := p.acc.String()
			innerIdx := strings.Index(current, tags.innerOpen)
			toolIdx := strings.Index(current, tags.toolOpen)
			if innerIdx >= 0 && (toolIdx < 0 || innerIdx < toolIdx) {
				before := cleanApertusContentWithGrammar(current[:innerIdx], tags)
				if before != "" {
					contentSB.WriteString(before)
				}
				p.acc.Reset()
				p.acc.WriteString(current[innerIdx+len(tags.innerOpen):])
				p.state = apertusParserStateThinking
				continue
			}

			if idx := toolIdx; idx >= 0 {
				before := cleanApertusContentWithGrammar(current[:idx], tags)
				if before != "" {
					contentSB.WriteString(before)
				}
				p.acc.Reset()
				p.acc.WriteString(current[idx+len(tags.toolOpen):])
				p.returnState = apertusParserStateContent
				p.state = apertusParserStateToolCalls
				continue
			}

			if done {
				cleaned := cleanApertusContentWithGrammar(current, tags)
				if p.looksLikeToolCall(cleaned) {
					parsed, parseErr := p.parseToolCalls(cleaned)
					if parseErr == nil {
						p.acc.Reset()
						calls = append(calls, parsed...)
						return contentSB.String(), thinkingSB.String(), calls, nil
					}
				}
				contentSB.WriteString(cleaned)
				p.acc.Reset()
				return contentSB.String(), thinkingSB.String(), calls, nil
			}

			if p.looksLikeToolCallStart(current) {
				return contentSB.String(), thinkingSB.String(), calls, nil
			}

			overlapLen := max(overlap(current, tags.toolOpen), overlap(current, tags.innerOpen))
			emitLen := len(current) - overlapLen
			if overlapLen == 0 {
				wsLen := trailingWhitespaceLen(current)
				emitLen = len(current) - wsLen
			}
			if emitLen > 0 {
				emit := current[:emitLen]
				if overlapLen > 0 {
					emit = strings.TrimRightFunc(emit, unicode.IsSpace)
				}
				contentSB.WriteString(cleanApertusContentWithGrammar(emit, tags))
				keep := current[emitLen:]
				p.acc.Reset()
				p.acc.WriteString(keep)
			}
			return contentSB.String(), thinkingSB.String(), calls, nil
		case apertusParserStateThinking:
			current := p.acc.String()
			innerIdx := strings.Index(current, tags.innerClose)
			toolIdx := strings.Index(current, tags.toolOpen)
			assistantIdx := strings.Index(current, tags.assistantClose)
			if toolIdx >= 0 && (innerIdx < 0 || toolIdx < innerIdx) && (assistantIdx < 0 || toolIdx < assistantIdx) {
				if p.thinking {
					thinkingSB.WriteString(current[:toolIdx])
				} else {
					contentSB.WriteString(cleanApertusContentWithGrammar(current[:toolIdx], tags))
				}
				p.acc.Reset()
				p.acc.WriteString(current[toolIdx+len(tags.toolOpen):])
				p.returnState = apertusParserStateThinking
				p.state = apertusParserStateToolCalls
				continue
			}
			closeIdx := innerIdx
			closeLen := len(tags.innerClose)
			if assistantIdx >= 0 && (closeIdx < 0 || assistantIdx < closeIdx) {
				closeIdx = assistantIdx
				closeLen = len(tags.assistantClose)
			}
			if closeIdx >= 0 {
				if p.thinking {
					thinkingSB.WriteString(current[:closeIdx])
				} else {
					contentSB.WriteString(cleanApertusContentWithGrammar(current[:closeIdx], tags))
				}
				after := strings.TrimLeftFunc(current[closeIdx+closeLen:], unicode.IsSpace)
				p.acc.Reset()
				p.acc.WriteString(after)
				p.state = apertusParserStateContent
				continue
			}
			if done {
				if p.thinking {
					thinkingSB.WriteString(current)
				} else {
					contentSB.WriteString(cleanApertusContentWithGrammar(current, tags))
				}
				p.acc.Reset()
				p.state = apertusParserStateContent
				return contentSB.String(), thinkingSB.String(), calls, nil
			}

			overlapLen := max(overlap(current, tags.innerClose), overlap(current, tags.toolOpen), overlap(current, tags.assistantClose))
			emitLen := len(current) - overlapLen
			if overlapLen == 0 {
				emitLen = len(current) - trailingWhitespaceLen(current)
			}
			if emitLen > 0 {
				emit := current[:emitLen]
				if p.thinking {
					thinkingSB.WriteString(emit)
				} else {
					contentSB.WriteString(cleanApertusContentWithGrammar(emit, tags))
				}
				keep := current[emitLen:]
				p.acc.Reset()
				p.acc.WriteString(keep)
			}
			return contentSB.String(), thinkingSB.String(), calls, nil
		case apertusParserStateToolCalls:
			current := p.acc.String()
			if idx := strings.Index(current, tags.toolClose); idx >= 0 {
				parsed, parseErr := p.parseToolCalls(current[:idx])
				if parseErr != nil {
					if isSoftApertusToolParseError(parseErr) {
						contentSB.WriteString(cleanApertusContentWithGrammar(current[:idx], tags))
						after := strings.TrimLeftFunc(current[idx+len(tags.toolClose):], unicode.IsSpace)
						p.acc.Reset()
						p.acc.WriteString(after)
						p.state = p.returnState
						continue
					}
					return "", "", nil, parseErr
				}
				calls = append(calls, parsed...)
				after := strings.TrimLeftFunc(current[idx+len(tags.toolClose):], unicode.IsSpace)
				p.acc.Reset()
				p.acc.WriteString(after)
				p.state = p.returnState
				continue
			}
			if done {
				parsed, parseErr := p.parseToolCalls(current)
				if parseErr != nil {
					if isSoftApertusToolParseError(parseErr) {
						contentSB.WriteString(cleanApertusContentWithGrammar(current, tags))
						p.acc.Reset()
						p.state = p.returnState
						return contentSB.String(), thinkingSB.String(), calls, nil
					}
					return "", "", nil, fmt.Errorf("unterminated apertus tool call: %w", parseErr)
				}
				p.acc.Reset()
				p.state = p.returnState
				calls = append(calls, parsed...)
				return contentSB.String(), thinkingSB.String(), calls, nil
			}
			return contentSB.String(), thinkingSB.String(), calls, nil
		default:
			return "", "", nil, fmt.Errorf("unknown apertus parser state %d", p.state)
		}
	}
}

func cleanApertusContentWithGrammar(s string, tags apertusGrammar) string {
	for _, tag := range []string{
		tags.assistantOpen,
		tags.assistantClose,
		tags.innerOpen,
		tags.innerClose,
		tags.toolOutputOpen,
		tags.toolOutputClose,
	} {
		if tag != "" {
			s = strings.ReplaceAll(s, tag, "")
		}
	}
	return strings.TrimRightFunc(s, unicode.IsSpace)
}

func (p *ApertusParser) HasToolSupport() bool {
	return true
}

func (p *ApertusParser) HasThinkingSupport() bool {
	return true
}

func (p *ApertusParser) PreservedTokens() []string {
	tags := p.parserGrammar()
	return []string{
		tags.toolOpen,
		tags.toolClose,
		tags.assistantOpen,
		tags.assistantClose,
		tags.innerOpen,
		tags.innerClose,
	}
}

func (p *ApertusParser) parseToolCalls(raw string) ([]api.ToolCall, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, fmt.Errorf("empty apertus tool call")
	}

	var entries []map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &entries); err != nil {
		var single map[string]json.RawMessage
		if singleErr := json.Unmarshal([]byte(raw), &single); singleErr != nil {
			return nil, err
		}
		entries = []map[string]json.RawMessage{single}
	}

	var calls []api.ToolCall
	for _, entry := range entries {
		if len(entry) != 1 {
			return nil, fmt.Errorf("apertus tool call object must contain exactly one function name")
		}
		for name, rawArgs := range entry {
			if len(p.allowedTool) > 0 && !p.allowedTool[name] {
				return nil, fmt.Errorf("unknown apertus tool %q", name)
			}

			args := api.NewToolCallFunctionArguments()
			if len(rawArgs) > 0 && string(rawArgs) != "null" {
				if err := json.Unmarshal(rawArgs, &args); err != nil {
					var decoded string
					if stringErr := json.Unmarshal(rawArgs, &decoded); stringErr != nil {
						return nil, err
					}
					if err := json.Unmarshal([]byte(decoded), &args); err != nil {
						return nil, err
					}
				}
			}

			calls = append(calls, api.ToolCall{
				Function: api.ToolCallFunction{
					Index:     p.callIndex,
					Name:      name,
					Arguments: args,
				},
			})
			p.callIndex++
		}
	}

	return calls, nil
}

func isSoftApertusToolParseError(err error) bool {
	if strings.Contains(err.Error(), "empty apertus tool call") {
		return true
	}

	var syntaxErr *json.SyntaxError
	if errors.As(err, &syntaxErr) {
		return true
	}

	var typeErr *json.UnmarshalTypeError
	return errors.As(err, &typeErr)
}

func (p *ApertusParser) looksLikeToolCall(s string) bool {
	if len(p.allowedTool) == 0 {
		return false
	}
	s = strings.TrimSpace(s)
	return strings.HasPrefix(s, "[{") || strings.HasPrefix(s, "{")
}

func (p *ApertusParser) looksLikeToolCallStart(s string) bool {
	if len(p.allowedTool) == 0 {
		return false
	}
	s = strings.TrimSpace(s)
	// The first two checks intentionally ask whether this chunk is a prefix of
	// a JSON tool-call start, so split chunks like "[" are buffered.
	return strings.HasPrefix("[{", s) || strings.HasPrefix("{", s) || strings.HasPrefix(s, "[{") || strings.HasPrefix(s, "{")
}
