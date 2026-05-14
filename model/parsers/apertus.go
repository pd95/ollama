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
	apertusToolOpenTag       = "<|tools_prefix|>"
	apertusToolCloseTag      = "<|tools_suffix|>"
	apertusAssistantOpenTag  = "<|assistant_start|>"
	apertusAssistantCloseTag = "<|assistant_end|>"
)

type apertusParserState int

const (
	apertusParserStateContent apertusParserState = iota
	apertusParserStateToolCalls
)

type ApertusParser struct {
	state       apertusParserState
	acc         strings.Builder
	allowedTool map[string]bool
	callIndex   int
}

func (p *ApertusParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.allowedTool = make(map[string]bool, len(tools))
	for _, tool := range tools {
		p.allowedTool[tool.Function.Name] = true
	}
	p.callIndex = 0
	return tools
}

func (p *ApertusParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.acc.WriteString(s)

	var sb strings.Builder
	for {
		switch p.state {
		case apertusParserStateContent:
			current := p.acc.String()
			if idx := strings.Index(current, apertusToolOpenTag); idx >= 0 {
				before := cleanApertusContent(current[:idx])
				if before != "" {
					sb.WriteString(before)
				}
				p.acc.Reset()
				p.acc.WriteString(current[idx+len(apertusToolOpenTag):])
				p.state = apertusParserStateToolCalls
				continue
			}

			if done {
				cleaned := cleanApertusContent(current)
				if p.looksLikeToolCall(cleaned) {
					parsed, parseErr := p.parseToolCalls(cleaned)
					if parseErr == nil {
						p.acc.Reset()
						calls = append(calls, parsed...)
						return sb.String(), "", calls, nil
					}
				}
				sb.WriteString(cleaned)
				p.acc.Reset()
				return sb.String(), "", calls, nil
			}

			if p.looksLikeToolCallStart(current) {
				return sb.String(), "", calls, nil
			}

			overlapLen := overlap(current, apertusToolOpenTag)
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
				sb.WriteString(cleanApertusContent(emit))
				keep := current[emitLen:]
				p.acc.Reset()
				p.acc.WriteString(keep)
			}
			return sb.String(), "", calls, nil
		case apertusParserStateToolCalls:
			current := p.acc.String()
			if idx := strings.Index(current, apertusToolCloseTag); idx >= 0 {
				parsed, parseErr := p.parseToolCalls(current[:idx])
				if parseErr != nil {
					if isSoftApertusToolParseError(parseErr) {
						sb.WriteString(cleanApertusContent(current[:idx]))
						after := strings.TrimLeftFunc(current[idx+len(apertusToolCloseTag):], unicode.IsSpace)
						p.acc.Reset()
						p.acc.WriteString(after)
						p.state = apertusParserStateContent
						continue
					}
					return "", "", nil, parseErr
				}
				calls = append(calls, parsed...)
				after := strings.TrimLeftFunc(current[idx+len(apertusToolCloseTag):], unicode.IsSpace)
				p.acc.Reset()
				p.acc.WriteString(after)
				p.state = apertusParserStateContent
				continue
			}
			if done {
				parsed, parseErr := p.parseToolCalls(current)
				if parseErr != nil {
					if isSoftApertusToolParseError(parseErr) {
						sb.WriteString(cleanApertusContent(current))
						p.acc.Reset()
						p.state = apertusParserStateContent
						return sb.String(), "", calls, nil
					}
					return "", "", nil, fmt.Errorf("unterminated apertus tool call: %w", parseErr)
				}
				p.acc.Reset()
				p.state = apertusParserStateContent
				calls = append(calls, parsed...)
				return sb.String(), "", calls, nil
			}
			return sb.String(), "", calls, nil
		default:
			return "", "", nil, fmt.Errorf("unknown apertus parser state %d", p.state)
		}
	}
}

func cleanApertusContent(s string) string {
	s = strings.ReplaceAll(s, apertusAssistantOpenTag, "")
	s = strings.ReplaceAll(s, apertusAssistantCloseTag, "")
	return strings.TrimRightFunc(s, unicode.IsSpace)
}

func (p *ApertusParser) HasToolSupport() bool {
	return true
}

func (p *ApertusParser) HasThinkingSupport() bool {
	return false
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
