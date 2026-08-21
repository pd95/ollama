package parsers

import (
	"bytes"
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
	// A tool call is one model response fragment. Keep malformed streams from
	// retaining unbounded output while allowing substantially larger calls than
	// the model's normal tool grammar produces.
	maxApertusToolCallBytes = 1 << 20
)

type apertusParserState uint8

const (
	apertusContent apertusParserState = iota
	apertusThinking
	apertusToolCalls
)

type ApertusParser struct {
	state       apertusParserState
	acc         strings.Builder
	allowedTool map[string]struct{}
	initErr     error
	callIndex   int
	pendingBare bool
	thinking    bool
}

func (p *ApertusParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.state = apertusContent
	p.acc.Reset()
	p.allowedTool = make(map[string]struct{}, len(tools))
	p.initErr = nil
	p.callIndex = 0
	p.pendingBare = false
	p.thinking = thinkValue != nil && thinkValue.Bool()
	for _, tool := range tools {
		name := tool.Function.Name
		if !apertusIdentifier(name) {
			p.initErr = fmt.Errorf("invalid apertus tool name %q", name)
			continue
		}
		if _, exists := p.allowedTool[name]; exists {
			p.initErr = fmt.Errorf("duplicate apertus tool name %q", name)
			continue
		}
		p.allowedTool[name] = struct{}{}
	}
	return tools
}

func (p *ApertusParser) Add(s string, done bool) (content, thinking string, calls []api.ToolCall, err error) {
	if p.initErr != nil {
		return "", "", nil, p.initErr
	}
	if p.pendingFragmentWouldExceed(s) {
		return "", "", nil, fmt.Errorf("apertus tool call exceeds %d bytes", maxApertusToolCallBytes)
	}
	p.acc.WriteString(s)
	if p.state == apertusToolCalls && p.acc.Len() > maxApertusToolCallBytes {
		return "", "", nil, fmt.Errorf("apertus tool call exceeds %d bytes", maxApertusToolCallBytes)
	}

	var out, thought strings.Builder
	for {
		current := p.acc.String()
		switch p.state {
		case apertusContent:
			innerIdx := strings.Index(current, apertusInnerOpenTag)
			toolIdx := strings.Index(current, apertusToolOpenTag)
			if innerIdx >= 0 && (toolIdx < 0 || innerIdx < toolIdx) {
				out.WriteString(cleanApertusContent(current[:innerIdx]))
				p.acc.Reset()
				p.acc.WriteString(current[innerIdx+len(apertusInnerOpenTag):])
				p.state = apertusThinking
				continue
			}
			if idx := toolIdx; idx >= 0 {
				p.pendingBare = false
				out.WriteString(cleanApertusContent(current[:idx]))
				p.acc.Reset()
				p.acc.WriteString(current[idx+len(apertusToolOpenTag):])
				p.state = apertusToolCalls
				if p.acc.Len() > maxApertusToolCallBytes {
					return "", "", nil, fmt.Errorf("apertus tool call exceeds %d bytes", maxApertusToolCallBytes)
				}
				continue
			}
			if done {
				cleaned := cleanApertusContent(current)
				if p.looksLikeToolCall(cleaned) {
					if parsed, parseErr := p.parseToolCalls(cleaned); parseErr == nil {
						p.acc.Reset()
						p.pendingBare = false
						return out.String(), thought.String(), parsed, nil
					} else if !isSoftApertusToolParseError(parseErr) {
						return "", "", nil, parseErr
					}
				}
				p.acc.Reset()
				p.pendingBare = false
				out.WriteString(cleaned)
				return out.String(), thought.String(), calls, nil
			}
			if p.looksLikeToolCallStart(current) {
				p.pendingBare = true
				if p.acc.Len() > maxApertusToolCallBytes {
					return "", "", nil, fmt.Errorf("apertus tool call exceeds %d bytes", maxApertusToolCallBytes)
				}
				return out.String(), thought.String(), nil, nil
			}
			p.pendingBare = false
			overlapLen := max(
				overlap(current, apertusToolOpenTag),
				overlap(current, apertusAssistantOpenTag),
				overlap(current, apertusAssistantCloseTag),
				overlap(current, apertusInnerOpenTag),
			)
			n := len(current) - overlapLen
			if n == len(current) {
				n -= trailingWhitespaceLen(current)
			}
			if n > 0 {
				emit := current[:n]
				if n < len(current) {
					emit = strings.TrimRightFunc(emit, unicode.IsSpace)
				}
				out.WriteString(cleanApertusContent(emit))
				p.acc.Reset()
				p.acc.WriteString(current[n:])
			}
			return out.String(), thought.String(), calls, nil

		case apertusThinking:
			if idx := strings.Index(current, apertusInnerCloseTag); idx >= 0 {
				inner := current[:idx]
				if p.thinking {
					thought.WriteString(inner)
				} else {
					out.WriteString(cleanApertusContent(inner))
				}
				p.acc.Reset()
				p.acc.WriteString(strings.TrimLeftFunc(current[idx+len(apertusInnerCloseTag):], unicode.IsSpace))
				p.state = apertusContent
				continue
			}
			if done {
				if p.thinking {
					thought.WriteString(current)
				} else {
					out.WriteString(cleanApertusContent(current))
				}
				p.acc.Reset()
				p.state = apertusContent
				return out.String(), thought.String(), calls, nil
			}
			n := len(current) - overlap(current, apertusInnerCloseTag)
			if n > 0 {
				emit := current[:n]
				if p.thinking {
					thought.WriteString(emit)
				} else {
					out.WriteString(cleanApertusContent(emit))
				}
				p.acc.Reset()
				p.acc.WriteString(current[n:])
			}
			return out.String(), thought.String(), calls, nil

		case apertusToolCalls:
			if idx := strings.Index(current, apertusToolCloseTag); idx >= 0 {
				parsed, parseErr := p.parseToolCalls(current[:idx])
				if parseErr != nil {
					if !isSoftApertusToolParseError(parseErr) {
						return "", "", nil, parseErr
					}
					out.WriteString(cleanApertusContent(current[:idx]))
				} else {
					calls = append(calls, parsed...)
				}
				p.acc.Reset()
				p.acc.WriteString(strings.TrimLeftFunc(current[idx+len(apertusToolCloseTag):], unicode.IsSpace))
				p.state = apertusContent
				p.pendingBare = false
				continue
			}
			if done {
				parsed, parseErr := p.parseToolCalls(current)
				if parseErr != nil {
					if !isSoftApertusToolParseError(parseErr) {
						return "", "", nil, fmt.Errorf("unterminated apertus tool call: %w", parseErr)
					}
					out.WriteString(cleanApertusContent(current))
				} else {
					calls = append(calls, parsed...)
				}
				p.acc.Reset()
				p.state = apertusContent
				p.pendingBare = false
				return out.String(), thought.String(), calls, nil
			}
			return out.String(), thought.String(), calls, nil
		}
	}
}

func (p *ApertusParser) pendingFragmentWouldExceed(s string) bool {
	if p.acc.Len() > maxApertusToolCallBytes {
		return true
	}
	if p.state == apertusToolCalls || p.pendingBare {
		return len(s) > maxApertusToolCallBytes-p.acc.Len()
	}
	if p.acc.Len() > 0 && strings.HasPrefix(apertusToolOpenTag, p.acc.String()) {
		remainingOpener := len(apertusToolOpenTag) - p.acc.Len()
		return len(s) > remainingOpener+maxApertusToolCallBytes
	}

	probeLen := min(len(s), maxApertusToolCallBytes+1-p.acc.Len())
	probe := p.acc.String() + s[:probeLen]
	if idx := strings.Index(probe, apertusToolOpenTag); idx >= 0 {
		return p.acc.Len()+len(s)-idx-len(apertusToolOpenTag) > maxApertusToolCallBytes
	}
	return apertusPendingBarePrefix(probe) && p.acc.Len()+len(s) > maxApertusToolCallBytes
}

func apertusPendingBarePrefix(s string) bool {
	s = strings.TrimLeftFunc(s, unicode.IsSpace)
	for {
		switch {
		case strings.HasPrefix(s, apertusAssistantOpenTag):
			s = strings.TrimLeftFunc(strings.TrimPrefix(s, apertusAssistantOpenTag), unicode.IsSpace)
		case strings.HasPrefix(s, apertusAssistantCloseTag):
			s = strings.TrimLeftFunc(strings.TrimPrefix(s, apertusAssistantCloseTag), unicode.IsSpace)
		default:
			return strings.TrimSpace(s) == "" || strings.HasPrefix("[{", s) || strings.HasPrefix("{", s) || strings.HasPrefix(s, "[{") || strings.HasPrefix(s, "{")
		}
	}
}

func cleanApertusContent(s string) string {
	s = strings.ReplaceAll(s, apertusAssistantOpenTag, "")
	s = strings.ReplaceAll(s, apertusAssistantCloseTag, "")
	s = strings.ReplaceAll(s, apertusInnerOpenTag, "")
	s = strings.ReplaceAll(s, apertusInnerCloseTag, "")
	s = strings.ReplaceAll(s, apertusToolOutputOpenTag, "")
	s = strings.ReplaceAll(s, apertusToolOutputCloseTag, "")
	s = strings.ReplaceAll(s, apertusToolOutputOpenTag, "")
	s = strings.ReplaceAll(s, apertusToolOutputCloseTag, "")
	return strings.TrimRightFunc(s, unicode.IsSpace)
}
func (p *ApertusParser) HasToolSupport() bool     { return true }
func (p *ApertusParser) HasThinkingSupport() bool { return true }
func (p *ApertusParser) PreservedTokens() []string {
	return []string{apertusToolOpenTag, apertusToolCloseTag, apertusAssistantOpenTag, apertusAssistantCloseTag, apertusInnerOpenTag, apertusInnerCloseTag}
}

func (p *ApertusParser) parseToolCalls(raw string) ([]api.ToolCall, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, errors.New("empty apertus tool call")
	}
	var entries []json.RawMessage
	if raw[0] == '[' {
		if err := json.Unmarshal([]byte(raw), &entries); err != nil {
			return nil, err
		}
	} else {
		entries = []json.RawMessage{json.RawMessage(raw)}
	}
	var calls []api.ToolCall
	for _, entry := range entries {
		name, rawArgs, err := apertusCallEntry(entry)
		if err != nil {
			return nil, err
		}
		if _, ok := p.allowedTool[name]; !ok {
			return nil, fmt.Errorf("unknown apertus tool %q", name)
		}
		args := api.NewToolCallFunctionArguments()
		if len(rawArgs) > 0 && !bytes.Equal(rawArgs, []byte("null")) {
			if err := json.Unmarshal(rawArgs, &args); err != nil {
				var encoded string
				if stringErr := json.Unmarshal(rawArgs, &encoded); stringErr != nil {
					return nil, err
				}
				if err := json.Unmarshal([]byte(encoded), &args); err != nil {
					return nil, err
				}
			}
		}
		calls = append(calls, api.ToolCall{Function: api.ToolCallFunction{Index: p.callIndex, Name: name, Arguments: args}})
		p.callIndex++
	}
	return calls, nil
}

func apertusCallEntry(raw json.RawMessage) (string, json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return "", nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return "", nil, errors.New("apertus tool call must be an object")
	}
	if !dec.More() {
		return "", nil, errors.New("apertus tool call object must contain exactly one function name")
	}
	nameTok, err := dec.Token()
	if err != nil {
		return "", nil, err
	}
	name, ok := nameTok.(string)
	if !ok {
		return "", nil, errors.New("invalid apertus tool name")
	}
	var args json.RawMessage
	if err := dec.Decode(&args); err != nil {
		return "", nil, err
	}
	if dec.More() {
		return "", nil, errors.New("apertus tool call object must contain exactly one function name")
	}
	if _, err := dec.Token(); err != nil {
		return "", nil, err
	}
	if dec.More() {
		return "", nil, errors.New("invalid trailing apertus tool call data")
	}
	return name, args, nil
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
	s = strings.TrimSpace(s)
	return len(p.allowedTool) > 0 && (strings.HasPrefix(s, "[{") || strings.HasPrefix(s, "{"))
}

func (p *ApertusParser) looksLikeToolCallStart(s string) bool {
	s = strings.TrimSpace(s)
	return len(p.allowedTool) > 0 && s != "" && (strings.HasPrefix("[{", s) || strings.HasPrefix("{", s) || strings.HasPrefix(s, "[{") || strings.HasPrefix(s, "{"))
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
