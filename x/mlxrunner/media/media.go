// Package media provides architecture-neutral helpers for preparing media
// prompts and replacing token embeddings in MLX models.
package media

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

var markerPattern = regexp.MustCompile(`\[img-([^]]*)\]`)

// Binding associates an input with its stable prompt marker and replacement.
type Binding struct {
	Media          llm.MediaData
	MarkerStart    int
	MarkerEnd      int
	Sequence       string
	ExpectedTokens int
}

// Item describes the token span occupied by one media input.
type Item struct {
	ID             int
	Kind           llm.MediaKind
	Span           batch.TokenSpan
	ExpectedTokens int
}

// Replacement associates a media span with its encoded MLX features.
type Replacement struct {
	Span     batch.TokenSpan
	Features *mlx.Array
}

// BindMarkers binds Ollama's stable [img-N] markers to media inputs. Returned
// bindings follow prompt order, irrespective of input order.
func BindMarkers(prompt string, inputs []llm.MediaData) ([]Binding, error) {
	if len(inputs) == 0 {
		return nil, errors.New("media request contains no media")
	}
	if strings.Contains(prompt, "[img]") {
		return nil, errors.New("prompt contains an unnumbered [img] marker")
	}

	byID := make(map[int]llm.MediaData, len(inputs))
	for _, input := range inputs {
		if input.ID < 0 {
			return nil, fmt.Errorf("media ID must be non-negative, got %d", input.ID)
		}
		if _, ok := byID[input.ID]; ok {
			return nil, fmt.Errorf("duplicate media ID %d", input.ID)
		}
		byID[input.ID] = input
	}

	matches := markerPattern.FindAllStringSubmatchIndex(prompt, -1)
	bindings := make([]Binding, 0, len(matches))
	seen := make(map[int]bool, len(matches))
	for _, match := range matches {
		rawID := prompt[match[2]:match[3]]
		id, err := strconv.Atoi(rawID)
		if err != nil || id < 0 || strconv.Itoa(id) != rawID {
			return nil, fmt.Errorf("invalid media marker %q", prompt[match[0]:match[1]])
		}
		input, ok := byID[id]
		if !ok {
			return nil, fmt.Errorf("prompt marker [img-%d] has no matching media", id)
		}
		if seen[id] {
			return nil, fmt.Errorf("prompt contains multiple [img-%d] markers", id)
		}
		seen[id] = true
		bindings = append(bindings, Binding{Media: input, MarkerStart: match[0], MarkerEnd: match[1]})
	}

	withoutMarkers := markerPattern.ReplaceAllString(prompt, "")
	if strings.Contains(withoutMarkers, "[img-") {
		return nil, errors.New("prompt contains a malformed media marker")
	}
	for _, input := range inputs {
		if !seen[input.ID] {
			return nil, fmt.Errorf("expected one [img-%d] marker in prompt", input.ID)
		}
	}
	if len(bindings) != len(inputs) {
		return nil, fmt.Errorf("prompt contains %d media markers for %d inputs", len(bindings), len(inputs))
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].MarkerStart < bindings[j].MarkerStart })
	return bindings, nil
}

// ExpandPrompt replaces bound markers while preserving their prompt order.
func ExpandPrompt(prompt string, bindings []Binding) (string, error) {
	cursor := 0
	replacements := make([]replacementSize, len(bindings))
	for i, binding := range bindings {
		if binding.MarkerStart < cursor || binding.MarkerEnd < binding.MarkerStart || binding.MarkerEnd > len(prompt) {
			return "", errors.New("media markers overlap or are out of range")
		}
		replacements[i] = replacementSize{removed: binding.MarkerEnd - binding.MarkerStart, added: len(binding.Sequence)}
		cursor = binding.MarkerEnd
	}
	expandedLen, err := checkedExpandedLength(len(prompt), replacements)
	if err != nil {
		return "", err
	}

	var expanded strings.Builder
	expanded.Grow(expandedLen)
	cursor = 0
	for _, binding := range bindings {
		expanded.WriteString(prompt[cursor:binding.MarkerStart])
		expanded.WriteString(binding.Sequence)
		cursor = binding.MarkerEnd
	}
	expanded.WriteString(prompt[cursor:])
	return expanded.String(), nil
}

type replacementSize struct {
	removed int
	added   int
}

func checkedExpandedLength(promptLength int, replacements []replacementSize) (int, error) {
	if promptLength < 0 {
		return 0, errors.New("expanded media prompt is too large")
	}
	total := promptLength
	for _, replacement := range replacements {
		if replacement.removed < 0 || replacement.removed > total || replacement.added < 0 || replacement.added > math.MaxInt-total+replacement.removed {
			return 0, errors.New("expanded media prompt is too large")
		}
		total += replacement.added - replacement.removed
	}
	return total, nil
}

// AddTokenBudget adds a media token count without overflowing int and rejects
// non-positive counts.
func AddTokenBudget(total, count int) (int, error) {
	if count <= 0 {
		return 0, fmt.Errorf("invalid media token count %d", count)
	}
	if total < 0 || count > math.MaxInt-total {
		return 0, errors.New("media token budget overflows int")
	}
	return total + count, nil
}

// AssignTokenSpans assigns contiguous placeholder runs to bindings by kind and
// validates that the resulting spans still follow marker order.
func AssignTokenSpans(tokens []int32, bindings []Binding, tokenIDs map[llm.MediaKind]int32) ([]Item, error) {
	usedKinds := make(map[llm.MediaKind]bool)
	ids := make(map[int32]llm.MediaKind)
	for _, binding := range bindings {
		if binding.ExpectedTokens <= 0 {
			return nil, fmt.Errorf("invalid expected token count %d for media %d", binding.ExpectedTokens, binding.Media.ID)
		}
		id, ok := tokenIDs[binding.Media.Kind]
		if !ok {
			return nil, fmt.Errorf("missing placeholder token ID for media kind %q", binding.Media.Kind)
		}
		if other, exists := ids[id]; exists && other != binding.Media.Kind {
			return nil, fmt.Errorf("media kinds %q and %q use the same placeholder token ID %d", other, binding.Media.Kind, id)
		}
		ids[id] = binding.Media.Kind
		usedKinds[binding.Media.Kind] = true
	}

	spans := make(map[llm.MediaKind][]batch.TokenSpan, len(usedKinds))
	for kind := range usedKinds {
		spans[kind] = tokenSpans(tokens, tokenIDs[kind])
	}
	indices := make(map[llm.MediaKind]int, len(usedKinds))
	items := make([]Item, 0, len(bindings))
	previousEnd := 0
	for _, binding := range bindings {
		kindSpans := spans[binding.Media.Kind]
		index := indices[binding.Media.Kind]
		if index >= len(kindSpans) {
			return nil, fmt.Errorf("prompt contains no %s token span for media %d", binding.Media.Kind, binding.Media.ID)
		}
		span := kindSpans[index]
		indices[binding.Media.Kind]++
		if span.Start < previousEnd {
			return nil, fmt.Errorf("media token spans do not follow prompt marker order at media %d", binding.Media.ID)
		}
		if got := span.End - span.Start; got != binding.ExpectedTokens {
			return nil, fmt.Errorf("media %d token count = %d, want %d", binding.Media.ID, got, binding.ExpectedTokens)
		}
		items = append(items, Item{ID: binding.Media.ID, Kind: binding.Media.Kind, Span: span, ExpectedTokens: binding.ExpectedTokens})
		previousEnd = span.End
	}
	for kind, kindSpans := range spans {
		if indices[kind] != len(kindSpans) {
			return nil, fmt.Errorf("prompt contains unexpected %s token spans: %d/%d", kind, indices[kind], len(kindSpans))
		}
	}
	return items, nil
}

// ValidateItems validates IDs, order, overlap, bounds, and expected span sizes.
func ValidateItems(items []Item, tokenCount int) error {
	if tokenCount <= 0 {
		return errors.New("invalid empty prepared tokens")
	}
	if len(items) == 0 {
		return errors.New("media payload contains no items")
	}
	previousEnd := 0
	seenIDs := make(map[int]bool, len(items))
	for _, item := range items {
		if item.ID < 0 || seenIDs[item.ID] {
			return fmt.Errorf("invalid or duplicate media ID %d", item.ID)
		}
		seenIDs[item.ID] = true
		if item.Span.Start < previousEnd || item.Span.Start < 0 || item.Span.End <= item.Span.Start || item.Span.End > tokenCount {
			return fmt.Errorf("invalid or overlapping media %d span [%d,%d) for %d tokens", item.ID, item.Span.Start, item.Span.End, tokenCount)
		}
		if item.ExpectedTokens <= 0 || item.Span.End-item.Span.Start != item.ExpectedTokens {
			return fmt.Errorf("invalid media %d span length %d, want %d", item.ID, item.Span.End-item.Span.Start, item.ExpectedTokens)
		}
		previousEnd = item.Span.End
	}
	return nil
}

// MergeEmbeddings replaces media spans in [1, sequence, hidden] token
// embeddings and preserves the original sequence length.
func MergeEmbeddings(embeddings *mlx.Array, tokenCount int, replacements []Replacement) (*mlx.Array, error) {
	if embeddings == nil || embeddings.NumDims() != 3 || embeddings.Dim(0) != 1 || embeddings.Dim(1) != tokenCount {
		return nil, errors.New("invalid token embeddings; want shape [1,sequence,hidden]")
	}
	if tokenCount > math.MaxInt32 {
		return nil, fmt.Errorf("input length %d exceeds MLX index range", tokenCount)
	}
	hidden := embeddings.Dim(2)
	parts := make([]*mlx.Array, 0, 2*len(replacements)+1)
	cursor := 0
	for _, replacement := range replacements {
		span := replacement.Span
		if span.Start < cursor || span.Start < 0 || span.End <= span.Start || span.End > tokenCount {
			return nil, fmt.Errorf("invalid or overlapping media span [%d,%d) for %d tokens", span.Start, span.End, tokenCount)
		}
		if replacement.Features == nil || replacement.Features.NumDims() != 3 || replacement.Features.Dim(0) != 1 ||
			replacement.Features.Dim(1) != span.End-span.Start || replacement.Features.Dim(2) != hidden {
			return nil, fmt.Errorf("invalid media features for span [%d,%d); want shape [1,%d,%d]", span.Start, span.End, span.End-span.Start, hidden)
		}
		if cursor < span.Start {
			parts = append(parts, mlx.SliceStartStop(embeddings, []int32{0, int32(cursor), 0}, []int32{1, int32(span.Start), int32(hidden)}))
		}
		parts = append(parts, replacement.Features)
		cursor = span.End
	}
	if cursor < tokenCount {
		parts = append(parts, mlx.SliceStartStop(embeddings, []int32{0, int32(cursor), 0}, []int32{1, int32(tokenCount), int32(hidden)}))
	}
	if len(replacements) == 0 {
		return nil, errors.New("media payload contains no embedding replacements")
	}
	if len(parts) == 1 {
		return parts[0], nil
	}
	return mlx.Concatenate(parts, 1), nil
}

func tokenSpans(tokens []int32, tokenID int32) []batch.TokenSpan {
	var spans []batch.TokenSpan
	for i := 0; i < len(tokens); {
		if tokens[i] != tokenID {
			i++
			continue
		}
		start := i
		for i < len(tokens) && tokens[i] == tokenID {
			i++
		}
		spans = append(spans, batch.TokenSpan{Start: start, End: i})
	}
	return spans
}
