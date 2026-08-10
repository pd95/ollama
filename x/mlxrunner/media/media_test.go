package media

import (
	"math"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestBindMarkersPreservesIdentityAndOrder(t *testing.T) {
	inputs := []llm.MediaData{{ID: 0, Kind: llm.MediaKindImage}, {ID: 7, Kind: llm.MediaKindAudio}}
	bindings, err := BindMarkers("a [img-7] b [img-0]", inputs)
	if err != nil {
		t.Fatal(err)
	}
	if len(bindings) != 2 || bindings[0].Media.ID != 7 || bindings[1].Media.ID != 0 {
		t.Fatalf("bindings = %#v, want IDs 7, 0", bindings)
	}
	bindings[0].Sequence, bindings[1].Sequence = "A", "I"
	got, err := ExpandPrompt("a [img-7] b [img-0]", bindings)
	if err != nil || got != "a A b I" {
		t.Fatalf("ExpandPrompt() = %q, %v", got, err)
	}
}

func TestBindMarkersRejectsMalformedMappings(t *testing.T) {
	input := llm.MediaData{ID: 0, Kind: llm.MediaKindImage}
	for _, tc := range []struct {
		name, prompt, want string
		inputs             []llm.MediaData
	}{
		{name: "empty", prompt: "text", want: "no media"},
		{name: "unnumbered", prompt: "[img]", inputs: []llm.MediaData{input}, want: "unnumbered"},
		{name: "missing", prompt: "text", inputs: []llm.MediaData{input}, want: "expected one"},
		{name: "unknown", prompt: "[img-1]", inputs: []llm.MediaData{input}, want: "no matching"},
		{name: "duplicate marker", prompt: "[img-0][img-0]", inputs: []llm.MediaData{input}, want: "multiple"},
		{name: "duplicate input", prompt: "[img-0]", inputs: []llm.MediaData{input, input}, want: "duplicate"},
		{name: "negative input", prompt: "[img-0]", inputs: []llm.MediaData{{ID: -1}}, want: "non-negative"},
		{name: "noncanonical", prompt: "[img-00]", inputs: []llm.MediaData{input}, want: "invalid"},
		{name: "malformed", prompt: "[img-zero]", inputs: []llm.MediaData{input}, want: "invalid"},
		{name: "unterminated", prompt: "[img-0", inputs: []llm.MediaData{input}, want: "malformed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := BindMarkers(tc.prompt, tc.inputs)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestExpansionAndBudgetOverflow(t *testing.T) {
	if _, err := checkedExpandedLength(math.MaxInt, []replacementSize{{added: 1}}); err == nil {
		t.Fatal("checkedExpandedLength() accepted overflow")
	}
	if _, err := AddTokenBudget(math.MaxInt, 1); err == nil {
		t.Fatal("AddTokenBudget() accepted overflow")
	}
	if _, err := AddTokenBudget(0, 0); err == nil {
		t.Fatal("AddTokenBudget() accepted zero tokens")
	}
	if _, err := ExpandPrompt("abc", []Binding{{MarkerStart: 1, MarkerEnd: 4}}); err == nil {
		t.Fatal("ExpandPrompt() accepted out-of-range marker")
	}
}

func TestAssignTokenSpansMixedKinds(t *testing.T) {
	bindings := []Binding{
		{Media: llm.MediaData{ID: 8, Kind: llm.MediaKindAudio}, ExpectedTokens: 2},
		{Media: llm.MediaData{ID: 3, Kind: llm.MediaKindImage}, ExpectedTokens: 3},
		{Media: llm.MediaData{ID: 9, Kind: llm.MediaKindAudio}, ExpectedTokens: 1},
	}
	tokens := []int32{0, 4, 4, 0, 3, 3, 3, 0, 4}
	items, err := AssignTokenSpans(tokens, bindings, map[llm.MediaKind]int32{llm.MediaKindImage: 3, llm.MediaKindAudio: 4})
	if err != nil {
		t.Fatal(err)
	}
	want := []Span{{Start: 1, End: 3}, {Start: 4, End: 7}, {Start: 8, End: 9}}
	for i := range want {
		if items[i].Span != want[i] || items[i].ID != bindings[i].Media.ID {
			t.Fatalf("item %d = %#v, want span %#v", i, items[i], want[i])
		}
	}

	tests := []struct {
		name string
		toks []int32
		ids  map[llm.MediaKind]int32
		want string
	}{
		{name: "same IDs", toks: tokens, ids: map[llm.MediaKind]int32{llm.MediaKindImage: 3, llm.MediaKindAudio: 3}, want: "same"},
		{name: "missing span", toks: []int32{0}, ids: map[llm.MediaKind]int32{llm.MediaKindImage: 3, llm.MediaKindAudio: 4}, want: "no audio"},
		{name: "extra span", toks: append(tokens, 0, 3), ids: map[llm.MediaKind]int32{llm.MediaKindImage: 3, llm.MediaKindAudio: 4}, want: "unexpected"},
		{name: "wrong length", toks: append([]int32(nil), tokens[:2]...), ids: map[llm.MediaKind]int32{llm.MediaKindImage: 3, llm.MediaKindAudio: 4}, want: "token count"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := AssignTokenSpans(tc.toks, bindings, tc.ids)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestValidateItems(t *testing.T) {
	valid := []Item{
		{ID: 1, Kind: llm.MediaKindImage, Span: Span{Start: 1, End: 3}, ExpectedTokens: 2},
		{ID: 0, Kind: llm.MediaKindAudio, Span: Span{Start: 4, End: 5}, ExpectedTokens: 1},
	}
	if err := ValidateItems(valid, 6); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		edit func([]Item)
		want string
	}{
		{name: "duplicate", edit: func(v []Item) { v[1].ID = v[0].ID }, want: "duplicate"},
		{name: "overlap", edit: func(v []Item) { v[1].Span.Start = 2 }, want: "overlapping"},
		{name: "bounds", edit: func(v []Item) { v[1].Span.End = 7 }, want: "span"},
		{name: "length", edit: func(v []Item) { v[1].ExpectedTokens = 2 }, want: "span length"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			items := append([]Item(nil), valid...)
			tc.edit(items)
			if err := ValidateItems(items, 6); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestMergeEmbeddingsOrdered(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("MLX is only available on Darwin")
	}
	embeddings := mlx.FromValues([]float32{0, 1, 2, 3, 4, 5, 6, 7}, 1, 4, 2)
	replacements := []Replacement{
		{Span: Span{Start: 1, End: 2}, Features: mlx.FromValues([]float32{10, 11}, 1, 1, 2)},
		{Span: Span{Start: 3, End: 4}, Features: mlx.FromValues([]float32{20, 21}, 1, 1, 2)},
	}
	got, err := MergeEmbeddings(embeddings, 4, replacements)
	if err != nil {
		t.Fatal(err)
	}
	mlx.Eval(got)
	if want := []float32{0, 1, 10, 11, 4, 5, 20, 21}; !slices.Equal(got.Floats(), want) {
		t.Fatalf("merged = %v, want %v", got.Floats(), want)
	}
	for _, tc := range []struct {
		name string
		reps []Replacement
		want string
	}{
		{name: "empty", want: "no embedding"},
		{name: "overlap", reps: []Replacement{replacements[0], {Span: Span{Start: 1, End: 3}, Features: mlx.FromValues(make([]float32, 4), 1, 2, 2)}}, want: "overlapping"},
		{name: "shape", reps: []Replacement{{Span: Span{Start: 1, End: 2}, Features: mlx.FromValues(make([]float32, 3), 1, 1, 3)}}, want: "shape"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := MergeEmbeddings(embeddings, 4, tc.reps)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}
