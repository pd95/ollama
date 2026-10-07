package tokenizer

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
)

func TestEmptyAddedToken(t *testing.T) {
	tok, err := LoadFromBytes([]byte(`{
		"model":{"type":"BPE","vocab":{"a":0},"merges":[]},
		"added_tokens":[
			{"id":1,"content":"","special":true},
			{"id":2,"content":"","special":false,"normalized":true}
		]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := tok.GetSpecialToken(""); ok {
		t.Fatal("empty added token must not be registered: it cannot consume input")
	}
	if got := tok.Encode("a", false); !slices.Equal(got, []int32{0}) {
		t.Fatalf("Encode(a) = %v, want [0]", got)
	}
}

func TestEmptyMetaspaceInput(t *testing.T) {
	for _, scheme := range []string{"always", "first", "never"} {
		t.Run(scheme, func(t *testing.T) {
			data := fmt.Sprintf(`{
				"model":{"type":"BPE","vocab":{"▁":0,"a":1},"merges":[]},
				"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":%q,"split":true}
			}`, scheme)
			tok, err := LoadFromBytes([]byte(data))
			if err != nil {
				t.Fatal(err)
			}
			if got := tok.Encode("", false); len(got) != 0 {
				t.Fatalf("Encode(empty) = %v, want no tokens", got)
			}
		})
	}
}

func TestLoadFromBytesRejectsWordPiece(t *testing.T) {
	data := []byte(`{
		"model": {
			"type": "WordPiece",
			"vocab": {"[UNK]": 0, "hello": 1}
		},
		"added_tokens": []
	}`)

	_, err := LoadFromBytes(data)
	if err == nil {
		t.Fatal("expected WordPiece load to fail")
	}
	if !strings.Contains(err.Error(), "unsupported tokenizer type: WordPiece") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestMetaspace(t *testing.T) {
	data := []byte(`{
		"model": {"type":"BPE", "vocab":{"▁":0,"a":1,"b":2,"▁a":3,"▁b":4,"▁▁":5,"<s>":6},"merges":["▁ a","▁ b","▁ ▁"]},
		"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},
		"decoder":{"type":"Replace","pattern":{"String":"▁"},"content":" "},
		"added_tokens":[{"id":6,"content":"<s>","special":true}]
	}`)
	for _, scheme := range []string{"always", "first", "never"} {
		t.Run(scheme, func(t *testing.T) {
			tok, err := LoadFromBytes([]byte(strings.ReplaceAll(string(data), `"always"`, `"`+scheme+`"`)))
			if err != nil {
				t.Fatal(err)
			}
			for _, tc := range []struct {
				text                 string
				always, first, never []int32
			}{
				{"", nil, nil, nil},
				{"a b", []int32{3, 4}, []int32{3, 4}, []int32{1, 4}},
				{"  a", []int32{0, 3}, []int32{0, 3}, []int32{0, 3}},
				{"a  b", []int32{3, 0, 4}, []int32{3, 0, 4}, []int32{1, 0, 4}},
				{"a<s>b", []int32{3, 6, 4}, []int32{3, 6, 2}, []int32{1, 6, 2}},
				{"<s>a", []int32{6, 3}, []int32{6, 1}, []int32{6, 1}},
			} {
				want := map[string][]int32{"always": tc.always, "first": tc.first, "never": tc.never}[scheme]
				if got := tok.Encode(tc.text, false); !slices.Equal(got, want) {
					t.Errorf("Encode(%q) = %v, want %v", tc.text, got, want)
				}
			}
		})
	}
}

func TestIgnoreMerges(t *testing.T) {
	for _, tc := range []struct {
		name    string
		setting string
		want    []int32
	}{
		{"default", "", []int32{0, 4}},
		{"false", `,"ignore_merges":false`, []int32{0, 4}},
		{"true", `,"ignore_merges":true`, []int32{5}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The higher-priority b+c merge prevents ab+c, despite abc being
			// in the vocabulary. HF BPE emits [a, bc] unless ignore_merges is set.
			data := `{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4,"abc":5},"merges":[["b","c"],["a","b"],["ab","c"]]` + tc.setting + `}}`
			tok, err := LoadFromBytes([]byte(data))
			if err != nil {
				t.Fatal(err)
			}
			for _, input := range []struct {
				text string
				want []int32
			}{
				{"abc", tc.want},
				{"abca", []int32{0, 4, 0}},
			} {
				if got := tok.Encode(input.text, false); !slices.Equal(got, input.want) {
					t.Errorf("Encode(%q) = %v, want %v", input.text, got, input.want)
				}
			}
		})
	}
}

func TestLoadFromBytesRejectsConflictingAddedTokenID(t *testing.T) {
	data := []byte(`{"model":{"type":"BPE","vocab":{"base":0}},"added_tokens":[{"id":0,"content":"replacement"}]}`)
	if _, err := LoadFromBytes(data); err == nil || !strings.Contains(err.Error(), "conflicting base and added content") {
		t.Fatalf("error = %v, want conflicting base and added content", err)
	}
}

func tokenizerJSON(vocab, addedTokens string) []byte {
	return []byte(fmt.Sprintf(`{"model":{"type":"BPE","vocab":%s,"merges":[]},"added_tokens":%s}`, vocab, addedTokens))
}

func TestValidateTokenizerID(t *testing.T) {
	tests := []struct {
		name string
		id   int32
		want string
	}{
		{name: "negative", id: -1, want: "must not be negative"},
		{name: "minimum int32", id: math.MinInt32, want: "must not be negative"},
		{name: "maximum int32", id: math.MaxInt32, want: "exceeds maximum"},
		{name: "exclusive cap", id: maxTokenizerVocabularySize, want: "exceeds maximum"},
		{name: "inclusive upper ID", id: maxTokenizerVocabularySize - 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateTokenizerID(tt.id)
			if tt.want == "" {
				if err != nil {
					t.Fatalf("validateTokenizerID(%d): %v", tt.id, err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("validateTokenizerID(%d) error = %v, want containing %q", tt.id, err, tt.want)
			}
		})
	}
}

func TestValidateTokenizerRecordCount(t *testing.T) {
	if err := validateTokenizerRecordCount(maxTokenizerVocabularySize-1, 1); err != nil {
		t.Fatalf("maximum record count rejected: %v", err)
	}
	if err := validateTokenizerRecordCount(maxTokenizerVocabularySize, 1); err == nil {
		t.Fatal("expected aggregate record count above cap to fail")
	}
}

func TestLoadFromBytesRejectsInvalidTokenizerIDs(t *testing.T) {
	tests := []struct {
		name  string
		data  []byte
		wants []string
	}{
		{name: "negative base", data: tokenizerJSON(`{"a":-1}`, `[]`), wants: []string{"invalid base token", "tokenizer ID -1"}},
		{name: "minimum base", data: tokenizerJSON(fmt.Sprintf(`{"a":%d}`, int64(math.MinInt32)), `[]`), wants: []string{"invalid base token", fmt.Sprint(math.MinInt32)}},
		{name: "negative added", data: tokenizerJSON(`{}`, `[{"id":-1,"content":"a"}]`), wants: []string{"invalid added token", "tokenizer ID -1"}},
		{name: "minimum added", data: tokenizerJSON(`{}`, fmt.Sprintf(`[{"id":%d,"content":"a"}]`, int64(math.MinInt32))), wants: []string{"invalid added token", fmt.Sprint(math.MinInt32)}},
		{name: "base at exclusive cap", data: tokenizerJSON(fmt.Sprintf(`{"a":%d}`, maxTokenizerVocabularySize), `[]`), wants: []string{"invalid base token", "exceeds maximum"}},
		{name: "added at exclusive cap", data: tokenizerJSON(`{}`, fmt.Sprintf(`[{"id":%d,"content":"a"}]`, maxTokenizerVocabularySize)), wants: []string{"invalid added token", "exceeds maximum"}},
		{name: "above int32", data: tokenizerJSON(`{"a":2147483648}`, `[]`), wants: []string{"failed to parse tokenizer", "cannot unmarshal number"}},
		{name: "below int32", data: tokenizerJSON(`{"a":-2147483649}`, `[]`), wants: []string{"failed to parse tokenizer", "cannot unmarshal number"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := LoadFromBytes(tt.data)
			if err == nil {
				t.Fatal("expected tokenizer load to fail")
			}
			for _, want := range tt.wants {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("error = %v, want containing %q", err, want)
				}
			}
		})
	}
}

func TestLoadFromBytesWithConfigFiltersAddedTokenIDs(t *testing.T) {
	data := tokenizerJSON(`{"base":0}`, `[
		{"id":-1,"content":"negative"},
		{"id":1,"content":"kept"},
		{"id":2,"content":"at-limit"},
		{"id":3,"content":"above-limit"}
	]`)
	tok, err := LoadFromBytesWithConfig(data, &TokenizerConfig{AddedTokenIDLimit: 2})
	if err != nil {
		t.Fatal(err)
	}

	if got := tok.VocabSize(); got != 2 {
		t.Fatalf("VocabSize() = %d, want 2", got)
	}
	if got := tok.Decode([]int32{0, 1}); got != "basekept" {
		t.Fatalf("Decode([0 1]) = %q, want %q", got, "basekept")
	}
	if id, ok := tok.GetSpecialToken("kept"); !ok || id != 1 {
		t.Fatalf("GetSpecialToken(kept) = (%d, %v), want (1, true)", id, ok)
	}
	for _, content := range []string{"negative", "at-limit", "above-limit"} {
		if id, ok := tok.GetSpecialToken(content); ok {
			t.Fatalf("GetSpecialToken(%q) = (%d, true), want absent", content, id)
		}
	}
}

func TestLoadFromBytesWithConfigAddedTokenIDLimitDoesNotFilterBaseVocabulary(t *testing.T) {
	data := tokenizerJSON(`{"base":3}`, `[
		{"id":1,"content":"added"},
		{"id":3,"content":"filtered-added"}
	]`)
	tok, err := LoadFromBytesWithConfig(data, &TokenizerConfig{AddedTokenIDLimit: 2})
	if err != nil {
		t.Fatal(err)
	}

	if got := tok.VocabSize(); got != 4 {
		t.Fatalf("VocabSize() = %d, want 4", got)
	}
	if got := tok.Decode([]int32{3, 1}); got != "baseadded" {
		t.Fatalf("Decode([3 1]) = %q, want %q", got, "baseadded")
	}
	if _, ok := tok.GetSpecialToken("filtered-added"); ok {
		t.Fatal("added token at the limit was retained")
	}
}

func TestLoadFromBytesWithConfigAddedTokenIDLimitDisabled(t *testing.T) {
	data := tokenizerJSON(`{}`, `[{"id":-1,"content":"negative"}]`)
	for name, config := range map[string]*TokenizerConfig{
		"nil config": nil,
		"zero limit": {},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := LoadFromBytesWithConfig(data, config)
			if err == nil || !strings.Contains(err.Error(), "invalid added token") {
				t.Fatalf("error = %v, want invalid added token", err)
			}
		})
	}
}

func TestLoadFromBytesWithConfigRejectsNegativeAddedTokenIDLimit(t *testing.T) {
	_, err := LoadFromBytesWithConfig(tokenizerJSON(`{}`, `[]`), &TokenizerConfig{AddedTokenIDLimit: -1})
	if err == nil || err.Error() != "added token ID limit must not be negative: -1" {
		t.Fatalf("error = %v, want negative limit error", err)
	}
}

func TestLoadFromBytesWithConfigValidatesOnlyRetainedAddedTokens(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{
			name: "duplicate IDs are filtered",
			data: tokenizerJSON(`{"base":0}`, `[
				{"id":2,"content":"first"},
				{"id":2,"content":"second"},
				{"id":1,"content":"kept"}
			]`),
		},
		{
			name: "duplicate content is filtered",
			data: tokenizerJSON(`{"base":0}`, `[
				{"id":2,"content":"duplicate"},
				{"id":3,"content":"duplicate"},
				{"id":1,"content":"kept"}
			]`),
		},
		{
			name: "base collision is filtered",
			data: tokenizerJSON(`{"base":0}`, `[
				{"id":2,"content":"base"},
				{"id":1,"content":"kept"}
			]`),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tok, err := LoadFromBytesWithConfig(tt.data, &TokenizerConfig{AddedTokenIDLimit: 2})
			if err != nil {
				t.Fatal(err)
			}
			if got := tok.VocabSize(); got != 2 {
				t.Fatalf("VocabSize() = %d, want 2", got)
			}
		})
	}
}

func TestLoadFromBytesWithConfigRejectsRetainedAddedTokenCollision(t *testing.T) {
	data := tokenizerJSON(`{"base":0}`, `[
		{"id":1,"content":"first"},
		{"id":1,"content":"second"},
		{"id":2,"content":"filtered"}
	]`)
	_, err := LoadFromBytesWithConfig(data, &TokenizerConfig{AddedTokenIDLimit: 2})
	if err == nil || err.Error() != "duplicate added token ID 1" {
		t.Fatalf("error = %v, want retained-token collision", err)
	}
}

func TestLoadFromBytesSupportsBoundedSparseIDs(t *testing.T) {
	tests := []struct {
		name        string
		data        []byte
		wantSize    int
		decodeID    int32
		wantDecoded string
		special     string
		wantSpecial int32
	}{
		{
			name:        "inclusive upper base ID",
			data:        tokenizerJSON(fmt.Sprintf(`{"edge":%d}`, maxTokenizerVocabularySize-1), `[]`),
			wantSize:    maxTokenizerVocabularySize,
			decodeID:    maxTokenizerVocabularySize - 1,
			wantDecoded: "edge",
		},
		{name: "sparse base", data: tokenizerJSON(`{"base":1000}`, `[]`), wantSize: 1001, decodeID: 1000, wantDecoded: "base"},
		{name: "sparse added", data: tokenizerJSON(`{}`, `[{"id":1000,"content":"added"}]`), wantSize: 1001, decodeID: 1000, wantDecoded: "added", special: "added", wantSpecial: 1000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tok, err := LoadFromBytes(tt.data)
			if err != nil {
				t.Fatal(err)
			}
			if got := tok.VocabSize(); got != tt.wantSize {
				t.Fatalf("VocabSize() = %d, want %d", got, tt.wantSize)
			}
			if got := tok.Decode([]int32{tt.decodeID}); got != tt.wantDecoded {
				t.Fatalf("Decode(%d) = %q, want %q", tt.decodeID, got, tt.wantDecoded)
			}
			if tt.special != "" {
				id, ok := tok.GetSpecialToken(tt.special)
				if !ok || id != tt.wantSpecial {
					t.Fatalf("GetSpecialToken(%q) = (%d, %v), want (%d, true)", tt.special, id, ok, tt.wantSpecial)
				}
			}
		})
	}
}

func TestLoadFromBytesRejectsTokenizerCollisions(t *testing.T) {
	tests := []struct {
		name    string
		inputs  [][]byte
		wantErr string
	}{
		{
			name: "duplicate base ID",
			inputs: [][]byte{
				tokenizerJSON(`{"a":1,"b":1}`, `[]`),
				tokenizerJSON(`{"b":1,"a":1}`, `[]`),
			},
			wantErr: "duplicate base token ID 1",
		},
		{
			name: "duplicate added ID",
			inputs: [][]byte{
				tokenizerJSON(`{}`, `[{"id":1,"content":"a"},{"id":1,"content":"b"}]`),
				tokenizerJSON(`{}`, `[{"id":1,"content":"b"},{"id":1,"content":"a"}]`),
			},
			wantErr: "duplicate added token ID 1",
		},
		{
			name: "repeated added entry",
			inputs: [][]byte{
				tokenizerJSON(`{}`, `[{"id":1,"content":"a"},{"id":1,"content":"a"}]`),
			},
			wantErr: "duplicate added token ID 1",
		},
		{
			name: "duplicate added content",
			inputs: [][]byte{
				tokenizerJSON(`{}`, `[{"id":1,"content":"a"},{"id":2,"content":"a"}]`),
				tokenizerJSON(`{}`, `[{"id":2,"content":"a"},{"id":1,"content":"a"}]`),
			},
			wantErr: `duplicate added token content "a" with IDs 1 and 2`,
		},
		{
			name: "cross-source ID conflict",
			inputs: [][]byte{
				tokenizerJSON(`{"a":1}`, `[{"id":1,"content":"b"}]`),
			},
			wantErr: "token ID 1 has conflicting base and added content",
		},
		{
			name: "cross-source content conflict",
			inputs: [][]byte{
				tokenizerJSON(`{"a":1}`, `[{"id":2,"content":"a"}]`),
			},
			wantErr: `token content "a" has conflicting base and added IDs 1 and 2`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i, data := range tt.inputs {
				_, err := LoadFromBytes(data)
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("input %d error = %v, want %q", i, err, tt.wantErr)
				}
			}
		})
	}
}

func TestLoadFromBytesBaseValidationIsDeterministic(t *testing.T) {
	tests := []struct {
		name    string
		inputs  [][]byte
		wantErr string
	}{
		{
			name: "range error precedes duplicate ID",
			inputs: [][]byte{
				tokenizerJSON(fmt.Sprintf(`{"duplicate-a":5,"negative":-2,"over-cap":%d,"duplicate-b":5}`, maxTokenizerVocabularySize), `[]`),
				tokenizerJSON(fmt.Sprintf(`{"duplicate-b":5,"over-cap":%d,"negative":-2,"duplicate-a":5}`, maxTokenizerVocabularySize), `[]`),
			},
			wantErr: "invalid base token ID: tokenizer ID -2 must not be negative",
		},
		{
			name: "lowest duplicate ID wins",
			inputs: [][]byte{
				tokenizerJSON(`{"high-a":7,"low-a":2,"high-b":7,"low-b":2}`, `[]`),
				tokenizerJSON(`{"low-b":2,"high-b":7,"low-a":2,"high-a":7}`, `[]`),
			},
			wantErr: "duplicate base token ID 2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i, data := range tt.inputs {
				for attempt := range 100 {
					_, err := LoadFromBytes(data)
					if err == nil || err.Error() != tt.wantErr {
						t.Fatalf("input %d attempt %d error = %v, want %q", i, attempt, err, tt.wantErr)
					}
				}
			}
		})
	}
}

func TestLoadFromBytesAcceptsExactAddedTokenPromotion(t *testing.T) {
	tok, err := LoadFromBytes(tokenizerJSON(`{"a":1}`, `[{"id":1,"content":"a"}]`))
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Decode([]int32{1}); got != "a" {
		t.Fatalf("Decode(1) = %q, want a", got)
	}
	id, ok := tok.GetSpecialToken("a")
	if !ok || id != 1 {
		t.Fatalf("GetSpecialToken(a) = (%d, %v), want (1, true)", id, ok)
	}
}

func TestLoadFromBytesAcceptsEmptyAndContiguousVocabulary(t *testing.T) {
	for name, data := range map[string][]byte{
		"empty":      tokenizerJSON(`{}`, `[]`),
		"contiguous": tokenizerJSON(`{"a":0,"b":1}`, `[]`),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadFromBytes(data); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestSparseAddedTokensPreserveUpstreamMatching(t *testing.T) {
	data := []byte(`{"normalizer":{"type":"NFC"},"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"added_tokens":[{"id":1000,"content":"é","normalized":true},{"id":1001,"content":"é","normalized":true,"special":true},{"id":1048575,"content":""},{"id":0,"content":""}]}`)
	tok, err := LoadFromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	if tok.VocabSize() != 1002 {
		t.Fatalf("extent = %d, want 1002", tok.VocabSize())
	}
	if got := tok.Encode("éé", false); !slices.Equal(got, []int32{1001, 1001}) {
		t.Fatalf("normalized precedence: %v", got)
	}
	if _, ok := tok.GetSpecialToken(""); ok {
		t.Fatal("empty token registered")
	}
	if tok.vocab.Values[500] != "" {
		t.Fatal("sparse hole filled")
	}
	if err := ValidateTokenizerJSONIDs(data); err != nil {
		t.Fatal(err)
	}
}

func TestEmptyAddedTokensStillValidateNumericIDs(t *testing.T) {
	for _, id := range []int{-1, 1048576} {
		data := []byte(fmt.Sprintf(`{"model":{"type":"BPE","vocab":{"a":0}},"added_tokens":[{"id":%d,"content":""}]}`, id))
		if _, err := LoadFromBytes(data); err == nil {
			t.Fatalf("empty token ID %d accepted", id)
		}
		if err := ValidateTokenizerJSONIDs(data); err == nil {
			t.Fatalf("import accepted empty token ID %d", id)
		}
	}
}
