package tokenizer

import (
	"encoding/base64"
	"encoding/json"
	"slices"
	"testing"
)

func tekkenFixture(t *testing.T, mutate func(map[string]any)) []byte {
	t.Helper()
	var vocab []any
	for b := range 256 {
		vocab = append(vocab, map[string]any{"rank": b, "token_bytes": base64.StdEncoding.EncodeToString([]byte{byte(b)})})
	}
	for i, text := range []string{"ab", "abc"} {
		vocab = append(vocab, map[string]any{"rank": 256 + i, "token_bytes": base64.StdEncoding.EncodeToString([]byte(text))})
	}
	raw := map[string]any{"config": map[string]any{"default_vocab_size": 1258, "default_num_special_tokens": 1000, "pattern": `\p{L}+|\p{N}|[^\p{L}\p{N}]+`}, "special_tokens": []any{map[string]any{"rank": 1, "token_str": "<s>"}, map[string]any{"rank": 24, "token_str": "[AUDIO]"}}, "vocab": vocab}
	if mutate != nil {
		mutate(raw)
	}
	data, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestTekkenRankedBytes(t *testing.T) {
	tok, err := LoadTekkenFromBytes(tekkenFixture(t, nil))
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("abca", false); !slices.Equal(got, []int32{1257, 1097}) {
		t.Fatalf("rank merge: %v", got)
	}
	if got := tok.Encode("abc", false); !slices.Equal(got, []int32{1257}) {
		t.Fatalf("whole-piece match: %v", got)
	}
	for _, id := range tok.Encode("[AUDIO]", false) {
		if id < 1000 {
			t.Fatal("ordinary text injected a control token")
		}
	}
	if id, ok := tok.GetSpecialToken("[AUDIO]"); !ok || id != 24 {
		t.Fatal("missing explicit control-token lookup")
	}
}

func TestTekkenRejectsBrokenData(t *testing.T) {
	for _, mutate := range []func(map[string]any){
		func(m map[string]any) { m["config"].(map[string]any)["default_vocab_size"] = 1 << 28 },
		func(m map[string]any) { m["vocab"].([]any)[0].(map[string]any)["token_bytes"] = "bad base64" },
		func(m map[string]any) { m["vocab"].([]any)[1].(map[string]any)["rank"] = 0 },
	} {
		if _, err := LoadTekkenFromBytes(tekkenFixture(t, mutate)); err == nil {
			t.Fatal("accepted malformed Tekken")
		}
	}
}
