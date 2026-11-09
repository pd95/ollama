package tokenizer

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

func TestWhitespacePretokenizerReference(t *testing.T) {
	raw, err := os.ReadFile("testdata/whitespace.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Input  string
			Pieces []string
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	stages, err := loadPretokenizers(json.RawMessage(`{"type":"Whitespace"}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range fixture.Cases {
		if got := stages[0].split(c.Input); !slices.Equal(got, c.Pieces) {
			t.Errorf("%q => %q, want %q", c.Input, got, c.Pieces)
		}
	}
}

func TestUnicodeBPE(t *testing.T) {
	raw := []byte(`{"model":{"type":"BPE","vocab":{"[UNK]":0,"é":1,"e":2,"́":3,"éé":4,"?":5},"merges":[["é","é"]],"unk_token":"[UNK]","fuse_unk":false},"pre_tokenizer":{"type":"Whitespace"},"decoder":null}`)
	tok, err := LoadFromBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("éé é??? ΩΩ", false); !slices.Equal(got, []int32{4, 2, 3, 5, 5, 5, 0, 0}) {
		t.Fatalf("Unicode BPE encoded %v", got)
	}
	if got := tok.Decode([]int32{4, 2, 3, 5}); got != "ééé?" {
		t.Fatalf("Unicode BPE decoded %q", got)
	}
}
