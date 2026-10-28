package tokenizer

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

// LoadTekkenFromBytes loads Mistral's byte-rank BPE data. Encode treats input
// as ordinary text; callers explicitly assemble speech control-token framing.
func LoadTekkenFromBytes(data []byte) (*Tokenizer, error) {
	if len(data) > 64<<20 {
		return nil, fmt.Errorf("Tekken metadata exceeds 64 MiB")
	}
	var raw struct {
		Config struct {
			Pattern    string `json:"pattern"`
			Vocabulary int    `json:"default_vocab_size"`
			Special    int    `json:"default_num_special_tokens"`
		} `json:"config"`
		Vocab []struct {
			Rank  int    `json:"rank"`
			Bytes string `json:"token_bytes"`
		} `json:"vocab"`
		Special []struct {
			Rank int    `json:"rank"`
			Text string `json:"token_str"`
		} `json:"special_tokens"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("Tekken metadata: %w", err)
	}
	size, special := raw.Config.Vocabulary, raw.Config.Special
	if size <= special || size > maxTokenizerVocabularySize || special < 1 || special > size || len(raw.Special) > special || len(raw.Vocab) > maxTokenizerVocabularySize {
		return nil, fmt.Errorf("invalid Tekken vocabulary extent")
	}
	ordinary := size - special
	if len(raw.Vocab) < ordinary {
		return nil, fmt.Errorf("missing Tekken vocabulary records")
	}
	pt, err := compileSplit(raw.Config.Pattern, "Isolated", false)
	if err != nil {
		return nil, err
	}
	vocab := &Vocabulary{Values: make([]string, size), Reverse: make(map[string]int32, ordinary), BOS: -1, PAD: -1}
	t := &Tokenizer{vocab: vocab, typ: TokenizerBPE, pretokenizer: []pretokenizer{pt}, specialTokens: map[string]int32{}, ignoreMerges: true, mergeByTokenRank: true}
	occupied := make([]bool, size)
	for _, record := range raw.Special {
		if record.Rank < 0 || record.Rank >= special || record.Text == "" || occupied[record.Rank] {
			return nil, fmt.Errorf("invalid Tekken special-token ID")
		}
		if _, ok := t.specialTokens[record.Text]; ok {
			return nil, fmt.Errorf("duplicate Tekken special token")
		}
		occupied[record.Rank] = true
		vocab.Values[record.Rank] = record.Text
		t.specialTokens[record.Text] = int32(record.Rank)
	}
	for _, record := range raw.Vocab {
		if record.Rank < 0 {
			return nil, fmt.Errorf("negative Tekken rank")
		}
		if record.Rank >= ordinary {
			continue
		}
		id := record.Rank + special
		if occupied[id] {
			return nil, fmt.Errorf("duplicate Tekken rank")
		}
		bytes, err := base64.StdEncoding.DecodeString(record.Bytes)
		if err != nil || len(bytes) == 0 {
			return nil, fmt.Errorf("invalid Tekken byte token")
		}
		var encoded strings.Builder
		for _, b := range bytes {
			encoded.WriteRune(byteToRune[b])
		}
		value := encoded.String()
		if _, ok := vocab.Reverse[value]; ok {
			return nil, fmt.Errorf("duplicate Tekken byte string")
		}
		occupied[id] = true
		vocab.Values[id] = value
		vocab.Reverse[value] = int32(id)
	}
	for id := special; id < size; id++ {
		if !occupied[id] {
			return nil, fmt.Errorf("missing Tekken rank %d", id-special)
		}
	}
	for b, r := range byteToRune {
		id, ok := vocab.Reverse[string(r)]
		if !ok {
			return nil, fmt.Errorf("Tekken lacks byte %d", b)
		}
		vocab.byteTokens[b] = id
	}
	if bos, ok := t.specialTokens["<s>"]; ok {
		vocab.BOS = bos
	}
	if eos, ok := t.specialTokens["</s>"]; ok {
		vocab.EOS = []int32{eos}
	}
	return t, nil
}
