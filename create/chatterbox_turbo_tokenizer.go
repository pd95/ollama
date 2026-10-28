package create

import (
	"bufio"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func chatterboxTurboTokenizer(dir string) (*validatedTokenizerSource, error) {
	read := func(name string, value any) error {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return fmt.Errorf("Chatterbox Turbo tokenizer requires %s: %w", name, err)
		}
		if err := json.Unmarshal(data, value); err != nil {
			return fmt.Errorf("Chatterbox Turbo parse %s: %w", name, err)
		}
		return nil
	}
	var vocab map[string]int32
	if err := read("vocab.json", &vocab); err != nil {
		return nil, err
	}
	var cfg struct {
		Class string `json:"tokenizer_class"`
		Added map[string]struct {
			Content    string `json:"content"`
			Special    bool   `json:"special"`
			Normalized bool   `json:"normalized"`
			LStrip     bool   `json:"lstrip"`
			RStrip     bool   `json:"rstrip"`
			SingleWord bool   `json:"single_word"`
		} `json:"added_tokens_decoder"`
		AddPrefixSpace bool `json:"add_prefix_space"`
	}
	if err := read("tokenizer_config.json", &cfg); err != nil {
		return nil, err
	}
	if cfg.Class != "GPT2Tokenizer" {
		return nil, fmt.Errorf("unsupported Chatterbox Turbo tokenizer class %q", cfg.Class)
	}
	var added []map[string]any
	for _, key := range slices.Sorted(maps.Keys(cfg.Added)) {
		token := cfg.Added[key]
		id, err := strconv.ParseInt(key, 10, 32)
		if err != nil || token.Content == "" || token.LStrip || token.RStrip || token.SingleWord {
			return nil, fmt.Errorf("unsupported Chatterbox Turbo added token %q", key)
		}
		added = append(added, map[string]any{"id": id, "content": token.Content, "special": token.Special, "normalized": token.Normalized})
	}
	merges, err := os.Open(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, fmt.Errorf("Chatterbox Turbo tokenizer requires merges.txt: %w", err)
	}
	defer merges.Close()
	var pairs [][]string
	scanner := bufio.NewScanner(merges)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#version:") {
			continue
		}
		pair := strings.Fields(line)
		if len(pair) != 2 {
			return nil, fmt.Errorf("invalid Chatterbox Turbo merge in merges.txt")
		}
		for _, name := range []string{pair[0], pair[1], pair[0] + pair[1]} {
			if _, ok := vocab[name]; !ok {
				return nil, fmt.Errorf("Chatterbox Turbo merges.txt references unknown token %q", name)
			}
		}
		pairs = append(pairs, pair)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read Chatterbox Turbo merges.txt: %w", err)
	}

	data, err := json.Marshal(map[string]any{
		"model":         map[string]any{"type": "BPE", "vocab": vocab, "merges": pairs},
		"added_tokens":  added,
		"pre_tokenizer": map[string]any{"type": "ByteLevel", "add_prefix_space": cfg.AddPrefixSpace, "use_regex": true},
		"decoder":       map[string]any{"type": "ByteLevel"},
	})
	if err != nil {
		return nil, err
	}
	if err := tokenizer.ValidateTokenizerJSONIDs(data); err != nil {
		return nil, fmt.Errorf("%w: Chatterbox Turbo synthesized tokenizer: %v", ErrInvalidTokenizer, err)
	}
	return &validatedTokenizerSource{data: data, generated: true}, nil
}
