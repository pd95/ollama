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

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
	"github.com/ollama/ollama/types/model"
)

const qwen3TTSArchitecture = "Qwen3TTSForConditionalGeneration"

func qwen3TTSAsset(dir, name string) (string, error) {
	info, err := os.Stat(filepath.Join(dir, filepath.FromSlash(name)))
	if err != nil {
		return "", fmt.Errorf("Qwen3-TTS requires %s: %w", name, err)
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("Qwen3-TTS %s must be a regular file", name)
	}
	return name, nil
}

func qwen3TTSSource(dir string) (bool, error) {
	if _, err := os.Stat(filepath.Join(dir, "config.json")); os.IsNotExist(err) {
		return false, nil
	}
	cfg, _, err := readSourceModelConfig(dir)
	if err != nil {
		return false, err
	}
	return cfg.Architecture() == qwen3TTSArchitecture, nil
}

// prepareQwen3TTSInventory keeps the separately published speech tokenizer in
// its own namespace and at its own source precision.
func prepareQwen3TTSInventory(inv Inventory) (Inventory, error) {
	if inv.Config.Architecture() != qwen3TTSArchitecture {
		return inv, nil
	}
	file, err := qwen3TTSAsset(inv.Dir, "speech_tokenizer/model.safetensors")
	if err != nil {
		return Inventory{}, err
	}
	config, err := os.ReadFile(filepath.Join(inv.Dir, "speech_tokenizer/config.json"))
	if err != nil {
		return Inventory{}, fmt.Errorf("Qwen3-TTS requires speech_tokenizer/config.json: %w", err)
	}
	if !json.Valid(config) {
		return Inventory{}, fmt.Errorf("Qwen3-TTS speech_tokenizer/config.json is invalid JSON")
	}
	reader, err := safetensors.OpenForExtraction(filepath.Join(inv.Dir, filepath.FromSlash(file)))
	if err != nil {
		return Inventory{}, fmt.Errorf("open Qwen3-TTS speech tokenizer: %w", err)
	}
	defer reader.Close()
	names := reader.ListTensors()
	if len(names) == 0 {
		return Inventory{}, fmt.Errorf("Qwen3-TTS speech tokenizer has no tensors")
	}
	inv.Tensors = maps.Clone(inv.Tensors)
	for _, name := range names {
		key := "speech_tokenizer." + name
		if inv.Has(key) {
			return Inventory{}, fmt.Errorf("Qwen3-TTS checkpoint already defines %s", key)
		}
		tensor, err := reader.GetTensor(name)
		if err != nil {
			return Inventory{}, err
		}
		inv.Tensors[key] = SourceTensor{Name: name, Dtype: tensor.Dtype, Shape: tensor.Shape, File: file}
	}
	return inv, nil
}

type qwen3TTSImportTransform struct {
	voices         []string
	languages      []string
	voiceLanguages map[string][]string
}

func newQwen3TTSImportTransform(raw json.RawMessage) (quantizePolicy, error) {
	var cfg struct {
		Talker struct {
			Speakers  map[string]json.RawMessage `json:"spk_id"`
			Languages map[string]int32           `json:"codec_language_id"`
			Dialects  map[string]json.RawMessage `json:"spk_is_dialect"`
		} `json:"talker_config"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	codes := map[string]string{"chinese": "zh", "english": "en", "german": "de", "french": "fr", "russian": "ru", "portuguese": "pt", "spanish": "es", "italian": "it", "korean": "ko", "japanese": "ja"}
	t := qwen3TTSImportTransform{voices: slices.Sorted(maps.Keys(cfg.Talker.Speakers))}
	for name := range cfg.Talker.Languages {
		if code := codes[name]; code != "" {
			t.languages = append(t.languages, code)
		}
	}
	slices.Sort(t.languages)
	if len(t.voices) > 0 && len(t.languages) > 0 {
		t.voiceLanguages = map[string][]string{}
		for _, voice := range t.voices {
			languages := slices.Clone(t.languages)
			var dialect string
			if json.Unmarshal(cfg.Talker.Dialects[voice], &dialect) == nil && dialect != "" {
				if _, ok := cfg.Talker.Languages[dialect]; !ok {
					// A checkpoint must supply the selected dialect codec ID
					// before that Chinese pairing can be advertised.
					languages = slices.DeleteFunc(languages, func(s string) bool { return s == "zh" })
				}
			}
			t.voiceLanguages[voice] = languages
		}
	}
	return t, nil
}

func (t qwen3TTSImportTransform) augmentConfig(cfg *model.ConfigV2) error {
	cfg.Voices = t.voices
	cfg.Languages = t.languages
	cfg.VoiceLanguages = t.voiceLanguages
	return nil
}

func (qwen3TTSImportTransform) quantizationType(name string, shape []int32, requested string) string {
	if !strings.HasPrefix(name, "talker.") {
		return ""
	}
	return GetTensorQuantization(name, shape, requested)
}

// qwen3TTSTokenizer normalizes the split Qwen2 tokenizer for the runner without
// writing a tokenizer.json into the owner's source snapshot.
func qwen3TTSTokenizer(dir string) (*validatedTokenizerSource, error) {
	read := func(name string, value any) error {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return fmt.Errorf("Qwen3-TTS tokenizer requires %s: %w", name, err)
		}
		if err := json.Unmarshal(data, value); err != nil {
			return fmt.Errorf("Qwen3-TTS parse %s: %w", name, err)
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
	if cfg.Class != "Qwen2Tokenizer" {
		return nil, fmt.Errorf("unsupported Qwen3-TTS tokenizer class %q", cfg.Class)
	}
	var added []map[string]any
	for _, key := range slices.Sorted(maps.Keys(cfg.Added)) {
		token := cfg.Added[key]
		id, err := strconv.ParseInt(key, 10, 32)
		if err != nil || token.Content == "" || token.LStrip || token.RStrip || token.SingleWord {
			return nil, fmt.Errorf("unsupported Qwen3-TTS added token %q", key)
		}
		added = append(added, map[string]any{"id": id, "content": token.Content, "special": token.Special, "normalized": token.Normalized})
	}
	merges, err := os.Open(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, fmt.Errorf("Qwen3-TTS tokenizer requires merges.txt: %w", err)
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
			return nil, fmt.Errorf("invalid Qwen3-TTS merge in merges.txt")
		}
		for _, name := range []string{pair[0], pair[1], pair[0] + pair[1]} {
			if _, ok := vocab[name]; !ok {
				return nil, fmt.Errorf("Qwen3-TTS merges.txt references unknown token %q", name)
			}
		}
		pairs = append(pairs, pair)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read Qwen3-TTS merges.txt: %w", err)
	}
	const pattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	data, err := json.Marshal(map[string]any{
		"model":        map[string]any{"type": "BPE", "vocab": vocab, "merges": pairs},
		"added_tokens": added,
		"normalizer":   map[string]any{"type": "NFC"},
		"pre_tokenizer": map[string]any{"type": "Sequence", "pretokenizers": []any{
			map[string]any{"type": "Split", "pattern": map[string]string{"Regex": pattern}, "behavior": "Isolated", "invert": false},
			map[string]any{"type": "ByteLevel", "add_prefix_space": cfg.AddPrefixSpace, "use_regex": false},
		}},
		"decoder": map[string]any{"type": "ByteLevel"},
	})
	if err != nil {
		return nil, err
	}
	if err := tokenizer.ValidateTokenizerJSONIDs(data); err != nil {
		return nil, fmt.Errorf("%w: Qwen3-TTS synthesized tokenizer: %v", ErrInvalidTokenizer, err)
	}
	return &validatedTokenizerSource{data: data, generated: true}, nil
}
