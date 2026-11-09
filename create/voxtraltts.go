package create

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	voxtralconfig "github.com/ollama/ollama/mlxrunner/model/voxtraltts/config"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
	"github.com/ollama/ollama/types/model"
)

const voxtralTTSArchitecture = voxtralconfig.Architecture

var voxtralEnglishGermanVoices = []string{"casual_female", "casual_male", "cheerful_female", "de_female", "de_male", "neutral_female", "neutral_male"}

func normalizeVoxtralTTSConfig(raw []byte) ([]byte, error) {
	if _, err := voxtralconfig.Parse(raw); err != nil {
		return nil, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, err
	}
	fields["architectures"], _ = json.Marshal([]string{voxtralTTSArchitecture})
	return json.Marshal(fields)
}

func voxtralVoiceFiles(dir string) ([]string, error) {
	var files []string
	for _, name := range voxtralconfig.VoiceNames() {
		path, err := speechTensorFile(dir, "voice_embedding/"+name, ".safetensors", ".pt")
		if errors.Is(err, os.ErrNotExist) && !officialVoxtralLayout(dir) && !slices.Contains(voxtralEnglishGermanVoices, name) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("Voxtral TTS voice %s: %w", name, err)
		}
		files = append(files, path)
	}
	return files, nil
}

func prepareVoxtralTTSInventory(inv Inventory) (Inventory, *validatedTokenizerSource, error) {
	if inv.Config.Architecture() != voxtralTTSArchitecture {
		return inv, nil, nil
	}
	if officialVoxtralLayout(inv.Dir) {
		var err error
		inv, err = prepareOfficialVoxtralTensors(inv)
		if err != nil {
			return Inventory{}, nil, err
		}
	}
	cfg, err := voxtralconfig.Parse(inv.RawConfig)
	if err != nil {
		return Inventory{}, nil, err
	}
	path := filepath.Join(inv.Dir, "tekken.json")
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() > 64<<20 {
		return Inventory{}, nil, fmt.Errorf("Voxtral TTS requires bounded tekken.json")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return Inventory{}, nil, err
	}
	tok, err := tokenizer.LoadTekkenFromBytes(raw)
	if err != nil {
		return Inventory{}, nil, err
	}
	for name, want := range map[string]int32{"<s>": 1, "[AUDIO]": 24, "[BEGIN_AUDIO]": 25, "[NEXT_AUDIO_TEXT]": 36, "[REPEAT_AUDIO_TEXT]": 35} {
		got, ok := tok.GetSpecialToken(name)
		if !ok || got != want {
			return Inventory{}, nil, fmt.Errorf("Voxtral TTS requires %s token %d", name, want)
		}
	}
	if tok.VocabSize() != cfg.Vocab {
		return Inventory{}, nil, fmt.Errorf("Voxtral TTS tokenizer vocabulary mismatch")
	}
	var metadata struct {
		Audio struct {
			Voices map[string]int `json:"voice_num_audio_tokens"`
		} `json:"audio"`
	}
	if err = json.Unmarshal(raw, &metadata); err != nil {
		return Inventory{}, nil, err
	}
	files, err := voxtralVoiceFiles(inv.Dir)
	if err != nil {
		return Inventory{}, nil, err
	}
	inv.Tensors = maps.Clone(inv.Tensors)
	voices := map[string]int{}
	for _, file := range files {
		name := strings.TrimSuffix(filepath.Base(file), filepath.Ext(file))
		count := metadata.Audio.Voices[name]
		if count < 1 || count > 512 {
			return Inventory{}, nil, fmt.Errorf("Voxtral TTS voice %s has invalid prompt length", name)
		}
		if _, ok := cfg.Multimodal.Codec.Voices[name]; !ok {
			return Inventory{}, nil, fmt.Errorf("Voxtral TTS voice %s absent from configuration", name)
		}
		reader, err := openTensorSource(context.Background(), filepath.Join(inv.Dir, file))
		if err != nil {
			return Inventory{}, nil, err
		}
		tensorName := "embedding"
		if filepath.Ext(file) == ".pt" && slices.Equal(reader.ListTensors(), []string{"tensor"}) {
			tensorName = "tensor"
		}
		if !slices.Equal(reader.ListTensors(), []string{tensorName}) {
			reader.Close()
			return Inventory{}, nil, fmt.Errorf("invalid Voxtral TTS voice %s", name)
		}
		tensor, err := reader.GetTensor(tensorName)
		reader.Close()
		if err != nil {
			return Inventory{}, nil, err
		}
		if (tensor.Dtype != "BF16" && tensor.Dtype != "F32") || !slices.Equal(tensor.Shape, []int32{int32(count), 3072}) {
			return Inventory{}, nil, fmt.Errorf("Voxtral TTS voice %s shape/type mismatch", name)
		}
		key := "voices." + name
		if inv.Has(key) {
			return Inventory{}, nil, fmt.Errorf("Voxtral TTS tensor collision: %s", key)
		}
		inv.Tensors[key] = SourceTensor{Name: tensorName, Dtype: tensor.Dtype, Shape: tensor.Shape, File: file}
		voices[name] = count
	}
	var fields map[string]json.RawMessage
	if err = json.Unmarshal(inv.RawConfig, &fields); err != nil {
		return Inventory{}, nil, err
	}
	fields["ollama_voice_num_audio_tokens"], _ = json.Marshal(voices)
	inv.RawConfig, err = json.Marshal(fields)
	return inv, &validatedTokenizerSource{tekken: raw}, err
}

type voxtralTTSImportTransform struct{ voices []string }

func newVoxtralTTSImportTransform(raw json.RawMessage) (quantizePolicy, error) {
	cfg, err := voxtralconfig.Parse(raw)
	if err != nil {
		return nil, err
	}
	voices := make([]string, 0, len(cfg.Voices))
	for name := range cfg.Voices {
		voices = append(voices, name)
	}
	slices.Sort(voices)
	return voxtralTTSImportTransform{voices: voices}, nil
}

func (p voxtralTTSImportTransform) augmentConfig(cfg *model.ConfigV2) error {
	cfg.Voices = p.voices
	cfg.VoiceLanguages = map[string][]string{}
	for _, voice := range p.voices {
		language := voxtralconfig.VoiceLanguage(voice)
		if language == "" {
			return fmt.Errorf("unknown Voxtral TTS preset %q", voice)
		}
		cfg.VoiceLanguages[voice] = []string{language}
		if !slices.Contains(cfg.Languages, language) {
			cfg.Languages = append(cfg.Languages, language)
		}
	}
	slices.Sort(cfg.Languages)
	return nil
}
func (voxtralTTSImportTransform) quantizationType(string, []int32, string) string { return "" }
