package create

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	kokoroconfig "github.com/ollama/ollama/mlxrunner/model/kokoro/config"
	"github.com/ollama/ollama/types/model"
)

const kokoroArchitecture = kokoroconfig.Architecture

var englishKokoroVoice = regexp.MustCompile(`^[ab][fm]_[a-z0-9_]+$`)

// Detect the publisher's architecture-less schema narrowly and normalize only
// the imported config bytes. A directory name is not model identity evidence.
func normalizeKokoroConfig(raw json.RawMessage) (json.RawMessage, bool, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, false, err
	}
	for _, key := range []string{"istftnet", "plbert", "vocab"} {
		if fields[key] == nil {
			return raw, false, nil
		}
	}
	if _, err := kokoroconfig.Parse(raw); err != nil {
		return nil, false, err
	}
	fields["model_type"] = json.RawMessage(`"kokoro"`)
	fields["architectures"], _ = json.Marshal([]string{kokoroArchitecture})
	normalized, err := json.Marshal(fields)
	return normalized, true, err
}

func kokoroVoiceFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(filepath.Join(dir, "voices"))
	if err != nil {
		return nil, fmt.Errorf("Kokoro requires English voice packs: %w", err)
	}
	voices := make(map[string]string)
	for _, entry := range entries {
		ext := filepath.Ext(entry.Name())
		name := strings.TrimSuffix(entry.Name(), ext)
		if (ext != ".safetensors" && ext != ".pt") || !englishKokoroVoice.MatchString(name) || entry.IsDir() {
			continue
		}
		if previous, ok := voices[name]; ok {
			return nil, fmt.Errorf("ambiguous Kokoro voice %s: %s and %s", name, previous, entry.Name())
		}
		voices[name] = "voices/" + entry.Name()
	}
	if voices["af_heart"] == "" {
		return nil, fmt.Errorf("Kokoro requires voices/af_heart.pt or voices/af_heart.safetensors")
	}
	files := make([]string, 0, len(voices))
	for _, name := range voices {
		files = append(files, name)
	}
	slices.Sort(files)
	return files, nil
}

func prepareKokoroInventory(inv Inventory) (Inventory, error) {
	if inv.Config.Architecture() != kokoroArchitecture {
		return inv, nil
	}
	var err error
	inv, err = prepareOriginalKokoroInventory(inv)
	if err != nil {
		return Inventory{}, err
	}
	files, err := kokoroVoiceFiles(inv.Dir)
	if err != nil {
		return Inventory{}, err
	}
	inv.Tensors = maps.Clone(inv.Tensors)
	var voices []string
	for _, file := range files {
		reader, err := openTensorSource(context.Background(), filepath.Join(inv.Dir, file))
		if err != nil {
			return Inventory{}, err
		}
		tensorName := "voice"
		if filepath.Ext(file) == ".pt" {
			tensorName = "tensor"
		}
		names := reader.ListTensors()
		if !slices.Equal(names, []string{tensorName}) {
			reader.Close()
			return Inventory{}, fmt.Errorf("Kokoro %s must contain only voice", file)
		}
		tensor, err := reader.GetTensor(tensorName)
		if err != nil {
			reader.Close()
			return Inventory{}, err
		}
		if tensor.Dtype != "F32" || !slices.Equal(tensor.Shape, []int32{510, 1, 256}) {
			reader.Close()
			return Inventory{}, fmt.Errorf("unsupported Kokoro voice %s: %s %v", file, tensor.Dtype, tensor.Shape)
		}
		name := strings.TrimSuffix(filepath.Base(file), filepath.Ext(file))
		key := "voices." + name
		if inv.Has(key) {
			reader.Close()
			return Inventory{}, fmt.Errorf("Kokoro tensor collision: %s", key)
		}
		inv.Tensors[key] = SourceTensor{Name: tensorName, Dtype: tensor.Dtype, Shape: tensor.Shape, File: file}
		voices = append(voices, name)
		reader.Close()
	}
	var cfg map[string]json.RawMessage
	if err := json.Unmarshal(inv.RawConfig, &cfg); err != nil {
		return Inventory{}, err
	}
	cfg["ollama_voices"], err = json.Marshal(voices)
	if err != nil {
		return Inventory{}, err
	}
	inv.RawConfig, err = json.Marshal(cfg)
	return inv, err
}

type kokoroImportTransform struct{ voices []string }

func newKokoroImportTransform(raw json.RawMessage) (quantizePolicy, error) {
	cfg, err := kokoroconfig.Parse(raw)
	if err != nil {
		return nil, err
	}
	return kokoroImportTransform{voices: cfg.Voices}, nil
}

func (p kokoroImportTransform) augmentConfig(cfg *model.ConfigV2) error {
	cfg.Voices = p.voices
	cfg.VoiceLanguages = map[string][]string{}
	for _, voice := range p.voices {
		language := "en-US"
		if strings.HasPrefix(voice, "b") {
			language = "en-GB"
		}
		cfg.VoiceLanguages[voice] = []string{language}
		if !slices.Contains(cfg.Languages, language) {
			cfg.Languages = append(cfg.Languages, language)
		}
	}
	slices.Sort(cfg.Languages)
	return nil
}
func (kokoroImportTransform) quantizationType(string, []int32, string) string { return "" }
