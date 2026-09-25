package create

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

// ErrInvalidTokenizer marks tokenizer metadata that cannot be safely imported.
var ErrInvalidTokenizer = errors.New("invalid tokenizer")

type validatedTokenizerSource struct {
	data []byte
}

// validateTokenizerSource checks the exact tokenizer.json bytes that will be
// imported. Some test and legacy sources do not contain this optional file.
func validateTokenizerSource(modelDir string) (*validatedTokenizerSource, error) {
	path := filepath.Join(modelDir, "tokenizer.json")
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}
	if err := tokenizer.ValidateTokenizerJSONIDs(data); err != nil {
		return nil, fmt.Errorf("%w: tokenizer.json: %v", ErrInvalidTokenizer, err)
	}
	return &validatedTokenizerSource{data: data}, nil
}

// PipelineOptions controls the source-specific stages of a safetensors import.
type PipelineOptions struct {
	Quantize      string
	Parser        string
	Renderer      string
	Requires      string
	DraftDir      string
	DraftQuantize string
	Validation    MLXValidationOptions
}

// Create imports a safetensors model through the full pipeline: read the
// source into an inventory, classify it, plan the output blobs, write them
// through store, import the config files, and write the manifest. It is the
// shared local and server entry point; the caller supplies blob storage (store)
// and manifest assembly (writeManifest). store, writeManifest, and fn must be
// non-nil.
func Create(ctx context.Context, modelName, modelDir string, opts PipelineOptions, store BlobStore, writeManifest ManifestWriter, fn func(status string)) error {
	defer releaseMLXCache()

	if err := checkContext(ctx); err != nil {
		return err
	}
	inv, err := ReadInventory(modelDir)
	if err != nil {
		return fmt.Errorf("read model: %w", err)
	}
	modelConfig, err := inferSafetensorsConfig(modelDir, inv.Config, opts.Parser, opts.Renderer)
	if err != nil {
		return err
	}
	if opts.Requires != "" {
		modelConfig.Requires, err = validateRequires(opts.Requires)
		if err != nil {
			return err
		}
	}
	if err := validateMLXSource(inv.Config, false, opts.Validation); err != nil {
		return err
	}
	validatedTokenizer, err := validateTokenizerSource(modelDir)
	if err != nil {
		return err
	}
	if err := checkContext(ctx); err != nil {
		return err
	}
	class, err := Classify(inv, opts.Quantize)
	if err != nil {
		return err
	}
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		return fmt.Errorf("build quantization policy for %q: %w", inv.Config.Architecture(), err)
	}
	specs, err := Plan(inv, class, policy)
	if err != nil {
		return fmt.Errorf("plan model: %w", err)
	}

	var draftLayers []LayerInfo
	if opts.DraftDir != "" {
		draftLayers, err = createDraftLayers(ctx, opts.DraftDir, "draft.", "draft/", opts.DraftQuantize, opts.Validation, store, fn)
		if err != nil {
			return err
		}
	}

	fn(fmt.Sprintf("importing %s (%d tensors%s)", modelName, len(inv.Tensors), quantizeStatus(class)))
	layers, err := WriteBlobs(ctx, specs, modelDir, store)
	if err != nil {
		return err
	}

	// Import config files (config.json, tokenizer, etc.) as JSON blobs.
	configLayers, configLayer, err := importConfigBlobs(ctx, modelDir, "", validatedTokenizer, store, fn)
	if err != nil {
		return err
	}
	layers = append(layers, configLayers...)
	layers = append(layers, draftLayers...)
	if configLayer.Digest == "" {
		return fmt.Errorf("config.json not found in %s", modelDir)
	}
	if err := checkContext(ctx); err != nil {
		return err
	}

	fn(fmt.Sprintf("writing manifest for %s", modelName))
	if err := writeManifest(ctx, modelName, ManifestInfo{ModelConfig: modelConfig, ConfigLayer: configLayer, Layers: layers, Class: class}); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	fn(fmt.Sprintf("successfully imported %s with %d layers", modelName, len(layers)))
	return nil
}

func checkContext(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("nil context")
	}
	return ctx.Err()
}

const mediaTypeImageJSON = "application/vnd.ollama.image.json"

// importConfigBlobs writes every .json in modelDir (except the shard index) as an
// image.json blob, prefixing each blob name with namePrefix, and returns the
// resulting layers along with the config.json layer (zero value if absent). The
// target import passes "" for namePrefix; a draft import passes "draft/" so its
// config sits beside the target's. tokenizer.json is written from the exact
// bytes validated before tensor import, even if the source file changes later.
func importConfigBlobs(ctx context.Context, modelDir, namePrefix string, validatedTokenizer *validatedTokenizerSource, store BlobStore, fn func(status string)) ([]LayerInfo, LayerInfo, error) {
	entries, err := os.ReadDir(modelDir)
	if err != nil {
		return nil, LayerInfo{}, err
	}
	var layers []LayerInfo
	var configLayer LayerInfo
	seenTokenizer := false
	for _, entry := range entries {
		if err := checkContext(ctx); err != nil {
			return nil, LayerInfo{}, err
		}
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") || entry.Name() == "model.safetensors.index.json" {
			continue
		}
		name := entry.Name()
		fn(fmt.Sprintf("importing config %s", name))
		var source io.Reader
		var closeSource func() error
		if name == "tokenizer.json" {
			if validatedTokenizer == nil {
				return nil, LayerInfo{}, fmt.Errorf("%w: tokenizer.json appeared after preflight", ErrInvalidTokenizer)
			}
			seenTokenizer = true
			source = bytes.NewReader(validatedTokenizer.data)
		} else {
			f, err := os.Open(filepath.Join(modelDir, name))
			if err != nil {
				return nil, LayerInfo{}, fmt.Errorf("open %s: %w", name, err)
			}
			source = f
			closeSource = f.Close
		}
		layer, err := store.WriteBlob(ReaderWithContext(ctx, source), mediaTypeImageJSON, namePrefix+name)
		var closeErr error
		if closeSource != nil {
			closeErr = closeSource()
		}
		if err != nil {
			return nil, LayerInfo{}, fmt.Errorf("write config %s: %w", name, err)
		}
		if closeErr != nil {
			return nil, LayerInfo{}, fmt.Errorf("close config %s: %w", name, closeErr)
		}
		if name == "config.json" {
			configLayer = layer
		}
		layers = append(layers, layer)
	}
	if validatedTokenizer != nil && !seenTokenizer {
		return nil, LayerInfo{}, fmt.Errorf("%w: tokenizer.json disappeared after preflight", ErrInvalidTokenizer)
	}
	return layers, configLayer, nil
}

func quantizeStatus(c Classification) string {
	switch c.Kind {
	case SourceBlockFP8:
		return ", converting fp8 to mxfp8"
	case SourcePrequantized:
		return ", preserving source quantization"
	default:
		if c.Quantize != "" {
			return ", quantizing to " + c.Quantize
		}
		return ""
	}
}
