package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/tokenizer"
)

type Request struct {
	TextCompletionsRequest
	Responses chan CompletionResponse
	Pipeline  func(Request) error

	Ctx context.Context

	Sampler *sample.Sampler
}

type TextCompletionsRequest struct {
	Prompt  string `json:"prompt"`
	Options struct {
		Temperature     float32 `json:"temperature"`
		TopP            float32 `json:"top_p"`
		MinP            float32 `json:"min_p"`
		TopK            int     `json:"top_k"`
		RepeatLastN     int     `json:"repeat_last_n"`
		PresencePenalty float32 `json:"presence_penalty"`
		MaxTokens       int     `json:"max_tokens"`

		// Deprecated: use MaxTokens instead
		NumPredict int `json:"num_predict"`
	} `json:"options"`
}

type Runner struct {
	Model         base.Model
	Tokenizer     *tokenizer.Tokenizer
	Requests      chan Request
	cache         kvCache
	contextLength int
}

func (r *Runner) Load(modelName string) error {
	root, err := model.Open(modelName)
	if err != nil {
		return err
	}
	defer root.Close()

	m, err := base.New(root)
	if err != nil {
		return err
	}

	// Load all tensor blobs from manifest
	tensors, err := loadTensorsFromManifest(root)
	if err != nil {
		return err
	}

	// Assign weights to model (model-specific logic)
	loadWeights := base.Weights(m)
	if err := loadWeights(tensors); err != nil {
		return err
	}

	r.Model = m
	r.Tokenizer = m.Tokenizer()
	r.contextLength = m.MaxContextLength()
	return nil
}

// loadTensorsFromManifest loads all tensor blobs from the manifest into a
// flat map, deduplicating by digest and remapping safetensors key suffixes.
//
// Uses a two-phase approach: first loads all raw tensors, then remaps
// .bias → _qbias with complete knowledge of which base names have .scale
// entries. This avoids a race condition where Go map iteration order could
// cause .bias to be processed before .scale within the same blob.
func loadTensorsFromManifest(root *model.Root) (map[string]*mlx.Array, error) {
	// Phase 1: Load all tensors raw from all blobs
	rawTensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	for _, layer := range root.Manifest.GetTensorLayers("") {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		blobPath := root.Manifest.BlobPath(layer.Digest)
		for name, arr := range mlx.Load(blobPath) {
			rawTensors[name] = arr
		}
	}

	// Phase 2: Identify all base names that have .scale tensors and remap them
	scaleBaseNames := make(map[string]bool)
	allTensors := make(map[string]*mlx.Array, len(rawTensors))
	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") {
			baseName := strings.TrimSuffix(name, ".scale")
			allTensors[baseName+"_scale"] = arr
			scaleBaseNames[baseName] = true
		}
	}

	// Phase 3: Process remaining tensors with complete scale knowledge
	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") {
			continue // already handled
		}
		if strings.HasSuffix(name, ".bias") && !strings.HasSuffix(name, ".weight_qbias") {
			baseName := strings.TrimSuffix(name, ".bias")
			if scaleBaseNames[baseName] {
				allTensors[baseName+"_qbias"] = arr
			} else {
				allTensors[name] = arr
			}
		} else {
			allTensors[name] = arr
		}
	}

	for name, arr := range allTensors {
		if !shouldSafeCopyDenseBF16(name, arr, scaleBaseNames) {
			continue
		}
		blobPath, ok := tensorBlobPath(root, name)
		if !ok {
			continue
		}
		safe, err := loadDenseTensorFromRaw(blobPath, name)
		if err != nil {
			return nil, fmt.Errorf("safe-load dense bf16 tensor %s: %w", name, err)
		}
		allTensors[name] = safe
	}

	slog.Info("Loaded tensors from manifest", "count", len(allTensors))
	return allTensors, nil
}

func shouldSafeCopyDenseBF16(name string, arr *mlx.Array, scaleBaseNames map[string]bool) bool {
	if arr == nil || !arr.Valid() || arr.DType() != mlx.DTypeBFloat16 {
		return false
	}
	if scaleBaseNames[name] {
		return false
	}
	switch {
	case name == "token_embd.weight":
		return true
	case strings.HasSuffix(name, "embed_tokens.weight"):
		return true
	default:
		return false
	}
}

func tensorBlobPath(root *model.Root, tensorName string) (string, bool) {
	for _, layer := range root.Manifest.GetTensorLayers("") {
		if layer.Name == tensorName {
			return root.Manifest.BlobPath(layer.Digest), true
		}
	}
	return "", false
}

func loadDenseTensorFromRaw(blobPath, name string) (*mlx.Array, error) {
	extractor, err := safetensors.OpenForExtraction(blobPath)
	if err != nil {
		return nil, err
	}
	defer extractor.Close()

	td, err := extractor.GetTensor(name)
	if err != nil {
		return nil, err
	}
	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		return nil, err
	}

	shape := make([]int, len(td.Shape))
	for i, d := range td.Shape {
		shape[i] = int(d)
	}
	return mlx.FromRawBytes(raw, mlx.DTypeBFloat16, shape...), nil
}

func (r *Runner) Run(host, port string, mux http.Handler) error {
	g, ctx := errgroup.WithContext(context.Background())

	g.Go(func() error {
		for {
			select {
			case <-ctx.Done():
				return nil
			case request := <-r.Requests:
				if err := request.Pipeline(request); err != nil {
					slog.Info("Request terminated", "error", err)
					var statusErr api.StatusError
					if !errors.As(err, &statusErr) {
						statusErr = api.StatusError{
							StatusCode:   http.StatusInternalServerError,
							ErrorMessage: err.Error(),
						}
					}
					select {
					case request.Responses <- CompletionResponse{Error: &statusErr}:
					case <-request.Ctx.Done():
					}
				}

				close(request.Responses)
			}
		}
	})

	g.Go(func() error {
		slog.Info("Starting HTTP server", "host", host, "port", port)
		return http.ListenAndServe(net.JoinHostPort(host, port), mux)
	})

	return g.Wait()
}
