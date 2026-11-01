// Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
// SPDX-License-Identifier: Apache-2.0
// Native Go/MLX adaptations for Ollama.

package qwen3asr

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/model/qwen3"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() { model.Register(Architecture, newModel) }

type Model struct {
	// Arrays exposes promoted weights to the runner's load-scope collector.
	Arrays  []*mlx.Array
	decoder *qwen3.Model
	encoder *encoder
	cfg     *config
	raw     []byte
	tok     *tokenizer.Tokenizer
}

func newModel(root *model.Root) (model.Model, error) {
	raw, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	cfg, err := parseConfig(raw)
	if err != nil {
		return nil, err
	}
	preprocessor, err := root.Manifest.ReadConfig("preprocessor_config.json")
	if err != nil {
		return nil, err
	}
	if err := ValidatePreprocessor(preprocessor); err != nil {
		return nil, err
	}
	if len(root.AllTensorQuant()) != 0 {
		return nil, fmt.Errorf("Qwen3-ASR requires original floating-point weights")
	}
	data, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, err
	}
	generation, err := root.Manifest.ReadConfig("generation_config.json")
	if err != nil {
		return nil, err
	}
	tokenConfig, err := root.Manifest.ReadConfig("tokenizer_config.json")
	if err != nil {
		return nil, err
	}
	tok, err := LoadTokenizer(raw, data, tokenConfig, generation)
	if err != nil {
		return nil, err
	}
	return &Model{cfg: cfg, raw: raw, tok: tok}, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	shapes, err := TensorShapes(m.raw)
	if err != nil {
		return err
	}
	if len(tensors) != len(shapes) {
		return fmt.Errorf("Qwen3-ASR requires %d tensors, got %d", len(shapes), len(tensors))
	}
	for n, shape := range shapes {
		a := tensors[n]
		if a == nil || !slices.Equal(a.Dims(), intDims(shape)) {
			return fmt.Errorf("Qwen3-ASR tensor %s requires shape %v", n, shape)
		}
		if a.DType() != mlx.DTypeBFloat16 && a.DType() != mlx.DTypeFloat32 {
			return fmt.Errorf("Qwen3-ASR unsupported dtype for %s", n)
		}
	}
	// The first qualified milestone executes in F32. Import preserves the
	// original BF16 blobs; promotion is native and in-memory only. BF16 neural
	// execution has not passed the independent provider prefill/logit gate.
	promoted := make(map[string]*mlx.Array, len(tensors))
	m.Arrays = make([]*mlx.Array, 0, len(tensors))
	for name, value := range tensors {
		promoted[name] = value.AsType(mlx.DTypeFloat32)
		m.Arrays = append(m.Arrays, promoted[name])
	}
	tensors = promoted
	decoderWeights := map[string]*mlx.Array{}
	for n, a := range tensors {
		if strings.HasPrefix(n, "thinker.model.") || n == "thinker.lm_head.weight" {
			decoderWeights[strings.TrimPrefix(n, "thinker.")] = a
		}
	}
	m.decoder = &qwen3.Model{Config: &m.cfg.Text, Layers: make([]*qwen3.Layer, m.cfg.Text.NumHiddenLayers)}
	if err := m.decoder.LoadWeights(decoderWeights); err != nil {
		return err
	}
	m.encoder = &encoder{cfg: m.cfg.Audio, tensors: tensors}
	return nil
}

func intDims(shape []int32) []int {
	out := make([]int, len(shape))
	for i, v := range shape {
		out[i] = int(v)
	}
	return out
}
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }
func (m *Model) MaxContextLength() int           { return int(m.cfg.Text.MaxPositionEmbeddings) }

// RetainRequestCache keeps complete-audio generations independent. KV state
// still lives throughout each request, then its prefix history is released.
func (*Model) RetainRequestCache() bool { return false }

func (m *Model) NewCaches() []cache.Cache        { return m.decoder.NewCaches() }
func (m *Model) Unembed(x *mlx.Array) *mlx.Array { return m.decoder.Unembed(x) }

func (m *Model) PrepareMedia(segments []model.Segment) (*model.PreparedRequest, error) {
	return m.PrepareMediaContext(context.Background(), segments)
}

// PrepareMediaContext cancels CPU preparation before requests reach the MLX worker.
func (m *Model) PrepareMediaContext(ctx context.Context, segments []model.Segment) (*model.PreparedRequest, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	audioCount := 0
	for _, s := range segments {
		if s.Kind == "audio" {
			audioCount++
		}
	}
	if audioCount > 1 {
		return nil, fmt.Errorf("Qwen3-ASR requires one audio clip per generation")
	}
	req := &model.PreparedRequest{}
	for source, s := range segments {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if s.Kind == "" && s.Data == nil {
			req.Tokens = append(req.Tokens, s.Tokens...)
			continue
		}
		if s.Kind != "audio" {
			return nil, fmt.Errorf("Qwen3-ASR requires audio input")
		}
		samples, rate, err := audio.DecodeLimitContext(ctx, s.Data, maxAudioSeconds)
		if err != nil {
			return nil, err
		}
		if len(samples) > maxAudioSeconds*rate {
			return nil, fmt.Errorf("Qwen3-ASR accepts up to %d seconds per generation; use audio transcriptions for longer recordings", maxAudioSeconds)
		}
		samples, err = audio.ResampleContext(ctx, samples, rate, 16000)
		if err != nil {
			return nil, err
		}
		f, err := extractFeaturesContext(ctx, samples)
		if err != nil {
			return nil, err
		}
		start := len(req.Tokens)
		for range audioTokens(f.Frames) {
			req.Tokens = append(req.Tokens, audioPad)
		}
		req.Items = append(req.Items, model.PreparedItem{Source: source, Range: [2]int{start, len(req.Tokens)}, MediaData: f.Data, Dims: []int{128, f.Frames}, Causal: true})
	}
	return req, nil
}

func (m *Model) EncodeMedia(_ *model.PreparedItem, data *mlx.Array) *mlx.Array {
	return m.encoder.forward(data)
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) (*mlx.Array, *mlx.Array) {
	h := m.decoder.EmbedTokens.Forward(b.InputIDs)
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		// Items may overlap only part of a causal prefill chunk.
		off := int(b.SeqOffsets[item.Seq])
		n := item.Features.Dim(0)
		lo, hi := max(item.Pos, off), min(item.Pos+n, off+b.InputIDs.Dim(1))
		if lo >= hi {
			continue
		}
		part := mlx.ExpandDims(item.Features.Slice(mlx.Slice(lo-item.Pos, hi-item.Pos), mlx.Slice()), 0)
		h = h.SliceUpdate(part, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(lo-off, hi-off), mlx.Slice())
	}
	return m.forwardEmbeddings(h, b, caches)
}

func (m *Model) forwardEmbeddings(h *mlx.Array, b *batch.Batch, caches []cache.Cache) (*mlx.Array, *mlx.Array) {
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	for i, layer := range m.decoder.Layers {
		var c cache.Cache
		if caches != nil {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, int32(h.Dim(0)), int32(h.Dim(1)), m.decoder.Config)
	}
	out := m.decoder.Norm.Forward(h, m.cfg.Text.RMSNormEps)
	return out, out
}
