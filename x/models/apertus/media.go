package apertus

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	shared "github.com/ollama/ollama/x/mlxrunner/media"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const (
	apertusBOI          = "<|img_start|>"
	apertusEOI          = "<|img_end|>"
	apertusImageWrapper = "<|img_token_start|>"
	apertusImageEOL     = "<|img_end_of_row|>"
	apertusImageToken   = "<|image|>"
	apertusBOA          = "<|audio_start|>"
	apertusEOA          = "<|audio_end|>"
	apertusAudioToken   = "<|audio|>"
)

type apertusMediaItem struct {
	id       int
	kind     llm.MediaKind
	spans    []batch.TokenSpan
	expected int
	image    *apertusImageInput
	audio    *apertusAudioInput
}

type apertusMediaPayload struct{ items []apertusMediaItem }

func (m *Model) PrepareMediaPrompt(ctx context.Context, prompt string, inputs []llm.MediaData) (*batch.PreparedInput, error) {
	if !isApertus1p5Config(*m.Config) {
		return nil, errors.New("Apertus media is only supported by Apertus 1.5")
	}
	bindings, err := shared.BindMarkers(prompt, inputs)
	if err != nil {
		return nil, fmt.Errorf("Apertus 1.5 media markers: %w", err)
	}
	items := make([]apertusMediaItem, len(bindings))
	for i := range bindings {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		binding := &bindings[i]
		item := apertusMediaItem{id: binding.Media.ID, kind: binding.Media.Kind}
		switch binding.Media.Kind {
		case llm.MediaKindImage:
			if m.Vision == nil {
				return nil, errors.New("Apertus 1.5 image input requires complete vision tokenizer weights")
			}
			item.image, err = inspectApertusImage(ctx, binding.Media.Data)
			if err != nil {
				return nil, fmt.Errorf("Apertus 1.5 image %d: %w", binding.Media.ID, err)
			}
		case llm.MediaKindAudio:
			if m.Audio == nil {
				return nil, errors.New("Apertus 1.5 audio input requires complete audio tokenizer weights")
			}
			item.audio, err = preprocessApertusAudio(ctx, binding.Media.Data, m.AudioTokenizer)
			if err != nil {
				return nil, fmt.Errorf("Apertus 1.5 audio %d: %w", binding.Media.ID, err)
			}
		default:
			return nil, fmt.Errorf("Apertus 1.5 media %d has unsupported kind %q", binding.Media.ID, binding.Media.Kind)
		}
		items[i] = item
	}
	if err := m.fitApertusImages(items); err != nil {
		return nil, err
	}
	total := 0
	for i := range items {
		item, binding := &items[i], &bindings[i]
		switch item.kind {
		case llm.MediaKindImage:
			item.expected = item.image.gridWidth * item.image.gridHeight
			rows := make([]string, item.image.gridHeight)
			row := strings.Repeat(apertusImageToken, item.image.gridWidth)
			for j := range rows {
				rows[j] = row
			}
			binding.Sequence = apertusBOI + fmt.Sprintf("%d*%d", item.image.gridHeight, item.image.gridWidth) + apertusImageWrapper + strings.Join(rows, apertusImageEOL) + apertusEOI
		case llm.MediaKindAudio:
			item.expected = item.audio.codes
			binding.Sequence = apertusBOA + strings.Repeat(apertusAudioToken, item.expected) + apertusEOA
		}
		binding.ExpectedTokens = item.expected
		total, err = shared.AddTokenBudget(total, item.expected)
		if err != nil {
			return nil, fmt.Errorf("Apertus 1.5 media budget: %w", err)
		}
	}
	expanded, err := shared.ExpandPrompt(prompt, bindings)
	if err != nil {
		return nil, fmt.Errorf("Apertus 1.5 prompt expansion: %w", err)
	}
	tokens := m.tok.Encode(expanded, m.tok.AddBOS())
	if len(tokens) == 0 {
		return nil, errors.New("Apertus 1.5 media prompt tokenized to empty input")
	}
	imageSpans := placeholderSpans(tokens, m.ImageTokenID)
	audioSpans := placeholderSpans(tokens, m.AudioTokenID)
	imageIndex, audioIndex, previousEnd, assigned := 0, 0, 0, 0
	for i := range items {
		item := &items[i]
		switch item.kind {
		case llm.MediaKindImage:
			for range item.image.gridHeight {
				if imageIndex >= len(imageSpans) {
					return nil, fmt.Errorf("Apertus 1.5 image %d has missing placeholder row", item.id)
				}
				span := imageSpans[imageIndex]
				imageIndex++
				if span.End-span.Start != item.image.gridWidth {
					return nil, fmt.Errorf("Apertus 1.5 image %d row token count = %d, want %d", item.id, span.End-span.Start, item.image.gridWidth)
				}
				item.spans = append(item.spans, span)
			}
		case llm.MediaKindAudio:
			if audioIndex >= len(audioSpans) {
				return nil, fmt.Errorf("Apertus 1.5 audio %d has missing placeholder span", item.id)
			}
			span := audioSpans[audioIndex]
			audioIndex++
			if span.End-span.Start != item.expected {
				return nil, fmt.Errorf("Apertus 1.5 audio %d token count = %d, want %d", item.id, span.End-span.Start, item.expected)
			}
			item.spans = []batch.TokenSpan{span}
		}
		if len(item.spans) == 0 || item.spans[0].Start < previousEnd {
			return nil, fmt.Errorf("Apertus 1.5 media spans do not follow marker order at media %d", item.id)
		}
		previousEnd = item.spans[len(item.spans)-1].End
		assigned += item.expected
	}
	if imageIndex != len(imageSpans) || audioIndex != len(audioSpans) {
		return nil, errors.New("Apertus 1.5 prompt contains unexpected media placeholder tokens")
	}
	if assigned != total {
		return nil, fmt.Errorf("Apertus 1.5 assigned media tokens %d, want %d", assigned, total)
	}
	return &batch.PreparedInput{Tokens: tokens, Payload: &apertusMediaPayload{items: items}}, nil
}

func (m *Model) fitApertusImages(items []apertusMediaItem) error {
	inputs := make([]*apertusImageInput, 0, len(items))
	for i := range items {
		if items[i].image != nil {
			inputs = append(inputs, items[i].image)
		}
	}
	if len(inputs) == 0 || m.mediaMemoryLimit == 0 {
		return nil
	}
	canonicalPeak := estimateApertusImagePeak(m.mediaResident, inputs)
	if canonicalPeak <= m.mediaMemoryLimit {
		return nil
	}
	for _, input := range inputs {
		setApertusImageArea(input, minApertusImageArea)
	}
	minimumPeak := estimateApertusImagePeak(m.mediaResident, inputs)
	if minimumPeak > m.mediaMemoryLimit {
		return fmt.Errorf("Apertus image minimum resolution requires an estimated %s peak but the safe media budget is %s; unload other models or use the NVFP4 model",
			format.HumanBytes2(minimumPeak), format.HumanBytes2(m.mediaMemoryLimit))
	}
	best := minApertusImageArea
	for low, high := minApertusImageArea, maxApertusImageArea; low <= high; {
		candidate := low + (high-low)/2
		for _, input := range inputs {
			setApertusImageArea(input, candidate)
		}
		if estimateApertusImagePeak(m.mediaResident, inputs) <= m.mediaMemoryLimit {
			best = candidate
			low = candidate + 1
		} else {
			high = candidate - 1
		}
	}
	var reductions []string
	for _, input := range inputs {
		setApertusImageArea(input, best)
		if input.width != input.canonicalWidth || input.height != input.canonicalHeight {
			reductions = append(reductions, fmt.Sprintf("original=%dx%d canonical=%dx%d final=%dx%d (%d->%d tokens)",
				input.originalWidth, input.originalHeight, input.canonicalWidth, input.canonicalHeight, input.width, input.height,
				input.canonicalWidth/16*(input.canonicalHeight/16), input.gridWidth*input.gridHeight))
		}
	}
	slog.Warn("Apertus image resolution reduced to fit memory budget",
		"images", strings.Join(reductions, ", "),
		"estimated_peak", format.HumanBytes2(estimateApertusImagePeak(m.mediaResident, inputs)),
		"safe_budget", format.HumanBytes2(m.mediaMemoryLimit))
	return nil
}

func placeholderSpans(tokens []int32, id int32) []batch.TokenSpan {
	var spans []batch.TokenSpan
	for i := 0; i < len(tokens); {
		if tokens[i] != id {
			i++
			continue
		}
		start := i
		for i < len(tokens) && tokens[i] == id {
			i++
		}
		spans = append(spans, batch.TokenSpan{Start: start, End: i})
	}
	return spans
}

func (m *Model) PrepareMediaEmbeddings(ctx context.Context, prepared *batch.PreparedInput) error {
	if prepared == nil {
		return errors.New("Apertus 1.5 media input is nil")
	}
	payload, ok := prepared.Payload.(*apertusMediaPayload)
	if !ok || payload == nil || len(payload.items) == 0 {
		return errors.New("Apertus 1.5 media payload is invalid")
	}
	if len(prepared.Tokens) == 0 {
		return errors.New("Apertus 1.5 media tokens are empty")
	}
	ids := mlx.Reshape(mlx.FromValues(prepared.Tokens, len(prepared.Tokens)), 1, int32(len(prepared.Tokens)))
	embeddings := m.EmbedTokens.Forward(ids)
	mlx.Eval(embeddings)
	mlx.Pin(embeddings)
	pinned := []*mlx.Array{embeddings}
	defer func() { mlx.Unpin(pinned...) }()
	replacements := make([]shared.Replacement, 0, len(payload.items))
	previousEnd := 0
	seen := map[int]bool{}
	for _, item := range payload.items {
		if err := ctx.Err(); err != nil {
			return err
		}
		if item.id < 0 || seen[item.id] || len(item.spans) == 0 {
			return fmt.Errorf("Apertus 1.5 media has invalid or duplicate ID %d", item.id)
		}
		seen[item.id] = true
		if item.spans[0].Start < previousEnd {
			return fmt.Errorf("Apertus 1.5 media %d overlaps a previous span", item.id)
		}
		var codes *mlx.Array
		var offset int32
		switch item.kind {
		case llm.MediaKindImage:
			if m.Vision == nil || item.image == nil {
				return fmt.Errorf("Apertus 1.5 image %d payload is invalid", item.id)
			}
			pixels, err := materializeApertusImage(ctx, item.image)
			if err != nil {
				return fmt.Errorf("Apertus 1.5 image %d decode: %w", item.id, err)
			}
			codes, err = m.Vision.encode(ctx, pixels, item.image.width, item.image.height)
			if err != nil {
				return fmt.Errorf("Apertus 1.5 image %d encode: %w", item.id, err)
			}
			offset = m.ImageTokenOffset
		case llm.MediaKindAudio:
			if m.Audio == nil || item.audio == nil {
				return fmt.Errorf("Apertus 1.5 audio %d payload is invalid", item.id)
			}
			var err error
			codes, err = m.Audio.encode(ctx, item.audio.samples)
			if err != nil {
				return fmt.Errorf("Apertus 1.5 audio %d encode: %w", item.id, err)
			}
			offset = m.AudioTokenOffset
		default:
			return fmt.Errorf("Apertus 1.5 media %d kind %q is invalid", item.id, item.kind)
		}
		if codes == nil || codes.NumDims() != 2 || codes.Dim(0) != 1 || codes.Dim(1) != item.expected {
			return fmt.Errorf("Apertus 1.5 media %d encoded shape %v, want [1,%d]", item.id, codes.Dims(), item.expected)
		}
		vocabIDs := mlx.Add(codes, mlx.NewScalarArray(float32(offset)).AsType(mlx.DTypeInt32))
		features := m.EmbedTokens.Forward(vocabIDs)
		mlx.Eval(features)
		mlx.Pin(features)
		mlx.Sweep()
		featureCursor := 0
		for _, span := range item.spans {
			if span.Start < previousEnd || span.Start < 0 || span.End <= span.Start || span.End > len(prepared.Tokens) {
				mlx.Unpin(features)
				return fmt.Errorf("Apertus 1.5 media %d has invalid span [%d,%d)", item.id, span.Start, span.End)
			}
			count := span.End - span.Start
			part := features.Slice(mlx.Slice(), mlx.Slice(featureCursor, featureCursor+count), mlx.Slice())
			mlx.Eval(part)
			mlx.Pin(part)
			pinned = append(pinned, part)
			replacements = append(replacements, shared.Replacement{Span: span, Features: part})
			featureCursor += count
			previousEnd = span.End
		}
		if featureCursor != item.expected {
			mlx.Unpin(features)
			return fmt.Errorf("Apertus 1.5 media %d spans cover %d tokens, want %d", item.id, featureCursor, item.expected)
		}
		mlx.Unpin(features)
	}
	if !slices.IsSortedFunc(replacements, func(a, b shared.Replacement) int { return a.Span.Start - b.Span.Start }) {
		return errors.New("Apertus 1.5 media replacements are unordered")
	}
	merged, err := shared.MergeEmbeddings(embeddings, len(prepared.Tokens), replacements)
	if err != nil {
		return fmt.Errorf("Apertus 1.5 embedding merge: %w", err)
	}
	mlx.Eval(merged)
	prepared.InputEmbeddings = merged
	return nil
}
