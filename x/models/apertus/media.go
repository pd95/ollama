package apertus

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
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

type apertusMediaPayload struct {
	kind       string
	spans      [][2]int
	expected   int
	width      int
	height     int
	gridWidth  int
	gridHeight int
	samples    int
}

// PrepareMedia expands each media segment in stream order and retains one
// cache-identity item per source segment. Image geometry is inspected for the
// whole request before pixels are materialized so every image shares one
// deterministic memory-budget decision.
func (m *Model) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	ctx := context.Background()
	if !isApertus1p5Config(*m.Config) {
		return nil, errors.New("Apertus media is only supported by Apertus 1.5")
	}
	images := make(map[int]*apertusImageInput)
	var imageInputs []*apertusImageInput
	for source, seg := range segments {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if seg.Data == nil || seg.Kind != "image" {
			continue
		}
		if m.Vision == nil {
			return nil, errors.New("Apertus 1.5 image input requires complete vision tokenizer weights")
		}
		image, err := inspectApertusImage(ctx, seg.Data)
		if err != nil {
			return nil, fmt.Errorf("Apertus 1.5 image: %w", err)
		}
		images[source] = image
		imageInputs = append(imageInputs, image)
	}
	if err := m.fitApertusImages(imageInputs); err != nil {
		return nil, err
	}

	prepared := &base.PreparedRequest{}
	for source, seg := range segments {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if seg.Data == nil {
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
			continue
		}

		start := len(prepared.Tokens)
		payload := apertusMediaPayload{kind: seg.Kind}
		var mediaData []float32
		var dims []int
		var expansion string
		switch seg.Kind {
		case "image":
			image := images[source]
			pixels, err := materializeApertusImage(ctx, image)
			if err != nil {
				return nil, fmt.Errorf("Apertus 1.5 image: %w", err)
			}
			payload.expected = image.gridWidth * image.gridHeight
			payload.width, payload.height = image.width, image.height
			payload.gridWidth, payload.gridHeight = image.gridWidth, image.gridHeight
			rows := make([]string, image.gridHeight)
			row := strings.Repeat(apertusImageToken, image.gridWidth)
			for i := range rows {
				rows[i] = row
			}
			expansion = apertusBOI + fmt.Sprintf("%d*%d", image.gridHeight, image.gridWidth) + apertusImageWrapper + strings.Join(rows, apertusImageEOL) + apertusEOI
			mediaData = pixels
			dims = []int{1, image.height, image.width, 3}
		case "audio":
			if m.Audio == nil {
				return nil, errors.New("Apertus 1.5 audio input requires complete audio tokenizer weights")
			}
			audio, err := preprocessApertusAudio(ctx, seg.Data, m.AudioTokenizer)
			if err != nil {
				return nil, fmt.Errorf("Apertus 1.5 audio: %w", err)
			}
			payload.expected = audio.codes
			payload.samples = len(audio.samples)
			expansion = apertusBOA + strings.Repeat(apertusAudioToken, payload.expected) + apertusEOA
			mediaData = audio.samples
			dims = []int{1, len(audio.samples), 1}
		default:
			return nil, fmt.Errorf("Apertus 1.5 does not support %s input", seg.Kind)
		}

		expansionTokens := m.tok.Encode(expansion, false)
		placeholder := m.ImageTokenID
		if seg.Kind == "audio" {
			placeholder = m.AudioTokenID
		}
		payload.spans = placeholderSpans(expansionTokens, placeholder)
		count := 0
		for _, span := range payload.spans {
			count += span[1] - span[0]
		}
		if count != payload.expected {
			return nil, fmt.Errorf("Apertus 1.5 %s expansion has %d placeholder tokens, want %d", seg.Kind, count, payload.expected)
		}
		if seg.Kind == "image" {
			if len(payload.spans) != payload.gridHeight {
				return nil, fmt.Errorf("Apertus 1.5 image has %d placeholder rows, want %d", len(payload.spans), payload.gridHeight)
			}
			for _, span := range payload.spans {
				if span[1]-span[0] != payload.gridWidth {
					return nil, fmt.Errorf("Apertus 1.5 image row has %d placeholder tokens, want %d", span[1]-span[0], payload.gridWidth)
				}
			}
		}
		prepared.Tokens = append(prepared.Tokens, expansionTokens...)
		prepared.Items = append(prepared.Items, base.PreparedItem{
			Range:     [2]int{start, len(prepared.Tokens)},
			Source:    source,
			MediaData: mediaData,
			Dims:      dims,
			Opaque:    payload,
			Causal:    true,
		})
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return prepared, nil
}

func (m *Model) fitApertusImages(inputs []*apertusImageInput) error {
	if len(inputs) == 0 || m.mediaMemoryLimit == 0 {
		return nil
	}
	if estimateApertusImagePeak(m.mediaResident, inputs) <= m.mediaMemoryLimit {
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

func placeholderSpans(tokens []int32, id int32) [][2]int {
	var spans [][2]int
	for i := 0; i < len(tokens); {
		if tokens[i] != id {
			i++
			continue
		}
		start := i
		for i < len(tokens) && tokens[i] == id {
			i++
		}
		spans = append(spans, [2]int{start, i})
	}
	return spans
}

// EncodeMedia builds a lazy feature graph from the runner-owned media data.
func (m *Model) EncodeMedia(item *base.PreparedItem, data *mlx.Array) *mlx.Array {
	return m.encodeMedia(item, data, nil)
}

func (m *Model) EncodeMediaStaged(item *base.PreparedItem, data *mlx.Array, materialize base.MediaMaterializer) *mlx.Array {
	return m.encodeMedia(item, data, materialize)
}

func (m *Model) encodeMedia(item *base.PreparedItem, data *mlx.Array, materialize base.MediaMaterializer) *mlx.Array {
	payload := item.Opaque.(apertusMediaPayload)
	var codes *mlx.Array
	var err error
	var offset int32
	switch payload.kind {
	case "image":
		codes, err = m.Vision.encodeStaged(data, payload.width, payload.height, materialize)
		offset = m.ImageTokenOffset
	case "audio":
		codes, err = m.Audio.encodeStaged(data, payload.samples, materialize)
		offset = m.AudioTokenOffset
	default:
		panic(fmt.Sprintf("invalid Apertus 1.5 media kind %q", payload.kind))
	}
	if err != nil {
		panic(err)
	}
	if codes == nil {
		panic(fmt.Sprintf("Apertus 1.5 %s encoder returned no codes", payload.kind))
	}
	if codes.NumDims() != 2 || codes.Dim(0) != 1 || codes.Dim(1) != payload.expected {
		panic(fmt.Sprintf("Apertus 1.5 %s encoded shape %v, want [1,%d]", payload.kind, codes.Dims(), payload.expected))
	}
	vocabIDs := mlx.Add(codes, mlx.NewScalarArray(float32(offset)).AsType(mlx.DTypeInt32))
	return mlx.Squeeze(m.EmbedTokens.Forward(vocabIDs), 0)
}

func (m *Model) scatterMedia(h *mlx.Array, b *batch.Batch) *mlx.Array {
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		payload := item.Opaque.(apertusMediaPayload)
		featureOffset := 0
		for _, span := range payload.spans {
			start, end := item.Pos+span[0], item.Pos+span[1]
			basePos := int(b.SeqOffsets[item.Seq])
			lo := max(start, basePos)
			hi := min(end, basePos+int(b.SeqQueryLens[item.Seq]))
			if hi > lo {
				features := item.Features.Slice(mlx.Slice(featureOffset+lo-start, featureOffset+hi-start), mlx.Slice())
				features = mlx.Reshape(features.AsType(h.DType()), 1, int32(hi-lo), m.HiddenSize)
				h = h.SliceUpdate(features, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(lo-basePos, hi-basePos), mlx.Slice())
			}
			featureOffset += span[1] - span[0]
		}
	}
	return h
}

var (
	_ base.MediaModel       = (*Model)(nil)
	_ base.StagedMediaModel = (*Model)(nil)
)
