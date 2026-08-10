package apertus

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	shared "github.com/ollama/ollama/x/mlxrunner/media"
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

type apertusMediaItem struct {
	kind  llm.MediaKind
	image *apertusImageInput
	audio *apertusAudioInput
}

type apertusSpan struct {
	Start int
	End   int
}

type apertusPreparedMedia struct {
	kind     llm.MediaKind
	spans    []apertusSpan // relative to the prepared item's Range start
	expected int
	image    *apertusImageInput
	audio    *apertusAudioInput
}

var _ base.MediaModel = (*Model)(nil)

func (m *Model) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	if !isApertus1p5Config(*m.Config) {
		return nil, errors.New("Apertus media is only supported by Apertus 1.5")
	}

	states := make(map[int]*apertusPreparedMedia)
	fitItems := make([]apertusMediaItem, 0)
	for source, segment := range segments {
		if segment.Data == nil {
			continue
		}
		state := &apertusPreparedMedia{kind: llm.MediaKind(segment.Kind)}
		var err error
		switch state.kind {
		case llm.MediaKindImage:
			if m.Vision == nil {
				return nil, errors.New("Apertus 1.5 image input requires complete vision tokenizer weights")
			}
			state.image, err = inspectApertusImage(context.Background(), segment.Data)
		case llm.MediaKindAudio:
			if m.Audio == nil {
				return nil, errors.New("Apertus 1.5 audio input requires complete audio tokenizer weights")
			}
			state.audio, err = preprocessApertusAudio(context.Background(), segment.Data, m.AudioTokenizer)
		default:
			return nil, fmt.Errorf("Apertus 1.5 has unsupported media kind %q", segment.Kind)
		}
		if err != nil {
			return nil, fmt.Errorf("prepare Apertus 1.5 %s: %w", state.kind, err)
		}
		states[source] = state
		fitItems = append(fitItems, apertusMediaItem{kind: state.kind, image: state.image, audio: state.audio})
	}
	if err := m.fitApertusImages(fitItems); err != nil {
		return nil, err
	}

	prepared := &base.PreparedRequest{}
	total := 0
	for source, segment := range segments {
		if segment.Data == nil {
			prepared.Tokens = append(prepared.Tokens, segment.Tokens...)
			continue
		}
		state := states[source]
		var sequence string
		var tokenID int32
		var mediaData []float32
		switch state.kind {
		case llm.MediaKindImage:
			state.expected = state.image.gridWidth * state.image.gridHeight
			rows := make([]string, state.image.gridHeight)
			row := strings.Repeat(apertusImageToken, state.image.gridWidth)
			for i := range rows {
				rows[i] = row
			}
			sequence = apertusBOI + fmt.Sprintf("%d*%d", state.image.gridHeight, state.image.gridWidth) + apertusImageWrapper + strings.Join(rows, apertusImageEOL) + apertusEOI
			tokenID = m.ImageTokenID
			pixels, err := materializeApertusImage(context.Background(), state.image)
			if err != nil {
				return nil, fmt.Errorf("materialize Apertus 1.5 image: %w", err)
			}
			mediaData = pixels
			state.image.data = nil
		case llm.MediaKindAudio:
			state.expected = state.audio.codes
			sequence = apertusBOA + strings.Repeat(apertusAudioToken, state.expected) + apertusEOA
			tokenID = m.AudioTokenID
			mediaData = state.audio.samples
			state.audio.samples = nil
		}
		var err error
		total, err = shared.AddTokenBudget(total, state.expected)
		if err != nil {
			return nil, fmt.Errorf("Apertus 1.5 media budget: %w", err)
		}
		expansion := m.tok.Encode(sequence, false)
		spans := placeholderSpans(expansion, tokenID)
		if state.kind == llm.MediaKindImage && len(spans) != state.image.gridHeight {
			return nil, fmt.Errorf("Apertus 1.5 image placeholder rows = %d, want %d", len(spans), state.image.gridHeight)
		}
		if state.kind == llm.MediaKindAudio && len(spans) != 1 {
			return nil, fmt.Errorf("Apertus 1.5 audio placeholder spans = %d, want 1", len(spans))
		}
		if len(spans) == 0 {
			return nil, errors.New("Apertus 1.5 media expansion produced no placeholders")
		}
		first := spans[0].Start
		covered := 0
		for _, span := range spans {
			covered += span.End - span.Start
			state.spans = append(state.spans, apertusSpan{Start: span.Start - first, End: span.End - first})
		}
		if covered != state.expected {
			return nil, fmt.Errorf("Apertus 1.5 %s placeholders cover %d tokens, want %d", state.kind, covered, state.expected)
		}
		offset := len(prepared.Tokens)
		prepared.Tokens = append(prepared.Tokens, expansion...)
		prepared.Items = append(prepared.Items, base.PreparedItem{
			Range:     [2]int{offset + first, offset + spans[len(spans)-1].End},
			Source:    source,
			MediaData: mediaData,
			Dims:      []int{len(mediaData)},
			Opaque:    state,
			Causal:    true,
			Serial:    true,
		})
	}
	return prepared, nil
}

func (m *Model) EncodeMedia(item *base.PreparedItem, data *mlx.Array) *mlx.Array {
	state := item.Opaque.(*apertusPreparedMedia)
	var codes *mlx.Array
	var err error
	var offset int32
	switch state.kind {
	case llm.MediaKindImage:
		codes, err = m.Vision.encodeData(context.Background(), data, state.image.width, state.image.height)
		offset = m.ImageTokenOffset
	case llm.MediaKindAudio:
		codes, err = m.Audio.encodeData(context.Background(), data)
		offset = m.AudioTokenOffset
	}
	if err != nil {
		panic(fmt.Sprintf("encode Apertus 1.5 %s: %v", state.kind, err))
	}
	if codes.NumDims() != 2 || codes.Dim(0) != 1 || codes.Dim(1) != state.expected {
		panic(fmt.Sprintf("Apertus 1.5 %s encoded shape %v, want [1,%d]", state.kind, codes.Dims(), state.expected))
	}
	ids := mlx.Add(codes, mlx.NewScalarArray(float32(offset)).AsType(mlx.DTypeInt32))
	features := m.EmbedTokens.Forward(ids)
	mlx.Eval(features)
	mlx.Pin(features)
	mlx.Sweep()
	mlx.Unpin(features)
	return features
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

func placeholderSpans(tokens []int32, id int32) []apertusSpan {
	var spans []apertusSpan
	for i := 0; i < len(tokens); {
		if tokens[i] != id {
			i++
			continue
		}
		start := i
		for i < len(tokens) && tokens[i] == id {
			i++
		}
		spans = append(spans, apertusSpan{Start: start, End: i})
	}
	return spans
}
