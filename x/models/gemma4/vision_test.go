package gemma4

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"image"
	"image/color"
	"image/png"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/tokenizer"
)

func TestParseVisionConfigDefaults(t *testing.T) {
	cfg, err := parseVisionConfig([]byte(`{"vision_config":{}}`))
	if err != nil {
		t.Fatalf("parseVisionConfig() error = %v", err)
	}
	if cfg == nil {
		t.Fatal("parseVisionConfig() = nil, want config")
	}
	if cfg.HiddenSize != 768 || cfg.IntermediateSize != 3072 || cfg.NumHiddenLayers != 16 {
		t.Fatalf("unexpected default dimensions: hidden=%d intermediate=%d layers=%d", cfg.HiddenSize, cfg.IntermediateSize, cfg.NumHiddenLayers)
	}
	if cfg.PatchSize != 16 || cfg.PoolingKernelSize != 3 || cfg.DefaultOutputLength != 280 {
		t.Fatalf("unexpected image defaults: patch=%d pooling=%d output=%d", cfg.PatchSize, cfg.PoolingKernelSize, cfg.DefaultOutputLength)
	}
	if cfg.RopeParameters.RopeTheta != 100 {
		t.Fatalf("vision rope theta = %v, want 100", cfg.RopeParameters.RopeTheta)
	}
}

func TestParseUnifiedVisionConfig(t *testing.T) {
	cfg, err := parseVisionConfig([]byte(`{"vision_config":{"model_type":"gemma4_unified_vision","mm_embed_dim":3840,"mm_posemb_size":1120,"model_patch_size":48,"num_soft_tokens":280,"output_proj_dims":3840,"patch_size":16,"pooling_kernel_size":3}}`))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.unified() || cfg.ModelPatchSize != 48 || cfg.MMEmbedDim != 3840 || cfg.MMPosembSize != 1120 {
		t.Fatalf("unexpected unified config: %+v", cfg)
	}
	if cfg.DefaultOutputLength != 280 {
		t.Fatalf("DefaultOutputLength = %d, want 280", cfg.DefaultOutputLength)
	}
}

func TestParseUnifiedVisionConfigRejectsInvalidPatchSize(t *testing.T) {
	_, err := parseVisionConfig([]byte(`{"vision_config":{"model_type":"gemma4_unified_vision","patch_size":16,"pooling_kernel_size":3,"model_patch_size":32}}`))
	if err == nil || !strings.Contains(err.Error(), "model_patch_size") {
		t.Fatalf("parseVisionConfig() error = %v, want model patch mismatch", err)
	}
}

func TestPreprocessGemma4ImageRejectsLimitsAndCancellation(t *testing.T) {
	cfg := &VisionConfig{PatchSize: 16, PoolingKernelSize: 3, DefaultOutputLength: 1}
	if _, err := preprocessGemma4Image(context.Background(), make([]byte, maxGemma4ImageBytes+1), cfg, 1); err == nil || !strings.Contains(err.Error(), "bytes") {
		t.Fatalf("oversized image error = %v", err)
	}
	if err := validateGemma4ImageDimensions(maxGemma4ImageDimension+1, 1); err == nil {
		t.Fatal("oversized dimension error = nil")
	}
	if err := validateGemma4ImageDimensions(8192, 8193); err == nil {
		t.Fatal("oversized pixel count error = nil")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := preprocessGemma4Image(ctx, []byte("image"), cfg, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled preprocessing error = %v", err)
	}
}

func TestDecodeGemma4WebP(t *testing.T) {
	data, err := base64.StdEncoding.DecodeString("UklGRrIBAABXRUJQVlA4TKUBAAAvSsAYAA8w//M///MfeJAkbXvaSG7m8Q3GfYSBJekwQztm/IcZlgwnmWImn2BK7aFmBtnVir6q//8VOkFE/xm4baTIu8c48ArEo6+B3zFKYln3pqClSCKX0begFTAXFOLXHSyF8cCNcZEG4OywuA4KVVfJCiArU7GAgJI8+lJP/OKMT/fBAjevg1cYB7YVkFuWga2lyPi5I0HFy5YTpWIHg0RZpkniRVW9odHAKOwosWuOGdxIyn2OvaCDvhg/we6TwadPBPbqBV58MsLmMJ8yZnOWk8SRz4N+QoyPL+MnamzMvcE1rHNEr91F9GKZPVUcS9w7PhhH36suB9qPeYb/oLk6cuTiJ0wOK3m5h1cKjW6EVZCYMK7dxcKCBdgP9HkKr9gkAO2P8GKZGWVdIAatQa+1IDpt6qyorVwdy01xdW8Jkfk6xjEXmVQQ+HQdFr6OKhIN34dXWq0+0qr6EJSCeeVLH9+gvGTLyqM65PQ44ihzlTXxQKjKbAvshXgir7Lil9w4L2bvMycmjQcqXaMCO6BlY28i+FOLzbfI1vEqxAhotocAAA==")
	if err != nil {
		t.Fatal(err)
	}
	cfg, format, err := decodeGemma4ImageConfig(context.Background(), data)
	if err != nil {
		t.Fatalf("decodeGemma4ImageConfig() error = %v", err)
	}
	if format != "webp" || cfg.Width <= 0 || cfg.Height <= 0 {
		t.Fatalf("WebP config = %dx%d %q", cfg.Width, cfg.Height, format)
	}
	img, format, err := decodeGemma4Image(context.Background(), data)
	if err != nil {
		t.Fatalf("decodeGemma4Image() error = %v", err)
	}
	if format != "webp" || img.Bounds().Dx() != cfg.Width || img.Bounds().Dy() != cfg.Height {
		t.Fatalf("WebP image = %v %q", img.Bounds(), format)
	}
}

func TestGemma4ResizeDimensions(t *testing.T) {
	gotW, gotH, err := gemma4ResizeDimensions(1024, 768, 16, 280*9, 3)
	if err != nil {
		t.Fatalf("gemma4ResizeDimensions() error = %v", err)
	}
	if gotW != 912 || gotH != 672 {
		t.Fatalf("gemma4ResizeDimensions() = %dx%d, want 912x672", gotW, gotH)
	}
	if gotW%48 != 0 || gotH%48 != 0 {
		t.Fatalf("resize dimensions must be multiples of pooled patch size, got %dx%d", gotW, gotH)
	}
}

func TestGemma4ResizeDimensionsRejectsUnboundedWork(t *testing.T) {
	_, _, err := gemma4ResizeDimensions(1, 1, 16, maxGemma4ResizePixels, 3)
	if err == nil || !strings.Contains(err.Error(), "resize target") {
		t.Fatalf("gemma4ResizeDimensions() error = %v, want resize work limit", err)
	}
}

func TestImageToCHWFloat32UsesBoundsAndChannelOrder(t *testing.T) {
	img := image.NewRGBA(image.Rect(10, 20, 12, 22))
	img.SetRGBA(10, 20, color.RGBA{R: 255, A: 255})
	img.SetRGBA(11, 20, color.RGBA{G: 128, A: 255})
	img.SetRGBA(10, 21, color.RGBA{B: 64, A: 255})
	img.SetRGBA(11, 21, color.RGBA{R: 32, G: 16, B: 8, A: 255})

	got := imageToCHWFloat32(img)
	want := []float32{
		1, 0, 0, 32.0 / 255.0,
		0, 128.0 / 255.0, 0, 16.0 / 255.0,
		0, 0, 64.0 / 255.0, 8.0 / 255.0,
	}
	if len(got) != len(want) {
		t.Fatalf("len(imageToCHWFloat32()) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("pixel[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestImageToUnifiedPatchesUsesHWCModelPatchOrder(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 2))
	for y := range 2 {
		for x := range 4 {
			img.SetRGBA(x, y, color.RGBA{R: uint8(1 + x + 4*y), G: uint8(11 + x + 4*y), B: uint8(21 + x + 4*y), A: 255})
		}
	}
	patches, positions, err := imageToUnifiedPatchesContext(context.Background(), img, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := positions, []int32{0, 0, 1, 0}; !slices.Equal(got, want) {
		t.Fatalf("positions = %v, want %v", got, want)
	}
	wantBytes := []uint8{1, 11, 21, 2, 12, 22, 5, 15, 25, 6, 16, 26, 3, 13, 23, 4, 14, 24, 7, 17, 27, 8, 18, 28}
	if len(patches) != len(wantBytes) {
		t.Fatalf("patch length = %d, want %d", len(patches), len(wantBytes))
	}
	for i, want := range wantBytes {
		if math.Abs(float64(patches[i]-float32(want)/255)) > 1e-6 {
			t.Fatalf("patch[%d] = %f, want %f", i, patches[i], float32(want)/255)
		}
	}
}

func TestImageTokenSpan(t *testing.T) {
	start, end, err := imageTokenSpan([]int32{1, 258880, 258880, 2}, 258880)
	if err != nil {
		t.Fatalf("imageTokenSpan() error = %v", err)
	}
	if start != 1 || end != 3 {
		t.Fatalf("imageTokenSpan() = %d, %d; want 1, 3", start, end)
	}

	if _, _, err := imageTokenSpan([]int32{258880, 2, 258880}, 258880); err == nil {
		t.Fatal("imageTokenSpan() error = nil, want split-token error")
	}
	if _, _, err := imageTokenSpan([]int32{1, 2, 3}, 258880); err == nil {
		t.Fatal("imageTokenSpan() error = nil, want missing-token error")
	}
}

func TestPreprocessGemma4ImageSoftTokenBudget(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 8, 4))
	src.SetRGBA(0, 0, color.RGBA{R: 255, A: 255})

	var buf bytes.Buffer
	if err := png.Encode(&buf, src); err != nil {
		t.Fatalf("png.Encode() error = %v", err)
	}

	img, err := preprocessGemma4Image(context.Background(), buf.Bytes(), &VisionConfig{PatchSize: 16, PoolingKernelSize: 3, DefaultOutputLength: 1}, 1)
	if err != nil {
		t.Fatalf("preprocessGemma4Image() error = %v", err)
	}
	if img.Width != 48 || img.Height != 48 {
		t.Fatalf("preprocessed dimensions = %dx%d, want 48x48", img.Width, img.Height)
	}
	if img.PatchWidth != 3 || img.PatchHeight != 3 || img.SoftTokens != 1 {
		t.Fatalf("patch layout = %dx%d soft=%d, want 3x3 soft=1", img.PatchWidth, img.PatchHeight, img.SoftTokens)
	}
	if len(img.Pixels) != 3*48*48 {
		t.Fatalf("pixel count = %d, want %d", len(img.Pixels), 3*48*48)
	}
}

func TestBindGemma4MediaPreservesMarkerIdentity(t *testing.T) {
	media := []llm.MediaData{
		{ID: 0, Kind: llm.MediaKindImage, Data: []byte("image")},
		{ID: 1, Kind: llm.MediaKindAudio, Data: []byte("audio")},
	}
	bindings, err := bindGemma4Media("before [img-1] middle [img-0] after", media)
	if err != nil {
		t.Fatal(err)
	}
	if len(bindings) != 2 || bindings[0].Media.ID != 1 || bindings[1].Media.ID != 0 {
		t.Fatalf("binding order = %#v, want media IDs 1, 0", bindings)
	}
}

func TestBindGemma4MediaRejectsInvalidMappings(t *testing.T) {
	image := llm.MediaData{ID: 0, Kind: llm.MediaKindImage, Data: []byte("image")}
	tests := []struct {
		name   string
		prompt string
		media  []llm.MediaData
		want   string
	}{
		{name: "no media", prompt: "text", want: "no media"},
		{name: "unnumbered", prompt: "[img]", media: []llm.MediaData{image}, want: "unnumbered"},
		{name: "missing", prompt: "text", media: []llm.MediaData{image}, want: "expected one"},
		{name: "unknown", prompt: "[img-1]", media: []llm.MediaData{image}, want: "no matching media"},
		{name: "duplicate marker", prompt: "[img-0][img-0]", media: []llm.MediaData{image}, want: "multiple"},
		{name: "duplicate id", prompt: "[img-0]", media: []llm.MediaData{image, image}, want: "duplicate"},
		{name: "noncanonical", prompt: "[img-00]", media: []llm.MediaData{image}, want: "invalid"},
		{name: "malformed", prompt: "[img-zero]", media: []llm.MediaData{image}, want: "invalid"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := bindGemma4Media(tt.prompt, tt.media)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("bindGemma4Media() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestAssignGemma4MixedMediaSpans(t *testing.T) {
	bindings := []gemma4MediaBinding{
		{Item: gemma4MediaItem{ID: 7, Kind: llm.MediaKindAudio, Audio: &gemma4AudioInput{SoftTokens: 2}}},
		{Item: gemma4MediaItem{ID: 3, Kind: llm.MediaKindImage, Image: &gemma4ImageInput{SoftTokens: 3}}},
		{Item: gemma4MediaItem{ID: 9, Kind: llm.MediaKindAudio, Audio: &gemma4AudioInput{SoftTokens: 1}}},
	}
	tokens := []int32{8, 4, 4, 8, 3, 3, 3, 8, 4, 8}
	items, err := assignGemma4MediaSpans(tokens, bindings, 3, 4)
	if err != nil {
		t.Fatal(err)
	}
	want := []gemma4Span{{Start: 1, End: 3}, {Start: 4, End: 7}, {Start: 8, End: 9}}
	for i := range want {
		if items[i].Span != want[i] {
			t.Fatalf("item %d span = %+v, want %+v", i, items[i].Span, want[i])
		}
	}

	tokens = append(tokens[:9], 4, 8)
	if _, err := assignGemma4MediaSpans(tokens, bindings, 3, 4); err == nil || !strings.Contains(err.Error(), "token count") {
		t.Fatalf("mismatched run error = %v", err)
	}
}

func TestPrepareMixedGemma4MediaPrompt(t *testing.T) {
	tok := testGemma4MediaTokenizer(t)
	audioCfg := defaultAudioProcessorConfig()
	frames := make([][]float64, 16000)
	for i := range frames {
		frames[i] = []float64{0.25}
	}
	m := &Model{
		TextConfig: &TextConfig{
			ImageTokenIDValue:     1,
			AudioTokenIDValue:     4,
			VisionSoftTokens:      1,
			MaxPositionEmbeddings: 2048,
		},
		VisionConfig: &VisionConfig{
			ModelType:           "gemma4_unified_vision",
			PatchSize:           16,
			PoolingKernelSize:   3,
			ModelPatchSize:      48,
			DefaultOutputLength: 1,
		},
		AudioConfig:          &AudioConfig{},
		AudioProcessorConfig: &audioCfg,
		tok:                  tok,
		mediaTokens:          defaultGemma4MediaTokens(),
	}
	media := []llm.MediaData{
		{ID: 0, Kind: llm.MediaKindImage, Data: makeTestGemma4PNG(t, 8, 4)},
		{ID: 1, Kind: llm.MediaKindAudio, Data: makeTestWAV(t, 1, 16, 16000, frames)},
	}
	prepared, err := m.prepareLegacyMediaPrompt(context.Background(), "<sep>[img-1]<sep>[img-0]<sep>", media)
	if err != nil {
		t.Fatal(err)
	}
	payload := prepared.Payload.(*gemma4MediaPayload)
	if len(payload.Items) != 2 || payload.Items[0].ID != 1 || payload.Items[1].ID != 0 {
		t.Fatalf("payload order = %#v, want audio 1 then image 0", payload.Items)
	}
	for _, item := range payload.Items {
		for i := item.Span.Start; i < item.Span.End; i++ {
			if prepared.PLEInputIDs[i] != 0 {
				t.Fatalf("PLEInputIDs[%d] = %d, want 0", i, prepared.PLEInputIDs[i])
			}
		}
	}
	if len(prepared.BidirectionalSpans) != 1 || prepared.BidirectionalSpans[0] != payload.Items[1].Span {
		t.Fatalf("bidirectional spans = %+v, want image span %+v", prepared.BidirectionalSpans, payload.Items[1].Span)
	}
}

func TestMergeGemma4MediaEmbeddings(t *testing.T) {
	skipIfNoMLX(t)
	embeddings := mlx.FromValues([]float32{
		0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
	}, 1, 8, 2)
	replacements := []gemma4MediaReplacement{
		{Span: gemma4Span{Start: 1, End: 3}, Features: mlx.FromValues([]float32{10, 11, 12, 13}, 1, 2, 2)},
		{Span: gemma4Span{Start: 5, End: 6}, Features: mlx.FromValues([]float32{20, 21}, 1, 1, 2)},
	}
	got, err := mergeGemma4MediaEmbeddings(embeddings, 8, replacements)
	if err != nil {
		t.Fatal(err)
	}
	mlx.Eval(got)
	want := []float32{0, 0, 10, 11, 12, 13, 3, 3, 4, 4, 20, 21, 6, 6, 7, 7}
	if !slices.Equal(got.Floats(), want) {
		t.Fatalf("merged embeddings = %v, want %v", got.Floats(), want)
	}
	if _, err := mergeGemma4MediaEmbeddings(embeddings, 8, []gemma4MediaReplacement{
		{Span: gemma4Span{Start: 2, End: 4}, Features: replacements[0].Features},
		{Span: gemma4Span{Start: 3, End: 4}, Features: replacements[1].Features},
	}); err == nil || !strings.Contains(err.Error(), "overlapping") {
		t.Fatalf("overlap error = %v", err)
	}
}

func TestPrepareMediaEmbeddingsHonorsCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := (&Model{}).prepareLegacyMediaEmbeddings(ctx, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("PrepareMediaEmbeddings() error = %v, want context.Canceled", err)
	}
}

func TestPrepareMediaEmbeddingsRejectsMalformedPayloads(t *testing.T) {
	validImage := func(id, start, end int) gemma4MediaItem {
		return gemma4MediaItem{
			ID: id, Kind: llm.MediaKindImage,
			Image: &gemma4ImageInput{SoftTokens: end - start},
			Span:  gemma4Span{Start: start, End: end},
		}
	}
	validAudio := func(id, start, end int) gemma4MediaItem {
		return gemma4MediaItem{
			ID: id, Kind: llm.MediaKindAudio,
			Audio: &gemma4AudioInput{
				Features: make([]float32, end-start), FeatureMask: make([]bool, end-start),
				FeatureSize: 1, Frames: end - start, SoftTokens: end - start,
			},
			Span: gemma4Span{Start: start, End: end},
		}
	}
	tests := []struct {
		name     string
		prepared *legacyPreparedInput
		want     string
	}{
		{name: "nil prepared", want: "nil"},
		{name: "empty tokens", prepared: &legacyPreparedInput{Payload: &gemma4MediaPayload{Items: []gemma4MediaItem{validImage(0, 0, 1)}}}, want: "empty"},
		{name: "nil payload", prepared: &legacyPreparedInput{Tokens: []int32{1}}, want: "payload"},
		{name: "out of bounds", prepared: &legacyPreparedInput{Tokens: []int32{1}, Payload: &gemma4MediaPayload{Items: []gemma4MediaItem{validImage(0, 0, 2)}}}, want: "span"},
		{name: "overlap", prepared: &legacyPreparedInput{Tokens: []int32{1, 2, 3}, Payload: &gemma4MediaPayload{Items: []gemma4MediaItem{validImage(0, 0, 2), validAudio(1, 1, 2)}}}, want: "overlapping"},
		{name: "duplicate id", prepared: &legacyPreparedInput{Tokens: []int32{1, 2}, Payload: &gemma4MediaPayload{Items: []gemma4MediaItem{validImage(0, 0, 1), validAudio(0, 1, 2)}}}, want: "duplicate"},
		{name: "audio backing", prepared: &legacyPreparedInput{Tokens: []int32{1}, Payload: &gemma4MediaPayload{Items: []gemma4MediaItem{{
			ID: 0, Kind: llm.MediaKindAudio,
			Audio: &gemma4AudioInput{Features: []float32{1}, FeatureMask: []bool{true}, FeatureSize: 2, Frames: 1, SoftTokens: 1},
			Span:  gemma4Span{Start: 0, End: 1},
		}}}}, want: "processor dimensions"},
		{name: "unknown kind", prepared: &legacyPreparedInput{Tokens: []int32{1}, Payload: &gemma4MediaPayload{Items: []gemma4MediaItem{{ID: 0, Span: gemma4Span{Start: 0, End: 1}}}}}, want: "media kind"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := (&Model{}).prepareLegacyMediaEmbeddings(context.Background(), tt.prepared)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("PrepareMediaEmbeddings() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func makeTestGemma4PNG(t *testing.T, width, height int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			img.SetRGBA(x, y, color.RGBA{R: uint8(1 + x%254), G: uint8(1 + y%254), B: 64, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func testGemma4MediaTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	data := []byte(`{
		"model":{"type":"BPE","vocab":{"<|image>":0,"<|image|>":1,"<image|>":2,"<|audio>":3,"<|audio|>":4,"<audio|>":5,"<sep>":6},"merges":[]},
		"added_tokens":[
			{"id":0,"content":"<|image>","special":true},
			{"id":1,"content":"<|image|>","special":true},
			{"id":2,"content":"<image|>","special":true},
			{"id":3,"content":"<|audio>","special":true},
			{"id":4,"content":"<|audio|>","special":true},
			{"id":5,"content":"<audio|>","special":true},
			{"id":6,"content":"<sep>","special":true}
		]
	}`)
	tok, err := tokenizer.LoadFromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}
