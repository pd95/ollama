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
	"github.com/ollama/ollama/x/mlxrunner/model/base"
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

func TestPrepareMixedGemma4MediaUsesUpstreamSegments(t *testing.T) {
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
		UnifiedVision:        &UnifiedVisionEmbedder{PatchDim: 3 * 48 * 48},
		EmbedVision:          &MultimodalEmbedder{},
		AudioConfig:          &AudioConfig{ModelType: "gemma4_audio"},
		AudioProcessorConfig: &audioCfg,
		Audio:                &AudioModel{},
		EmbedAudio:           &MultimodalEmbedder{},
		tok:                  tok,
		mediaTokens:          defaultGemma4MediaTokens(),
	}
	prepared, err := m.PrepareMedia([]base.Segment{
		{Tokens: []int32{9}},
		{Kind: string(llm.MediaKindAudio), Data: makeTestWAV(t, 1, 16, 16000, frames)},
		{Tokens: []int32{8}},
		{Kind: string(llm.MediaKindImage), Data: makeTestGemma4PNG(t, 8, 4)},
		{Tokens: []int32{7}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared.Items) != 2 {
		t.Fatalf("items = %d, want 2", len(prepared.Items))
	}
	audio, image := prepared.Items[0], prepared.Items[1]
	if audio.Source != 1 || image.Source != 3 || !audio.Causal || image.Causal {
		t.Fatalf("items = %+v, want ordered causal audio and bidirectional image", prepared.Items)
	}
	audioState, audioOK := audio.Opaque.(*gemma4PreparedMedia)
	imageState, imageOK := image.Opaque.(*gemma4PreparedMedia)
	if !audioOK || !imageOK || audioState.Kind != llm.MediaKindAudio || imageState.Kind != llm.MediaKindImage {
		t.Fatalf("states = %#v, %#v", audio.Opaque, image.Opaque)
	}
	if audio.Range[1] >= image.Range[0] {
		t.Fatalf("media ranges are not in stream order: %v, %v", audio.Range, image.Range)
	}
	if got := len(image.MediaData); got != imageState.SoftTokens*int(m.UnifiedVision.PatchDim) {
		t.Fatalf("image data = %d, want %d", got, imageState.SoftTokens*int(m.UnifiedVision.PatchDim))
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
