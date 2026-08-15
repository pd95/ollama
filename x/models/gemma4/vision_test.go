package gemma4

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func testVisionConfig(t *testing.T, outputLength int32) *VisionConfig {
	t.Helper()
	cfg, err := parseVisionConfig([]byte(`{"vision_config":{"default_output_length":1}}`))
	if err != nil {
		t.Fatal(err)
	}
	cfg.DefaultOutputLength = outputLength
	if err := validateVisionConfig(cfg); err != nil {
		t.Fatal(err)
	}
	return cfg
}

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

func TestParseVisionConfigRejectsUnsafeDimensions(t *testing.T) {
	tests := []struct {
		name string
		json string
		want string
	}{
		{"negative hidden", `{"vision_config":{"hidden_size":-1}}`, "hidden_size"},
		{"too many layers", `{"vision_config":{"num_hidden_layers":513}}`, "num_hidden_layers"},
		{"attention width mismatch", `{"vision_config":{"head_dim":65}}`, "attention width"},
		{"invalid kv heads", `{"vision_config":{"num_key_value_heads":7}}`, "attention heads"},
		{"unbounded soft tokens", `{"vision_config":{"default_output_length":16385}}`, "default_output_length"},
		{"unbounded pooled patch", `{"vision_config":{"patch_size":16384,"pooling_kernel_size":2}}`, "resize budget"},
		{"undersized position table", `{"vision_config":{"position_embedding_size":100}}`, "position table"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseVisionConfig([]byte(tt.json))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("parseVisionConfig() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestVisionStandardizationTensorRequirements(t *testing.T) {
	cfg := testVisionConfig(t, 1)
	if err := validateVisionStandardizationTensors(cfg, false, false); err != nil {
		t.Fatalf("non-standardized config error = %v", err)
	}
	cfg.Standardize = true
	for _, tt := range []struct {
		name              string
		hasBias, hasScale bool
		wantErr           bool
	}{
		{"complete", true, true, false},
		{"missing bias", false, true, true},
		{"missing scale", true, false, true},
		{"missing pair", false, false, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := validateVisionStandardizationTensors(cfg, tt.hasBias, tt.hasScale)
			if (err != nil) != tt.wantErr {
				t.Fatalf("validateVisionStandardizationTensors() error = %v, wantErr %t", err, tt.wantErr)
			}
		})
	}
}

func TestGemma4ImageBounds(t *testing.T) {
	if err := validateGemma4ImageDataSize(maxGemma4ImageBytes); err != nil {
		t.Fatalf("exact byte limit error = %v", err)
	}
	if err := validateGemma4ImageDataSize(maxGemma4ImageBytes + 1); err == nil {
		t.Fatal("over byte limit error = nil")
	}
	if err := validateGemma4ImageDimensions(maxGemma4ImageDimension, 1); err != nil {
		t.Fatalf("exact dimension limit error = %v", err)
	}
	if err := validateGemma4ImageDimensions(maxGemma4ImageDimension+1, 1); err == nil {
		t.Fatal("over dimension limit error = nil")
	}
	if err := validateGemma4ImageDimensions(8192, 8192); err != nil {
		t.Fatalf("exact pixel limit error = %v", err)
	}
	if err := validateGemma4ImageDimensions(8192, 8193); err == nil {
		t.Fatal("over pixel limit error = nil")
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

func TestGemma4ResizeDimensionsRejectsUnsafeArithmetic(t *testing.T) {
	if _, _, err := gemma4ResizeDimensions(1024, 1024, 16, maxGemma4ResizePixels/(16*16), 4); err != nil {
		t.Fatalf("exact resize work limit error = %v", err)
	}
	if _, _, err := gemma4ResizeDimensions(1, 1, 16, maxGemma4ResizePixels/(16*16)+1, 4); err == nil {
		t.Fatal("over resize work limit error = nil")
	}
	if _, _, err := gemma4ResizeDimensions(1, 1, math.MaxInt, math.MaxInt, math.MaxInt); err == nil {
		t.Fatal("overflowing resize parameters error = nil")
	}
}

func TestImageToCHWFloat32UsesBoundsAndChannelOrder(t *testing.T) {
	img := image.NewRGBA(image.Rect(10, 20, 12, 22))
	img.SetRGBA(10, 20, color.RGBA{R: 255, A: 255})
	img.SetRGBA(11, 20, color.RGBA{G: 128, A: 255})
	img.SetRGBA(10, 21, color.RGBA{B: 64, A: 255})
	img.SetRGBA(11, 21, color.RGBA{R: 32, G: 16, B: 8, A: 255})

	got, err := imageToCHWFloat32(img)
	if err != nil {
		t.Fatalf("imageToCHWFloat32() error = %v", err)
	}
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

func TestPreprocessGemma4ImageSoftTokenBudget(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 8, 4))
	src.SetRGBA(0, 0, color.RGBA{R: 255, A: 255})

	var buf bytes.Buffer
	if err := png.Encode(&buf, src); err != nil {
		t.Fatalf("png.Encode() error = %v", err)
	}

	img, err := preprocessGemma4Image(buf.Bytes(), testVisionConfig(t, 1), 1)
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

func TestPrepareMediaPreservesOrderedImageItems(t *testing.T) {
	pngData := func(c color.RGBA) []byte {
		t.Helper()
		img := image.NewRGBA(image.Rect(0, 0, 8, 4))
		img.SetRGBA(0, 0, c)
		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			t.Fatalf("png.Encode() error = %v", err)
		}
		return buf.Bytes()
	}

	m := &Model{
		TextConfig: &TextConfig{
			ImageTokenIDValue: 10,
			BOITokenIDValue:   11,
			EOITokenIDValue:   12,
			VisionSoftTokens:  1,
		},
		VisionConfig: testVisionConfig(t, 1),
		Vision:       &VisionModel{},
		EmbedVision:  &MultimodalEmbedder{},
	}
	segments := []base.Segment{
		{Tokens: []int32{1, 2}},
		{Kind: "image", Data: pngData(color.RGBA{R: 255, A: 255})},
		{Tokens: []int32{3}},
		{Kind: "image", Data: pngData(color.RGBA{B: 255, A: 255})},
	}
	got, err := m.PrepareMedia(segments)
	if err != nil {
		t.Fatalf("PrepareMedia() error = %v", err)
	}
	wantTokens := []int32{1, 2, 11, 10, 12, 3, 11, 10, 12}
	if !slices.Equal(got.Tokens, wantTokens) {
		t.Fatalf("PrepareMedia().Tokens = %v, want %v", got.Tokens, wantTokens)
	}
	if len(got.Items) != 2 {
		t.Fatalf("len(PrepareMedia().Items) = %d, want 2", len(got.Items))
	}
	for i, want := range []struct {
		range_ [2]int
		source int
	}{{[2]int{2, 5}, 1}, {[2]int{6, 9}, 3}} {
		item := got.Items[i]
		if item.Range != want.range_ || item.Source != want.source {
			t.Fatalf("item %d range/source = %v/%d, want %v/%d", i, item.Range, item.Source, want.range_, want.source)
		}
		if item.Causal {
			t.Fatalf("item %d is causal, want whole bidirectional image expansion", i)
		}
		if !slices.Equal(item.Dims, []int{1, 3, 48, 48}) || len(item.MediaData) != 3*48*48 {
			t.Fatalf("item %d media shape/data = %v/%d, want [1 3 48 48]/%d", i, item.Dims, len(item.MediaData), 3*48*48)
		}
		payload := item.Opaque.(gemma4MediaPayload)
		if payload.ImageStart != 1 || payload.ImageEnd != 2 || payload.Image.Pixels != nil {
			t.Fatalf("item %d payload = start %d end %d pixels %d", i, payload.ImageStart, payload.ImageEnd, len(payload.Image.Pixels))
		}
		media := batch.MediaItem{Pos: item.Range[0], Opaque: item.Opaque}
		start, end := gemma4ImageRun(media)
		if start != item.Range[0]+1 || end != item.Range[1]-1 {
			t.Fatalf("item %d scatter/PLE run = [%d,%d), want current image span [%d,%d)", i, start, end, item.Range[0]+1, item.Range[1]-1)
		}
	}
}

func TestPrepareMediaSequentialRequestsAreIsolated(t *testing.T) {
	pngData := func(c color.RGBA) []byte {
		t.Helper()
		img := image.NewRGBA(image.Rect(0, 0, 8, 4))
		img.SetRGBA(0, 0, c)
		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}
	m := &Model{
		TextConfig:   &TextConfig{ImageTokenIDValue: 10, BOITokenIDValue: 11, EOITokenIDValue: 12, VisionSoftTokens: 1},
		VisionConfig: testVisionConfig(t, 1),
		Vision:       &VisionModel{},
		EmbedVision:  &MultimodalEmbedder{},
	}
	first, err := m.PrepareMedia([]base.Segment{{Tokens: []int32{1}}, {Kind: "image", Data: pngData(color.RGBA{R: 255, A: 255})}})
	if err != nil {
		t.Fatal(err)
	}
	second, err := m.PrepareMedia([]base.Segment{{Tokens: []int32{2, 3}}, {Kind: "image", Data: pngData(color.RGBA{B: 255, A: 255})}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(first.Tokens, []int32{1, 11, 10, 12}) || !slices.Equal(second.Tokens, []int32{2, 3, 11, 10, 12}) {
		t.Fatalf("sequential tokens = %v / %v", first.Tokens, second.Tokens)
	}
	if len(first.Items) != 1 || len(second.Items) != 1 || &first.Items[0].MediaData[0] == &second.Items[0].MediaData[0] {
		t.Fatal("sequential prepares reused request/item storage")
	}
	firstTokens := slices.Clone(first.Tokens)
	firstPixels := slices.Clone(first.Items[0].MediaData)
	second.Tokens[0] = 99
	second.Items[0].MediaData[0] = 0.5
	if !slices.Equal(first.Tokens, firstTokens) || !slices.Equal(first.Items[0].MediaData, firstPixels) {
		t.Fatal("mutating the later prepared request changed the earlier request")
	}
}
