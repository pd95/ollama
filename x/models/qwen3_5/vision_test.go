package qwen3_5

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/png"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

type qwenPanickingImage struct{ payload any }

func (qwenPanickingImage) ColorModel() color.Model   { return color.NRGBAModel }
func (qwenPanickingImage) Bounds() image.Rectangle   { return image.Rect(0, 0, 4, 4) }
func (p qwenPanickingImage) At(int, int) color.Color { panic(p.payload) }

type qwenTestContext struct{}

func (qwenTestContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (qwenTestContext) Done() <-chan struct{}       { return nil }
func (qwenTestContext) Value(any) any               { return nil }

type qwenCountingContext struct {
	qwenTestContext
	calls atomic.Int32
}

func (c *qwenCountingContext) Err() error {
	c.calls.Add(1)
	return nil
}

type qwenCheckpointContext struct {
	qwenTestContext
	blockAt  int32
	calls    atomic.Int32
	canceled atomic.Bool
	reached  chan struct{}
	release  chan struct{}
	once     sync.Once
}

func newQwenCheckpointContext(blockAt int32) *qwenCheckpointContext {
	return &qwenCheckpointContext{
		blockAt: blockAt,
		reached: make(chan struct{}),
		release: make(chan struct{}),
	}
}

func (c *qwenCheckpointContext) Err() error {
	if c.calls.Add(1) == c.blockAt {
		c.once.Do(func() { close(c.reached) })
		<-c.release
	}
	if c.canceled.Load() {
		return context.Canceled
	}
	return nil
}

func (c *qwenCheckpointContext) cancel() {
	c.canceled.Store(true)
	close(c.release)
}

func testQwenVisionModel() *Model {
	cfg := &VisionConfig{PatchSize: 16, SpatialMergeSize: 2, TemporalPatchSize: 2, InChannels: 3, NumPositionEmbeddings: 2304}
	return &Model{
		Config:      &Config{},
		VisionTower: &VisionTower{},
		Vision:      cfg,
		MM: multimodalConfig{
			ImageTokenID:       9,
			VisionStartTokenID: 7,
			VisionEndTokenID:   8,
			VisionConfig:       cfg,
		},
	}
}

func TestVisionAdapterWeightsAreCollectable(t *testing.T) {
	mlxtest.Setup(t)
	weight := mlx.FromValue(float32(1))
	adapter := &VisionAdapter{Model: &Model{VisionTower: &VisionTower{PosEmbed: weight}}}
	if got := mlx.Collect(adapter); len(got) != 1 || got[0] != weight {
		t.Fatalf("mlx.Collect(adapter) = %v, want the tower weight", got)
	}
}

func TestSmartResize(t *testing.T) {
	// Goldens from the reference smart_resize at the family's processor
	// bounds (factor 32, pixels in [65536, 16777216]).
	cases := []struct{ h, w, wantH, wantW int32 }{
		{480, 640, 480, 640},
		{1080, 1920, 1088, 1920},
		{3024, 4032, 3008, 4032},
		{100, 100, 256, 256},
		{37, 5000, 32, 4992},
	}
	for _, c := range cases {
		h, w, err := smartResize(c.h, c.w, 32)
		if err != nil {
			t.Fatalf("smartResize(%d, %d): %v", c.h, c.w, err)
		}
		if h != c.wantH || w != c.wantW {
			t.Errorf("smartResize(%d, %d) = (%d, %d), want (%d, %d)", c.h, c.w, h, w, c.wantH, c.wantW)
		}
	}
	if _, _, err := smartResize(10, 5000, 32); err == nil {
		t.Error("aspect ratio over 200 should error")
	}
}

func TestParseMultimodalConfig(t *testing.T) {
	mm, err := parseMultimodalConfig([]byte(`{
		"image_token_id": 248056, "vision_start_token_id": 248053, "vision_end_token_id": 248054,
		"vision_config": {"depth": 27, "hidden_size": 1152, "deepstack_visual_indexes": []}}`))
	if err != nil {
		t.Fatal(err)
	}
	v := mm.VisionConfig
	if v == nil || v.PatchSize != 16 || v.SpatialMergeSize != 2 || v.TemporalPatchSize != 2 || v.NumHeads != 16 {
		t.Fatalf("defaults not applied: %+v", v)
	}

	// Deepstack injection is not implemented; a config requesting it must
	// fail the load rather than silently skip the injections.
	_, err = parseMultimodalConfig([]byte(`{
		"vision_config": {"depth": 27, "deepstack_visual_indexes": [8, 16, 24]}}`))
	if err == nil || !strings.Contains(err.Error(), "deepstack") {
		t.Fatalf("deepstack config accepted: %v", err)
	}
}

// TestMRopePositionRule drives a real image through PrepareMedia and checks
// the reference position rule: text advances all channels one per token; an
// image grid anchors T at the current position, offsets H/W by row/column,
// and advances the position by the grid's longer side.
func TestMRopePositionRule(t *testing.T) {
	m := testQwenVisionModel()

	prepared, err := m.PrepareMedia(context.Background(), []base.Segment{{Tokens: []int32{1, 2, 3}}})
	if err != nil {
		t.Fatal(err)
	}
	if prepared.Layout != nil {
		t.Fatal("text-only stream must carry no layout")
	}

	// A 64x64 PNG resizes to the 256x256 minimum-pixel floor: a 16x16 patch
	// grid, merged 8x8.
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 64, 64))); err != nil {
		t.Fatal(err)
	}
	prepared, err = m.PrepareMedia(context.Background(), []base.Segment{
		{Tokens: []int32{1, 2, 3}},
		{Kind: "image", Data: buf.Bytes()},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Three text tokens, vision_start, 64 pads, vision_end.
	const L = 69
	if len(prepared.Tokens) != L {
		t.Fatalf("tokens = %d, want %d", len(prepared.Tokens), L)
	}
	if len(prepared.Items) != 1 || prepared.Items[0].Range != [2]int{3, L} || !prepared.Items[0].Causal {
		t.Fatalf("items = %+v", prepared.Items)
	}

	layout := prepared.Layout.(*visionLayout)
	if layout.promptLen != L {
		t.Fatalf("promptLen = %d, want %d", layout.promptLen, L)
	}
	// The image anchors at position 4 and advances the cursor by its longer
	// side: vision_end lands at 12 and decode continues from maxPos+1.
	if want := int32(12 + 1 - L); layout.delta != want {
		t.Fatalf("delta = %d, want %d", layout.delta, want)
	}
	pos := func(i int32) [3]int32 {
		return [3]int32{layout.positions[i], layout.positions[L+i], layout.positions[2*L+i]}
	}
	checks := []struct {
		i    int32
		want [3]int32
	}{
		{0, [3]int32{0, 0, 0}},     // text
		{2, [3]int32{2, 2, 2}},     // text
		{3, [3]int32{3, 3, 3}},     // vision_start
		{4, [3]int32{4, 4, 4}},     // image row 0, col 0
		{5, [3]int32{4, 4, 5}},     // image row 0, col 1
		{12, [3]int32{4, 5, 4}},    // image row 1, col 0
		{67, [3]int32{4, 11, 11}},  // image row 7, col 7
		{68, [3]int32{12, 12, 12}}, // vision_end
	}
	for _, c := range checks {
		if got := pos(c.i); got != c.want {
			t.Errorf("positions[%d] = %v, want %v", c.i, got, c.want)
		}
	}
}

func TestPrepareMediaPreCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for name, mediaModel := range map[string]base.MediaModel{
		"model":   &Model{},
		"adapter": &VisionAdapter{Model: &Model{}},
	} {
		t.Run(name, func(t *testing.T) {
			prepared, err := mediaModel.PrepareMedia(ctx, []base.Segment{{Tokens: []int32{1}}})
			if !errors.Is(err, context.Canceled) || prepared != nil {
				t.Fatalf("PrepareMedia() = (%v, %v), want (nil, context.Canceled)", prepared, err)
			}
		})
	}
}

func TestVisionPatchLookups(t *testing.T) {
	// 4x6 pre-merge grid, merge 2: token order is block-major; rope
	// positions are the pre-merge (h, w) indices.
	ropePos, posIdx, posWt, err := visionPatchLookups(context.Background(), 4, 6, 2, 2304)
	if err != nil {
		t.Fatal(err)
	}
	if len(ropePos) != 2*24 || len(posIdx) != 4*24 || len(posWt) != 4*24 {
		t.Fatalf("lengths %d %d %d", len(ropePos), len(posIdx), len(posWt))
	}
	// First merge block covers (0,0) (0,1) (1,0) (1,1) in that order.
	want := []int32{0, 0, 0, 1, 1, 0, 1, 1}
	for i, w := range want {
		if ropePos[i] != w {
			t.Fatalf("ropePos[:8] = %v, want %v", ropePos[:8], want)
		}
	}
	// Second block continues at (0,2).
	if ropePos[8] != 0 || ropePos[9] != 2 {
		t.Fatalf("second block starts at (%d, %d), want (0, 2)", ropePos[8], ropePos[9])
	}
	// Bilinear weights sum to one per patch.
	for i := range 24 {
		var sum float32
		for c := range 4 {
			sum += posWt[4*i+c]
		}
		if sum < 0.999 || sum > 1.001 {
			t.Fatalf("patch %d weights sum %f", i, sum)
		}
	}
}

func TestImageReaderCancellation(t *testing.T) {
	ctx := newQwenCheckpointContext(1)
	result := make(chan error, 1)
	go func() {
		_, err := qwenImageContextReader{check: ctx.Err, r: bytes.NewReader([]byte("image"))}.Read(make([]byte, 1))
		result <- err
	}()
	<-ctx.reached
	ctx.cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("Read() error = %v, want context.Canceled", err)
	}
}

func TestDropAlphaCancellation(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 64, 64))
	src.SetNRGBA(0, 0, color.NRGBA{R: 1, A: 1})
	ctx := newQwenCheckpointContext(1)
	type result struct {
		image image.Image
		err   error
	}
	done := make(chan result, 1)
	go func() {
		img, err := dropAlpha(ctx, src, src.Bounds())
		done <- result{image: img, err: err}
	}()
	<-ctx.reached
	ctx.cancel()
	got := <-done
	if !errors.Is(got.err, context.Canceled) || got.image != nil {
		t.Fatalf("dropAlpha() = (%v, %v), want (nil, context.Canceled)", got.image, got.err)
	}
}

func TestResizeCancellation(t *testing.T) {
	ctx := newQwenCheckpointContext(2) // entry check, then the first destination pixel
	result := make(chan error, 1)
	go func() {
		_, err := resizeQwenImage(ctx, image.NewRGBA(image.Rect(0, 0, 4, 4)), image.Rect(0, 0, 4, 4), 64, 64)
		result <- err
	}()
	<-ctx.reached
	ctx.cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("resizeQwenImage() error = %v, want context.Canceled", err)
	}
}

func TestResizeRepanicsUnrelatedPayload(t *testing.T) {
	payload := &struct{ marker byte }{marker: 1}
	recovered := func() (got any) {
		defer func() { got = recover() }()
		_, _ = resizeQwenImage(context.Background(), qwenPanickingImage{payload: payload}, image.Rect(0, 0, 4, 4), 64, 64)
		return nil
	}()
	if recovered != payload {
		t.Fatalf("recovered payload = %v, want exact unrelated payload %v", recovered, payload)
	}
}

func TestPatchifyCancellation(t *testing.T) {
	ctx := newQwenCheckpointContext(2) // allocation check, then the first merge-block row
	type result struct {
		pixels []float32
		err    error
	}
	done := make(chan result, 1)
	go func() {
		pixels, err := patchifyImage(ctx, image.NewRGBA(image.Rect(0, 0, 64, 64)), 4, 4, 16, 2, 2)
		done <- result{pixels: pixels, err: err}
	}()
	<-ctx.reached
	ctx.cancel()
	got := <-done
	if !errors.Is(got.err, context.Canceled) || got.pixels != nil {
		t.Fatalf("patchifyImage() = (%v, %v), want (nil, context.Canceled)", got.pixels, got.err)
	}
}

func TestVisionPatchLookupsCancellation(t *testing.T) {
	ctx := newQwenCheckpointContext(2) // allocation check, then the first merge-block row
	type result struct {
		ropePos []int32
		posIdx  []int32
		posWt   []float32
		err     error
	}
	done := make(chan result, 1)
	go func() {
		ropePos, posIdx, posWt, err := visionPatchLookups(ctx, 16, 16, 2, 2304)
		done <- result{ropePos: ropePos, posIdx: posIdx, posWt: posWt, err: err}
	}()
	<-ctx.reached
	ctx.cancel()
	got := <-done
	if !errors.Is(got.err, context.Canceled) || got.ropePos != nil || got.posIdx != nil || got.posWt != nil {
		t.Fatalf("visionPatchLookups() returned partial results on cancellation: %+v", got)
	}
}

func TestPrepareMediaFinalCancellation(t *testing.T) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 64, 64))); err != nil {
		t.Fatal(err)
	}
	m := testQwenVisionModel()
	segments := []base.Segment{{Kind: "image", Data: buf.Bytes()}}
	counting := &qwenCountingContext{}
	if _, err := m.PrepareMedia(counting, segments); err != nil {
		t.Fatal(err)
	}

	ctx := newQwenCheckpointContext(counting.calls.Load())
	type result struct {
		prepared *base.PreparedRequest
		err      error
	}
	done := make(chan result, 1)
	go func() {
		prepared, err := m.PrepareMedia(ctx, segments)
		done <- result{prepared: prepared, err: err}
	}()
	<-ctx.reached
	ctx.cancel()
	got := <-done
	if !errors.Is(got.err, context.Canceled) || got.prepared != nil {
		t.Fatalf("PrepareMedia() = (%v, %v), want (nil, context.Canceled)", got.prepared, got.err)
	}
}
