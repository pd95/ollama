package glimmer

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/png"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

type glimmerPanickingImage struct{ payload any }

func (glimmerPanickingImage) ColorModel() color.Model   { return color.NRGBAModel }
func (glimmerPanickingImage) Bounds() image.Rectangle   { return image.Rect(0, 0, 4, 4) }
func (p glimmerPanickingImage) At(int, int) color.Color { panic(p.payload) }

type glimmerTestContext struct{}

func (glimmerTestContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (glimmerTestContext) Done() <-chan struct{}       { return nil }
func (glimmerTestContext) Value(any) any               { return nil }

type glimmerCountingContext struct {
	glimmerTestContext
	calls atomic.Int32
}

func (c *glimmerCountingContext) Err() error {
	c.calls.Add(1)
	return nil
}

type glimmerCheckpointContext struct {
	glimmerTestContext
	blockAt  int32
	calls    atomic.Int32
	canceled atomic.Bool
	reached  chan struct{}
	release  chan struct{}
	once     sync.Once
}

func newGlimmerCheckpointContext(blockAt int32) *glimmerCheckpointContext {
	return &glimmerCheckpointContext{
		blockAt: blockAt,
		reached: make(chan struct{}),
		release: make(chan struct{}),
	}
}

func (c *glimmerCheckpointContext) Err() error {
	if c.calls.Add(1) == c.blockAt {
		c.once.Do(func() { close(c.reached) })
		<-c.release
	}
	if c.canceled.Load() {
		return context.Canceled
	}
	return nil
}

func (c *glimmerCheckpointContext) cancel() {
	c.canceled.Store(true)
	close(c.release)
}

func testVisionModel() *Model {
	return &Model{
		VisionEncoder: &VisionEncoder{},
		Config: &Config{
			HasVision:              true,
			PatchTokenID:           7,
			ImageStartTokenID:      8,
			ImageEndTokenID:        9,
			VisionPatchSize:        14,
			VisionPatchTemporal:    1,
			VisionDownsampleFactor: 2,
		},
	}
}

func testPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewNRGBA(image.Rect(0, 0, w, h))); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestPrepareMediaSplicesExpansion(t *testing.T) {
	m := testVisionModel()
	prepared, err := m.PrepareMedia(context.Background(), []base.Segment{
		{Tokens: []int32{1, 2}},
		{Kind: "image", Data: testPNG(t, 56, 56)},
		{Tokens: []int32{3}},
	})
	if err != nil {
		t.Fatal(err)
	}

	// 56x56 at patch stride 28 (14*2) is a 2x2 downsampled grid: 4 output
	// tokens over a 4x4 patch grid.
	wantTokens := []int32{1, 2, 8, 7, 7, 7, 7, 9, 3}
	if len(prepared.Tokens) != len(wantTokens) {
		t.Fatalf("tokens = %v, want %v", prepared.Tokens, wantTokens)
	}
	for i, tok := range wantTokens {
		if prepared.Tokens[i] != tok {
			t.Fatalf("tokens = %v, want %v", prepared.Tokens, wantTokens)
		}
	}

	if len(prepared.Items) != 1 {
		t.Fatalf("items = %d, want 1", len(prepared.Items))
	}
	item := prepared.Items[0]
	if item.Range != [2]int{2, 8} {
		t.Fatalf("range = %v, want [2 8]", item.Range)
	}
	if item.Source != 1 {
		t.Fatalf("source = %d, want 1", item.Source)
	}
	if !item.Causal {
		t.Fatal("expected causal expansion")
	}

	geom := item.Opaque.(preparedImage)
	if geom.gridH != 4 || geom.gridW != 4 || geom.outputTokens != 4 {
		t.Fatalf("geometry = %+v, want 4x4 grid with 4 output tokens", geom)
	}
	want := 1
	for _, d := range item.Dims {
		want *= d
	}
	if len(item.MediaData) != want {
		t.Fatalf("media data length %d does not match dims %v", len(item.MediaData), item.Dims)
	}
}

func TestPrepareMediaRejectsUnsupportedKind(t *testing.T) {
	m := testVisionModel()
	_, err := m.PrepareMedia(context.Background(), []base.Segment{{Kind: "audio", Data: []byte{1}}})
	if err == nil {
		t.Fatal("expected error for audio input")
	}

	text := &Model{Config: &Config{}}
	_, err = text.PrepareMedia(context.Background(), []base.Segment{{Kind: "image", Data: []byte{1}}})
	if err == nil {
		t.Fatal("expected error for text-only model")
	}
}

func TestPrepareMediaPreCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	prepared, err := testVisionModel().PrepareMedia(ctx, []base.Segment{{Tokens: []int32{1}}})
	if !errors.Is(err, context.Canceled) || prepared != nil {
		t.Fatalf("PrepareMedia() = (%v, %v), want (nil, context.Canceled)", prepared, err)
	}
}

func TestImageReaderCancellation(t *testing.T) {
	ctx := newGlimmerCheckpointContext(1)
	result := make(chan error, 1)
	go func() {
		_, err := glimmerContextReader{check: ctx.Err, r: bytes.NewReader([]byte("image"))}.Read(make([]byte, 1))
		result <- err
	}()
	<-ctx.reached
	ctx.cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("Read() error = %v, want context.Canceled", err)
	}
}

func TestResizeCancellation(t *testing.T) {
	ctx := newGlimmerCheckpointContext(2) // entry check, then the first destination pixel
	result := make(chan error, 1)
	go func() {
		_, err := resizeGlimmerImage(ctx, image.NewNRGBA(image.Rect(0, 0, 4, 4)), image.Rect(0, 0, 4, 4), 64, 64)
		result <- err
	}()
	<-ctx.reached
	ctx.cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("resizeGlimmerImage() error = %v, want context.Canceled", err)
	}
}

func TestResizeRepanicsUnrelatedPayload(t *testing.T) {
	payload := &struct{ marker byte }{marker: 1}
	recovered := func() (got any) {
		defer func() { got = recover() }()
		_, _ = resizeGlimmerImage(context.Background(), glimmerPanickingImage{payload: payload}, image.Rect(0, 0, 4, 4), 64, 64)
		return nil
	}()
	if recovered != payload {
		t.Fatalf("recovered payload = %v, want exact unrelated payload %v", recovered, payload)
	}
}

func TestPrepareMediaFinalCancellation(t *testing.T) {
	m := testVisionModel()
	segments := []base.Segment{{Kind: "image", Data: testPNG(t, 56, 56)}}
	counting := &glimmerCountingContext{}
	if _, err := m.PrepareMedia(counting, segments); err != nil {
		t.Fatal(err)
	}

	ctx := newGlimmerCheckpointContext(counting.calls.Load())
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
