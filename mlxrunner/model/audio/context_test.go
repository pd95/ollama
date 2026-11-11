package audio

import (
	"context"
	"errors"
	"testing"
	"time"
)

// Cancel deterministically after work begins, without timing-sensitive sleeps.
type cancelAfterChecks struct {
	checks int
	limit  int
}

func (*cancelAfterChecks) Deadline() (time.Time, bool) { return time.Time{}, false }
func (*cancelAfterChecks) Done() <-chan struct{}       { return nil }
func (*cancelAfterChecks) Value(any) any               { return nil }

func (c *cancelAfterChecks) Err() error {
	c.checks++
	if c.checks > c.limit {
		return context.Canceled
	}
	return nil
}

func TestAudioContextMidWorkCancellation(t *testing.T) {
	ctx := &cancelAfterChecks{limit: 8}
	_, _, err := DecodeLimitContext(ctx, EncodeWAV(make([]float32, 32000), 16000), 2)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("WAV decode ignored mid-work cancellation after %d checks: %v", ctx.checks, err)
	}
	ctx = &cancelAfterChecks{limit: 3}
	_, err = ResampleContext(ctx, make([]float32, 32000), 48000, 16000)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("resampling ignored mid-work cancellation after %d checks: %v", ctx.checks, err)
	}
	ctx = &cancelAfterChecks{limit: 3}
	_, err = EncodeWAVContext(ctx, make([]float32, 32000), 16000)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("WAV encoding ignored cancellation: %v", err)
	}
	ctx = &cancelAfterChecks{limit: 5}
	_, err = SplitContext(ctx, make([]float32, 80001), 16000, 1)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cut-point search ignored cancellation: %v", err)
	}
}
