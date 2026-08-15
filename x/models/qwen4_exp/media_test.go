package qwen4_exp

import (
	"context"
	"errors"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/qwen3_5"
)

func TestPrepareMediaPassesCancellationToVision(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	m := &Model{Vision: &qwen3_5.VisionAdapter{Model: &qwen3_5.Model{}}}
	prepared, err := m.PrepareMedia(ctx, []base.Segment{{Tokens: []int32{1}}})
	if !errors.Is(err, context.Canceled) || prepared != nil {
		t.Fatalf("PrepareMedia() = (%v, %v), want (nil, context.Canceled)", prepared, err)
	}
}
