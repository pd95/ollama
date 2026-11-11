package mlxrunner

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func TestQwen3ASRRawRepeatedMedia(t *testing.T) {
	tok, err := tokenizer.LoadFromBytes([]byte(`{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`))
	if err != nil {
		t.Fatal(err)
	}
	r := &Runner{Tokenizer: tok}
	wav := audio.EncodeWAV(make([]float32, 16000), 16000)
	// Raw generation forwards the authored prompt and detected media directly
	// to expandMedia, including repeated references to the same source.
	_, _, err = r.expandMedia(context.Background(), &qwen3asr.Model{}, "[img-0][img-0]", []llm.MediaData{llm.NewMediaData(0, wav)})
	if err == nil || !strings.Contains(err.Error(), "one audio clip") {
		t.Fatalf("raw repeated markers bypassed the single-audio bound: %v", err)
	}
}

type cancelMediaPreparation struct {
	stubMediaModel
	cancel context.CancelFunc
}

func (m cancelMediaPreparation) PrepareMediaContext(ctx context.Context, _ []model.Segment) (*model.PreparedRequest, error) {
	m.cancel()
	return nil, ctx.Err()
}

func TestQwen3ASRPreparationReceivesRequestContext(t *testing.T) {
	r := mediaTestRunner(t)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	r.Model = cancelMediaPreparation{stubMediaModel: r.Model.(stubMediaModel), cancel: cancel}
	err := r.Prepare(&Request{Ctx: ctx, CompletionRequest: CompletionRequest{
		Prompt: "[img-0]", Media: []llm.MediaData{{ID: 0, Kind: llm.MediaKindImage, Data: []byte("media")}},
	}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("preparation did not receive caller cancellation: %v", err)
	}
}
