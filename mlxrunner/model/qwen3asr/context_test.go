package qwen3asr

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/audio"
)

type cancelFeatureWork struct {
	checks int
}

func (*cancelFeatureWork) Deadline() (time.Time, bool) { return time.Time{}, false }
func (*cancelFeatureWork) Done() <-chan struct{}       { return nil }
func (*cancelFeatureWork) Value(any) any               { return nil }

func TestQwen3ASRConcurrentPreparation(t *testing.T) {
	m := &Model{}
	wav := audio.EncodeWAV(make([]float32, 16000), 16000)
	type result struct {
		request  *model.PreparedRequest
		err      error
		canceled bool
	}
	results := make(chan result, 4)
	for i := range 4 {
		go func() {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if i%2 == 0 {
				cancel()
			}
			r, err := m.PrepareMediaContext(ctx, []model.Segment{{Kind: "audio", Data: wav}})
			results <- result{r, err, i%2 == 0}
		}()
	}
	var successful []*model.PreparedRequest
	for range 4 {
		r := <-results
		if r.canceled {
			if !errors.Is(r.err, context.Canceled) || r.request != nil {
				t.Fatalf("canceled preparation: %+v", r)
			}
		} else {
			if r.err != nil {
				t.Fatal(r.err)
			}
			successful = append(successful, r.request)
		}
	}
	successful[0].Items[0].MediaData[0] = 123
	if successful[1].Items[0].MediaData[0] != -1.5 {
		t.Fatal("concurrent feature buffers are shared")
	}
}

func (c *cancelFeatureWork) Err() error {
	c.checks++
	if c.checks > 13 {
		return context.Canceled
	}
	return nil
}

func TestQwen3ASRPreprocessingCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := (&Model{}).PrepareMediaContext(ctx, []model.Segment{{Kind: "audio", Data: []byte("must not decode")}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled input decoded: %v", err)
	}
	work := &cancelFeatureWork{}
	_, err = extractFeaturesContext(work, make([]float32, 32000))
	if !errors.Is(err, context.Canceled) || work.checks <= 13 {
		t.Fatalf("frontend did not cancel after work started: checks=%d err=%v", work.checks, err)
	}
}
