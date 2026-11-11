package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/openai"
)

type (
	timestampObserverKey   struct{}
	timestampProgressEvent struct {
		Type      string   `json:"type"`
		Stage     string   `json:"stage"`
		Processed float64  `json:"processed_seconds"`
		Scanned   float64  `json:"scanned_seconds"`
		Total     *float64 `json:"total_seconds,omitempty"`
		Percent   *float64 `json:"percent,omitempty"`
	}
	timestampBatch struct {
		Text     string
		Offset   int
		Words    []openai.TranscriptionWord
		Segments []openai.TranscriptionSegment
	}
	timestampObserver struct {
		mu            sync.Mutex
		progress      timestampProgressEvent
		shortDuration float64
		notify        chan struct{}
		batches       chan timestampBatch
	}
)

func timestampProgress(ctx context.Context, stage string, processed float64) {
	if o, ok := ctx.Value(timestampObserverKey{}).(*timestampObserver); ok {
		o.mu.Lock()
		o.progress.Stage = stage
		if processed >= 0 {
			o.progress.Processed = processed
		}
		o.mu.Unlock()
		select {
		case o.notify <- struct{}{}:
		default:
		}
	}
}

func (o *timestampObserver) snapshot() timestampProgressEvent {
	o.mu.Lock()
	defer o.mu.Unlock()
	p := o.progress
	if p.Total != nil {
		percent := 100 * p.Processed / *p.Total
		p.Percent = &percent
	}
	return p
}

func timestampEmit(ctx context.Context, b timestampBatch) error {
	o, ok := ctx.Value(timestampObserverKey{}).(*timestampObserver)
	if !ok {
		return nil
	}
	select {
	case o.batches <- b:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

type (
	timestampBlock       struct{ textStart, textEnd, wordStart, wordEnd, segmentStart, segmentEnd int }
	timestampPublication struct {
		blocks    []timestampBlock
		published []openai.TranscriptionWord
	}
)

func (p *timestampPublication) update(ctx context.Context, out *openai.TranscriptionTimestampResponse, textStart, wordStart, segmentStart int, watermark float64, final bool) error {
	if ctx.Value(timestampObserverKey{}) == nil {
		return nil
	}
	for i, w := range p.published {
		if out.Words[i] != w {
			return timestampFailure(422, "chunk join revised published words")
		}
	}
	if len(out.Text) > textStart {
		end := 0
		if out.Segments != nil {
			end = len(*out.Segments)
		}
		p.blocks = append(p.blocks, timestampBlock{textStart, len(out.Text), wordStart, len(out.Words), segmentStart, end})
	}
	for len(p.blocks) > 0 {
		b := p.blocks[0]
		safe := true
		for _, w := range out.Words[b.wordStart:b.wordEnd] {
			if w.End >= watermark {
				safe = false
				break
			}
		}
		if !final && !safe {
			break
		}
		batch := timestampBatch{Text: out.Text[b.textStart:b.textEnd], Offset: b.wordStart, Words: append([]openai.TranscriptionWord{}, out.Words[b.wordStart:b.wordEnd]...)}
		if out.Segments != nil {
			batch.Segments = append([]openai.TranscriptionSegment{}, (*out.Segments)[b.segmentStart:b.segmentEnd]...)
		}
		if err := timestampEmit(ctx, batch); err != nil {
			return err
		}
		p.published = append(p.published, batch.Words...)
		p.blocks = p.blocks[1:]
	}
	return nil
}

// Scan and reopen rather than retaining decoded PCM or trusting MP3 bitrate estimates.
func timestampScan(ctx context.Context, path string, o *timestampObserver) (float64, error) {
	source, err := audio.OpenSampleStream(ctx, path)
	if err != nil {
		return 0, timestampDecodeError(err)
	}
	defer source.Close()
	rate := source.SampleRate()
	if rate < 1 {
		return 0, timestampFailure(400, "invalid audio sample rate")
	}
	samples := make([]float32, 4096)
	count := int64(0)
	for {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		n, err := source.Read(samples)
		if n < 0 || n > len(samples) {
			return 0, errors.New("invalid audio decoder read")
		}
		count += int64(n)
		if count > int64(maxAudioSeconds)*int64(rate) {
			return 0, timestampFailure(400, "word timestamps accept recordings up to 2 hours")
		}
		o.mu.Lock()
		o.progress.Scanned = float64(count) / float64(rate)
		o.mu.Unlock()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return 0, timestampDecodeError(err)
		}
		if n == 0 {
			return 0, io.ErrNoProgress
		}
	}
	if count == 0 {
		return 0, timestampFailure(400, "audio file contains no samples")
	}
	o.shortDuration = (time.Duration(count) * time.Second / time.Duration(rate)).Seconds()
	return float64(count) / float64(rate), nil
}

func timestampDecodeError(err error) error {
	if errors.Is(err, audio.ErrInvalidData) || errors.Is(err, audio.ErrUnrecognized) {
		return timestampFailure(400, err.Error())
	}
	return err
}

func (s *Server) streamTimestampTranscription(c *gin.Context, form audioTextForm, asr, aligner *Model) {
	ctx, cancel := context.WithCancel(c.Request.Context())
	defer cancel()
	o := &timestampObserver{progress: timestampProgressEvent{Type: "transcript.progress", Stage: "queued"}, notify: make(chan struct{}, 1), batches: make(chan timestampBatch, 1)}
	ctx = context.WithValue(ctx, timestampObserverKey{}, o)
	job, err := s.timestampQueue().enqueue(ctx, func(work context.Context) (openai.TranscriptionTimestampResponse, error) {
		timestampProgress(work, "preparing", 0)
		total, err := timestampScan(work, form.timestampUpload, o)
		if err != nil {
			return openai.TranscriptionTimestampResponse{}, err
		}
		o.mu.Lock()
		o.progress.Total = &total
		o.mu.Unlock()
		timestampProgress(work, "decoding", 0)
		result, err := s.transcribeWithTimestamps(work, form, asr, aligner)
		if err != nil {
			return result, err
		}
		expected := total
		if total <= 60 {
			expected = o.shortDuration
		}
		if result.Duration != expected {
			return result, timestampFailure(502, "decoded duration changed after preparation")
		}
		// Short clips bypass the rolling-window publisher.
		if total <= 60 {
			var segments []openai.TranscriptionSegment
			if result.Segments != nil {
				segments = append([]openai.TranscriptionSegment{}, (*result.Segments)...)
			}
			if err := timestampEmit(work, timestampBatch{Text: result.Text, Words: append([]openai.TranscriptionWord{}, result.Words...), Segments: segments}); err != nil {
				return result, err
			}
		}
		timestampProgress(work, "finalizing", total)
		return result, nil
	})
	if err != nil {
		openAIError(c, audioTextErrorStatus(err, 500), err.Error())
		return
	}
	defer func() {
		job.cancel()
		q := s.timestampQueue()
		q.mu.Lock()
		defer q.mu.Unlock()
		for i, j := range q.pending {
			if j == job {
				copy(q.pending[i:], q.pending[i+1:])
				q.pending[len(q.pending)-1] = nil
				q.pending = q.pending[:len(q.pending)-1]
				break
			}
		}
	}()
	controller := http.NewResponseController(c.Writer)
	callbackDone := make(chan struct{})
	stop := context.AfterFunc(ctx, func() { defer close(callbackDone); _ = controller.SetWriteDeadline(time.Now()) })
	defer func() {
		if !stop() {
			<-callbackDone
		}
		_ = controller.SetWriteDeadline(time.Time{})
	}()
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("X-Accel-Buffering", "no")
	write := func(v any) error {
		data, err := json.Marshal(v)
		if err != nil {
			return err
		}
		_ = controller.SetWriteDeadline(time.Now().Add(30 * time.Second))
		if err := ctx.Err(); err != nil {
			return err
		}
		if _, err = fmt.Fprintf(c.Writer, "data: %s\n\n", data); err != nil {
			return err
		}
		return controller.Flush()
	}
	batch := func(b timestampBatch) error {
		if b.Text != "" {
			if err := write(map[string]any{"type": "transcript.text.delta", "delta": b.Text}); err != nil {
				return err
			}
		}
		return write(struct {
			Type     string                         `json:"type"`
			Offset   int                            `json:"word_offset"`
			Words    []openai.TranscriptionWord     `json:"words"`
			Segments *[]openai.TranscriptionSegment `json:"segments,omitempty"`
		}{"transcript.words.delta", b.Offset, b.Words, func() *[]openai.TranscriptionSegment {
			if form.segmentTimestamps {
				return &b.Segments
			}
			return nil
		}()})
	}
	if err := write(o.snapshot()); err != nil {
		return
	}
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if write(o.snapshot()) != nil {
				return
			}
		case <-o.notify:
			if write(o.snapshot()) != nil {
				return
			}
		case b := <-o.batches:
			if batch(b) != nil {
				return
			}
		case reply := <-job.done:
			for {
				select {
				case b := <-o.batches:
					if batch(b) != nil {
						return
					}
				default:
					goto drained
				}
			}
		drained:
			if reply.err != nil {
				_ = write(map[string]any{"type": "error", "status": audioTextErrorStatus(reply.err, 500), "complete": false, "error": openai.NewError(audioTextErrorStatus(reply.err, 500), reply.err.Error()).Error})
				return
			}
			if write(o.snapshot()) != nil {
				return
			}
			_ = write(map[string]any{"type": "transcript.text.done", "text": reply.response.Text, "result": reply.response})
			return
		}
	}
}
