package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/openai"
)

func testTimestampObserver(ctx context.Context) (context.Context, *timestampObserver) {
	o := &timestampObserver{notify: make(chan struct{}, 1), batches: make(chan timestampBatch, 1)}
	return context.WithValue(ctx, timestampObserverKey{}, o), o
}

func TestTimestampStreamPublication(t *testing.T) {
	ctx, o := testTimestampObserver(t.Context())
	p := timestampPublication{}
	out := openai.TranscriptionTimestampResponse{Text: "Why sky", Words: []openai.TranscriptionWord{{Word: "Why", Start: 29, End: 29.5}, {Word: "sky", Start: 29.6, End: 30.08}}}
	if err := p.update(ctx, &out, 0, 0, 0, 28, false); err != nil {
		t.Fatal(err)
	}
	if len(o.batches) != 0 {
		t.Fatal("published mutable boundary")
	}
	clip := openai.TranscriptionTimestampResponse{Text: "Why sky blue?", Words: []openai.TranscriptionWord{{Word: "Why", Start: .99, End: 1.52}, {Word: "sky", Start: 1.59, End: 2.05}, {Word: "blue", Start: 2.06, End: 2.5}}}
	oldText, oldWords := len(out.Text), len(out.Words)
	if err := appendTimestampWindow(&out, clip, 28, 30, 60, false, nil); err != nil {
		t.Fatal(err)
	}
	// Release the held first block; keep the new block pending to exercise EOF flush.
	if err := p.update(ctx, &out, oldText, oldWords, 0, 30.06, false); err != nil {
		t.Fatal(err)
	}
	first := <-o.batches
	if first.Words[1].End != 30.05 || first.Text != "Why sky" {
		t.Fatalf("%+v", first)
	}
	if err := p.update(ctx, &out, len(out.Text), len(out.Words), 0, 0, true); err != nil {
		t.Fatal(err)
	}
	last := <-o.batches
	if first.Text+last.Text != out.Text || last.Offset != 2 || last.Words[0] != out.Words[2] {
		t.Fatal("stream differs from final result")
	}
	out.Words[0].End = 29.9
	if err := p.update(ctx, &out, len(out.Text), len(out.Words), 0, 100, true); err == nil {
		t.Fatal("published revision accepted")
	}
}

func TestTimestampStreamScan(t *testing.T) {
	ctx, o := testTimestampObserver(t.Context())
	path := t.TempDir() + "/audio.wav"
	if err := os.WriteFile(path, audio.EncodeWAV(make([]float32, 95*16000), 16000), 0o600); err != nil {
		t.Fatal(err)
	}
	total, err := timestampScan(ctx, path, o)
	if err != nil || total != 95 {
		t.Fatalf("%v %v", total, err)
	}
	o.progress.Total = &total
	o.progress.Processed = 30
	p := o.snapshot()
	if p.Percent == nil || *p.Percent != 100*30.0/95 {
		t.Fatal(p)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := timestampScan(canceled, path, o); err == nil {
		t.Fatal("ignored cancellation")
	}
	if err := os.WriteFile(path, []byte("invalid"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := timestampScan(ctx, path, o); audioTextErrorStatus(err, 500) != 400 {
		t.Fatal(err)
	}
}

func TestTimestampStreamErrorAndAdmission(t *testing.T) {
	s := &Server{}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/", nil)
	s.streamTimestampTranscription(c, audioTextForm{timestampUpload: t.TempDir() + "/missing"}, nil, nil)
	if w.Code != 200 || !strings.Contains(w.Header().Get("Content-Type"), "text/event-stream") {
		t.Fatal(w.Code, w.Header())
	}
	done := false
	failure := false
	for _, line := range strings.Split(w.Body.String(), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var e map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &e); err != nil {
			t.Fatal(err)
		}
		done = done || e["type"] == "transcript.text.done"
		failure = failure || e["type"] == "error" && e["complete"] == false
	}
	if done || !failure {
		t.Fatal(w.Body.String())
	}
	s.timestampQueue().mu.Lock()
	s.timestampQueue().closed = true
	s.timestampQueue().mu.Unlock()
	w = httptest.NewRecorder()
	c, _ = gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/", nil)
	s.streamTimestampTranscription(c, audioTextForm{}, nil, nil)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatal(w.Code)
	}
}

func TestTimestampStreamQueueDeliversBeforeWork(t *testing.T) {
	s := &Server{}
	q := s.timestampQueue()
	q.limit = 1
	release := make(chan struct{})
	started := make(chan struct{})
	_, err := q.enqueue(t.Context(), func(context.Context) (openai.TranscriptionTimestampResponse, error) {
		close(started)
		<-release
		return openai.TranscriptionTimestampResponse{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer close(release)
	<-started
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, _ := gin.CreateTestContext(w)
		c.Request = r
		s.streamTimestampTranscription(c, audioTextForm{}, nil, nil)
	}))
	defer server.Close()
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", server.URL, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := make([]byte, 512)
	n, err := resp.Body.Read(buf)
	if err != nil || !strings.Contains(string(buf[:n]), "queued") {
		t.Fatalf("%s %v", buf[:n], err)
	}
	cancel()
}

func TestTimestampStreamPointAtWatermark(t *testing.T) {
	ctx, o := testTimestampObserver(t.Context())
	p := timestampPublication{}
	out := openai.TranscriptionTimestampResponse{Text: "point", Words: []openai.TranscriptionWord{{Word: "point", Start: 28, End: 28}}}
	if err := p.update(ctx, &out, 0, 0, 0, 28, false); err != nil {
		t.Fatal(err)
	}
	if len(o.batches) != 0 {
		t.Fatal("point at watermark was published")
	}
	if err := p.update(ctx, &out, len(out.Text), 1, 0, 58, false); err != nil {
		t.Fatal(err)
	}
	if (<-o.batches).Words[0] != out.Words[0] {
		t.Fatal("point changed")
	}
}

func TestTimestampStreamShortDurationGrid(t *testing.T) {
	for _, rate := range []int{44100, 288359} {
		ctx, o := testTimestampObserver(t.Context())
		path := t.TempDir() + "/audio.wav"
		count := rate + 1
		if err := os.WriteFile(path, audio.EncodeWAV(make([]float32, count), rate), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := timestampScan(ctx, path, o); err != nil {
			t.Fatal(err)
		}
		expected := (time.Duration(count) * time.Second / time.Duration(rate)).Seconds()
		if o.shortDuration != expected {
			t.Fatalf("short duration changed: %.17g %.17g", o.shortDuration, expected)
		}
	}
}

func TestTimestampStreamScanDurationLimit(t *testing.T) {
	ctx, o := testTimestampObserver(t.Context())
	path := t.TempDir() + "/audio.wav"
	if err := os.WriteFile(path, audio.EncodeWAV(make([]float32, maxAudioSeconds+1), 1), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := timestampScan(ctx, path, o); audioTextErrorStatus(err, 500) != 400 {
		t.Fatalf("duration cap: %v", err)
	}
}
