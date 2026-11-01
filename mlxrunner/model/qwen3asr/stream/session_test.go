package stream

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

type runeTokenizer struct{}

func (runeTokenizer) Encode(s string, _ bool) []int32 { return []int32(s) }
func (runeTokenizer) Decode(ids []int32) string       { return string(ids) }

func session(t *testing.T, f Infer) *Session {
	t.Helper()
	s, err := New(context.Background(), Config{}, runeTokenizer{}, f)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	return s
}

func TestReplayPacketization(t *testing.T) {
	// Cross update boundaries inside packets, with a one-sample final tail.
	input := make([]float32, 3*UpdateSamples+1)
	for i := range input {
		input[i] = float32(i%100) / 100
	}
	run := func(sizes []int) []Update {
		t.Helper()
		var ends []int
		s := session(t, func(_ context.Context, req Request) (string, error) {
			if !reflect.DeepEqual(req.Samples, input[:len(req.Samples)]) {
				t.Fatal("audio changed")
			}
			ends = append(ends, len(req.Samples))
			return "language German<asr_text>Die südafrikanische Juristin.", nil
		})
		var updates []Update
		for offset, packet := 0, 0; offset < len(input); packet++ {
			n := min(sizes[packet%len(sizes)], len(input)-offset)
			u, err := s.Push(offset, input[offset:offset+n])
			if err != nil {
				t.Fatal(err)
			}
			if u != nil {
				updates = append(updates, *u)
			}
			offset += n
		}
		final, err := s.Finish()
		if err != nil {
			t.Fatal(err)
		}
		updates = append(updates, final)
		if !reflect.DeepEqual(ends, []int{32000, 64000, 96000, 96001}) {
			t.Fatal(ends)
		}
		if s.pcm != nil || s.raw != "" || !s.closed {
			t.Fatal("session retained audio/text")
		}
		return updates
	}
	regular := run([]int{16000})
	irregular := run([]int{1, 319, 16000, 71, 7999})
	if !reflect.DeepEqual(regular, irregular) {
		t.Fatal("packetization changed hypotheses")
	}
	if !irregular[len(irregular)-1].Final || irregular[0].Language != "German" {
		t.Fatal(irregular)
	}
}

func TestReplayPrefixAndRevisions(t *testing.T) {
	var prefixes []string
	outputs := []string{"Hello wor", "Hello world", " revised", "!"}
	s := session(t, func(_ context.Context, r Request) (string, error) {
		prefixes = append(prefixes, r.Prefix)
		return outputs[len(prefixes)-1], nil
	})
	for offset := 0; offset < 3*UpdateSamples; offset += MaxPacketSamples {
		if _, err := s.Push(offset, make([]float32, MaxPacketSamples)); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := s.Push(3*UpdateSamples, []float32{0}); err != nil {
		t.Fatal(err)
	}
	final, err := s.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(prefixes, []string{"", "", "Hello ", "Hello  re"}) {
		t.Fatal(prefixes)
	}
	if final.Text != "Hello  re!" || !final.Final {
		t.Fatal(final)
	}
	if _, err := s.Push(final.AudioEnd, []float32{0}); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}

type splitTokenizer struct{ runeTokenizer }

func (splitTokenizer) Decode(ids []int32) string {
	if len(ids) == 7 {
		return "broken\ufffd"
	}
	return string(ids)
}

func TestReplayUnicodeRollback(t *testing.T) {
	var prefixes []string
	s, err := New(context.Background(), Config{}, splitTokenizer{}, func(_ context.Context, r Request) (string, error) {
		prefixes = append(prefixes, r.Prefix)
		return "Grüße世界abcde", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for offset := 0; offset < 3*UpdateSamples; offset += MaxPacketSamples {
		if _, err := s.Push(offset, make([]float32, MaxPacketSamples)); err != nil {
			t.Fatal(err)
		}
	}
	if prefixes[2] != "Grüße世" {
		t.Fatal(prefixes)
	}
}

func TestReplayBoundsAndFinalBoundary(t *testing.T) {
	calls := 0
	s := session(t, func(context.Context, Request) (string, error) { calls++; return "", nil })
	for _, p := range []struct {
		offset int
		data   []float32
	}{
		{1, []float32{0}},
		{0, nil},
		{0, make([]float32, MaxPacketSamples+1)},
		{0, []float32{float32(math.NaN())}},
		{0, []float32{float32(math.Inf(1))}},
		{0, []float32{1.01}},
	} {
		if _, err := s.Push(p.offset, p.data); err == nil {
			t.Fatal("accepted invalid input")
		}
		if len(s.pcm) != 0 {
			t.Fatal("invalid packet consumed")
		}
	}
	for offset := 0; offset < MaxSamples; offset += MaxPacketSamples {
		if _, err := s.Push(offset, make([]float32, MaxPacketSamples)); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := s.Push(MaxSamples, []float32{0}); err == nil {
		t.Fatal("accepted >60 seconds")
	}
	final, err := s.Finish()
	if err != nil || !final.Final || final.AudioEnd != MaxSamples || calls != 30 {
		t.Fatalf("%+v %v calls=%d", final, err, calls)
	}
	short := session(t, func(context.Context, Request) (string, error) { t.Fatal("decoded tiny input"); return "", nil })
	if _, err := short.Push(0, make([]float32, 200)); err != nil {
		t.Fatal(err)
	}
	if _, err := short.Finish(); err == nil || !short.closed {
		t.Fatal("tiny session not rejected/closed")
	}
}

func TestReplayCancelBusyIsolation(t *testing.T) {
	entered, finished := make(chan struct{}), make(chan error, 1)
	s := session(t, func(ctx context.Context, _ Request) (string, error) {
		close(entered)
		<-ctx.Done()
		return "", ctx.Err()
	})
	if _, err := s.Push(0, make([]float32, 16000)); err != nil {
		t.Fatal(err)
	}
	go func() { _, err := s.Push(16000, make([]float32, 16000)); finished <- err }()
	<-entered
	if _, err := s.Push(32000, []float32{0}); !errors.Is(err, ErrBusy) {
		t.Fatal(err)
	}
	if _, err := s.Finish(); !errors.Is(err, ErrBusy) {
		t.Fatal(err)
	}
	other := session(t, func(_ context.Context, r Request) (string, error) {
		if r.Prefix != "" {
			t.Fatal("session text leaked")
		}
		return "independent", nil
	})
	if _, err := other.Push(0, make([]float32, 201)); err != nil {
		t.Fatal(err)
	}
	u, err := other.Finish()
	if err != nil || u.Text != "independent" {
		t.Fatal(u, err)
	}
	s.Close()
	if err := <-finished; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if s.pcm != nil || s.raw != "" {
		t.Fatal("canceled state retained")
	}
}

func TestReplayFailureIsTerminal(t *testing.T) {
	for _, infer := range []Infer{
		func(context.Context, Request) (string, error) { return "partial", errors.New("inference failed") },
		func(context.Context, Request) (string, error) { return strings.Repeat("a", maxTextBytes+1), nil },
		func(context.Context, Request) (string, error) { return "\xff", nil },
	} {
		s := session(t, infer)
		if _, err := s.Push(0, make([]float32, 16000)); err != nil {
			t.Fatal(err)
		}
		if _, err := s.Push(16000, make([]float32, 16000)); err == nil {
			t.Fatal("accepted failed decode")
		}
		if !s.closed || s.pcm != nil {
			t.Fatal("failed session remains usable")
		}
	}
}

func TestReplayLiteralMarkerAndParentCancel(t *testing.T) {
	s := session(t, func(context.Context, Request) (string, error) { return "The token <asr_text> is literal.", nil })
	if _, err := s.Push(0, make([]float32, 201)); err != nil {
		t.Fatal(err)
	}
	u, err := s.Finish()
	if err != nil || u.Text != "The token <asr_text> is literal." {
		t.Fatal(u, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	other, err := New(ctx, Config{}, runeTokenizer{}, func(context.Context, Request) (string, error) { t.Fatal("decoded canceled session"); return "", nil })
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	if _, err := other.Push(0, []float32{0}); err != nil {
		t.Fatal(err)
	}
	cancel()
	if _, err := other.Push(1, []float32{0}); !errors.Is(err, context.Canceled) && !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if !other.closed || other.pcm != nil {
		t.Fatal("canceled session retained PCM")
	}
}

func TestReplayIdleAndParentRelease(t *testing.T) {
	for _, parentCancel := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		s, err := New(ctx, Config{IdleTimeout: 20 * time.Millisecond}, runeTokenizer{}, func(context.Context, Request) (string, error) { t.Error("idle session decoded"); return "", nil })
		if err != nil {
			cancel()
			t.Fatal(err)
		}
		if _, err := s.Push(0, make([]float32, 201)); err != nil {
			s.Close()
			cancel()
			t.Fatal(err)
		}
		if parentCancel {
			cancel()
		}
		select {
		case <-s.Done():
		case <-time.After(time.Second):
			s.Close()
			cancel()
			t.Fatal("idle/cancelled session did not release")
		}
		s.mu.Lock()
		retained := len(s.pcm) != 0 || s.raw != ""
		s.mu.Unlock()
		if retained {
			t.Error("closed idle session retained input")
		}
		s.Close()
		cancel()
	}
}

func TestReplayDrainKeepsLeaseUntilCleanup(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	lease, release := context.WithCancel(context.Background())
	defer release()
	entered, observedCancel, cleanupAllowed := make(chan struct{}), make(chan struct{}), make(chan struct{})
	s, err := New(parent, Config{IdleTimeout: 100 * time.Millisecond}, runeTokenizer{}, func(ctx context.Context, _ Request) (string, error) {
		close(entered)
		<-ctx.Done()
		close(observedCancel)
		<-cleanupAllowed // Models device-scope teardown/drain before callback return.
		return "must never publish", ctx.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	go func() { <-s.Done(); release() }()
	if _, err := s.Push(0, make([]float32, 16000)); err != nil {
		t.Fatal(err)
	}
	finished := make(chan error, 1)
	go func() { _, err := s.Push(16000, make([]float32, 16000)); finished <- err }()
	<-entered
	// A slow active callback is not idle; a pending packet cannot hold a model
	// worker, and the outer lease remains separate from cancellation of work.
	select {
	case <-observedCancel:
		t.Error("idle timeout canceled active inference")
	case <-time.After(150 * time.Millisecond):
	}
	cancel()
	<-observedCancel
	select {
	case <-s.Done():
		t.Error("drained before callback cleanup")
	default:
	}
	if lease.Err() != nil {
		t.Error("lease released while native work was draining")
	}
	close(cleanupAllowed)
	if err := <-finished; !errors.Is(err, context.Canceled) {
		t.Error(err)
	}
	select {
	case <-lease.Done():
	case <-time.After(time.Second):
		t.Fatal("drained session did not release lease")
	}
}

func TestReplayStaleIdleCallback(t *testing.T) {
	entered, proceed := make(chan struct{}), make(chan struct{})
	s := session(t, func(ctx context.Context, _ Request) (string, error) {
		close(entered)
		<-proceed
		return "current", ctx.Err()
	})
	if _, err := s.Push(0, make([]float32, 16000)); err != nil {
		t.Fatal(err)
	}
	result := make(chan error, 1)
	go func() { _, err := s.Push(16000, make([]float32, 16000)); result <- err }()
	<-entered
	// Emulate an old timer callback queued before the operation stopped it.
	s.mu.Lock()
	s.idleDeadline = time.Now().Add(-time.Second)
	s.mu.Unlock()
	s.expireIdle()
	if s.ctx.Err() != nil {
		t.Error("stale expiry canceled active inference")
	}
	close(proceed)
	if err := <-result; err != nil {
		t.Fatal(err)
	}
	// The same stale callback can also arrive after finishOperation refreshed
	// the deadline; it must not close the renewed session.
	s.expireIdle()
	if _, err := s.Push(32000, []float32{0}); err != nil {
		t.Fatal(err)
	}
}

func TestReplayLanguageFraming(t *testing.T) {
	for _, tc := range []struct{ language, raw, text, detected string }{
		{"", "language None<asr_text>", "", ""},
		{"", "language German<asr_text> Hallo.", "Hallo.", "German"},
		{"English", "language None<asr_text>", "language None<asr_text>", "English"},
	} {
		s, err := New(context.Background(), Config{Language: tc.language}, runeTokenizer{}, func(context.Context, Request) (string, error) { return tc.raw, nil })
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.Push(0, make([]float32, 201)); err != nil {
			s.Close()
			t.Fatal(err)
		}
		u, err := s.Finish()
		if err != nil || u.Text != tc.text || u.Language != tc.detected {
			t.Fatal(u, err)
		}
	}
}

func TestNegotiatedCadence(t *testing.T) {
	for _, interval := range []int{SampleRate / 2, SampleRate, UpdateSamples} {
		for _, packet := range []int{16000, 7919} {
			var ends []int
			s, err := New(t.Context(), Config{UpdateSamples: interval}, runeTokenizer{}, func(_ context.Context, r Request) (string, error) {
				ends = append(ends, len(r.Samples))
				return "hello world", nil
			})
			if err != nil {
				t.Fatal(err)
			}
			for offset := 0; offset < 64017; {
				n := min(packet, interval, 64017-offset)
				if _, err := s.Push(offset, make([]float32, n)); err != nil {
					t.Fatal(err)
				}
				offset += n
			}
			u, err := s.Finish()
			if err != nil || !u.Final || u.AudioEnd != 64017 {
				t.Fatal(u, err)
			}
			var expected []int
			for end := interval; end <= 64017; end += interval {
				expected = append(expected, end)
			}
			expected = append(expected, 64017)
			if !reflect.DeepEqual(ends, expected) {
				t.Fatal(ends, expected)
			}
		}
	}
	for _, bad := range []int{-1, 4000, 16001, 64000} {
		if s, err := New(t.Context(), Config{UpdateSamples: bad}, runeTokenizer{}, func(context.Context, Request) (string, error) { return "", nil }); err == nil {
			s.Close()
			t.Fatal("accepted invalid cadence", bad)
		}
	}
}

func TestReplayHalfSecondBoundsAndPrefixPolicy(t *testing.T) {
	input := make([]float32, 24017)
	for i := range input {
		input[i] = float32(i%100) / 100
	}
	var prefixes []string
	var ends []int
	outputs := []string{"Hello wor", "Hello world", " revised", "!"}
	s, err := New(t.Context(), Config{UpdateSamples: 8000}, runeTokenizer{}, func(_ context.Context, request Request) (string, error) {
		if !reflect.DeepEqual(request.Samples, input[:len(request.Samples)]) {
			t.Fatal("irregular packets changed audio prefix")
		}
		prefixes = append(prefixes, request.Prefix)
		ends = append(ends, len(request.Samples))
		return outputs[len(prefixes)-1], nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if u, err := s.Push(0, input[:1000]); err != nil || u != nil {
		t.Fatal(u, err)
	}
	if _, err := s.Push(1000, input[1000:17000]); err == nil {
		t.Fatal("one-second packet accepted at half-second cadence")
	}
	if len(s.pcm) != 1000 || len(ends) != 0 {
		t.Fatal("oversized packet consumed input or entered inference")
	}
	offset, revision := 1000, 0
	for _, count := range []int{7000, 8000, 23, 7977, 17} {
		u, err := s.Push(offset, input[offset:offset+count])
		if err != nil {
			t.Fatal(err)
		}
		offset += count
		if u != nil {
			revision++
			if u.Revision != revision || u.AudioEnd != revision*8000 {
				t.Fatal("push crossed more than one update boundary", u)
			}
		}
	}
	final, err := s.Finish()
	if err != nil || !final.Final || final.AudioEnd != len(input) || final.Revision != 4 || final.Text != "Hello  re!" {
		t.Fatal(final, err)
	}
	if !reflect.DeepEqual(ends, []int{8000, 16000, 24000, 24017}) || !reflect.DeepEqual(prefixes, []string{"", "", "Hello ", "Hello  re"}) {
		t.Fatal("cadence changed controlled prefix policy", ends, prefixes)
	}
}
