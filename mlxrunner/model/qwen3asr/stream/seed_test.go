package stream

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sync"
	"testing"
	"time"
)

func TestReplaySeedFreshSingleDecode(t *testing.T) {
	for _, interval := range []int{8000, 16000, 32000} {
		for _, count := range []int{201, 16001, MaxSeedSamples} {
			input := make([]float32, count)
			for i := range input {
				input[i] = float32(i%1024-512) / 512
			}
			calls := 0
			s, err := New(t.Context(), Config{UpdateSamples: interval}, runeTokenizer{}, func(_ context.Context, request Request) (string, error) {
				calls++
				if request.Prefix != "" || !reflect.DeepEqual(request.Samples, input) {
					t.Fatal("seed used a previous prefix or changed PCM")
				}
				return "language German<asr_text>Frischer Text", nil
			})
			if err != nil {
				t.Fatal(err)
			}
			u, err := s.Seed(input)
			if err != nil || u == nil || u.Final || u.Revision != 1 || u.AudioEnd != count || u.Text != "Frischer Text" {
				t.Fatal(u, err)
			}
			if _, err := s.Seed(input); err == nil {
				t.Fatal("second seed accepted")
			}
			final, err := s.Finish()
			if err != nil || !final.Final || final.Revision != 2 || final.AudioEnd != count || final.Text != u.Text || calls != 1 {
				t.Fatal(final, err, calls)
			}
			select {
			case <-s.Done():
			default:
				t.Fatal("seeded final did not drain")
			}
		}
	}
}

func TestReplaySeedValidationDoesNotConsume(t *testing.T) {
	s := session(t, func(context.Context, Request) (string, error) { return "fresh", nil })
	for _, pcm := range [][]float32{nil, make([]float32, 200), make([]float32, MaxSeedSamples+1), append(make([]float32, 200), float32(math.NaN())), append(make([]float32, 200), float32(math.Inf(1))), append(make([]float32, 200), float32(-1.01))} {
		if _, err := s.Seed(pcm); err == nil || len(s.pcm) != 0 || s.steps != 0 {
			t.Fatal("invalid seed accepted or consumed input", err)
		}
	}
	if _, err := s.Push(0, make([]float32, 201)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Seed(make([]float32, 201)); err == nil || len(s.pcm) != 201 {
		t.Fatal("seed accepted after ordinary Push", err)
	}
}

func TestReplaySeedCancellationWaitsForInferenceDrain(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	s := session(t, func(ctx context.Context, _ Request) (string, error) {
		close(entered)
		<-ctx.Done()
		<-release
		return "", ctx.Err()
	})
	result := make(chan error, 1)
	go func() { _, err := s.Seed(make([]float32, 256000)); result <- err }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("seed did not enter inference")
	}
	if _, err := s.Seed(make([]float32, 201)); !errors.Is(err, ErrBusy) {
		t.Fatal("concurrent seed was admitted", err)
	}
	s.Close()
	select {
	case <-s.Done():
		t.Fatal("seed acknowledged cancellation before inference drained")
	default:
	}
	releaseOnce.Do(func() { close(release) })
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	select {
	case <-s.Done():
	case <-time.After(time.Second):
		t.Fatal("seed did not acknowledge drain")
	}
}
