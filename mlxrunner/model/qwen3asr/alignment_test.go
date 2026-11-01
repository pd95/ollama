package qwen3asr

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestOriginalAlignerConfig(t *testing.T) {
	raw, err := os.ReadFile("testdata/config-aligner.json")
	if err != nil {
		t.Fatal(err)
	}
	if !IsForcedAligner(raw) {
		t.Fatal("aligner not identified")
	}
	shapes, err := TensorShapes(raw)
	if err != nil || len(shapes) != 708 || !reflect.DeepEqual(shapes["thinker.lm_head.weight"], []int32{5000, 1024}) {
		t.Fatalf("classification schema: %v", err)
	}
	bad := strings.Replace(string(raw), `"classify_num": 5000`, `"classify_num": 151936`, 1)
	if _, err := TensorShapes([]byte(bad)); err == nil {
		t.Fatal("vocabulary head accepted for classification")
	}
	cfg, err := parseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{cfg: cfg}
	if _, err := m.PrepareMedia(nil); err == nil {
		t.Fatal("aligner admitted autoregressive preparation")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.AlignContext(ctx, nil, "Hello", "English"); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestAlignmentTextSpans(t *testing.T) {
	text := "  Grüße, can't foo-bar 中A文! Cafe\u0301  "
	units, prompt, err := AlignmentPrompt(text, "German")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"Grüße", "can't", "foobar", "中", "A", "文", "Cafe"}
	var actual []string
	for _, u := range units {
		actual = append(actual, u.Text)
		if u.ByteStart < 0 || u.ByteEnd > len(text) || u.ByteStart >= u.ByteEnd {
			t.Fatal("invalid source span")
		}
	}
	if !reflect.DeepEqual(actual, want) || strings.Count(prompt, "<timestamp>") != 14 {
		t.Fatalf("%v / %s", actual, prompt)
	}
	if text[units[2].ByteStart:units[2].ByteEnd] != "foo-bar" {
		t.Fatal("original punctuation lost")
	}
	for _, invalid := range []string{"", "!!!", strings.Repeat("a ", 1025)} {
		if _, _, err := AlignmentPrompt(invalid, "English"); err == nil {
			t.Fatal("invalid text admitted")
		}
	}
	if _, _, err := AlignmentPrompt("hello", "Japanese"); err == nil {
		t.Fatal("unqualified segmentation admitted")
	}
}

func TestAlignmentRepair(t *testing.T) {
	for _, c := range []struct{ raw, want []int }{
		{[]int{0, 80, 160, 240}, []int{0, 80, 160, 240}},
		{[]int{800, 80, 160, 240}, []int{80, 80, 160, 240}},
		{[]int{0, 800, 160, 240}, []int{0, 0, 160, 240}},
		{[]int{0, 800, 720, 640, 320, 400, 480, 560}, []int{0, 80, 160, 240, 320, 400, 480, 560}},
	} {
		before := append([]int(nil), c.raw...)
		if got := repairTimestamps(c.raw); !reflect.DeepEqual(got, c.want) {
			t.Fatalf("%v: %v != %v", c.raw, got, c.want)
		}
		if !reflect.DeepEqual(before, c.raw) {
			t.Fatal("raw predictions mutated")
		}
	}
	u, _, _ := AlignmentPrompt("a b", "English")
	r, err := alignmentResult("a b", u, []int32{0, 10, 2, 3}, .1)
	if err != nil || !r.Items[0].Repaired || !r.Items[1].OutsideAudio || r.Items[0].RawEndMS != 800 {
		t.Fatalf("lost anomaly semantics: %+v %v", r, err)
	}
}
