package audio

import (
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestPlanSpeech(t *testing.T) {
	for _, tc := range []struct {
		text  string
		limit int
		want  []string
	}{
		{"One. Two! Three? Four.", 11, []string{"One. Two!", "Three?", "Four."}},
		{"Dr. Smith paid 3.14 euros. Next sentence.", 26, []string{"Dr. Smith paid 3.14 euros.", "Next sentence."}},
		{"He said “Hello.” Then left.", 20, []string{"He said “Hello.”", "Then left."}},
		{"你好。再见！谢谢？", 6, []string{"你好。再见！", "谢谢？"}},
		{"first line\nsecond line", 12, []string{"first line", "second line"}},
		{"  \n ", 10, nil},
	} {
		got, err := PlanSpeech(tc.text, tc.limit, utf8.RuneCountInString)
		if err != nil || !slices.Equal(got, tc.want) {
			t.Errorf("%q: got %q / %v, want %q", tc.text, got, err, tc.want)
		}
	}
}

func TestPlanSpeechCoverageAndModelBudget(t *testing.T) {
	count := func(s string) int { return utf8.RuneCountInString(s) * 2 }
	text := strings.Repeat("First clause, then more words. ", 12) + strings.Repeat("語", 45)
	parts, err := PlanSpeech(text, 42, count)
	if err != nil {
		t.Fatal(err)
	}
	for _, part := range parts {
		if count(part) > 42 || !utf8.ValidString(part) {
			t.Fatalf("oversized/invalid %q", part)
		}
	}
	compact := func(s string) string { return strings.Join(strings.Fields(s), "") }
	if compact(strings.Join(parts, "")) != compact(text) {
		t.Fatal("text lost or duplicated")
	}
	if _, err := PlanSpeech("語", 1, count); err == nil {
		t.Fatal("accepted unsplittable oversized token")
	}
}

func TestSplitSpeech(t *testing.T) {
	for _, tc := range []struct{ text, left, right string }{
		{"First sentence. Second sentence.", "First sentence.", "Second sentence."},
		{"first clause, second clause continues", "first clause,", "second clause continues"},
		{"one two three four", "one two", "three four"},
		{"語語語語", "語語", "語語"},
	} {
		left, right := SplitSpeech(tc.text)
		if left != tc.left || right != tc.right {
			t.Errorf("%q: %q / %q", tc.text, left, right)
		}
	}
}
