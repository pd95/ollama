package english

import (
	"strings"
	"testing"
)

func TestPronunciations(t *testing.T) {
	for _, tc := range []struct {
		text string
		gb   bool
		want string
	}{
		{"Hello world.", false, "həlˈO wˈɜɹld."},
		{"Hello world.", true, "həlˈQ wˈɜːld."},
		{"42", false, "fˈɔɹTi tˈu"},
		{"the apple", false, "ði ˈæpᵊl"},
		{"the cat", false, "ðə kˈæt"},
	} {
		t.Run(tc.text, func(t *testing.T) {
			got, err := Phonemes(tc.text, tc.gb)
			if err != nil || got != tc.want {
				t.Fatalf("%q, %v; want %q", got, err, tc.want)
			}
		})
	}
}

func TestContextAndFallback(t *testing.T) {
	for _, tc := range []struct{ text, contains string }{
		{"a record", "ɹˈɛkəɹd"},
		{"to record", "ɹəkˈɔɹd"},
		{"the wind", "wˈɪnd"},
		{"to wind", "wˈInd"},
		{"read yesterday", "ɹˈɛd"},
		{"to read", "ɹˈid"},
		{"lead pipes", "lˈɛd"},
		{"zorbaflump", "zˈɑɹbæflʌmp"},
	} {
		got, err := Phonemes(tc.text, false)
		if err != nil || !strings.Contains(got, tc.contains) {
			t.Errorf("%s = %q, %v; missing %q", tc.text, got, err, tc.contains)
		}
	}
	if _, err := Phonemes("hello 🤖", false); err == nil {
		t.Fatal("silently omitted unsupported symbol")
	}
}

func TestNumbersAndAbbreviations(t *testing.T) {
	for _, pair := range [][2]string{
		{"3.14", "three point one four"},
		{"21st", "twenty first"},
		{"$12.50", "twelve dollars and fifty cents"},
		{"Dr. Smith", "doctor Smith"},
		{"1,234", "one thousand two hundred thirty four"},
		{"007", "zero zero seven"},
	} {
		got, err := Phonemes(pair[0], false)
		if err != nil {
			t.Fatal(err)
		}
		want, err := Phonemes(pair[1], false)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("%s = %q, want %q", pair[0], got, want)
		}
	}
}
