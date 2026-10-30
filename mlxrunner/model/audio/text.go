package audio

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// PlanSpeech keeps whole sentences together when they fit the model's budget.
// count measures the candidate in the model's units (tokens, runes or phonemes).
// An oversized sentence is fitted at sentence/clause/word boundaries, then at
// Unicode code points. The caller retains its generation and duration limits.
func PlanSpeech(text string, limit int, count func(string) int) ([]string, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, nil
	}
	if limit < 1 || count == nil {
		return nil, fmt.Errorf("invalid speech text budget")
	}
	var out []string
	var current string
	flush := func() {
		if current != "" {
			out = append(out, current)
			current = ""
		}
	}
	for _, sentence := range speechSentences(text) {
		if count(sentence) > limit {
			flush()
			pending := []string{sentence}
			for len(pending) > 0 {
				part := pending[0]
				pending = pending[1:]
				if count(part) <= limit {
					out = append(out, part)
					continue
				}
				left, right := SplitSpeech(part)
				if left == "" || right == "" {
					return nil, fmt.Errorf("speech text cannot fit the %d-unit budget", limit)
				}
				pending = append([]string{left, right}, pending...)
			}
			continue
		}
		next := sentence
		if current != "" {
			sep := " "
			if last, _ := utf8.DecodeLastRuneInString(current); strings.ContainsRune("。！？", last) {
				sep = ""
			}
			next = current + sep + sentence
			if count(next) > limit {
				flush()
				next = sentence
			}
		}
		current = next
	}
	flush()
	return out, nil
}

// SplitSpeech divides a candidate that exceeds a model's duration/context
// budget. Prefer a balanced sentence boundary, then a clause, then whitespace.
// Neither side is empty unless the input is a single Unicode code point.
func SplitSpeech(text string) (string, string) {
	runes := []rune(strings.TrimSpace(text))
	for _, kind := range []int{0, 1, 2} {
		best, distance := 0, len(runes)+1
		for i, r := range runes {
			end := i + 1
			valid := kind == 2 && unicode.IsSpace(r)
			if kind == 0 {
				end, valid = speechSentenceEnd(runes, i)
			}
			if kind == 1 {
				valid = strings.ContainsRune(",;:，；：", r)
			}
			if !valid || end >= len(runes) || strings.TrimSpace(string(runes[:end])) == "" || strings.TrimSpace(string(runes[end:])) == "" {
				continue
			}
			d := end - (len(runes) - end)
			if d < 0 {
				d = -d
			}
			if d < distance {
				best, distance = end, d
			}
		}
		if best > 0 {
			return strings.TrimSpace(string(runes[:best])), strings.TrimSpace(string(runes[best:]))
		}
	}
	if len(runes) < 2 {
		return string(runes), ""
	}
	return string(runes[:len(runes)/2]), string(runes[len(runes)/2:])
}

func speechSentences(text string) []string {
	runes := []rune(text)
	var out []string
	start := 0
	for i := 0; i < len(runes); i++ {
		end, stop := speechSentenceEnd(runes, i)
		if runes[i] == '\n' || runes[i] == '\r' {
			end, stop = i+1, true
		}
		if stop {
			if part := strings.TrimSpace(string(runes[start:end])); part != "" {
				out = append(out, part)
			}
			start, i = end, end-1
		}
	}
	if part := strings.TrimSpace(string(runes[start:])); part != "" {
		out = append(out, part)
	}
	return out
}

func speechSentenceEnd(runes []rune, i int) (int, bool) {
	r := runes[i]
	if !strings.ContainsRune(".!?…。！？", r) {
		return 0, false
	}
	if r == '.' {
		start := i
		for start > 0 && !unicode.IsSpace(runes[start-1]) {
			start--
		}
		word := strings.ToLower(strings.TrimLeft(string(runes[start:i]), "\"'([{“‘"))
		switch word {
		case "dr", "mr", "mrs", "ms", "prof", "st", "sr", "jr", "vs", "e.g", "i.e", "z.b", "d.h", "bzw", "u.a":
			return 0, false
		}
		if len(word) == 1 && word[0] >= 'a' && word[0] <= 'z' {
			return 0, false
		}
	}
	end := i + 1
	for end < len(runes) && strings.ContainsRune(".!?…。！？\"'”’)]}」』»,", runes[end]) {
		end++
	}
	fullWidth := strings.ContainsRune("。！？", r)
	return end, fullWidth || end == len(runes) || unicode.IsSpace(runes[end])
}
