// Package english supplies an offline English frontend for Kokoro. Pronunciations
// come from attributed Misaki dictionaries; context and fallback rules are native Go.
package english

import (
	"compress/gzip"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

//go:embed data/*.json.gz
var dictionaries embed.FS

type entry struct {
	text string
	tags map[string]string
}
type lexicon map[string]entry

var (
	american = sync.OnceValues(func() (lexicon, error) { return load("us") })
	british  = sync.OnceValues(func() (lexicon, error) { return load("gb") })
)

func load(locale string) (lexicon, error) {
	result := lexicon{}
	for _, kind := range []string{"silver", "gold"} {
		f, err := dictionaries.Open("data/" + locale + "_" + kind + ".json.gz")
		if err != nil {
			return nil, err
		}
		compressed, err := gzip.NewReader(f)
		if err != nil {
			f.Close()
			return nil, err
		}
		raw, err := io.ReadAll(compressed)
		compressed.Close()
		f.Close()
		if err != nil {
			return nil, err
		}
		var rows map[string]json.RawMessage
		if err = json.Unmarshal(raw, &rows); err != nil {
			return nil, err
		}
		for key, value := range rows {
			var e entry
			if len(value) > 0 && value[0] == '"' {
				err = json.Unmarshal(value, &e.text)
			} else {
				err = json.Unmarshal(value, &e.tags)
			}
			if err != nil {
				return nil, fmt.Errorf("English dictionary %q: %w", key, err)
			}
			result[key] = e
		}
	}
	return result, nil
}

var (
	tokens        = regexp.MustCompile(`[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?(?:st|nd|rd|th)?|[\p{L}]+(?:['’][\p{L}]+)*|[^\s]`)
	abbreviations = map[string]string{"dr": "doctor", "mr": "mister", "mrs": "missus", "ms": "miss", "prof": "professor", "st": "saint", "vs": "versus"}
)

// Phonemes preserves punctuation and expands numbers before dictionary lookup.
// Unknown Latin words use conservative grapheme rules, never silent omission.
func Phonemes(text string, gb bool) (string, error) {
	var dictionary lexicon
	var err error
	if gb {
		dictionary, err = british()
	} else {
		dictionary, err = american()
	}
	if err != nil {
		return "", err
	}
	items := tokens.FindAllString(strings.ReplaceAll(text, "’", "'"), -1)
	var expanded []string
	for i := 0; i < len(items); i++ {
		word := items[i]
		if word == "$" || word == "£" || word == "€" {
			if i+1 >= len(items) || items[i+1][0] < '0' || items[i+1][0] > '9' {
				return "", fmt.Errorf("currency requires an amount")
			}
			amount := strings.SplitN(strings.ReplaceAll(items[i+1], ",", ""), ".", 2)
			expanded = append(expanded, numberWords(amount[0])...)
			unit := map[string]string{"$": "dollar", "£": "pound", "€": "euro"}[word]
			if amount[0] != "1" {
				unit += "s"
			}
			expanded = append(expanded, unit)
			if len(amount) == 2 {
				if len(amount[1]) != 2 {
					return "", fmt.Errorf("currency fraction must have two digits")
				}
				if amount[1] != "00" {
					expanded = append(expanded, "and")
					expanded = append(expanded, numberWords(strings.TrimLeft(amount[1], "0"))...)
					cent := "cents"
					if amount[1] == "01" {
						cent = "cent"
					}
					expanded = append(expanded, cent)
				}
			}
			i++
			continue
		}
		if name := abbreviations[strings.ToLower(word)]; name != "" && i+1 < len(items) && items[i+1] == "." {
			expanded = append(expanded, name)
			i++
			continue
		}
		if word[0] >= '0' && word[0] <= '9' {
			expanded = append(expanded, numberWords(word)...)
			continue
		}
		switch word {
		case "%":
			word = "percent"
		case "&":
			word = "and"
		case "+":
			word = "plus"
		case "@":
			word = "at"
		case "/":
			word = "slash"
		case "-", "—", "–":
			word = ","
		}
		expanded = append(expanded, word)
	}
	var out strings.Builder
	needSpace := false
	for i, word := range expanded {
		if strings.Contains(";:,.!?…\"()“”", word) && len([]rune(word)) == 1 {
			out.WriteString(word)
			needSpace = word != "(" && word != "“"
			continue
		}
		phonemes := pronounce(dictionary, expanded, i, gb)
		if phonemes == "" {
			return "", fmt.Errorf("cannot pronounce English token %q", word)
		}
		if needSpace {
			out.WriteByte(' ')
		}
		out.WriteString(strings.ReplaceAll(phonemes, "ɾ", "T"))
		needSpace = true
	}
	return strings.TrimSpace(out.String()), nil
}

func pronounce(d lexicon, words []string, index int, gb bool) string {
	word := words[index]
	lower := strings.ToLower(word)
	previous, next := "", ""
	if index > 0 {
		previous = strings.ToLower(words[index-1])
	}
	if index+1 < len(words) {
		next = strings.ToLower(words[index+1])
	}
	if lower == "a" {
		return "ɐ"
	}
	if lower == "an" {
		return "ɐn"
	}
	if lower == "the" {
		if strings.HasPrefix(plainLookup(d, next, ""), "ˈ") {
			next = strings.TrimPrefix(plainLookup(d, next, ""), "ˈ")
		} else {
			next = plainLookup(d, next, "")
		}
		if next != "" && strings.ContainsRune("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ", []rune(next)[0]) {
			return "ði"
		}
		return "ðə"
	}
	if lower == "to" {
		return "tə"
	}
	tag := ""
	if strings.Contains(" to can could will would should must may might please ", " "+previous+" ") {
		tag = "VERB"
	}
	if lower == "live" && strings.Contains(" i we they you ", " "+previous+" ") {
		tag = "VERB"
	}
	if lower == "read" && (previous == "had" || previous == "have" || previous == "has" || previous == "yesterday" || next == "yesterday") {
		tag = "VBD"
	}
	if lower == "lead" && (next == "pipe" || next == "pipes" || next == "pencil" || next == "metal") {
		return "lˈɛd"
	}
	if p := plainLookup(d, word, tag); p != "" {
		return p
	}
	if p := plainLookup(d, lower, tag); p != "" {
		return p
	}
	// Initialisms use dictionary letter names; unknown ordinary words use LTS.
	if word == strings.ToUpper(word) && len(word) > 1 {
		var spelled strings.Builder
		for _, r := range word {
			p := plainLookup(d, string(r), "")
			if p == "" {
				return ""
			}
			spelled.WriteString(p)
		}
		return spelled.String()
	}
	if strings.HasSuffix(lower, "'s") {
		if p := plainLookup(d, strings.TrimSuffix(lower, "'s"), ""); p != "" {
			last := []rune(p)[len([]rune(p))-1]
			switch {
			case strings.ContainsRune("ptkfθ", last):
				return p + "s"
			case strings.ContainsRune("szʃʒʧʤ", last):
				return p + "ᵻz"
			default:
				return p + "z"
			}
		}
	}
	return fallback(lower, gb)
}

func plainLookup(d lexicon, word, tag string) string {
	e, ok := d[word]
	if !ok {
		return ""
	}
	if e.text != "" {
		return e.text
	}
	if p := e.tags[tag]; p != "" {
		return p
	}
	return e.tags["DEFAULT"]
}

func fallback(word string, gb bool) string {
	var ascii strings.Builder
	for _, r := range norm.NFD.String(word) {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		if r == '\'' {
			continue
		}
		if r < 'a' || r > 'z' {
			return ""
		}
		ascii.WriteRune(r)
	}
	word = ascii.String()
	if word == "" {
		return ""
	}
	var out strings.Builder
	for i := 0; i < len(word); {
		remaining := word[i:]
		p := ""
		advance := 1
		for _, rule := range [][2]string{{"tion", "ʃən"}, {"sion", "ʒən"}, {"tch", "ʧ"}, {"dge", "ʤ"}, {"igh", "I"}, {"ch", "ʧ"}, {"sh", "ʃ"}, {"th", "θ"}, {"ph", "f"}, {"ng", "ŋ"}, {"qu", "kw"}, {"ee", "i"}, {"ea", "i"}, {"oo", "u"}, {"ai", "A"}, {"ay", "A"}, {"oi", "Y"}, {"oy", "Y"}, {"ow", "W"}, {"ou", "W"}} {
			if strings.HasPrefix(remaining, rule[0]) {
				p = rule[1]
				advance = len(rule[0])
				break
			}
		}
		if p == "" {
			c := word[i]
			switch c {
			case 'a':
				p = "æ"
			case 'e':
				if i == len(word)-1 && i > 1 {
					p = ""
				} else {
					p = "ɛ"
				}
			case 'i':
				p = "ɪ"
			case 'o':
				if gb {
					p = "ɒ"
				} else {
					p = "ɑ"
				}
			case 'u':
				p = "ʌ"
			case 'y':
				p = "i"
			case 'c':
				p = "k"
				if i+1 < len(word) && strings.ContainsRune("eiy", rune(word[i+1])) {
					p = "s"
				}
			case 'g':
				p = "ɡ"
			case 'j':
				p = "ʤ"
			case 'r':
				p = "ɹ"
			case 'x':
				p = "ks"
			default:
				p = string(c)
			}
			// Basic consonant-vowel-consonant-e long vowels.
			if i+2 < len(word) && i+3 == len(word) && word[i+2] == 'e' && !strings.ContainsRune("aeiouy", rune(word[i+1])) {
				if long := map[byte]string{'a': "A", 'e': "i", 'i': "I", 'o': "O", 'u': "u"}[c]; long != "" {
					p = long
					if gb && c == 'o' {
						p = "Q"
					}
				}
			}
			if i > 0 && c == word[i-1] && !strings.ContainsRune("aeiouy", rune(c)) {
				p = ""
			}
		}
		out.WriteString(p)
		i += advance
	}
	phonemes := out.String()
	for index, r := range phonemes {
		if strings.ContainsRune("AIOQWYaiuæɑɒɔəɛɜɪʊʌ", r) {
			return phonemes[:index] + "ˈ" + phonemes[index:]
		}
	}
	return phonemes
}

var (
	ones = []string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"}
	tens = []string{"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
)

func numberWords(raw string) []string {
	raw = strings.ReplaceAll(raw, ",", "")
	ordinal := false
	for _, suffix := range []string{"st", "nd", "rd", "th"} {
		if strings.HasSuffix(raw, suffix) {
			raw = strings.TrimSuffix(raw, suffix)
			ordinal = true
			break
		}
	}
	if parts := strings.SplitN(raw, ".", 2); len(parts) == 2 {
		out := numberWords(parts[0])
		out = append(out, "point")
		for _, r := range parts[1] {
			out = append(out, ones[r-'0'])
		}
		return out
	}
	n, err := strconv.ParseUint(raw, 10, 64)
	if raw == "" {
		return []string{"zero"}
	}
	if err != nil || len(raw) > 12 || (len(raw) > 1 && raw[0] == '0') {
		var out []string
		for _, r := range raw {
			if r >= '0' && r <= '9' {
				out = append(out, ones[r-'0'])
			}
		}
		return out
	}
	var speak func(uint64) []string
	speak = func(n uint64) []string {
		if n < 20 {
			return []string{ones[n]}
		}
		if n < 100 {
			out := []string{tens[n/10]}
			if n%10 != 0 {
				out = append(out, ones[n%10])
			}
			return out
		}
		for _, scale := range []struct {
			n    uint64
			name string
		}{{1e9, "billion"}, {1e6, "million"}, {1000, "thousand"}, {100, "hundred"}} {
			if n >= scale.n {
				out := append(speak(n/scale.n), scale.name)
				if n%scale.n != 0 {
					out = append(out, speak(n%scale.n)...)
				}
				return out
			}
		}
		return nil
	}
	out := speak(n)
	if ordinal {
		last := len(out) - 1
		irregular := map[string]string{"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
		if word := irregular[out[last]]; word != "" {
			out[last] = word
		} else if strings.HasSuffix(out[last], "y") {
			out[last] = strings.TrimSuffix(out[last], "y") + "ieth"
		} else {
			out[last] += "th"
		}
	}
	return out
}
