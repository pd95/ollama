package voxtraltts

import (
	"fmt"
	"html"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

const white = `[\p{Z}\x{85}\x{9}-\x{D}]`

var markdownRules = []struct {
	pattern     *regexp.Regexp
	replacement string
}{
	{regexp.MustCompile("(?s)```.*?```|~~~.*?~~~"), " Code example omitted. "},
	{regexp.MustCompile(`(?i)<br\s*/?>`), " "},
	{regexp.MustCompile(`\[([^\]]+)\]\(([^)]*)\)`), "$1"},
	{regexp.MustCompile(`<https?://[^>]+>`), "link"},
	{regexp.MustCompile(`https?://[^\p{Z}\x{85}\x{9}-\x{D}]+`), "link"},
	{regexp.MustCompile("`([^`]+)`"), "$1"},
	{regexp.MustCompile(`(?m)^` + white + `{0,3}#{1,6}` + white + `*`), ""},
	{regexp.MustCompile(`(?m)^` + white + `*[-*+]` + white + `+`), ""},
	{regexp.MustCompile(`(?m)^` + white + `*[0-9]{1,4}[.)]` + white + `+`), ""},
	{regexp.MustCompile(`(?m)^` + white + `{0,3}>` + white + `?`), ""},
	{regexp.MustCompile(`</?[a-zA-Z][^>]*>`), " "},
}

var (
	imageLink            = regexp.MustCompile(`!\[([^\]]*)\]\(([^)]*)\)`)
	parenthetical        = regexp.MustCompile(white + `*\(([^()]+)\)` + white + `*`)
	numericParenthetical = regexp.MustCompile(`^\s*[$€£¥₹¢]?\s*[+−–-]?[0-9]+(?:[.,][0-9]+)?\s*%?\s*$`)
	parentheticalRules   = []struct {
		pattern     *regexp.Regexp
		replacement string
	}{
		{regexp.MustCompile(`(?i)\bN\s*/\s*A\b`), "not available"},
		{regexp.MustCompile(`\s*(?:\+/-|±)\s*`), " plus or minus "},
		{regexp.MustCompile(`\s*>=\s*`), " greater than or equal to "},
		{regexp.MustCompile(`\s*<=\s*`), " less than or equal to "},
		{regexp.MustCompile(`\s*>\s*`), " greater than "},
		{regexp.MustCompile(`\s*<\s*`), " less than "},
		{regexp.MustCompile(`\s*~\s*`), " about "},
		{regexp.MustCompile(`#\s*([0-9]+)\b`), "number $1"},
		{regexp.MustCompile(`(?i)([0-9]+(?:\.[0-9]+)?)\s*ms\b`), "${1} milliseconds"},
		{regexp.MustCompile(`(?i)([0-9]+(?:\.[0-9]+)?)\s*h\b`), "${1} hours"},
		{regexp.MustCompile(`([0-9]+(?:\.[0-9]+)?)\s*%`), "${1} percent"},
		{regexp.MustCompile(`\s*\+\s*`), " plus "},
		{regexp.MustCompile(`\s*=\s*`), " equals "},
	}
)

var (
	punctuationRuns = regexp.MustCompile(`!{2,}|\?{2,}`)
	dots            = regexp.MustCompile(`\.{3,}`)
	hyphens         = regexp.MustCompile(`[-–−]{2,}`)
	numberToken     = regexp.MustCompile(`[$£€¥₹¢][0-9][0-9,]*(?:\.[0-9]+)?|-?[0-9][0-9,]*(?:\.[0-9]+)?\b`)
)

// normalizeText mirrors the pinned MLX-Audio demonstration preprocessing before
// Tekken encoding. It intentionally leaves ordinary small numbers for the LM.
func normalizeText(text string) (string, error) {
	text = markdownRules[0].pattern.ReplaceAllString(text, markdownRules[0].replacement)
	text = html.UnescapeString(text)
	text = markdownRules[1].pattern.ReplaceAllString(text, markdownRules[1].replacement)
	text = imageLink.ReplaceAllStringFunc(text, func(match string) string {
		alt := strings.TrimSpace(imageLink.FindStringSubmatch(match)[1])
		if alt == "" {
			return "image"
		}
		return alt
	})
	for _, rule := range markdownRules[2:] {
		text = rule.pattern.ReplaceAllString(text, rule.replacement)
	}
	text = strings.Map(func(r rune) rune {
		if r == 0x200b || (r >= 0x200e && r <= 0x200f) || (r >= 0x2028 && r <= 0x2029) || (r >= 0x2060 && r <= 0x2064) || r == 0xad || r == 0x180e || r == 0xfeff || (r >= 0xfff9 && r <= 0xfffb) {
			return -1
		}
		if unicode.IsSpace(r) {
			return ' '
		}
		return r
	}, text)
	text = normalizeNumbers(text)
	text = parenthetical.ReplaceAllStringFunc(text, func(match string) string {
		inner := strings.TrimSpace(parenthetical.FindStringSubmatch(match)[1])
		if inner == "" || numericParenthetical.MatchString(inner) {
			return match
		}
		for _, rule := range parentheticalRules {
			inner = rule.pattern.ReplaceAllString(inner, rule.replacement)
		}
		return "—" + strings.Join(strings.Fields(inner), " ") + "—"
	})
	text = strings.NewReplacer("‐", "-", "‑", "-").Replace(text)
	text = dots.ReplaceAllString(text, "...")
	text = punctuationRuns.ReplaceAllStringFunc(text, func(s string) string { return s[:1] })
	text = hyphens.ReplaceAllString(text, "—")
	runes := []rune(text)
	for i, r := range runes {
		if i == 0 || i+1 >= len(runes) || !strings.ContainsRune("-–−", r) || !unicode.IsSpace(runes[i-1]) || !unicode.IsSpace(runes[i+1]) {
			continue
		}
		next := i + 1
		for next < len(runes) && unicode.IsSpace(runes[next]) {
			next++
		}
		if next < len(runes) && !unicode.IsDigit(runes[next]) && !strings.ContainsRune("$€£¥₹¢", runes[next]) {
			runes[i] = '—'
		}
	}
	text = strings.Join(strings.Fields(string(runes)), " ")
	if text == "" {
		return "", fmt.Errorf("speech input is empty after normalization")
	}
	end := []rune(text)
	if !strings.ContainsRune(".!?…؟。！？", end[len(end)-1]) {
		text += "."
	}
	return text, nil
}

var (
	small   = []string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"}
	decades = []string{"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
	scales  = []string{"", "thousand", "million", "billion", "trillion", "quadrillion"}
)

func subThousand(n uint64, and bool) string {
	var words []string
	if n >= 100 {
		words = append(words, small[n/100], "hundred")
		if and && n%100 != 0 {
			words = append(words, "and")
		}
	}
	n %= 100
	if n > 0 && n < 20 {
		words = append(words, small[n])
	} else if n >= 20 {
		word := decades[n/10]
		if n%10 != 0 {
			word += "-" + small[n%10]
		}
		words = append(words, word)
	}
	if len(words) == 0 {
		return "zero"
	}
	return strings.Join(words, " ")
}

func integerWords(n uint64, and bool) string {
	if n == 0 {
		return "zero"
	}
	var groups []uint64
	for n > 0 {
		groups = append(groups, n%1000)
		n /= 1000
	}
	var words []string
	for i := len(groups) - 1; i >= 0; i-- {
		if groups[i] != 0 {
			piece := subThousand(groups[i], and && i == 0 && len(groups) > 1)
			if i > 0 {
				piece += " " + scales[i]
			}
			words = append(words, piece)
		}
	}
	return strings.Join(words, " ")
}

func decimalWords(whole uint64, fraction string) string {
	digits := make([]string, 0, len(fraction))
	for _, r := range fraction {
		digits = append(digits, small[r-'0'])
	}
	return integerWords(whole, false) + " point " + strings.Join(digits, "-")
}

var units = map[rune][4]string{'$': {"dollar", "dollars", "cent", "cents"}, '£': {"pound", "pounds", "penny", "pence"}, '€': {"euro", "euros", "cent", "cents"}, '¥': {"yen", "yen", "", ""}, '₹': {"rupee", "rupees", "paise", "paise"}, '¢': {"cent", "cents", "", ""}}

func normalizeNumbers(text string) string {
	matches := numberToken.FindAllStringIndex(text, -1)
	onlyDollars := true
	currencies := 0
	for _, loc := range matches {
		r := []rune(text[loc[0]:loc[1]])[0]
		if _, ok := units[r]; ok {
			currencies++
			if r != '$' {
				onlyDollars = false
			}
		}
	}
	dollarDone := false
	var out strings.Builder
	offset := 0
	for _, loc := range matches {
		token := text[loc[0]:loc[1]]
		replacement := token
		first := []rune(token)[0]
		u, currency := units[first]
		amount := token
		if currency {
			amount = strings.TrimPrefix(amount, string(first))
		}
		parts := strings.SplitN(amount, ".", 2)
		whole, err := strconv.ParseUint(strings.ReplaceAll(strings.TrimPrefix(parts[0], "-"), ",", ""), 10, 64)
		if err == nil && whole < 1e18 {
			if currency {
				major := u[1]
				if whole == 1 {
					major = u[0]
				}
				short := onlyDollars && currencies > 1 && dollarDone
				if first == '$' {
					dollarDone = true
				}
				fraction := ""
				if len(parts) > 1 {
					fraction = parts[1]
				}
				if len(fraction) > 2 || ((first == '¥' || first == '¢') && strings.Trim(fraction, "0") != "") {
					replacement = decimalWords(whole, fraction) + " " + major
				} else {
					minor := uint64(0)
					if fraction != "" {
						minor, _ = strconv.ParseUint((fraction + "00")[:2], 10, 64)
					}
					switch {
					case short:
						replacement = integerWords(whole, false)
						if minor != 0 {
							replacement += " " + integerWords(minor, false)
						}
					case minor == 0:
						replacement = integerWords(whole, true) + " " + major
					default:
						unit := u[3]
						if minor == 1 {
							unit = u[2]
						}
						replacement = integerWords(minor, false) + " " + unit
						if whole != 0 {
							replacement = integerWords(whole, true) + " " + major + " and " + replacement
						}
					}
				}
			} else if whole >= 1e6 && strings.Contains(parts[0], ",") {
				// The publisher only automatically expands large comma-grouped numbers.
				before := rune(0)
				if loc[0] > 0 {
					prefix := []rune(text[:loc[0]])
					before = prefix[len(prefix)-1]
				}
				if !unicode.IsLetter(before) && !unicode.IsNumber(before) && before != '_' {
					replacement = integerWords(whole, true)
					if len(parts) > 1 {
						replacement += " point "
						for i, r := range parts[1] {
							if i > 0 {
								replacement += "-"
							}
							replacement += small[r-'0']
						}
					}
					if strings.HasPrefix(token, "-") {
						replacement = "negative " + replacement
					}
				}
			}
		}
		out.WriteString(text[offset:loc[0]])
		out.WriteString(replacement)
		offset = loc[1]
	}
	out.WriteString(text[offset:])
	return out.String()
}
