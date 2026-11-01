package server

import (
	"unicode"
	"unicode/utf8"

	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

// asrPublicUpdate adds presentation metadata without changing native replay
// output. Agreement can retreat on later corrections; only Final is immutable.
type asrPublicUpdate struct {
	asrstream.Update
	StableText string `json:"stable_text"`
	DraftText  string `json:"draft_text"`
}

// asrTextStability retains only the last three successful hypotheses from one
// detected language. Callers serialize access and supply validated UTF-8 text.
type asrTextStability struct {
	language string
	history  [3]string
	count    int
}

func (s *asrTextStability) present(update asrstream.Update) asrPublicUpdate {
	result := asrPublicUpdate{Update: update, DraftText: update.Text}
	if update.Final {
		result.StableText, result.DraftText = update.Text, ""
		*s = asrTextStability{}
		return result
	}
	if s.language != update.Language {
		*s = asrTextStability{language: update.Language}
	}
	if s.count < len(s.history) {
		s.history[s.count] = update.Text
		s.count++
	} else {
		s.history[0], s.history[1], s.history[2] = s.history[1], s.history[2], update.Text
	}
	if s.count == len(s.history) {
		prefix := commonASRPrefix(commonASRPrefix(s.history[0], s.history[1]), s.history[2])
		end := 0
		for offset, r := range prefix {
			if unicode.IsSpace(r) {
				end = offset + utf8.RuneLen(r)
			}
		}
		result.StableText, result.DraftText = update.Text[:end], update.Text[end:]
	}
	return result
}

func commonASRPrefix(a, b string) string {
	end := 0
	for end < min(len(a), len(b)) && a[end] == b[end] {
		end++
	}
	// Equal leading bytes can stop inside otherwise distinct Unicode runes.
	for end > 0 && end < len(a) && !utf8.RuneStart(a[end]) {
		end--
	}
	return a[:end]
}
