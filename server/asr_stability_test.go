package server

import (
	"encoding/json"
	"testing"
	"unicode/utf8"

	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

func TestQwen3ASRStabilityAgreementAndRetreat(t *testing.T) {
	var history asrTextStability
	for i, test := range []struct{ text, stable string }{
		{"hello world one", ""},
		{"hello world two", ""},
		{"hello world three", "hello world "},
		{"hello brave four", "hello "},
		{"hello brave five", "hello "},
		{"hello brave six", "hello brave "},
	} {
		u := history.present(asrstream.Update{Revision: i + 1, Text: test.text, Language: "English"})
		if u.StableText != test.stable || u.StableText+u.DraftText != test.text || u.Revision != i+1 {
			t.Fatalf("revision %d: %+v", i+1, u)
		}
	}
}

func TestQwen3ASRStabilityExactUnicode(t *testing.T) {
	for _, test := range []struct {
		name, first, second, third, stable string
	}{
		{"unicode whitespace", "Grüße\u2003aus Köln!", "Grüße\u2003aus Koblenz!", "Grüße\u2003aus Konstanz!", "Grüße\u2003aus "},
		{"partial UTF-8 prefix", "äpfel", "äpfel", "öfen", ""},
		{"no whitespace", "日本語", "日本語", "日本語", ""},
		{"no normalization", "Café heute", "Café heute", "Cafe\u0301 heute", ""},
		{"exact whitespace", "  word\tnew", "  word\tnewer", "  word\tnews", "  word\t"},
		{"controls preserved on wire", "hello\nworld\x1b", "hello\nworld\x1b", "hello\nworld\x1b", "hello\n"},
		{"empty", "", "", "", ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			var history asrTextStability
			history.present(asrstream.Update{Text: test.first})
			history.present(asrstream.Update{Text: test.second})
			u := history.present(asrstream.Update{Text: test.third})
			if u.StableText != test.stable || u.StableText+u.DraftText != test.third || !utf8.ValidString(u.StableText) || !utf8.ValidString(u.DraftText) {
				t.Fatalf("invalid exact split: %+v", u)
			}
		})
	}
}

func TestQwen3ASRStabilityLanguageSessionAndFinal(t *testing.T) {
	var history asrTextStability
	for range 3 {
		history.present(asrstream.Update{Text: "same words ", Language: "English"})
	}
	for i := range 3 {
		u := history.present(asrstream.Update{Text: "same words ", Language: "German"})
		if (u.StableText != "") != (i == 2) {
			t.Fatalf("language history not reset, revision %d: %+v", i+1, u)
		}
	}
	u := history.present(asrstream.Update{Text: "final\u00a0text", Final: true})
	if u.StableText != u.Text || u.DraftText != "" {
		t.Fatal("final was not wholly stable", u)
	}
	u = history.present(asrstream.Update{Text: "new session ", Language: "German"})
	if u.StableText != "" || u.DraftText != u.Text {
		t.Fatal("final retained prior history", u)
	}
	var independent asrTextStability
	u = independent.present(asrstream.Update{Text: "same words ", Language: "German"})
	if u.StableText != "" {
		t.Fatal("session history crossed instances", u)
	}
}

func TestQwen3ASRStabilityOnlyPublicWire(t *testing.T) {
	backend := asrstream.Update{Revision: 1, AudioEnd: 16000, Text: "draft", Language: "English"}
	var history asrTextStability
	data, err := json.Marshal(history.present(backend))
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	if fields["stable_text"] != "" || fields["draft_text"] != "draft" || fields["text"] != "draft" || fields["revision"] != float64(1) {
		t.Fatal(fields)
	}
	data, err = json.Marshal(backend)
	if err != nil {
		t.Fatal(err)
	}
	clear(fields)
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	if _, ok := fields["stable_text"]; ok {
		t.Fatal("presentation leaked into backend update")
	}
}
