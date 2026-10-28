package config

import "testing"

func TestPresetLanguages(t *testing.T) {
	voices := VoiceNames()
	if len(voices) != 20 {
		t.Fatalf("got %d voices", len(voices))
	}
	languages := map[string]bool{}
	for _, voice := range voices {
		language := VoiceLanguage(voice)
		languages[language] = true
		got, lang, err := ResolveVoice(voice, "")
		if err != nil || got != voice || lang != language {
			t.Fatalf("%s: %s %s %v", voice, got, lang, err)
		}
		if _, _, err := ResolveVoice(voice, language); err != nil {
			t.Fatal(err)
		}
		if _, _, err := ResolveVoice(voice, "ja"); err == nil {
			t.Fatal("accepted unsupported language")
		}
		if language != "en" {
			if _, _, err := ResolveVoice(voice, "en"); err == nil {
				t.Fatal("accepted mismatched voice/language")
			}
		}
	}
	if len(languages) != 9 || languages[""] {
		t.Fatalf("languages: %v", languages)
	}
	for alias, language := range map[string]string{"French": "fr", "pt-BR": "pt", "hi_IN": "hi", "Arabic": "ar", "en-GB": "en", "de-DE": "de"} {
		voice, lang, err := ResolveVoice("", alias)
		if err != nil || lang != language || VoiceLanguage(voice) != language {
			t.Fatalf("%s: %s %s %v", alias, voice, lang, err)
		}
	}
	if _, _, err := ResolveVoice("unknown", ""); err == nil {
		t.Fatal("accepted unknown voice")
	}
}
