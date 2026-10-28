package config

import (
	"fmt"
	"maps"
	"slices"
	"strings"
)

// Preset languages are defined by the provider's voice catalog, not inferred
// from the request text. The language belongs to the conditioning recording.
var presetLanguages = map[string]string{
	"casual_female": "en", "casual_male": "en", "cheerful_female": "en",
	"neutral_female": "en", "neutral_male": "en",
	"ar_male": "ar", "de_female": "de", "de_male": "de",
	"es_female": "es", "es_male": "es", "fr_female": "fr", "fr_male": "fr",
	"hi_female": "hi", "hi_male": "hi", "it_female": "it", "it_male": "it",
	"nl_female": "nl", "nl_male": "nl", "pt_female": "pt", "pt_male": "pt",
}

func VoiceNames() []string              { return slices.Sorted(maps.Keys(presetLanguages)) }
func VoiceLanguage(voice string) string { return presetLanguages[voice] }

// ResolveVoice keeps voice selection explicit. A language without a voice
// chooses that language's male preset; omission of both preserves casual_male.
func ResolveVoice(voice, language string) (string, string, error) {
	language = strings.ToLower(strings.ReplaceAll(language, "_", "-"))
	switch language {
	case "", "auto":
		language = ""
	case "en", "english", "en-us", "en-gb":
		language = "en"
	case "de", "german", "de-de":
		language = "de"
	case "fr", "french", "fr-fr", "fr-ca":
		language = "fr"
	case "es", "spanish", "es-es", "es-mx":
		language = "es"
	case "pt", "portuguese", "pt-pt", "pt-br":
		language = "pt"
	case "it", "italian", "it-it":
		language = "it"
	case "nl", "dutch", "nl-nl", "nl-be":
		language = "nl"
	case "ar", "arabic", "ar-sa", "ar-eg":
		language = "ar"
	case "hi", "hindi", "hi-in":
		language = "hi"
	default:
		return "", "", fmt.Errorf("Voxtral TTS supports ar, de, en, es, fr, hi, it, nl and pt")
	}
	voice = strings.ToLower(voice)
	if voice == "" {
		voice = "casual_male"
		if language != "" && language != "en" {
			voice = language + "_male"
		}
	}
	actual := VoiceLanguage(voice)
	if actual == "" {
		return "", "", fmt.Errorf("unknown Voxtral TTS voice %q", voice)
	}
	if language != "" && actual != language {
		return "", "", fmt.Errorf("Voxtral TTS language must match its preset voice (%s)", actual)
	}
	return voice, actual, nil
}
