package renderers

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"golang.org/x/text/language"

	"github.com/ollama/ollama/api"
)

func TestQwen3ASRProviderPromptFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/qwen3_asr_prompt.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	p := AudioPrompterForRenderer("qwen3-asr")
	audio, err := p.AudioPrompt(AudioRequest{Source: language.English, Audio: api.ImageData{1}})
	if err != nil {
		t.Fatal(err)
	}
	got, err := RenderWithRenderer("qwen3-asr", audio.Messages, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != fixture.Text {
		t.Fatalf("rendered prompt %q != pinned provider %q", got, fixture.Text)
	}
}

func TestQwen3ASRPrompt(t *testing.T) {
	for _, source := range []language.Tag{language.Und, language.English, language.German} {
		p := AudioPrompterForRenderer("qwen3-asr")
		if p.AudioClipLimit() != 60*time.Second {
			t.Fatal("ASR must bound each complete-audio generation to 60 seconds")
		}
		prompt, err := p.AudioPrompt(AudioRequest{Task: AudioTranscribe, Source: source, Prompt: "Navi Pillay", Audio: api.ImageData{1}})
		if err != nil {
			t.Fatal(err)
		}
		got, err := RenderWithRenderer("qwen3-asr", prompt.Messages, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		want := "<|im_start|>system\nNavi Pillay<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"
		if source != language.Und {
			want += "language " + LanguageName(source) + "<asr_text>"
		}
		if got != want {
			t.Fatalf("source=%s prompt %q != provider %q", source, got, want)
		}
		for _, raw := range []string{"language German<asr_text> Hallo Welt. ", "Hallo Welt."} {
			if got := p.AudioText(AudioRequest{}, raw); got != "Hallo Welt." {
				t.Fatalf("language framing leaked: %q", got)
			}
		}
		literal := "The token <asr_text> is literal."
		if got := p.AudioText(AudioRequest{}, literal); got != literal {
			t.Fatalf("ordinary added-token text was stripped: %q", got)
		}
	}
}

func TestQwen3ASRRejectsUnsupportedTasks(t *testing.T) {
	p := AudioPrompterForRenderer("qwen3-asr")
	for _, req := range []AudioRequest{{Task: AudioTranslate}, {Task: AudioTranscribe, Source: language.French}} {
		if _, err := p.AudioPrompt(req); err == nil {
			t.Fatal("unsupported request accepted")
		}
	}
}

func TestQwen3ASRMediaMarker(t *testing.T) {
	r := &Qwen3ASRRenderer{useImgTags: true}
	prompt, err := r.AudioPrompt(AudioRequest{Source: language.German, Audio: api.ImageData{1}})
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.Render(prompt.Messages, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "<|audio_start|>[img-0]<|audio_end|>") {
		t.Fatalf("invalid indexed media marker: %q", got)
	}
}
