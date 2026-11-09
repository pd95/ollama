package chatterbox

import (
	"context"
	"encoding/binary"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestSpeechValidation(t *testing.T) {
	data := make([]byte, 44+4800)
	copy(data, "RIFF")
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	copy(data[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(data[16:], 16)
	binary.LittleEndian.PutUint16(data[20:], 1)
	binary.LittleEndian.PutUint16(data[22:], 1)
	binary.LittleEndian.PutUint32(data[24:], 24000)
	binary.LittleEndian.PutUint32(data[28:], 48000)
	binary.LittleEndian.PutUint16(data[32:], 2)
	binary.LittleEndian.PutUint16(data[34:], 16)
	copy(data[36:], "data")
	binary.LittleEndian.PutUint32(data[40:], 4800)
	m := &Model{}
	request := llm.SpeechRequest{Input: "Hello", ReferenceAudio: &llm.ReferenceAudio{Format: "wav", Data: data}}
	if err := m.ValidateSpeechRequest(request); err != nil {
		t.Fatal(err)
	}
	for _, alter := range []func(*llm.SpeechRequest){func(r *llm.SpeechRequest) { r.ReferenceAudio = nil }, func(r *llm.SpeechRequest) { r.Voice = "Ryan" }, func(r *llm.SpeechRequest) { r.ReferenceText = "transcript" }, func(r *llm.SpeechRequest) { r.Instructions = "whisper" }, func(r *llm.SpeechRequest) { r.Language = "fr" }, func(r *llm.SpeechRequest) { r.Input = " " }} {
		r := request
		alter(&r)
		if m.ValidateSpeechRequest(r) == nil {
			t.Fatal("accepted unsupported options")
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := m.Speak(ctx, request, func([]float32) error { t.Fatal("emitted after cancellation"); return nil }); err != context.Canceled {
		t.Fatal(err)
	}
}

func TestChunking(t *testing.T) {
	text := strings.Repeat("Hallo! Grüße aus Köln. ", 40)
	compact := func(s string) string { return strings.Join(strings.Fields(s), "") }
	chunks := textChunks(text)
	for _, s := range chunks {
		if len([]rune(s)) > 160 {
			t.Fatal("chunk exceeds bound")
		}
	}
	if compact(strings.Join(chunks, "")) != compact(text) {
		t.Fatal("text lost or repeated")
	}
}
