package llm

import "context"

// SpeechRequest asks a speech model to read Input aloud.
type SpeechRequest struct {
	Input          string          `json:"input"`
	Voice          string          `json:"voice,omitempty"`
	Instructions   string          `json:"instructions,omitempty"`
	Language       string          `json:"language,omitempty"`
	Seed           *uint64         `json:"seed,omitempty"`
	Speed          *float64        `json:"speed,omitempty"`
	ReferenceAudio *ReferenceAudio `json:"reference_audio,omitempty"`
	ReferenceText  string          `json:"reference_text,omitempty"`
}

// ReferenceAudio carries a voice-cloning example. JSON encodes Data as base64;
// paths are resolved by the client, never by the server.
type ReferenceAudio struct {
	Format string `json:"format"`
	Data   []byte `json:"data"`
}

// MaxSpeechRequestBytes includes base64 audio and the bounded text fields.
const MaxSpeechRequestBytes = 4 << 20

// Speaker is implemented by runners that serve speech models. Speech streams
// 24 kHz mono 16-bit little-endian PCM to fn as it is generated.
type Speaker interface {
	Speech(ctx context.Context, req SpeechRequest, fn func(pcm []byte) error) error
}
