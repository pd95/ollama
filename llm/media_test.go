package llm

import "testing"

func TestAudioFormatAndMediaKind(t *testing.T) {
	for _, tc := range []struct {
		name, format string
		data         []byte
		kind         MediaKind
	}{
		{name: "wav", format: "wav", data: []byte("RIFF\x00\x00\x00\x00WAVE"), kind: MediaKindAudio},
		{name: "mp3 id3", format: "mp3", data: []byte("ID3\x04\x00\x00"), kind: MediaKindAudio},
		{name: "mp3 frame", format: "mp3", data: []byte{0xff, 0xfb, 0x90, 0x64}, kind: MediaKindAudio},
		{name: "flac", data: []byte("fLaC"), kind: MediaKindUnknown},
		{name: "garbage", data: []byte("not media"), kind: MediaKindUnknown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			format, ok := AudioFormat(tc.data)
			if format != tc.format || ok != (tc.format != "") {
				t.Fatalf("AudioFormat() = %q, %v", format, ok)
			}
			if got := DetectMediaKind(tc.data); got != tc.kind {
				t.Fatalf("DetectMediaKind() = %q, want %q", got, tc.kind)
			}
		})
	}
}
