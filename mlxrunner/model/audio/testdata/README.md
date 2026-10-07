`synthetic_silence.mp3.base64` is generated test data with no external source:
ten 768-byte MPEG-1 Layer III frames, each consisting of the public MP3 frame
header `ff fb d4 c4` followed by a zero-filled payload. It contains no recorded
audio or third-party creative content. The decoded binary SHA-256 is
`2c5d72bd22f75ba5dd043f8b46d4451b3372ee4ab6fa95fad134d45724456510`.

It is stored as base64 so the fixture remains reviewable by text-only patch and
source tooling.
