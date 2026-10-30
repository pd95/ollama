# Qwen3-TTS native speech

CustomVoice supports the supplied 0.6B affine 4/8-bit and 1.7B bf16 checkpoints.
Base supports the supplied 0.6B/1.7B bf16 checkpoints with per-request reference
audio and a transcript. VoiceDesign is unsupported. Runtime inference uses Go/MLX;
Python MLX-Audio 0.6.2 is the compatibility reference, with its MIT notice in
`MLX_AUDIO_LICENSE`. The Base codec encoder follows its Mimi/SEANet components.

Base receives mono PCM16 WAV at 16–48 kHz, at most 15 seconds, through
`reference_audio: {"format":"wav","data":"<base64>"}` plus `reference_text`.
The shared sinc resampler produces 24 kHz input; Base needs at least 1280 samples
at that rate for its reflected TDNN context. Transcript and target text are each
limited to 4096 characters. Base rejects preset voices and style instructions;
CustomVoice rejects reference audio. Both variants support English/German via
their language conditioning and optional request seeds; speed changes are rejected.

Speaker embeddings, reference codec IDs, transcript conditioning and generation
caches belong to one request. Target text is chunked without dropping words.
The decoder prepends reference codes and trims their exact sample duration from
each target waveform. Reference profiles are never retained on the model.
The 512-frame target-utterance limit reports an error if EOS has not occurred.

The text planner groups complete sentences within a conservative budget of
160 Unicode code points and 160 text tokens. Oversized sentences prefer clauses,
then words, then Unicode code points. Framing/reference conditioning and decoder
state remain model-specific; changing text grouping does not carry generation
caches across utterances or relax the 512-frame cap.

`TestBaseReference` compares mels, ECAPA-TDNN blocks, speaker pooling/embedding,
SEANet, encoder attention, downsampling, reference codes, ICL prompt/talker and
fixed-code decoded audio against pinned fixtures when source/reference environment
variables are set. Keep channel/time layouts through reductions: materializing an
otherwise identical transposed feature can change float32 reduction rounding.
Pure Go tests also cover request admission and supported configurations.
