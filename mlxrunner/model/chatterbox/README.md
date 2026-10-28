# Native Chatterbox multilingual v3

Native Go/MLX speech with per-request reference WAV conditioning. The imported
manifest contains v3 weights/tokenizer and the separately supplied S3TokenizerV2.
No Python process, model download or original HF directory is used at inference.
Turbo has a different T3/flow architecture and requires separate qualification.

The port covers Unicode full-case NFKD text, Llama-520M T3 with Perceiver CFG,
S3 tokenizer, voice LSTM, CAMPPlus speaker encoder, Conformer, ten-step flow
solver and HiFT vocoder. English/German are enabled. Requests require a mono
PCM16 WAV, 16–48 kHz, 0.1–15 seconds; references stay local to each request.
Presets, transcript, instructions and speed changes are rejected. Text chunks
are bounded to 160 runes and 512 generated codes with explicit EOS failure.
Cancellation is checked between generation/flow steps and waveform operations.

Request sampling uses PCG, temperature .8, min-p .05 and full-history repetition
penalty 1.2. S3Gen uses a separate fixed PCG-zero noise table. Stochastic sequences
do not claim Python RNG identity. Fixed-input neural stages are compared to
MLX-Audio 0.6.2; only inverse-FFT overlap-add has rounding differences (<4.5e-8).
The MIT notice is in `MLX_AUDIO_LICENSE`. The vendored MLX-LM Llama transformer
carries Copyright © 2023–2024 Apple Inc.

`TestReference` checks preprocessing, reference encoders, conditioning, prefill,
cached generation, flow inputs/all steps and fixed-source waveform stages.
Set `OLLAMA_CHATTERBOX_SOURCE`, `OLLAMA_S3_SOURCE` and
`OLLAMA_SPEECH_REFERENCE_DIR` to pinned inputs for native numerical checks.
