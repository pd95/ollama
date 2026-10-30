# Native Voxtral TTS 2603

Go/MLX port of the MLX-Audio 0.6.2 Voxtral TTS implementation:
https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/voxtral_tts

Source implementation copyright (c) 2024 Prince Canuma, MIT; retained in
LICENSE.MLX-Audio. The port uses the same fixed model mathematics. Model weights
are separate local inputs, with their own model-card license; they are not
included in this source package.
The [speech notice](../SPEECH_NOTICE) also credits the vendored MLX-LM transformer
source and describes native adaptations. Full notices/licenses ship with the
runtime payload independently of user-imported models.

Config/ validates supplied bf16 and affine 4/6-bit geometry. The importer freezes
Tekken data and namespaces seven English/German voice packs. Runtime speech uses
ordinary-text byte-rank BPE, a causal Mistral backbone with interleaved RoPE,
seven-step classifier-free acoustic flow and the causal ALiBi codec.

Text preprocessing follows the pinned demonstration path, including Markdown,
parentheticals and safe number/currency normalization. Literal control-token
spellings in speech input remain ordinary text. English/German only are initially
advertised; voice locale must match any explicit language. Speed is fixed at 1
and style instructions are rejected.

A request-local CPU RNG supplies flow noise. It intentionally does not promise
identical draws to Python MLX. Paired fixed-noise operators and fixed-code waveform
checks qualify the mathematics independently of sampled utterances. Short native
regression WAVs do not replace human intelligibility review.

Streaming decodes 25 new frames with 32 context frames to cover the composed
convolution/attention receptive field. All 1920 samples per codec frame are
preserved. Same-shaped native/reference windows match exactly; full-vs-chunked
matrix shapes can change float32 rounding by a few PCM quantization units, checked separately against the measured
reference baseline (four-unit maximum, half-unit RMS).
Utterances are bounded to 160 characters and 256 frames, with cancellation checks
between frames/chunks and an error if EOS is absent at the bound.
