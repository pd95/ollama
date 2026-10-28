# Chatterbox native components

The multilingual-v3 candidate currently implements the separately supplied
S3TokenizerV2, reference voice encoder, Unicode text preparation, Perceiver
conditioning, Llama-520M T3 prefill and cached continuation. It is not registered
as a speech runtime until S3 waveform generation and import are complete.
Turbo uses a different T3/flow architecture and is a separate qualification.

The reference is MLX-Audio 0.6.2, with its MIT notice in `MLX_AUDIO_LICENSE`.
Its vendored MLX-LM Llama transformer carries Copyright © 2023–2024 Apple Inc.
Reference tests bind the exact local model/S3 checkpoints and backend externally;
no model download or Python process is used by the native components.

`TestReference` checks polyphase resampling, S3 mels/hidden/codes, voice trimming,
mels/windows, all LSTM states, speaker embeddings, text IDs, T3 conditioning,
prefill/logits and a cached next step. Set `OLLAMA_CHATTERBOX_SOURCE`,
`OLLAMA_S3_SOURCE`, and `OLLAMA_SPEECH_REFERENCE_DIR` for those checks.
The shared Slaney filters have also been checked against Qwen Base fixtures.
