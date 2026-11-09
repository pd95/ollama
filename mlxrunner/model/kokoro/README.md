# Native Kokoro speech

Go/MLX adaptation of the Kokoro implementation in
[MLX-Audio 0.6.2](https://github.com/Blaizzy/mlx-audio/tree/v0.6.2/mlx_audio/tts/models/kokoro),
copyright Prince Canuma and contributors, under the retained
[MIT license](MLX_AUDIO_LICENSE). Native import, request lifecycle and
context/duration-aware sentence fitting are implemented for Ollama.

The self-contained [English frontend](english/README.md) uses attributed
Misaki 0.9.4 dictionaries with their [Apache-2.0 license](english/data/LICENSE).
It supports American/British English, imported preset voices and speed control;
unknown-word/context pronunciation remains approximate.

[Speech attribution](../SPEECH_NOTICE) lists the upstream sources and changes.
The build stages the notice and full license texts with the runtime payload.
Model weights and voice packs are imported user inputs and are not bundled.
