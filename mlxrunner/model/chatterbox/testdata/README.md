# Provider-v3 HiFT activation controls

`provider-hift-en.wav` and `provider-hift-de.wav` are 24 kHz mono PCM16
fixed-input controls from the original v3 checkpoint
`ResembleAI/chatterbox@5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`.
They hold the captured `vocoder_stage_2` inputs and imported weights fixed,
changing only the final pre-convolution activation from 0.1 to 0.01.
No fade is applied. PCM uses the runner's 32767 scaling and round-away rule.

The activation follows the provider HiFT decoder at source commit
`5de7a54aa4e5e2baadb0182dde554908b48b85c2`, `src/chatterbox/models/s3gen/hifigan.py`.
All earlier native/MLX-Audio fixtures remain separate. These samples establish
the selected activation on fixed inputs, not whole-provider neural fidelity.
The native check permits at most one PCM code point of quantization/float error.
