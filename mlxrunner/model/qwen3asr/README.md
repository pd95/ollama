# Qwen3-ASR original 0.6B complete-audio baseline

Imports the original `Qwen3ASRForConditionalGeneration` BF16 checkpoint without
changing source bytes. Runtime promotion to F32 is deliberate: the initial
independent provider qualification covers F32 execution, not BF16 performance.
Parameter memory is approximately 3.49 GiB after promotion. Shared allocator
defaults and existing speech implementations are unchanged. Initial admission
uses the validated schema to budget BF16 source plus F32 promoted tensor memory
(~5.24 GiB); after startup, live MLX allocations replace that estimate. This
is not a request activation budget or a hard RSS cap.

Reuses the existing Qwen3 decoder layers with explicit 128-dimensional heads,
16 query heads, 8 KV heads and a separate trained output projection. Audio
uses its own Whisper frontend and 18-layer bidirectional encoder; Qwen3-TTS
conditioning, codec, talker and code predictor are not reused. The original
MRoPE's three equal position axes reduce to the existing half-rotation RoPE.

The input bound is 60 seconds per generation after native mono/16 kHz
resampling, minimum 201 samples. The encoder follows the pinned provider eager
backend's global attention; FlashAttention windowing is not qualified.
Preprocessing is CPU-only and request-local. The runner's normal media ownership
and causal decoder cache interfaces are reused. The qwen3-asr renderer exposes
the existing transcription route for English, German or automatic detection.
Translation and word alignment are not implemented. Experimental PCM sessions
and fresh-window HTTP routes are described below. PrepareMedia rejects
multiple audio occurrences before decoding, including repeated raw media markers.
Importer and startup share the EOS and special-token sidecar checks.

Real-audio provider controls cover complete recordings through 44.78 seconds,
frontend/encoder outputs, full prefill logits and cached greedy IDs. Shared Go
resampling differs from provider librosa/soxr and can change transcription.
`TestQwen3ASRAudioReference` uses `OLLAMA_QWEN3_ASR_AUDIO_REFERENCE` and the source
path to run these optional native fixture checks. F32 component checks disable
TF32 in their worker only; runtime defaults remain unchanged.

`TestReference` compares original-provider frontend/encoder/decoder fixtures,
all 28 decoder layers at identical inputs, prefill states, vocabulary logits,
and greedy cached decoding. Set `OLLAMA_QWEN3_ASR_SOURCE` and
`OLLAMA_QWEN3_ASR_REFERENCE` to run it. With `OLLAMA_QWEN3_ASR_IMPORTED`, it
loads tokenizer/config/weights through that imported manifest instead; the
source directory still supplies the independent checkpoint identity.

Provider source: QwenLM/Qwen3-ASR commit
`7c6daf77a2421100f5fb066495372c00129d39ff`, Apache-2.0. See
`QWEN_ASR_LICENSE` and the parent `SPEECH_NOTICE`. The original provider Python
runtime and checkpoint assets are not bundled or needed by shipped inference.

CPU audio preparation honors request cancellation through contextual decoding,
resampling and frontend extraction. The HTTP transcription handler propagates
cancellation through upload-file reads, quiet-point splitting and WAV encoding.
Legacy background-context helpers preserve DSP arithmetic. Complete-audio ASR
opts out of retaining generation history between requests; the runner frees KV/
prefix state and waits for device completion callbacks on the serialized worker.
Within-request KV caching and other models' retention/allocator policies remain
unchanged. SSE transport emits completed clip text after full-file upload.

The 0.6B encoder uses an embedded, non-learned 13x896 sinusoidal position buffer
captured from the pinned original provider CPU construction. This fixes platform
exp/trig phase differences without changing checkpoint bytes or requiring Python
at inference time. `positions.json` records source, dtype, environment and digest;
generation is a bounded original-provider validation step. It is specific to the
strict validated width and at most 13 positions per 100-frame CNN chunk.
Expanded internal replay keeps the strict logit diagnostic visible separately.
The owner approved replay-only atol1e-3/rtol1e-4; original and identical-input
component gates are unchanged. Both the initial eleven and expanded37 prefixes
pass the approved profile with exact prompt/greedy IDs.

Default-disabled experimental PCM sessions and fresh-window recognition are now
implemented in dedicated server/runner ASR files. The server coordinator supplies
scheduler leases, admission, cancellation and positive device-drain cleanup;
`OLLAMA_EXPERIMENT=asr-streaming` enables those routes. The ordinary complete-file
endpoint remains separate. The client live-caption commitment experiment is
closed after actual-radio joining failures; recovery success does not qualify
reliable word commitment. The outer workspace's ASR closure and model-extension
handoff record current direction and evidence. Larger ASR and alignment models
require their own import/execution qualification; this baseline is0.6B only.
