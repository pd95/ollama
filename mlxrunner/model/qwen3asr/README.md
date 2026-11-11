# Native Qwen3-ASR and ForcedAligner

The original ASR 0.6B, ASR 1.7B and ForcedAligner 0.6B checkpoints are supported
through strict, model-specific configuration and tensor schemas. ASR 1.7B has a
24-layer encoder with width 1024 and output width 2048; its decoder width is
2048. The aligner has a 24-layer encoder with width 1024 and output width 1024,
a decoder with width 1024 and an 8,192-token context limit, and a 5,000-class
timestamp head. It uses timestamp token ID 151705 and an 80 ms interval between
timestamp classes. Both new schemas contain 708 trained tensors. Original BF16
bytes are preserved; execution promotes weights to F32 in memory. Their
width-1024 provider position buffer is separate from the ASR 0.6B model's
width-896 buffer.

`Model.AlignContext` performs one supplied-text/audio classification, preserving
source text and UTF-8 byte spans, raw classes/milliseconds, provider-repaired
seconds and explicit repair/outside-audio flags. The supported inputs are
English or German, clips of up to 60 seconds, text of up to 64 KiB, 1–1,024
normalized units and at most 8,192 expanded tokens. It rejects ordinary ASR
media preparation. The importer preserves the pinned aligner wrapper's
Transformers 4.57.6 regex patch without changing ASR/TTS tokenization. The
imported model advertises `alignment,audio`. There is no standalone public
supplied-text alignment route. The existing `/v1/audio/transcriptions` endpoint
can use this model after ASR to return complete-file word timestamps; see
`docs/api/openai-compatibility.mdx` for its standard timestamp fields and optional
Ollama `alignment_model` extension.
Use the outer stream native Go driver in `tools/align/main.go` for supplied
pairs.

Optional `TestForcedAlignerReference` compares all components, exact timestamp
classes, wrapper prompts/token IDs, normalization, anomaly repair and repeated
AlignContext/cancellation against independently executed original provider code.
The outer stream model-extension guide/evidence records source identities,
per-model gates, exact commands and limits. New BF16 execution, annotated
alignment accuracy and multilingual segmentation remain unqualified. No
TTS/LFM3/allocator math or closed live-caption behavior is changed.

## Preserved 0.6B baseline

Imports the original `Qwen3ASRForConditionalGeneration` BF16 checkpoint without
changing source bytes. Runtime promotion to F32 is deliberate: the initial
independent provider qualification covers F32 execution, not BF16 performance.
Parameter memory is approximately 3.49 GiB after promotion. Shared allocator
defaults and existing speech implementations are unchanged. Initial admission
uses the validated schema to budget BF16 source plus F32 promoted tensor memory
(~5.24 GiB); after startup, live MLX allocations replace that estimate. This is
not a request activation budget or a hard RSS cap.

Reuses the existing Qwen3 decoder layers with explicit 128-dimensional heads, 16
query heads, 8 KV heads and a separate trained output projection. Audio uses its
own Whisper frontend and 18-layer bidirectional encoder; Qwen3-TTS conditioning,
codec, talker and code predictor are not reused. The original MRoPE's three
equal position axes reduce to the existing half-rotation RoPE.

The input bound is 60 seconds per generation after native mono/16 kHz
resampling, minimum 201 samples. The encoder follows the pinned provider eager
backend's global attention; FlashAttention windowing is not qualified.
Preprocessing is CPU-only and request-local. The runner's normal media ownership
and causal decoder cache interfaces are reused. The qwen3-asr renderer exposes
the existing transcription route for English, German or automatic detection.
Translation and word alignment are not implemented. Experimental PCM sessions
and fresh-window HTTP routes are described below. PrepareMedia rejects multiple
audio occurrences before decoding, including repeated raw media markers.
Importer and startup share the EOS and special-token sidecar checks.

Real-audio provider controls cover complete recordings through 44.78 seconds,
frontend/encoder outputs, full prefill logits and cached greedy IDs. Shared Go
resampling differs from provider librosa/soxr and can change transcription.
`TestQwen3ASRAudioReference` uses `OLLAMA_QWEN3_ASR_AUDIO_REFERENCE` and the
source path to run these optional native fixture checks. F32 component checks
disable TF32 in their worker only; runtime defaults remain unchanged.

`TestReference` compares original-provider frontend/encoder/decoder fixtures,
all 28 decoder layers at identical inputs, prefill states, vocabulary logits,
and greedy cached decoding. Set `OLLAMA_QWEN3_ASR_SOURCE` and
`OLLAMA_QWEN3_ASR_REFERENCE` to run it. With `OLLAMA_QWEN3_ASR_IMPORTED`, it
loads tokenizer/config/weights through that imported manifest instead; the
source directory still supplies the independent checkpoint identity.

Provider source: QwenLM/Qwen3-ASR commit
`7c6daf77a2421100f5fb066495372c00129d39ff`, Apache-2.0. See `QWEN_ASR_LICENSE`
and the parent `SPEECH_NOTICE`. The original provider Python runtime and
checkpoint assets are not bundled or needed by shipped inference.

CPU audio preparation honors request cancellation through contextual decoding,
resampling and frontend extraction. The HTTP transcription handler propagates
cancellation through upload-file reads, quiet-point splitting and WAV encoding.
Legacy background-context helpers preserve DSP arithmetic. Complete-audio ASR
opts out of retaining generation history between requests; the runner frees KV/
prefix state and waits for device completion callbacks on the serialized worker.
Within-request KV caching and other models' retention/allocator policies remain
unchanged. SSE transport emits completed clip text after full-file upload.

The 0.6B encoder uses an embedded, non-learned 13 × 896 sinusoidal position
buffer captured from the pinned original provider CPU construction. This fixes
platform exp/trig phase differences without changing checkpoint bytes or
requiring Python at inference time. `positions.json` records source, dtype,
environment and digest; generation is a bounded original-provider validation
step. It is specific to the strict validated width and at most 13 positions per
100-frame CNN chunk. Expanded internal replay keeps the strict logit diagnostic
visible separately. The owner approved replay-only atol 1e-3/rtol 1e-4; original
and identical-input component gates are unchanged. Both the initial eleven and
expanded 37 prefixes pass the approved profile with exact prompt/greedy IDs.

Default-disabled experimental PCM sessions and fresh-window recognition are now
implemented in dedicated server/runner ASR files. The server coordinator
supplies scheduler leases, admission, cancellation and positive device-drain
cleanup; `OLLAMA_EXPERIMENT=asr-streaming` enables those routes. The ordinary
complete-file endpoint remains separate. The client live-caption commitment
experiment is closed after actual-radio joining failures; recovery success does
not qualify reliable word commitment. The outer workspace's ASR closure and
model-extension handoff record current direction and evidence. The baseline
sections above remain specific to 0.6B; the new model profiles and independent
checks are summarized at the top of this file.
