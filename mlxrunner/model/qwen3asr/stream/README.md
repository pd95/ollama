# Native bounded ASR replay sessions

`Session` accepts normalized mono 16 kHz float PCM, in packets of at most one
second. Each packet carries its absolute starting sample offset; duplicates,
gaps, reordering, non-finite/out-of-range PCM and utterances beyond 60 seconds
are rejected. The caller serializes accepted packets and supplies native inference.
Calls apply synchronous backpressure; overlapping operations return `ErrBusy`.
There is no hidden queue, resampler, microphone I/O, route or transport. An
idle timer enforces the lifecycle policy described below.

By default every two seconds the session replays the complete accumulated audio. The first
two updates use no previous text. Later updates trim five tokenizer tokens,
backing off incomplete Unicode, from the previous raw output and append the
remaining prefix through a controlled inference callback. Audio preprocessing,
bidirectional encoder outputs and decoder prefill must be recomputed; only
within-update KV caching is permitted. The callback must enforce EOS and bounded
generation. Persistent state contains PCM and bounded text, never MLX tensors.

Each `Update` replaces the entire hypothesis. Revisions increase, but no word
is immutable until `Finish` returns `Final=true`. `AudioEnd` is an audio sample
offset, not a word timestamp. Finish consumes an unpadded tail; at an exact
two-second boundary it finalizes the previous result without decoding again.
The minimum total input is 201 samples. Finish, inference errors and observed
cancellation release retained PCM/text. `Close` also cancels in-flight inference
and may be called concurrently; it cannot preempt a running GPU kernel.

Parent cancellation and idle expiry automatically close the session; the default
idle timeout is 15 seconds, configurable up to one minute. Expiry only applies
between operations, with current-deadline checks suppressing stale timer callbacks.
`Done` acknowledges closure and callback return. The native callback must drain
its device cleanup before returning, and a separate model lease must stay held
until Done, independently of work cancellation. The caller closes explicit disconnects and supplies model scheduling/leases,
aggregate admission and an absolute session lifetime. The dedicated experimental
ASR server coordinator now implements those responsibilities for public HTTP
PCM sessions and fresh windows, disabled by default. The complete-file
transcription API remains separate.
Word timestamps and forced alignment remain separate. The native test uses
recordings replayed without wall-clock pacing, not a microphone latency test.

The prefix policy is adapted from QwenLM/Qwen3-ASR commit
`7c6daf77a2421100f5fb066495372c00129d39ff`, Apache-2.0. The provider streaming
wrapper uses vLLM, whereas the native model follows eager/global attention.
The native final tail deliberately uses the same Unicode-safe rollback as normal
updates instead of the provider's distinct `max(1, len-K)` final-tail rule.
Provider repetition cleanup is not implemented. Exact vLLM streaming parity is
not claimed. F32 stays enabled; shared allocator/cache policies stay unchanged.

Run the ordinary state/lifecycle tests on Linux:

```sh
go -C ./ollama test -race ./mlxrunner/model/qwen3asr/stream
```

`TestReplayNative` is an opt-in supervised macOS check using the existing private
component import and hash-verified prepared WAV receipts. Its English/German
eight-second prefixes plus a 17-sample tail and one-second silence are replayed
with regular, irregular and repeated session packetization. It records native
frontend/encoder/prefill tensors for independent original-provider checks.
The expanded fixture retains strict logit diagnostics separately from the owner-
approved replay-only1e-3/1e-4 profile. Original/component gates stay unchanged.
The eleven-prefix and seven-case37-prefix corpora pass that profile with exact
prompt/greedy IDs. The runner-local adapter/registry uses the existing finite
request queue and terminal response-channel closure, including after cancellation.
Private loopback IPC is distinct from the default-disabled public ASR coordinator.
Evidence and reproduction are routed through the outer workspace's Qwen3-ASR
stream. Its live-caption commitment experiment is closed after real-radio joining
failures; those client heuristics do not redefine this native session contract.

`Config.UpdateSamples` optionally selects8000(half a second),16000(one second) or
32000(two seconds).
Zero preserves the two-second default; other values fail. The existing first-two-
updates/five-token rollback policy is unchanged at every cadence. Public stable/
draft annotation belongs to the outer coordinator, not this inference protocol.

Half-second updates are opt-in for ordinary sessions. The outer segment client
default is one second; experimental overlap explicitly selects half a second.
The public API omitted interval stays two seconds. See the outer stream
`fixtures/half-second-2026-11-01/README.md` for bounded provider, pacing and
lifecycle qualification; more frequent drafts do not imply immutable words.


`Seed` accepts201..256000 normalized samples only on an empty session and runs
one fresh decode with an empty prefix. `Finish` reuses that processed result.
The same operation, cancellation and drain contracts apply; the server window
coordinator retains admission/lease ownership until positive cleanup. This is a
finite16s context input primitive, not a word-timing or safe-commitment guarantee.
