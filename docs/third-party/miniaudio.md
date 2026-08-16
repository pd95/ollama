# miniaudio

Ollama vendors miniaudio v0.11.25 for in-process MP3 decoding in native MLX
media runners. The source is `x/mlxrunner/media/miniaudio.h`, copied from the
same miniaudio revision present in llama.cpp b9888, pinned by
`LLAMA_CPP_VERSION` when this copy was introduced.

- Upstream: <https://github.com/mackron/miniaudio>
- Version: 0.11.25 (2026-03-04)
- SHA-256: `ac7af4de748b7e26b777f37e01cee313a308a7296a3eb080e2906b320cc55c89`
- License selection: MIT No Attribution (MIT-0)

The vendored header contains the complete upstream license statements. Ollama
uses the MIT-0 option reproduced below:

> Copyright 2026 David Reid
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

WAV decoding remains model-owned for now. A future consolidation must preserve
the existing pinned WAV processor outputs before switching it to this shared
decoder.
