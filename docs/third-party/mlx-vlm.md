# MLX-VLM

Portions of `x/models/gemma4/vision.go` are adapted from the Gemma 4
implementation in MLX-VLM:

- Repository: https://github.com/Blaizzy/mlx-vlm
- Revision: `61990c9054f2bc7bb8f32541e3238b4a58fe64e5`
- Source paths: `mlx_vlm/models/gemma4/gemma4.py`,
  `mlx_vlm/models/gemma4/vision.py`, and
  `mlx_vlm/models/gemma4_unified/gemma4_unified.py`

The native Gemma 4 audio encoder in `x/models/gemma4/audio.go` is adapted from
the same project's Gemma 4 Conformer implementation at revision
`84f43753380355c0455a2bafb291d4b7cbcf81d1` (MLX-VLM v0.6.5), source path
`mlx_vlm/models/gemma4/audio.py`. Its architecture and numerical behavior were
also cross-checked against Hugging Face Transformers revision
`dff4572dfa4bfa9f00cc8414e4b84877552fefe9`. No MLX-VLM or Transformers runtime
dependency is included.

The adapted implementation is distributed under the following license.

```text
MIT License

Copyright © 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
