// Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
// SPDX-License-Identifier: Apache-2.0
// Native Go/MLX adaptations for Ollama.

package qwen3asr

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
)

type encoder struct {
	cfg     audioConfig
	tensors map[string]*mlx.Array
}

func encoderGELU(x *mlx.Array) *mlx.Array {
	return mlx.GELU(x.AsType(mlx.DTypeFloat32)).AsType(x.DType())
}

func (e *encoder) linear(x *mlx.Array, name string) *mlx.Array {
	p := "thinker.audio_tower." + name
	w := mlx.Transpose(e.tensors[p+".weight"], 1, 0)
	if b := e.tensors[p+".bias"]; b != nil {
		// Match the provider's 2D F.linear/addmm bias accumulation and the
		// existing native linear layers, rather than rounding a separate add.
		return b.Addmm(x, w, 1, 1)
	}
	return mlx.Matmul(x, w)
}

func (e *encoder) norm(x *mlx.Array, name string) *mlx.Array {
	p := "thinker.audio_tower." + name
	return mlx.LayerNormFn(x, e.tensors[p+".weight"], e.tensors[p+".bias"], 1e-5)
}

func (e *encoder) convolve(data *mlx.Array) *mlx.Array {
	frames := data.Dim(1)
	width := min(100, frames)
	var chunks []*mlx.Array
	for start := 0; start < frames; start += 100 {
		count := min(100, frames-start)
		x := data.Slice(mlx.Slice(), mlx.Slice(start, start+count))
		if count < width {
			x = mlx.Concatenate([]*mlx.Array{x, mlx.Zeros(data.DType(), 128, width-count)}, 1)
		}
		x = mlx.Reshape(x, 1, 128, int32(width), 1)
		for i := 1; i <= 3; i++ {
			p := fmt.Sprintf("thinker.audio_tower.conv2d%d", i)
			w := mlx.Transpose(e.tensors[p+".weight"], 0, 2, 3, 1)
			x = encoderGELU(mlx.Add(mlx.Conv2d(x, w, 2, 2, 1, 1, 1, 1, 1), e.tensors[p+".bias"]))
		}
		// PyTorch flattens channels before frequency after its NCHW convolution.
		x = mlx.Transpose(x, 0, 2, 3, 1)
		x = mlx.Reshape(x, int32(x.Dim(1)), int32(e.cfg.Channels*16))
		x = e.linear(x, "conv_out")
		steps := audioTokens(count)
		x = x.Slice(mlx.Slice(0, steps), mlx.Slice())
		pos := encoderPositions(steps, e.cfg.Width)
		chunks = append(chunks, mlx.Add(x, mlx.FromValues(pos, steps, e.cfg.Width).AsType(x.DType())))
	}
	return mlx.Concatenate(chunks, 0)
}

func (e *encoder) layer(x *mlx.Array, i int) *mlx.Array {
	p := fmt.Sprintf("layers.%d.", i)
	norm := e.norm(x, p+"self_attn_layer_norm")
	heads, head := e.cfg.Heads, e.cfg.Width/e.cfg.Heads
	project := func(n string) *mlx.Array {
		return mlx.Transpose(mlx.Reshape(e.linear(norm, p+"self_attn."+n), 1, int32(x.Dim(0)), int32(heads), int32(head)), 0, 2, 1, 3)
	}
	q, k, v := project("q_proj"), project("k_proj"), project("v_proj")
	// The pinned provider eager backend receives no encoder attention mask:
	// its cu_seqlens argument only constrains FlashAttention, which is not
	// the reference backend here. All supplied tokens attend bidirectionally.
	scores := mlx.Mul(mlx.Matmul(q, mlx.Transpose(k, 0, 1, 3, 2)), mlx.FromValue(float32(1/math.Sqrt(float64(head)))))
	probs := mlx.SoftmaxAxis(scores.AsType(mlx.DTypeFloat32), -1, true).AsType(q.DType())
	attn := mlx.Reshape(mlx.Transpose(mlx.Matmul(probs, v), 0, 2, 1, 3), int32(x.Dim(0)), int32(e.cfg.Width))
	x = mlx.Add(x, e.linear(attn, p+"self_attn.out_proj"))
	return mlx.Add(x, e.linear(encoderGELU(e.linear(e.norm(x, p+"final_layer_norm"), p+"fc1")), p+"fc2"))
}

func (e *encoder) forward(data *mlx.Array) *mlx.Array {
	x := e.convolve(data.AsType(e.tensors["thinker.audio_tower.conv2d1.weight"].DType()))
	for i := range e.cfg.Layers {
		x = e.layer(x, i)
	}
	return e.linear(encoderGELU(e.linear(e.norm(x, "ln_post"), "proj1")), "proj2")
}
