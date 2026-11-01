// Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
// SPDX-License-Identifier: Apache-2.0
// Native Go/MLX adaptations for Ollama.

package qwen3asr

import "fmt"

// TensorShapes defines every trained tensor of the admitted original checkpoint.
// Import checks headers before writing; runtime checks the loaded arrays again.
func TensorShapes(raw []byte) (map[string][]int32, error) {
	cfg, err := parseConfig(raw)
	if err != nil {
		return nil, err
	}
	shapes := map[string][]int32{}
	add := func(name string, dims ...int32) { shapes[name] = dims }
	linear := func(name string, out, in int32, bias bool) {
		add(name+".weight", out, in)
		if bias {
			add(name+".bias", out)
		}
	}
	norm := func(name string, n int32, bias bool) {
		add(name+".weight", n)
		if bias {
			add(name+".bias", n)
		}
	}
	a := cfg.Audio
	c, h := int32(a.Channels), int32(a.Width)
	for i := 1; i <= 3; i++ {
		in := c
		if i == 1 {
			in = 1
		}
		add(fmt.Sprintf("thinker.audio_tower.conv2d%d.weight", i), c, in, 3, 3)
		add(fmt.Sprintf("thinker.audio_tower.conv2d%d.bias", i), c)
	}
	linear("thinker.audio_tower.conv_out", h, c*16, false)
	norm("thinker.audio_tower.ln_post", h, true)
	linear("thinker.audio_tower.proj1", h, h, true)
	linear("thinker.audio_tower.proj2", int32(a.Output), h, true)
	for i := range a.Layers {
		p := fmt.Sprintf("thinker.audio_tower.layers.%d.", i)
		for _, n := range []string{"q_proj", "k_proj", "v_proj", "out_proj"} {
			linear(p+"self_attn."+n, h, h, true)
		}
		norm(p+"self_attn_layer_norm", h, true)
		norm(p+"final_layer_norm", h, true)
		linear(p+"fc1", int32(a.FFN), h, true)
		linear(p+"fc2", h, int32(a.FFN), true)
	}
	d := cfg.Text
	add("thinker.model.embed_tokens.weight", d.VocabSize, d.HiddenSize)
	classes := d.VocabSize
	if cfg.Aligner {
		classes = 5000
	}
	linear("thinker.lm_head", classes, d.HiddenSize, false)
	norm("thinker.model.norm", d.HiddenSize, false)
	for i := range d.NumHiddenLayers {
		p := fmt.Sprintf("thinker.model.layers.%d.", i)
		norm(p+"input_layernorm", d.HiddenSize, false)
		norm(p+"post_attention_layernorm", d.HiddenSize, false)
		norm(p+"self_attn.q_norm", d.HeadDim, false)
		norm(p+"self_attn.k_norm", d.HeadDim, false)
		linear(p+"self_attn.q_proj", d.NumAttentionHeads*d.HeadDim, d.HiddenSize, false)
		linear(p+"self_attn.k_proj", d.NumKeyValueHeads*d.HeadDim, d.HiddenSize, false)
		linear(p+"self_attn.v_proj", d.NumKeyValueHeads*d.HeadDim, d.HiddenSize, false)
		linear(p+"self_attn.o_proj", d.HiddenSize, d.NumAttentionHeads*d.HeadDim, false)
		linear(p+"mlp.gate_proj", d.IntermediateSize, d.HiddenSize, false)
		linear(p+"mlp.up_proj", d.IntermediateSize, d.HiddenSize, false)
		linear(p+"mlp.down_proj", d.HiddenSize, d.IntermediateSize, false)
	}
	return shapes, nil
}
