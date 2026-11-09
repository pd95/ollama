package chatterbox

import (
	"regexp"
	"strings"

	"github.com/ollama/ollama/mlx"
)

var (
	turboIndexed = regexp.MustCompile(`\.(encoders|up_encoders|down_blocks|mid_blocks|up_blocks|transformer_blocks)\.(\d+)`)
	turboHiConv  = regexp.MustCompile(`(\.(?:conv_pre|conv_post)|\.(?:ups|source_downs|convs1|convs2|condnet)\.\d+)\.conv\.`)
)

// Turbo retains PyTorch module-container names. Alias only those layouts whose
// operations are qualified against Turbo's own weights and fixed inputs.
func turboTensorName(name string) string {
	for i, n := range []string{"lstm1", "lstm2", "lstm3"} {
		name = strings.ReplaceAll(name, "ve."+n+".", "ve.lstm.layers."+string(rune('0'+i))+".")
	}
	if strings.HasPrefix(name, "s3gen.") && !strings.HasPrefix(name, "s3gen.mel2wav.") && !strings.HasPrefix(name, "s3gen.speaker_encoder.") && !strings.HasPrefix(name, "s3gen.trim_fade") {
		name = "s3gen.flow." + strings.TrimPrefix(name, "s3gen.")
	}
	name = turboIndexed.ReplaceAllString(name, ".${1}_${2}")
	name = strings.ReplaceAll(name, ".transformer_blocks_", ".transformer_")
	pairs := [][2]string{{".attn1.to_q.", ".attn.query_proj."}, {".attn1.to_k.", ".attn.key_proj."}, {".attn1.to_v.", ".attn.value_proj."}, {".attn1.to_out.0.", ".attn.out_proj."}, {".ff.net.0.proj.", ".ff.layers.0."}, {".ff.net.1.", ".ff.layers.1."}, {".mlp.0.", ".mlp_linear."}, {".block1.block.0.conv.conv.", ".block1.conv.conv."}, {".block2.block.0.conv.conv.", ".block2.conv.conv."}, {".block1.block.1.", ".block1.norm."}, {".block2.block.1.", ".block2.norm."}, {".final_block.block.0.conv.conv.", ".final_block.conv.conv."}, {".final_block.block.1.", ".final_block.norm."}, {".res_conv.conv.", ".res_conv."}, {".downsample.conv.conv.", ".downsample.conv."}, {".upsample.conv.conv.", ".upsample.conv."}, {".final_proj.conv.", ".final_proj."}, {".scales", ".weight_scale"}, {".biases", ".weight_qbias"}}
	for _, p := range pairs {
		name = strings.ReplaceAll(name, p[0], p[1])
	}
	if strings.HasPrefix(name, "s3gen.mel2wav.") {
		name = turboHiConv.ReplaceAllString(name, "${1}.")
	}
	return name
}

func turboTensorAliases(tensors map[string]*mlx.Array) map[string]*mlx.Array {
	out := make(map[string]*mlx.Array, len(tensors))
	for name, a := range tensors {
		out[turboTensorName(name)] = a
	}
	return out
}
