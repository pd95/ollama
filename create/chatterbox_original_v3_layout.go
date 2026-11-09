package create

import (
	"fmt"
	"strconv"
	"strings"
)

var originalV3Renames = chatterboxRenames(
	`\.(down_blocks|mid_blocks|up_blocks)\.(\d+)\.0\.`, `.${1}_${2}.resnet.`,
	`\.(down_blocks|mid_blocks|up_blocks)\.(\d+)\.1\.(\d+)\.`, `.${1}_${2}.transformer_${3}.`,
	`\.down_blocks\.(\d+)\.2\.`, `.down_blocks_${1}.downsample.`,
	`\.up_blocks\.(\d+)\.2\.`, `.up_blocks_${1}.upsample.`,
	`\.(block1|block2)\.block\.0\.`, `.${1}.conv.conv.`,
	`\.(block1|block2)\.block\.2\.`, `.${1}.norm.`,
	`\.mlp\.1\.`, `.mlp_linear.`,
	`\.attn1\.to_q\.`, `.attn.query_proj.`,
	`\.attn1\.to_k\.`, `.attn.key_proj.`,
	`\.attn1\.to_v\.`, `.attn.value_proj.`,
	`\.attn1\.to_out\.0\.`, `.attn.out_proj.`,
	`\.ff\.net\.0\.proj\.`, `.ff.layers.0.`,
	`\.ff\.net\.2\.`, `.ff.layers.1.`,
	`\.(downsample|upsample)\.(weight|bias)$`, `.${1}.conv.${2}`,
	`\.final_block\.block\.0\.`, `.final_block.conv.conv.`,
	`\.final_block\.block\.2\.`, `.final_block.norm.`,
	`\.(up_embed|embed)\.out\.0\.`, `.${1}.linear.`,
	`\.(up_embed|embed)\.out\.1\.`, `.${1}.norm.`,
	`\.(up_encoders|encoders)\.(\d+)\.`, `.${1}_${2}.`,
)

func originalV3TensorLayout(group, name string, rank int) (originalChatterboxLayout, error) {
	if rank < 0 || rank > 4 || name == "" {
		return originalChatterboxLayout{}, fmt.Errorf("unsupported original v3 tensor %s/%s rank %d", group, name, rank)
	}
	result := originalChatterboxLayout{Axes: chatterboxIdentityAxes(rank)}
	switch group {
	case "t3":
		if strings.HasPrefix(name, "tfmr.layers.") || strings.HasPrefix(name, "tfmr.embed_tokens.") || strings.HasPrefix(name, "tfmr.norm.") {
			name = strings.Replace(name, "tfmr.", "tfmr.model.", 1)
		}
		if rank == 3 && strings.Contains(strings.ToLower(name), "conv") && strings.HasSuffix(name, ".weight") {
			result.Axes = []int{0, 2, 1}
		}
		result.Name = "t3." + name
	case "ve":
		result.Name = "ve." + name
		if match := originalChatterboxLSTM.FindStringSubmatch(name); match != nil {
			suffix := map[string]string{"weight_ih": "Wx", "weight_hh": "Wh", "bias_ih": "bias", "bias_hh": "bias"}[match[1]]
			result.Name = "ve.lstm.layers." + match[2] + "." + suffix
		} else if strings.HasPrefix(name, "lstm.") {
			return originalChatterboxLayout{}, fmt.Errorf("unsupported original v3 voice LSTM tensor %s", name)
		}
	case "s3gen":
		result.Axes = originalChatterboxConvAxes(name, rank)
		if rank == 3 && strings.HasPrefix(name, "mel2wav.ups.") && strings.HasSuffix(name, ".weight") {
			result.Axes = []int{1, 2, 0}
		}
		name = renameChatterboxTensor(name, originalV3Renames)
		name = originalCondnetIndex.ReplaceAllStringFunc(name, func(match string) string {
			index, _ := strconv.Atoi(originalCondnetIndex.FindStringSubmatch(match)[1])
			return fmt.Sprintf(".condnet.%d.", index/2)
		})
		result.Name = "s3gen." + name
	default:
		return originalChatterboxLayout{}, fmt.Errorf("unsupported original v3 component %q", group)
	}
	return result, nil
}
