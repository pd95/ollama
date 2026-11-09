package create

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// originalChatterboxLayout describes storage transformations, not neural
// execution. It is deliberately separate from the converted-checkpoint path:
// an original PyTorch convolution is transposed even when two axes have equal
// lengths. Shape equality alone cannot identify its storage convention.
type originalChatterboxLayout struct {
	Name string
	Axes []int
}

type chatterboxRename struct {
	pattern *regexp.Regexp
	value   string
}

func chatterboxRenames(rules ...string) []chatterboxRename {
	var result []chatterboxRename
	for i := 0; i < len(rules); i += 2 {
		result = append(result, chatterboxRename{regexp.MustCompile(rules[i]), rules[i+1]})
	}
	return result
}

func renameChatterboxTensor(name string, rules []chatterboxRename) string {
	for _, rule := range rules {
		name = rule.pattern.ReplaceAllString(name, rule.value)
	}
	return name
}

var (
	originalChatterboxLSTM = regexp.MustCompile(`^lstm\.(weight_ih|weight_hh|bias_ih|bias_hh)_l([0-2])$`)
	originalTurboRenames   = chatterboxRenames(
		`\.(up_embed|embed)\.out\.0\.`, `.${1}.linear.`,
		`\.(up_embed|embed)\.out\.1\.`, `.${1}.norm.`,
		`\.(down_blocks|mid_blocks|up_blocks)\.(\d+)\.0\.`, `.${1}.${2}.resnet.`,
		`\.(down_blocks|mid_blocks|up_blocks)\.(\d+)\.1\.(\d+)\.`, `.${1}.${2}.transformer_blocks.${3}.`,
		`\.(down_blocks|mid_blocks)\.(\d+)\.2\.`, `.${1}.${2}.downsample.`,
		`\.up_blocks\.(\d+)\.2\.`, `.up_blocks.${1}.upsample.`,
		`\.(block1|block2)\.block\.0\.`, `.${1}.block.0.conv.conv.`,
		`\.(block1|block2)\.block\.2\.`, `.${1}.block.1.`,
		`\.mlp\.1\.`, `.mlp.0.`,
		`\.ff\.net\.2\.`, `.ff.net.1.`,
		`\.final_block\.block\.0\.`, `.final_block.block.0.conv.conv.`,
		`\.final_block\.block\.2\.`, `.final_block.block.1.`,
		`\.(downsample|upsample)\.(weight|bias)$`, `.${1}.conv.conv.${2}`,
		`\.res_conv\.(weight|bias)$`, `.res_conv.conv.${1}`,
		`\.final_proj\.(weight|bias)$`, `.final_proj.conv.${1}`,
	)
	originalCondnetIndex = regexp.MustCompile(`\.condnet\.([02468])\.`)
	originalTurboHiFT    = regexp.MustCompile(`(\.(?:conv_pre|conv_post)|\.(?:ups|source_downs|convs1|convs2|condnet)\.\d+)\.(weight|bias)$`)
	originalCAMPIndex    = regexp.MustCompile(`xvector\.(block|transit)([1-9]\d*)\.`)
	originalCAMPLayer    = regexp.MustCompile(`\.tdnnd([1-9]\d*)\.`)
	originalCAMPRenames  = chatterboxRenames(
		`^xvector\.(tdnn|dense|out_nonlinear)\.`, `${1}.`,
		`\.nonlinear(\d+)\.batchnorm\.`, `.nonlinear${1}.0.`,
		`(^|\.)(nonlinear|out_nonlinear)\.batchnorm\.`, `${1}${2}.0.`,
	)
	originalS3Renames = chatterboxRenames(
		`quantizer\.(_codebook|codebook)\.`, `quantizer.fsq_codebook.`,
		`\.mlp\.(\d+)\.`, `.mlp.layers.${1}.`,
	)
)

func chatterboxIdentityAxes(rank int) []int {
	axes := make([]int, rank)
	for i := range axes {
		axes[i] = i
	}
	return axes
}

func originalChatterboxConvAxes(name string, rank int) []int {
	if strings.HasSuffix(name, ".weight") {
		switch rank {
		case 3:
			return []int{0, 2, 1}
		case 4:
			return []int{0, 2, 3, 1}
		}
	}
	return chatterboxIdentityAxes(rank)
}

// originalTurboTensorLayout accepts component-local names after weight norm
// pairs have been folded and the two LSTM biases have been added. In
// particular it does not authorize dropping the second member of a pair.
func originalTurboTensorLayout(group, name string, rank int) (originalChatterboxLayout, error) {
	if rank < 0 || rank > 4 || name == "" {
		return originalChatterboxLayout{}, fmt.Errorf("unsupported original Turbo tensor %s/%s rank %d", group, name, rank)
	}
	result := originalChatterboxLayout{Axes: chatterboxIdentityAxes(rank)}
	switch group {
	case "t3":
		result.Name = "t3." + name
		for _, suffix := range []string{".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight"} {
			if strings.HasSuffix(name, suffix) {
				if rank != 2 {
					return originalChatterboxLayout{}, fmt.Errorf("Turbo GPT-2 Conv1D must have rank two: %s", name)
				}
				result.Axes = []int{1, 0}
			}
		}
	case "ve":
		result.Name = "ve." + name
		if match := originalChatterboxLSTM.FindStringSubmatch(name); match != nil {
			layer, _ := strconv.Atoi(match[2])
			suffix := map[string]string{"weight_ih": "Wx", "weight_hh": "Wh", "bias_ih": "bias", "bias_hh": "bias"}[match[1]]
			result.Name = fmt.Sprintf("ve.lstm%d.%s", layer+1, suffix)
		} else if strings.HasPrefix(name, "lstm.") {
			return originalChatterboxLayout{}, fmt.Errorf("unsupported original voice LSTM tensor %s", name)
		}
	case "s3gen":
		name = strings.TrimPrefix(name, "flow.")
		result.Axes = originalChatterboxConvAxes(name, rank)
		if rank == 3 && strings.HasPrefix(name, "mel2wav.ups.") && strings.HasSuffix(name, ".weight") {
			result.Axes = []int{1, 2, 0}
		}
		name = renameChatterboxTensor(name, originalTurboRenames)
		if strings.HasPrefix(name, "mel2wav.") {
			name = originalCondnetIndex.ReplaceAllStringFunc(name, func(match string) string {
				index, _ := strconv.Atoi(originalCondnetIndex.FindStringSubmatch(match)[1])
				return fmt.Sprintf(".condnet.%d.", index/2)
			})
			name = originalTurboHiFT.ReplaceAllString(name, `${1}.conv.${2}`)
		}
		result.Name = "s3gen." + name
	default:
		return originalChatterboxLayout{}, fmt.Errorf("unsupported original Turbo component %q", group)
	}
	return result, nil
}

// originalCAMPLayout is shared by v3 and Turbo. Each family must supply its
// own speaker_encoder tensors; Turbo must never borrow v3's trained encoder.
func originalCAMPLayout(name string, rank int) originalChatterboxLayout {
	axes := originalChatterboxConvAxes(name, rank)
	name = originalCAMPIndex.ReplaceAllStringFunc(name, func(match string) string {
		parts := originalCAMPIndex.FindStringSubmatch(match)
		index, _ := strconv.Atoi(parts[2])
		plural := "blocks"
		if parts[1] == "transit" {
			plural = "transits"
		}
		return fmt.Sprintf("%s.%d.", plural, index-1)
	})
	name = originalCAMPLayer.ReplaceAllStringFunc(name, func(match string) string {
		index, _ := strconv.Atoi(originalCAMPLayer.FindStringSubmatch(match)[1])
		return fmt.Sprintf(".layers.%d.", index-1)
	})
	return originalChatterboxLayout{Name: renameChatterboxTensor(name, originalCAMPRenames), Axes: axes}
}

// Computed buffers and ONNX intermediates are classified by the caller. This
// mapper never silently drops an unknown tensor, including S3 frontend assets.
func originalChatterboxS3Layout(name string, rank int) originalChatterboxLayout {
	axes := chatterboxIdentityAxes(rank)
	if strings.Contains(name, ".conv1.") || strings.Contains(name, ".conv2.") || strings.Contains(name, ".fsmn_block.") {
		if rank == 3 && strings.HasSuffix(name, ".weight") {
			axes = []int{0, 2, 1}
		}
	}
	return originalChatterboxLayout{Name: renameChatterboxTensor(name, originalS3Renames), Axes: axes}
}
