package create

import (
	"encoding/json"
	"fmt"
	"strings"
)

type apertusImportTransform struct {
	tieWordEmbeddings bool
}

func newApertusImportTransform(rawConfig json.RawMessage) (quantizePolicy, error) {
	var cfg struct {
		TieWordEmbeddings *bool `json:"tie_word_embeddings"`
		TextConfig        *struct {
			TieWordEmbeddings *bool `json:"tie_word_embeddings"`
		} `json:"text_config"`
	}
	if len(rawConfig) > 0 {
		if err := json.Unmarshal(rawConfig, &cfg); err != nil {
			return nil, fmt.Errorf("apertus: parse config.json: %w", err)
		}
	}
	tied := false
	if cfg.TextConfig != nil {
		if cfg.TextConfig.TieWordEmbeddings != nil {
			tied = *cfg.TextConfig.TieWordEmbeddings
		}
	} else if cfg.TieWordEmbeddings != nil {
		tied = *cfg.TieWordEmbeddings
	}
	return apertusImportTransform{tieWordEmbeddings: tied}, nil
}

func (t apertusImportTransform) includeTensor(name string) bool {
	// Transformers treats tie_word_embeddings as authoritative even when an
	// export redundantly includes a byte-identical output projection.
	return !t.tieWordEmbeddings || name != "lm_head.weight"
}

func (apertusImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	if isApertus1p5MediaTensor(name) {
		return ""
	}
	base := normalizeQuantType(quantize)
	stackedExpert := isStackedExpertWeight(name)
	if !stackedExpert && !ShouldQuantize(name, "") {
		return ""
	}
	if len(shape) != 2 && !(len(shape) == 3 && stackedExpert) {
		return ""
	}

	var elems int64 = 1
	for _, d := range shape {
		elems *= int64(d)
	}
	if elems < 1024 || isRoutingGate(name) || !isAligned(shape, base) {
		return ""
	}

	return base
}

func isApertus1p5TextTensor(name string) bool {
	return strings.HasPrefix(name, "model.language_model.") || name == "lm_head.weight"
}

func isApertus1p5VisionTensor(name string) bool {
	return strings.HasPrefix(name, "model.vision_tokenizer.")
}

func isApertus1p5AudioTensor(name string) bool {
	return strings.HasPrefix(name, "model.audio_tokenizer.encoder.") ||
		name == "model.audio_tokenizer.quantizer.codebook.embed"
}

func isApertus1p5MediaTensor(name string) bool {
	return isApertus1p5VisionTensor(name) || isApertus1p5AudioTensor(name)
}

func isApertus1p5RuntimeTensor(name string) bool {
	return isApertus1p5TextTensor(name) || isApertus1p5MediaTensor(name)
}

func Apertus1p5VisionInventoryComplete(inv Inventory) bool {
	return completeApertus1p5VisionNames(inv.Has)
}

func completeApertus1p5VisionNames(has func(string) bool) bool {
	const p = "model.vision_tokenizer."
	pair := func(path string) bool { return has(path+".weight") && has(path+".bias") }
	if !pair(p+"encoder.conv_in") || !pair(p+"encoder.conv_out") || !pair(p+"encoder.norm_out") ||
		!pair(p+"quant_conv") || !has(p+"quantize.embedding.weight") {
		return false
	}
	for level := range 5 {
		for block := range 4 {
			path := fmt.Sprintf("%sencoder.down.%d.block.%d", p, level, block)
			if !pair(path+".norm1") || !pair(path+".conv1") || !pair(path+".norm2") || !pair(path+".conv2") {
				return false
			}
			if block == 0 && (level == 2 || level == 4) && !pair(path+".nin_shortcut") {
				return false
			}
			if level == 4 {
				attn := fmt.Sprintf("%sencoder.down.%d.attn.%d", p, level, block)
				if !pair(attn+".norm") || !pair(attn+".q") || !pair(attn+".k") || !pair(attn+".v") || !pair(attn+".proj_out") {
					return false
				}
			}
		}
		if level < 4 && !pair(fmt.Sprintf("%sencoder.down.%d.downsample.conv", p, level)) {
			return false
		}
	}
	for _, block := range []string{"block_1", "block_2"} {
		path := p + "encoder.mid." + block
		if !pair(path+".norm1") || !pair(path+".conv1") || !pair(path+".norm2") || !pair(path+".conv2") {
			return false
		}
	}
	attn := p + "encoder.mid.attn_1"
	return pair(attn+".norm") && pair(attn+".q") && pair(attn+".k") && pair(attn+".v") && pair(attn+".proj_out")
}

func Apertus1p5AudioInventoryComplete(inv Inventory) bool {
	return completeApertus1p5AudioNames(inv.Has)
}

func completeApertus1p5AudioNames(has func(string) bool) bool {
	const p = "model.audio_tokenizer.encoder.layers."
	conv := func(path string) bool {
		return has(path+".conv.bias") && has(path+".conv.parametrizations.weight.original0") && has(path+".conv.parametrizations.weight.original1")
	}
	if !conv(p+"0") || !conv(p+"15") || !has("model.audio_tokenizer.quantizer.codebook.embed") {
		return false
	}
	for _, index := range []int{1, 4, 7, 10} {
		path := fmt.Sprintf("%s%d", p, index)
		if !conv(path+".block.1") || !conv(path+".block.3") || !conv(path+".shortcut") {
			return false
		}
	}
	for _, index := range []int{3, 6, 9, 12} {
		if !conv(fmt.Sprintf("%s%d", p, index)) {
			return false
		}
	}
	for layer := range 2 {
		for _, kind := range []string{"weight_ih", "weight_hh", "bias_ih", "bias_hh"} {
			if !has(fmt.Sprintf("%s13.lstm.%s_l%d", p, kind, layer)) {
				return false
			}
		}
	}
	return true
}
