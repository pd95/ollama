package create

import (
	"encoding/json"
	"strings"
)

type apertusImportTransform struct{}

func newApertusImportTransform(json.RawMessage) (quantizePolicy, error) {
	return apertusImportTransform{}, nil
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
