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
	// This policy is deliberately limited to Apertus NVFP4 imports. Unknown or
	// mismatched requests remain at source precision rather than falling through
	// to a different shared quantization recipe.
	if normalizeQuantType(quantize) != "nvfp4" {
		return ""
	}
	stackedExpert := isStackedExpertWeight(name)
	if !stackedExpert && !ShouldQuantize(name, "") {
		return ""
	}
	if len(shape) != 2 && !(len(shape) == 3 && stackedExpert) {
		return ""
	}

	elems, ok := apertusElementCount(shape)
	if !ok || elems < 1024 || isRoutingGate(name) || isApertusNonlinearTensor(name) || !isAligned(shape, "nvfp4") {
		return ""
	}

	return "nvfp4"
}

func apertusElementCount(shape []int32) (uint64, bool) {
	elements := uint64(1)
	for _, dim := range shape {
		if dim <= 0 {
			return 0, false
		}
		d := uint64(dim)
		if elements > ^uint64(0)/d {
			return 0, false
		}
		elements *= d
	}
	return elements, true
}

func isApertusNonlinearTensor(name string) bool {
	return strings.Contains(name, ".act_fn.")
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
