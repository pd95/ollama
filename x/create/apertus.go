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
	base := normalizeQuantType(quantize)
	if base == "mxfp8" {
		// Apertus has a bespoke NVFP4 policy, but its MXFP8 imports use the
		// established general policy. Returning source precision here turns a
		// 70B MXFP8 request into a BF16-sized artifact that cannot be admitted.
		return GetTensorQuantization(name, shape, quantize)
	}
	if base != "nvfp4" {
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
