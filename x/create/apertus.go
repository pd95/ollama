package create

import "encoding/json"

type apertusImportTransform struct{}

func newApertusImportTransform(json.RawMessage) (quantizePolicy, error) {
	return apertusImportTransform{}, nil
}

func (apertusImportTransform) quantizationType(name string, shape []int32, quantize string) string {
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
