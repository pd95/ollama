// Package quant holds the quantization format facts shared by the model
// importer, the runtime loader, and `ollama show`. It deliberately has no
// dependency on the MLX C library, so any package can use it without pulling
// in cgo — which is what keeps these facts from drifting between separate
// hand-maintained copies.
package quant

import (
	"math"
	"strings"
)

type params struct {
	groupSize int
	bits      int
	mode      string
}

// byType maps each canonical quantization type to its parameters. Aliases are
// resolved to a canonical name by Canonical before lookup.
var byType = map[string]params{
	"nvfp4": {groupSize: 16, bits: 4, mode: "nvfp4"},
	"mxfp4": {groupSize: 32, bits: 4, mode: "mxfp4"},
	"int2":  {groupSize: 64, bits: 2, mode: "affine"},
	"int3":  {groupSize: 64, bits: 3, mode: "affine"},
	"int4":  {groupSize: 64, bits: 4, mode: "affine"},
	"int5":  {groupSize: 64, bits: 5, mode: "affine"},
	"int6":  {groupSize: 64, bits: 6, mode: "affine"},
	"mxfp8": {groupSize: 32, bits: 8, mode: "mxfp8"},
	"int8":  {groupSize: 64, bits: 8, mode: "affine"},
}

// Canonical returns the canonical name for a quantization type, resolving
// aliases (for example "FP8" and "Q8" both map to "int8"). It returns "" for
// the empty string and for any type it does not recognize.
func Canonical(quantType string) string {
	switch strings.ToUpper(strings.TrimSpace(quantType)) {
	case "NVFP4":
		return "nvfp4"
	case "MXFP4":
		return "mxfp4"
	case "MXFP8":
		return "mxfp8"
	case "INT2", "Q2":
		return "int2"
	case "INT3", "Q3":
		return "int3"
	case "INT4", "FP4", "Q4":
		return "int4"
	case "INT5", "Q5":
		return "int5"
	case "INT6", "Q6":
		return "int6"
	case "INT8", "FP8", "Q8":
		return "int8"
	default:
		return ""
	}
}

// Importable reports whether the runtime can load this quantization format.
func Importable(quantType string) bool {
	_, ok := byType[Canonical(quantType)]
	return ok
}

// Creatable reports whether Ollama may create this quantization from a float
// checkpoint. MLX can execute additional affine widths that are intentionally
// import-only because they are not exposed as Ollama quantization targets.
func Creatable(quantType string) bool {
	switch Canonical(quantType) {
	case "int4", "int8", "nvfp4", "mxfp4", "mxfp8":
		return true
	default:
		return false
	}
}

// Params returns the default group size, bit width, and mode for a
// quantization type. The empty string returns zeros. An unrecognized
// non-empty type falls back to 8-bit affine, matching the runtime loader's
// historical leniency toward unexpected metadata.
func Params(quantType string) (groupSize, bits int, mode string) {
	if strings.TrimSpace(quantType) == "" {
		return 0, 0, ""
	}
	if p, ok := byType[Canonical(quantType)]; ok {
		return p.groupSize, p.bits, p.mode
	}
	return 32, 8, "affine"
}

// Bits returns the bit width of a recognized quantization type, or 0 if the
// type is empty or unrecognized. Unlike Params it applies no fallback, so
// callers that size or display tensors never act on an unknown type.
func Bits(quantType string) int {
	if p, ok := byType[Canonical(quantType)]; ok {
		return p.bits
	}
	return 0
}

// PackFactor returns how many quantized values are packed into one 32-bit
// word, or 0 for an empty or unrecognized type. MLX stores quantized weights
// packed into U32 words, so a tensor's logical last dimension is its stored
// last dimension times this factor.
func PackFactor(quantType string) int {
	if b := Bits(quantType); b > 0 && 32%b == 0 {
		return 32 / b
	}
	return 0
}

// LogicalColumns expands a U32-packed last dimension using the exact MLX
// relationship logical_columns = packed_columns * 32 / bits. The result is
// rejected when the division is not exact or multiplication would overflow.
func LogicalColumns(packedColumns uint64, quantType string) (uint64, bool) {
	bits := Bits(quantType)
	if bits == 0 || packedColumns > math.MaxUint64/32 {
		return 0, false
	}
	n := packedColumns * 32
	if n%uint64(bits) != 0 {
		return 0, false
	}
	return n / uint64(bits), true
}

// InferAffineParams derives an affine group size from packed weight and scale
// columns. A low-bit hint makes every MLX-supported width resolvable; without
// one, only the historical unambiguous int4/int8 layouts are accepted.
func InferAffineParams(packedColumns, scaleColumns uint64, hintBits int) (groupSize, bits int, ok bool) {
	groupForBits := func(bits int) (int, bool) {
		if packedColumns == 0 || scaleColumns == 0 || packedColumns > math.MaxUint64/32 || scaleColumns > math.MaxUint64/uint64(bits) {
			return 0, false
		}
		n, d := packedColumns*32, scaleColumns*uint64(bits)
		if d == 0 || n%d != 0 || n/d > uint64(math.MaxInt) {
			return 0, false
		}
		groupSize := int(n / d)
		switch groupSize {
		case 32, 64, 128:
			return groupSize, true
		default:
			return 0, false
		}
	}

	switch hintBits {
	case 2, 3, 4, 5, 6, 8:
		if groupSize, ok := groupForBits(hintBits); ok {
			return groupSize, hintBits, true
		}
		return 0, 0, false
	}

	groupSize4, ok4 := groupForBits(4)
	groupSize8, ok8 := groupForBits(8)
	switch {
	case ok4 && groupSize4 == 32:
		return groupSize4, 4, true
	case ok8 && groupSize8 == 64:
		return groupSize8, 8, true
	case ok4 && !ok8:
		return groupSize4, 4, true
	case ok8 && !ok4:
		return groupSize8, 8, true
	default:
		return 0, 0, false
	}
}
