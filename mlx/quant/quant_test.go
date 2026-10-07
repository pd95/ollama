package quant

import "testing"

func TestParams(t *testing.T) {
	tests := []struct {
		in        string
		groupSize int
		bits      int
		mode      string
	}{
		{"nvfp4", 16, 4, "nvfp4"},
		{"NVFP4", 16, 4, "nvfp4"},
		{"mxfp4", 32, 4, "mxfp4"},
		{"mxfp8", 32, 8, "mxfp8"},
		{"int2", 64, 2, "affine"},
		{"q2", 64, 2, "affine"},
		{"int3", 64, 3, "affine"},
		{"q3", 64, 3, "affine"},
		{"int4", 64, 4, "affine"},
		{"int5", 64, 5, "affine"},
		{"q5", 64, 5, "affine"},
		{"int6", 64, 6, "affine"},
		{"q6", 64, 6, "affine"},
		{"int8", 64, 8, "affine"},
		{"fp4", 64, 4, "affine"},
		{"q4", 64, 4, "affine"},
		{"fp8", 64, 8, "affine"},
		{"q8", 64, 8, "affine"},
		{"", 0, 0, ""},
		// Unknown non-empty types fall back to 8-bit affine, matching the
		// runtime loader's historical behavior.
		{"something-else", 32, 8, "affine"},
	}
	for _, tt := range tests {
		gs, bits, mode := Params(tt.in)
		if gs != tt.groupSize || bits != tt.bits || mode != tt.mode {
			t.Errorf("Params(%q) = (%d, %d, %q), want (%d, %d, %q)", tt.in, gs, bits, mode, tt.groupSize, tt.bits, tt.mode)
		}
	}
}

func TestImportableAndCreatable(t *testing.T) {
	for _, qt := range []string{"int2", "int3", "int4", "int5", "int6", "int8", "nvfp4", "mxfp4", "mxfp8"} {
		if !Importable(qt) {
			t.Errorf("Importable(%q) = false", qt)
		}
	}
	for _, qt := range []string{"int4", "int8", "nvfp4", "mxfp4", "mxfp8"} {
		if !Creatable(qt) {
			t.Errorf("Creatable(%q) = false", qt)
		}
	}
	for _, qt := range []string{"int2", "int3", "int5", "int6", "int7", ""} {
		if Creatable(qt) {
			t.Errorf("Creatable(%q) = true", qt)
		}
	}
	if Importable("int7") {
		t.Error("Importable(int7) = true")
	}
}

func TestBitsAndPackFactor(t *testing.T) {
	tests := []struct {
		in         string
		bits       int
		packFactor int
	}{
		{"int4", 4, 8},
		{"nvfp4", 4, 8},
		{"mxfp4", 4, 8}, // regression: mxfp4 was missing from the old show.go switch
		{"int3", 3, 0},  // exact expansion is rational, not an integer factor
		{"int6", 6, 0},
		{"int8", 8, 4},
		{"mxfp8", 8, 4},
		{"FP8", 8, 4},
		{"", 0, 0},
		{"unknown", 0, 0}, // strict: no fallback for sizing/display
	}
	for _, tt := range tests {
		if got := Bits(tt.in); got != tt.bits {
			t.Errorf("Bits(%q) = %d, want %d", tt.in, got, tt.bits)
		}
		if got := PackFactor(tt.in); got != tt.packFactor {
			t.Errorf("PackFactor(%q) = %d, want %d", tt.in, got, tt.packFactor)
		}
	}
}

func TestLogicalColumns(t *testing.T) {
	tests := []struct {
		packed uint64
		qt     string
		want   uint64
		ok     bool
	}{
		{2, "int2", 32, true},
		{3, "int3", 32, true},
		{4, "int4", 32, true},
		{5, "int5", 32, true},
		{6, "int6", 32, true},
		{8, "int8", 32, true},
		{1, "int3", 0, false},
		{^uint64(0), "int4", 0, false},
		{4, "int7", 0, false},
	}
	for _, tt := range tests {
		got, ok := LogicalColumns(tt.packed, tt.qt)
		if got != tt.want || ok != tt.ok {
			t.Errorf("LogicalColumns(%d, %q) = (%d, %v), want (%d, %v)", tt.packed, tt.qt, got, ok, tt.want, tt.ok)
		}
	}
}

func TestInferAffineParams(t *testing.T) {
	for _, bits := range []int{2, 3, 4, 5, 6, 8} {
		packed := uint64(64 * bits / 32)
		groupSize, gotBits, ok := InferAffineParams(packed, 1, bits)
		if !ok || groupSize != 64 || gotBits != bits {
			t.Errorf("InferAffineParams(%d, 1, %d) = (%d, %d, %v), want (64, %d, true)", packed, bits, groupSize, gotBits, ok, bits)
		}
	}
	if _, _, ok := InferAffineParams(6, 1, 0); ok {
		t.Error("metadata-free int3/int6-ambiguous layout unexpectedly inferred")
	}
	if gs, bits, ok := InferAffineParams(4, 1, 0); !ok || gs != 32 || bits != 4 {
		t.Errorf("legacy int4 inference = (%d, %d, %v)", gs, bits, ok)
	}
	if gs, bits, ok := InferAffineParams(16, 1, 0); !ok || gs != 64 || bits != 8 {
		t.Errorf("legacy int8 inference = (%d, %d, %v)", gs, bits, ok)
	}
	if _, _, ok := InferAffineParams(^uint64(0), 1, 4); ok {
		t.Error("overflowing inference unexpectedly succeeded")
	}
}
