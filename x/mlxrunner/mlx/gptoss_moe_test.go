package mlx

import (
	"math"
	"testing"
)

func TestCheckedMoESpan(t *testing.T) {
	tests := []struct {
		name    string
		factors []uint64
		want    uint64
		ok      bool
	}{
		{name: "single", factors: []uint64{1}, want: 1, ok: true},
		{name: "maximum index", factors: []uint64{65536, 65536}, want: uint64(math.MaxUint32) + 1, ok: true},
		{name: "first past maximum index", factors: []uint64{65536, 65537}, ok: false},
		{name: "zero", factors: []uint64{1, 0}, ok: false},
		{name: "uint64 overflow", factors: []uint64{math.MaxUint64, 2}, ok: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := checkedMoESpan(tt.factors...)
			if ok != tt.ok || got != tt.want {
				t.Fatalf("checkedMoESpan(%v) = (%d, %v), want (%d, %v)", tt.factors, got, ok, tt.want, tt.ok)
			}
		})
	}
}

func TestValidateMoEIndexSpans(t *testing.T) {
	tests := []struct {
		name                                string
		experts, batch, rows, colVecs, topK int
		down, want                          bool
	}{
		{name: "valid ordinary gate", experts: 8, batch: 4, rows: 256, colVecs: 64, topK: 4, want: true},
		{name: "valid exact uint32 weight span", experts: 1, batch: 1, rows: 65536, colVecs: 65536, topK: 1, want: true},
		{name: "expert stride overflow", experts: 1, batch: 1, rows: 65536, colVecs: 65537, topK: 1},
		{name: "full expert weight scale overflow", experts: 65537, batch: 1, rows: 65536, colVecs: 1, topK: 1},
		{name: "full bias overflow", experts: 65536, batch: 1, rows: 65537, colVecs: 1, topK: 1},
		{name: "selector overflow", experts: 1, batch: 65536, rows: 1, colVecs: 1, topK: 65537},
		{name: "output overflow", experts: 1, batch: 65536, rows: 2, colVecs: 1, topK: 65536},
		{name: "gate input factor overflow", experts: 1, batch: 32768, rows: 1, colVecs: 16385, topK: 1},
		{name: "valid ordinary down", experts: 8, batch: 4, rows: 256, colVecs: 64, topK: 4, down: true, want: true},
		{name: "down input factor overflow", experts: 1, batch: 32768, rows: 1, colVecs: 16385, topK: 1, down: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := validateMoEIndexSpans(tt.experts, tt.batch, tt.rows, tt.colVecs, tt.topK, tt.down); got != tt.want {
				t.Fatalf("validateMoEIndexSpans() = %v, want %v", got, tt.want)
			}
		})
	}
}

type moeTestInputs struct {
	input, weight, scales, bias, expertIDs *Array
	upWeight, upScales, upBias             *Array
}

func newMoETestInputs(expertIDs []uint32) moeTestInputs {
	const experts, rows, colVecs, topK = 2, 4, 32, 1
	batch := len(expertIDs) / topK
	return moeTestInputs{
		input:     Zeros(DTypeFloat32, batch, colVecs*32),
		weight:    Zeros(DTypeUint32, experts, rows, colVecs*4),
		scales:    Zeros(DTypeUint8, experts, rows, colVecs),
		bias:      Zeros(DTypeBFloat16, experts, rows),
		expertIDs: FromValues(expertIDs, batch, topK),
		upWeight:  Zeros(DTypeUint32, experts, rows, colVecs*4),
		upScales:  Zeros(DTypeUint8, experts, rows, colVecs),
		upBias:    Zeros(DTypeBFloat16, experts, rows),
	}
}

func TestValidateMoEInputs(t *testing.T) {
	skipIfNoMLX(t)
	withMLXThread(t, func() {
		valid := newMoETestInputs([]uint32{1})
		if _, ok := validateMoEGateUpInputs(valid.input, valid.weight, valid.scales, valid.bias, valid.upWeight, valid.upScales, valid.upBias, valid.expertIDs, 4, 32, 1); !ok {
			t.Fatal("valid gate/up inputs rejected")
		}

		downInput := Zeros(DTypeFloat32, 1, 1, 32*32)
		if _, ok := validateMoEDownInputs(downInput, valid.weight, valid.scales, valid.bias, valid.expertIDs, 4, 32, 1); !ok {
			t.Fatal("valid down inputs rejected")
		}

		for _, tt := range []struct {
			name string
			fn   func() bool
		}{
			{name: "expert equal to count", fn: func() bool {
				bad := newMoETestInputs([]uint32{2})
				_, ok := validateMoEGateUpInputs(bad.input, bad.weight, bad.scales, bad.bias, bad.upWeight, bad.upScales, bad.upBias, bad.expertIDs, 4, 32, 1)
				return ok
			}},
			{name: "maximum expert", fn: func() bool {
				bad := newMoETestInputs([]uint32{math.MaxUint32})
				_, ok := validateMoEDownInputs(downInput, bad.weight, bad.scales, bad.bias, bad.expertIDs, 4, 32, 1)
				return ok
			}},
			{name: "wrong selector dtype", fn: func() bool {
				ids := FromValues([]int32{0}, 1, 1)
				_, ok := validateMoEDownInputs(downInput, valid.weight, valid.scales, valid.bias, ids, 4, 32, 1)
				return ok
			}},
			{name: "wrong input rank", fn: func() bool {
				input := Zeros(DTypeFloat32, 1, 1, 32*32)
				_, ok := validateMoEGateUpInputs(input, valid.weight, valid.scales, valid.bias, valid.upWeight, valid.upScales, valid.upBias, valid.expertIDs, 4, 32, 1)
				return ok
			}},
			{name: "wrong input dtype", fn: func() bool {
				input := Zeros(DTypeBFloat16, 1, 32*32)
				_, ok := validateMoEGateUpInputs(input, valid.weight, valid.scales, valid.bias, valid.upWeight, valid.upScales, valid.upBias, valid.expertIDs, 4, 32, 1)
				return ok
			}},
			{name: "wrong weight dtype", fn: func() bool {
				weight := Zeros(DTypeUint8, 2, 4, 32*4)
				_, ok := validateMoEDownInputs(downInput, weight, valid.scales, valid.bias, valid.expertIDs, 4, 32, 1)
				return ok
			}},
			{name: "wrong weight rank", fn: func() bool {
				weight := Zeros(DTypeUint32, 2, 4*32*4)
				_, ok := validateMoEDownInputs(downInput, weight, valid.scales, valid.bias, valid.expertIDs, 4, 32, 1)
				return ok
			}},
			{name: "gate up expert mismatch", fn: func() bool {
				up := Zeros(DTypeUint32, 3, 4, 32*4)
				_, ok := validateMoEGateUpInputs(valid.input, valid.weight, valid.scales, valid.bias, up, valid.upScales, valid.upBias, valid.expertIDs, 4, 32, 1)
				return ok
			}},
			{name: "malformed scale", fn: func() bool {
				scale := Zeros(DTypeUint8, 2, 4, 31)
				_, ok := validateMoEDownInputs(downInput, valid.weight, scale, valid.bias, valid.expertIDs, 4, 32, 1)
				return ok
			}},
			{name: "malformed bias", fn: func() bool {
				bias := Zeros(DTypeBFloat16, 2, 5)
				_, ok := validateMoEDownInputs(downInput, valid.weight, valid.scales, bias, valid.expertIDs, 4, 32, 1)
				return ok
			}},
			{name: "missing scale", fn: func() bool {
				_, ok := validateMoEDownInputs(downInput, valid.weight, nil, valid.bias, valid.expertIDs, 4, 32, 1)
				return ok
			}},
			{name: "missing bias", fn: func() bool {
				_, ok := validateMoEDownInputs(downInput, valid.weight, valid.scales, nil, valid.expertIDs, 4, 32, 1)
				return ok
			}},
			{name: "wrong selector shape", fn: func() bool {
				ids := FromValues([]uint32{0}, 1)
				_, ok := validateMoEDownInputs(downInput, valid.weight, valid.scales, valid.bias, ids, 4, 32, 1)
				return ok
			}},
			{name: "zero topk", fn: func() bool {
				_, ok := validateMoEDownInputs(downInput, valid.weight, valid.scales, valid.bias, valid.expertIDs, 4, 32, 0)
				return ok
			}},
			{name: "oversized topk", fn: func() bool {
				_, ok := validateMoEDownInputs(downInput, valid.weight, valid.scales, valid.bias, valid.expertIDs, 4, 32, 3)
				return ok
			}},
		} {
			t.Run(tt.name, func(t *testing.T) {
				if tt.fn() {
					t.Fatal("malformed inputs accepted")
				}
			})
		}
	})
}

func TestMoEFusedZeroReferenceAndValidationIsolation(t *testing.T) {
	skipIfNoMLX(t)
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("Metal is not available")
		}

		bad := newMoETestInputs([]uint32{2})
		if out, ok := MoEFusedGateUpSwiGLU(bad.input, bad.weight, bad.scales, bad.bias, bad.upWeight, bad.upScales, bad.upBias, bad.expertIDs, 4, 32, 1, -7, 7); ok || out != nil {
			t.Fatal("out-of-range gate/up selector accepted")
		}

		valid := newMoETestInputs([]uint32{1})
		gateOut, ok := MoEFusedGateUpSwiGLU(valid.input, valid.weight, valid.scales, valid.bias, valid.upWeight, valid.upScales, valid.upBias, valid.expertIDs, 4, 32, 1, -7, 7)
		if !ok {
			t.Fatal("valid gate/up call failed after malformed call")
		}
		Eval(gateOut)
		assertMoEZeroReference(t, gateOut, []int{1, 1, 4})

		downInput := Zeros(DTypeFloat32, 1, 1, 32*32)
		if out, ok := MoEFusedDown(downInput, valid.weight, valid.scales, valid.bias, bad.expertIDs, 4, 32, 1); ok || out != nil {
			t.Fatal("out-of-range down selector accepted")
		}
		downOut, ok := MoEFusedDown(downInput, valid.weight, valid.scales, valid.bias, valid.expertIDs, 4, 32, 1)
		if !ok {
			t.Fatal("valid down call failed after malformed call")
		}
		Eval(downOut)
		assertMoEZeroReference(t, downOut, []int{1, 1, 4})
	})
}

func assertMoEZeroReference(t *testing.T, got *Array, wantShape []int) {
	t.Helper()
	if dims := got.Dims(); len(dims) != len(wantShape) {
		t.Fatalf("shape = %v, want %v", dims, wantShape)
	} else {
		for i := range dims {
			if dims[i] != wantShape[i] {
				t.Fatalf("shape = %v, want %v", dims, wantShape)
			}
		}
	}
	for i, value := range got.Floats() {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value)) > 1e-6 {
			t.Fatalf("output[%d] = %v, want zero reference within 1e-6", i, value)
		}
	}
}
