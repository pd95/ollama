package mlx

// #include "generated.h"
import "C"
import "math"

func GELUApprox(t *Array) *Array {
	return t.Multiply(
		FromValue[float32](0.5),
	).Multiply(
		t.Add(
			t.Power(FromValue[float32](3.0)).Multiply(FromValue[float32](0.044715)),
		).Multiply(
			FromValue(float32(math.Sqrt(2 / math.Pi))),
		).Tanh().Add(FromValue[float32](1.0)),
	).AsType(t.DType())
}

func SILU(t *Array) *Array {
	return t.Multiply(t.Sigmoid()).AsType(t.DType())
}

func clipScalar(t *Array, low, high float32, hasLow, hasHigh bool) *Array {
	out := t
	if hasLow {
		lowv := FromValue[float32](low).AsType(t.DType())
		out = Where(out.Less(lowv), lowv, out)
	}
	if hasHigh {
		highv := FromValue[float32](high).AsType(t.DType())
		out = Where(highv.Less(out), highv, out)
	}
	return out
}

func SwiGLUAlphaLimit(gate, up *Array, alpha, limit float32) *Array {
	gateClipped := clipScalar(gate, 0, limit, false, true)
	upClipped := clipScalar(up, -limit, limit, true, true)
	outGate := gateClipped.Multiply(MulScalar(gateClipped, alpha).Sigmoid())
	return outGate.Multiply(AddScalar(upClipped, 1.0))
}
