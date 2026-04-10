package mlx

// #include "generated.h"
import "C"

import (
	"unsafe"
)

func FastScaledDotProductAttention(q, k, v *Array, scale float32, mode string, mask *Array) *Array {
	sinks := New("")
	cMode := C.CString(mode)

	var maskCtx C.mlx_array
	if mask != nil {
		maskCtx = mask.ctx
	} else {
		empty := New("")
		maskCtx = empty.ctx
	}

	out := New("FAST_SDPA")
	C.mlx_fast_scaled_dot_product_attention(&out.ctx, q.ctx, k.ctx, v.ctx, C.float(scale), cMode, maskCtx, sinks.ctx, DefaultStream().ctx)
	return out
}

func ScaledDotProductAttention(query, key, value, mask *Array, scale float32) *Array {
	return ScaledDotProductAttentionWithSinks(query, key, value, scale, "causal", mask, nil)
}

func ScaledDotProductAttentionWithSinks(query, key, value *Array, scale float32, maskMode string, mask, sinks *Array) *Array {
	if mask == nil {
		mask = New("")
	}
	if sinks == nil {
		sinks = New("")
	}

	cMode := C.CString(maskMode)
	defer C.free(unsafe.Pointer(cMode))

	out := New("FAST_SDPA")
	C.mlx_fast_scaled_dot_product_attention(&out.ctx, query.ctx, key.ctx, value.ctx, C.float(scale), cMode, mask.ctx, sinks.ctx, DefaultStream().ctx)
	return out
}

type LayerNorm struct {
	Weight *Array `weight:"weight"`
	Bias   *Array `weight:"bias"`
}

func (r *LayerNorm) Forward(x *Array, eps float32) *Array {
	out := New("FAST_LAYERNORM")
	C.mlx_fast_layer_norm(&out.ctx, x.ctx, r.Weight.ctx, r.Bias.ctx, C.float(eps), DefaultStream().ctx)
	return out
}

type RMSNorm struct {
	Weight *Array `weight:"weight"`
}

func (r *RMSNorm) Forward(x *Array, eps float32) *Array {
	out := New("FAST_RMSNORM")
	C.mlx_fast_rms_norm(&out.ctx, x.ctx, r.Weight.ctx, C.float(eps), DefaultStream().ctx)
	return out
}
