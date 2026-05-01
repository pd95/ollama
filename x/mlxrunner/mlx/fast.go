package mlx

// #include "generated.h"
import "C"

import (
	"unsafe"
)

func ScaledDotProductAttention(query, key, value, mask *Array, scale float32) *Array {
	return ScaledDotProductAttentionWithSinks(query, key, value, scale, "causal", mask, nil)
}

func ScaledDotProductAttentionWithSinks(query, key, value *Array, scale float32, maskMode string, mask, sinks *Array) *Array {
	return fastScaledDotProductAttention(query, key, value, scale, maskMode, mask, sinks)
}

func FastScaledDotProductAttention(q, k, v *Array, scale float32, mode string, mask *Array) *Array {
	return fastScaledDotProductAttention(q, k, v, scale, mode, mask, nil)
}

func FastScaledDotProductAttentionWithSinks(q, k, v *Array, scale float32, mode string, mask, sinks *Array) *Array {
	return fastScaledDotProductAttention(q, k, v, scale, mode, mask, sinks)
}

func fastScaledDotProductAttention(q, k, v *Array, scale float32, mode string, mask, sinks *Array) *Array {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))

	var maskCtx C.mlx_array
	if mask != nil {
		maskCtx = mask.ctx
	} else {
		empty := New("")
		maskCtx = empty.ctx
	}
	if sinks == nil {
		sinks = New("")
	}

	out := New("FAST_SDPA")
	C.mlx_fast_scaled_dot_product_attention(&out.ctx, q.ctx, k.ctx, v.ctx, C.float(scale), cMode, maskCtx, sinks.ctx, DefaultStream().ctx)
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
