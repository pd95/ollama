package mlx

// #include "generated.h"
import "C"

import (
	"fmt"
	"unsafe"
)

func FastScaledDotProductAttention(q, k, v *Array, scale float32, mode string, mask *Array, sinkArr ...*Array) *Array {
	sinks := New("")
	if len(sinkArr) > 1 {
		panic("mlx.FastScaledDotProductAttention: at most one sinks array is allowed")
	}
	if len(sinkArr) == 1 && sinkArr[0] != nil {
		sinks = sinkArr[0]
		if q == nil || !q.Valid() || sinks == nil || !sinks.Valid() {
			panic("mlx.FastScaledDotProductAttention: query and sinks must be valid arrays")
		}
		if q.NumDims() != 4 {
			panic(fmt.Sprintf("mlx.FastScaledDotProductAttention: query with sinks must have rank 4, got shape %v", q.Dims()))
		}
		if sinks.NumDims() != 1 || sinks.Dim(0) != q.Dim(1) || sinks.DType() != q.DType() {
			panic(fmt.Sprintf("mlx.FastScaledDotProductAttention: sinks must have shape [heads]=[%d] and dtype %s for query %v, got shape %v dtype %s", q.Dim(1), q.DType(), q.Dims(), sinks.Dims(), sinks.DType()))
		}
	}
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))

	var maskCtx C.mlx_array
	if mask != nil {
		maskCtx = mask.ctx
	} else {
		empty := New("")
		maskCtx = empty.ctx
	}

	out := New("FAST_SDPA")
	mlxCheck(C.mlx_fast_scaled_dot_product_attention(&out.ctx, q.ctx, k.ctx, v.ctx, C.float(scale), cMode, maskCtx, sinks.ctx, C.bool(false), DefaultStream().ctx))
	return out
}

type LayerNorm struct {
	Weight *Array `weight:"weight"`
	Bias   *Array `weight:"bias"`
}

func (r *LayerNorm) Forward(x *Array, eps float32) *Array {
	out := New("FAST_LAYERNORM")
	mlxCheck(C.mlx_fast_layer_norm(&out.ctx, x.ctx, r.Weight.ctx, r.Bias.ctx, C.float(eps), DefaultStream().ctx))
	return out
}

type RMSNorm struct {
	Weight *Array `weight:"weight"`
}

func (r *RMSNorm) Forward(x *Array, eps float32) *Array {
	out := New("FAST_RMSNORM")
	mlxCheck(C.mlx_fast_rms_norm(&out.ctx, x.ctx, r.Weight.ctx, C.float(eps), DefaultStream().ctx))
	return out
}
