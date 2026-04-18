package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

import (
	"sync"
	"unsafe"
)

var (
	moeSwiGLUOnce     sync.Once
	moeSwiGLUKernel   C.mlx_fast_metal_kernel
	moeSwiGLUDisabled bool

	moeDownOnce     sync.Once
	moeDownKernel   C.mlx_fast_metal_kernel
	moeDownDisabled bool
)

// moeSwiGLUSource is a fused gate+up+SwiGLU Metal kernel for MXFP4 MoE experts.
// Adapted from the gpt-oss reference kernel gptoss_f32_mf4w_moe_matmul_swiglu.
//
// Each threadgroup processes nSg rows total (even simdgroups compute gate,
// odd compute up). After a barrier, simdgroup pairs apply SwiGLU to produce
// nSg/2 output channels per threadgroup.
//
// Template args: NumColVecs (int), NumRows (int), NumTopK (int)
// Inputs (named): input(float), gate_w(uint), gate_s(bfloat), gate_b(bfloat),
//
//	up_w(uint), up_s(bfloat), up_b(bfloat),
//	expert_ids(uint), swiglu_params(float)
//
// Output (named): output(float)
const moeSwiGLUSource = `
uint3 gid = threadgroup_position_in_grid;
uint sgTid = thread_index_in_simdgroup;
uint sgIdx = simdgroup_index_in_threadgroup;
uint nSg = simdgroups_per_threadgroup;
constexpr uint sgSize = 32;

threadgroup float tg_buf[32];

uint outCh = gid.x * (nSg / 2) + (sgIdx / 2);
uint expertId = expert_ids[gid.y * NumTopK + gid.z];

uint rowStride = (uint)NumColVecs;
uint expertStride = (uint)NumRows * rowStride;

const device uint4* w;
const device uint8_t* ws;
const device bfloat* wb;

if (sgIdx % 2 == 0) {
    w = reinterpret_cast<const device uint4*>(gate_w) + expertId * expertStride + outCh * rowStride + sgTid;
    ws = gate_s + expertId * (uint)NumRows * (uint)NumColVecs + outCh * (uint)NumColVecs + sgTid;
    wb = gate_b + expertId * (uint)NumRows + outCh;
} else {
    w = reinterpret_cast<const device uint4*>(up_w) + expertId * expertStride + outCh * rowStride + sgTid;
    ws = up_s + expertId * (uint)NumRows * (uint)NumColVecs + outCh * (uint)NumColVecs + sgTid;
    wb = up_b + expertId * (uint)NumRows + outCh;
}

auto inp = reinterpret_cast<const device float4*>(input) + 8 * (gid.y * (uint)NumColVecs + sgTid);

uint numIter = ((uint)NumColVecs - sgTid + (sgSize - 1)) / sgSize;

float4 sum4 = 0.0f;
do {
    // Read 32 packed FP4 E2M1 values (uint4 = 4 × uint32) and E8M0 scale.
    // Each uint32 holds 8 nibbles: bits [3:0],[7:4],...,[31:28].
    // Dequant: (nibble & 7) << 9 → half, * 16384, sign from bit 3.
    const uint4 pk = *w;
    const float wscale = as_type<float>(uint(*ws) << 23);

    // Helper macro: extract 8 FP4 values from a uint32 into two float4 vectors.
    // lo = values from nibbles 0-3 (bits 0-15), hi = values from nibbles 4-7 (bits 16-31).
#define DEQUANT_U32(u, lo, hi) { \
    half _h0 = as_type<half>(ushort((u & 0x7u) << 9)) * 16384.0h; \
    half _h1 = as_type<half>(ushort(((u >> 4) & 0x7u) << 9)) * 16384.0h; \
    half _h2 = as_type<half>(ushort(((u >> 8) & 0x7u) << 9)) * 16384.0h; \
    half _h3 = as_type<half>(ushort(((u >> 12) & 0x7u) << 9)) * 16384.0h; \
    lo = float4((u & 0x8u) ? -float(_h0) : float(_h0), \
                ((u >> 4) & 0x8u) ? -float(_h1) : float(_h1), \
                ((u >> 8) & 0x8u) ? -float(_h2) : float(_h2), \
                ((u >> 12) & 0x8u) ? -float(_h3) : float(_h3)); \
    half _h4 = as_type<half>(ushort(((u >> 16) & 0x7u) << 9)) * 16384.0h; \
    half _h5 = as_type<half>(ushort(((u >> 20) & 0x7u) << 9)) * 16384.0h; \
    half _h6 = as_type<half>(ushort(((u >> 24) & 0x7u) << 9)) * 16384.0h; \
    half _h7 = as_type<half>(ushort(((u >> 28) & 0x7u) << 9)) * 16384.0h; \
    hi = float4(((u >> 16) & 0x8u) ? -float(_h4) : float(_h4), \
                ((u >> 20) & 0x8u) ? -float(_h5) : float(_h5), \
                ((u >> 24) & 0x8u) ? -float(_h6) : float(_h6), \
                ((u >> 28) & 0x8u) ? -float(_h7) : float(_h7)); \
}

    float4 wv0123, wv4567, wv89AB, wvCDEF, wvGHIJ, wvKLMN, wvOPQR, wvSTUV;
    DEQUANT_U32(pk.x, wv0123, wv4567)  // values 0-7
    DEQUANT_U32(pk.y, wv89AB, wvCDEF)  // values 8-15
    DEQUANT_U32(pk.z, wvGHIJ, wvKLMN)  // values 16-23
    DEQUANT_U32(pk.w, wvOPQR, wvSTUV)  // values 24-31
#undef DEQUANT_U32

    const float4 i0123 = inp[0];
    const float4 i4567 = inp[1];
    const float4 i89AB = inp[2];
    const float4 iCDEF = inp[3];
    const float4 iGHIJ = inp[4];
    const float4 iKLMN = inp[5];
    const float4 iOPQR = inp[6];
    const float4 iSTUV = inp[7];

    float4 psum0 = i0123 * wv0123;
    float4 psum1 = i4567 * wv4567;
    psum0 = fma(i89AB, wv89AB, psum0);
    psum1 = fma(iCDEF, wvCDEF, psum1);
    psum0 = fma(iGHIJ, wvGHIJ, psum0);
    psum1 = fma(iKLMN, wvKLMN, psum1);
    psum0 = fma(iOPQR, wvOPQR, psum0);
    psum1 = fma(iSTUV, wvSTUV, psum1);
    sum4 = fma(psum0, wscale, sum4);
    sum4 = fma(psum1, wscale, sum4);

    w += sgSize;
    ws += sgSize;
    inp += 8 * sgSize;
} while (--numIter != 0);

const float2 sum2 = sum4.xy + sum4.zw;
float sum = sum2.x + sum2.y;
sum = simd_sum(sum);

if (simd_is_first()) {
    sum += static_cast<float>(*wb);
    tg_buf[sgIdx] = sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint tid = sgIdx * sgSize + sgTid;
if (tid * 2 < nSg) {
    const float2 gu = reinterpret_cast<const threadgroup float2*>(tg_buf)[tid];
    const float smin = swiglu_params[0];
    const float smax = swiglu_params[1];
    const float gate_val = min(gu.x, smax);
    const float up_val = clamp(gu.y, smin, smax);
    const float alpha = 1.702f;
    const float swish = gate_val / (1.0f + precise::exp(-alpha * gate_val));
    const float result = fma(swish, up_val, swish);

    uint ch = gid.x * (nSg / 2) + tid;
    output[gid.y * (uint)NumTopK * (uint)NumRows + gid.z * (uint)NumRows + ch] = result;
}
`

func initMoESwiGLUKernel() {
	inputs, freeInputs, ok := cStringVector([]string{
		"input", "gate_w", "gate_s", "gate_b",
		"up_w", "up_s", "up_b",
		"expert_ids", "swiglu_params",
	})
	if !ok {
		moeSwiGLUDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"output"})
	if !ok {
		moeSwiGLUDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("gptoss_moe_swiglu")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(moeSwiGLUSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	moeSwiGLUKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),  // ensure_row_contiguous
		C.bool(false), // atomic_outputs
	)
}

// MoEFusedGateUpSwiGLU runs a fused gate+up+SwiGLU kernel for MXFP4 MoE experts.
// It returns (result, true) on success, or (nil, false) if the kernel is unavailable
// or the inputs are incompatible.
//
// Parameters:
//   - input: float32 [batch, hiddenSize] — flattened input hidden states
//   - gateWeight, gateScales, gateBias: MXFP4 gate projection weights
//   - upWeight, upScales, upBias: MXFP4 up projection weights
//   - expertIds: uint32 [batch, topK] — expert indices per token
//   - numRows: output dimension of gate/up projections
//   - numColVecs: hiddenSize / 32 (MXFP4 groups per row)
//   - topK: number of active experts per token
//   - swigluMin, swigluMax: SwiGLU clamp parameters
func MoEFusedGateUpSwiGLU(
	input, gateWeight, gateScales, gateBias,
	upWeight, upScales, upBias, expertIds *Array,
	numRows, numColVecs, topK int,
	swigluMin, swigluMax float32,
) (*Array, bool) {
	if moeSwiGLUDisabled {
		return nil, false
	}
	if input == nil || gateWeight == nil || gateScales == nil || gateBias == nil ||
		upWeight == nil || upScales == nil || upBias == nil || expertIds == nil {
		return nil, false
	}

	moeSwiGLUOnce.Do(initMoESwiGLUKernel)
	if moeSwiGLUDisabled {
		return nil, false
	}

	// Determine batch size from input shape
	inDims := input.Dims()
	if len(inDims) < 1 {
		return nil, false
	}
	batch := inDims[0]

	// Choose simdgroup count (must be even, numRows divisible by nSg/2)
	nSg := 8
	for nSg > 2 && numRows%(nSg/2) != 0 {
		nSg /= 2
	}
	if nSg < 2 || numRows%(nSg/2) != 0 {
		return nil, false
	}

	// Configure template args
	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)

	for _, tpl := range []struct {
		name  string
		value int
	}{
		{"NumColVecs", numColVecs},
		{"NumRows", numRows},
		{"NumTopK", topK},
	} {
		cn := C.CString(tpl.name)
		rc := C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cn, C.int(tpl.value))
		C.free(unsafe.Pointer(cn))
		if rc != 0 {
			moeSwiGLUDisabled = true
			return nil, false
		}
	}

	// Output shape: [batch, topK, numRows]
	outShape := []C.int{C.int(batch), C.int(topK), C.int(numRows)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(outShape), C.size_t(len(outShape)), C.mlx_dtype(DTypeFloat32)) != 0 {
		moeSwiGLUDisabled = true
		return nil, false
	}

	// MLX dispatch_threads takes total thread count, not threadgroup count.
	// We need numRows/(nSg/2) threadgroups, each with 32*nSg threads.
	tgSize := 32 * nSg
	numTGx := numRows / (nSg / 2)
	gridX := numTGx * tgSize
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(gridX), C.int(batch), C.int(topK)) != 0 {
		moeSwiGLUDisabled = true
		return nil, false
	}

	// Threadgroup: (32 * nSg, 1, 1)
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, C.int(tgSize), 1, 1) != 0 {
		moeSwiGLUDisabled = true
		return nil, false
	}

	// Build swiglu params array
	swigluParams := FromValues([]float32{swigluMin, swigluMax}, 2)

	// Assemble inputs
	inputs := []C.mlx_array{
		input.ctx,
		gateWeight.ctx,
		gateScales.ctx,
		gateBias.ctx,
		upWeight.ctx,
		upScales.ctx,
		upBias.ctx,
		expertIds.ctx,
		swigluParams.ctx,
	}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, moeSwiGLUKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		moeSwiGLUDisabled = true
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 1 {
		return nil, false
	}

	out := New("MOE_FUSED_SWIGLU")
	C.mlx_vector_array_get(&out.ctx, outVec, 0)
	return out, true
}

// moeDownSource is a fused down-projection Metal kernel for MXFP4 MoE experts.
// Adapted from the gpt-oss reference kernel gptoss_f32_mf4w_moe_matmul.
//
// Each simdgroup computes one output channel (dot-product of input row against
// one MXFP4-packed weight row, plus bias). One threadgroup produces nSg output
// channels. No SwiGLU — this is a plain matmul+bias.
//
// The input is per-expert: input[gid.y * NumTopK * NumInCols + gid.z * NumInCols + ...]
// where NumInCols is the intermediate_size (in float32 elements).
//
// Template args: NumColVecs (int), NumRows (int), NumTopK (int)
// Inputs (named): input(float), down_w(uint), down_s(bfloat), down_b(bfloat),
//
//	expert_ids(uint)
//
// Output (named): output(float)
const moeDownSource = `
uint3 gid = threadgroup_position_in_grid;
uint sgTid = thread_index_in_simdgroup;
uint sgIdx = simdgroup_index_in_threadgroup;
uint nSg = simdgroups_per_threadgroup;
constexpr uint sgSize = 32;

uint outCh = gid.x * nSg + sgIdx;
uint expertId = expert_ids[gid.y * NumTopK + gid.z];

uint rowStride = (uint)NumColVecs;
uint expertStride = (uint)NumRows * rowStride;

const device uint4* w = reinterpret_cast<const device uint4*>(down_w) + expertId * expertStride + outCh * rowStride + sgTid;
const device uint8_t* ws = down_s + expertId * (uint)NumRows * (uint)NumColVecs + outCh * (uint)NumColVecs + sgTid;
const device bfloat* wb = down_b + expertId * (uint)NumRows + outCh;

auto inp = reinterpret_cast<const device float4*>(input) + 8 * (gid.y * NumTopK * (uint)NumColVecs + gid.z * (uint)NumColVecs + sgTid);

uint numIter = ((uint)NumColVecs - sgTid + (sgSize - 1)) / sgSize;

float4 sum4 = 0.0f;
do {
    const uint4 pk = *w;
    const float wscale = as_type<float>(uint(*ws) << 23);

#define DEQUANT_U32(u, lo, hi) { \
    half _h0 = as_type<half>(ushort((u & 0x7u) << 9)) * 16384.0h; \
    half _h1 = as_type<half>(ushort(((u >> 4) & 0x7u) << 9)) * 16384.0h; \
    half _h2 = as_type<half>(ushort(((u >> 8) & 0x7u) << 9)) * 16384.0h; \
    half _h3 = as_type<half>(ushort(((u >> 12) & 0x7u) << 9)) * 16384.0h; \
    lo = float4((u & 0x8u) ? -float(_h0) : float(_h0), \
                ((u >> 4) & 0x8u) ? -float(_h1) : float(_h1), \
                ((u >> 8) & 0x8u) ? -float(_h2) : float(_h2), \
                ((u >> 12) & 0x8u) ? -float(_h3) : float(_h3)); \
    half _h4 = as_type<half>(ushort(((u >> 16) & 0x7u) << 9)) * 16384.0h; \
    half _h5 = as_type<half>(ushort(((u >> 20) & 0x7u) << 9)) * 16384.0h; \
    half _h6 = as_type<half>(ushort(((u >> 24) & 0x7u) << 9)) * 16384.0h; \
    half _h7 = as_type<half>(ushort(((u >> 28) & 0x7u) << 9)) * 16384.0h; \
    hi = float4(((u >> 16) & 0x8u) ? -float(_h4) : float(_h4), \
                ((u >> 20) & 0x8u) ? -float(_h5) : float(_h5), \
                ((u >> 24) & 0x8u) ? -float(_h6) : float(_h6), \
                ((u >> 28) & 0x8u) ? -float(_h7) : float(_h7)); \
}

    float4 wv0123, wv4567, wv89AB, wvCDEF, wvGHIJ, wvKLMN, wvOPQR, wvSTUV;
    DEQUANT_U32(pk.x, wv0123, wv4567)
    DEQUANT_U32(pk.y, wv89AB, wvCDEF)
    DEQUANT_U32(pk.z, wvGHIJ, wvKLMN)
    DEQUANT_U32(pk.w, wvOPQR, wvSTUV)
#undef DEQUANT_U32

    const float4 i0123 = inp[0];
    const float4 i4567 = inp[1];
    const float4 i89AB = inp[2];
    const float4 iCDEF = inp[3];
    const float4 iGHIJ = inp[4];
    const float4 iKLMN = inp[5];
    const float4 iOPQR = inp[6];
    const float4 iSTUV = inp[7];

    float4 psum0 = i0123 * wv0123;
    float4 psum1 = i4567 * wv4567;
    psum0 = fma(i89AB, wv89AB, psum0);
    psum1 = fma(iCDEF, wvCDEF, psum1);
    psum0 = fma(iGHIJ, wvGHIJ, psum0);
    psum1 = fma(iKLMN, wvKLMN, psum1);
    psum0 = fma(iOPQR, wvOPQR, psum0);
    psum1 = fma(iSTUV, wvSTUV, psum1);
    sum4 = fma(psum0, wscale, sum4);
    sum4 = fma(psum1, wscale, sum4);

    w += sgSize;
    ws += sgSize;
    inp += 8 * sgSize;
} while (--numIter != 0);

const float2 sum2 = sum4.xy + sum4.zw;
float sum = sum2.x + sum2.y;
sum = simd_sum(sum);

if (simd_is_first()) {
    sum += static_cast<float>(*wb);
    output[gid.y * (uint)NumTopK * (uint)NumRows + gid.z * (uint)NumRows + outCh] = sum;
}
`

func initMoEDownKernel() {
	inputs, freeInputs, ok := cStringVector([]string{
		"input", "down_w", "down_s", "down_b", "expert_ids",
	})
	if !ok {
		moeDownDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"output"})
	if !ok {
		moeDownDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("gptoss_moe_down")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(moeDownSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	moeDownKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),  // ensure_row_contiguous
		C.bool(false), // atomic_outputs
	)
}

// MoEFusedDown runs a fused down-projection kernel for MXFP4 MoE experts.
// It returns (result, true) on success, or (nil, false) if the kernel is unavailable
// or the inputs are incompatible.
//
// Parameters:
//   - input: float32 [batch, topK, intermediateSize] — per-expert SwiGLU output
//   - downWeight, downScales, downBias: MXFP4 down projection weights
//   - expertIds: uint32 [batch, topK] — expert indices per token
//   - numRows: output dimension (hiddenSize)
//   - numColVecs: intermediateSize / 32 (MXFP4 groups per row)
//   - topK: number of active experts per token
func MoEFusedDown(
	input, downWeight, downScales, downBias, expertIds *Array,
	numRows, numColVecs, topK int,
) (*Array, bool) {
	if moeDownDisabled {
		return nil, false
	}
	if input == nil || downWeight == nil || downScales == nil || downBias == nil || expertIds == nil {
		return nil, false
	}

	moeDownOnce.Do(initMoEDownKernel)
	if moeDownDisabled {
		return nil, false
	}

	// Determine batch size from input shape [batch, topK, intermediateSize]
	inDims := input.Dims()
	if len(inDims) < 1 {
		return nil, false
	}
	batch := inDims[0]

	// Each simdgroup computes one output channel. Choose nSg so numRows is divisible.
	nSg := 8
	for nSg > 1 && numRows%nSg != 0 {
		nSg /= 2
	}
	if nSg < 1 || numRows%nSg != 0 {
		return nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)

	for _, tpl := range []struct {
		name  string
		value int
	}{
		{"NumColVecs", numColVecs},
		{"NumRows", numRows},
		{"NumTopK", topK},
	} {
		cn := C.CString(tpl.name)
		rc := C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cn, C.int(tpl.value))
		C.free(unsafe.Pointer(cn))
		if rc != 0 {
			moeDownDisabled = true
			return nil, false
		}
	}

	// Output shape: [batch, topK, numRows]
	outShape := []C.int{C.int(batch), C.int(topK), C.int(numRows)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(outShape), C.size_t(len(outShape)), C.mlx_dtype(DTypeFloat32)) != 0 {
		moeDownDisabled = true
		return nil, false
	}

	// Grid: numRows/nSg threadgroups in X, batch in Y, topK in Z
	// Each threadgroup has 32*nSg threads
	tgSize := 32 * nSg
	numTGx := numRows / nSg
	gridX := numTGx * tgSize
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(gridX), C.int(batch), C.int(topK)) != 0 {
		moeDownDisabled = true
		return nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, C.int(tgSize), 1, 1) != 0 {
		moeDownDisabled = true
		return nil, false
	}

	inputs := []C.mlx_array{
		input.ctx,
		downWeight.ctx,
		downScales.ctx,
		downBias.ctx,
		expertIds.ctx,
	}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, moeDownKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		moeDownDisabled = true
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 1 {
		return nil, false
	}

	out := New("MOE_FUSED_DOWN")
	C.mlx_vector_array_get(&out.ctx, outVec, 0)
	return out, true
}
