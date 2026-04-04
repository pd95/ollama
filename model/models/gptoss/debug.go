package gptoss

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"sync/atomic"

	"github.com/ollama/ollama/ml"
)

var (
	stableDebugEnabled = os.Getenv("OLLAMA_GPTOSS_ACT_DEBUG") != ""
	stableDebugConfig  atomic.Bool
	stableDebugPrefill atomic.Bool
	stableDebugDecode  atomic.Bool
	stableDebugLayer   atomic.Int32
	stableDebugPhase   atomic.Int32
	stableDebugForce   atomic.Int32
)

const (
	stableDebugPhaseNone int32 = iota
	stableDebugPhasePrefill
	stableDebugPhaseDecode
)

type stableTopKV struct {
	Index int
	Value float32
}

func stableDebugSelectedIndex(batchSeq int) int {
	if stableDebugPhase.Load() == stableDebugPhasePrefill && batchSeq > 0 {
		return batchSeq - 1
	}
	return 0
}

func stableDebugBegin(batchSeq, batchSize int, positions []int32, opts *Options) int32 {
	if !stableDebugEnabled || len(positions) == 0 {
		stableDebugPhase.Store(stableDebugPhaseNone)
		return stableDebugPhaseNone
	}
	if stableDebugConfig.CompareAndSwap(false, true) {
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=config layer=-1 heads=%d kv_heads=%d head_dim=%d rope_base=%g rope_scale=%g sliding_window=%d experts=%d experts_used=%d\n",
			opts.numHeads, opts.numKVHeads, opts.headDim(), opts.ropeBase, opts.ropeScale, 0, opts.numExperts, opts.numExpertsUsed)
	}

	firstPos := positions[0]
	lastPos := positions[len(positions)-1]
	selectedIndex := stableDebugSelectedIndex(batchSeq)
	switch stableDebugForce.Swap(stableDebugPhaseNone) {
	case stableDebugPhasePrefill:
		stableDebugPrefill.Store(true)
		stableDebugPhase.Store(stableDebugPhasePrefill)
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=step layer=-1 semantic=prefill_last batch_seq=%d batch_size=%d pos_first=%d pos_last=%d selected_index=%d\n",
			batchSeq, batchSize, firstPos, lastPos, selectedIndex)
		return stableDebugPhasePrefill
	case stableDebugPhaseDecode:
		stableDebugDecode.Store(true)
		stableDebugPhase.Store(stableDebugPhaseDecode)
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=step layer=-1 semantic=decode_first batch_seq=%d batch_size=%d pos_first=%d pos_last=%d selected_index=%d\n",
			batchSeq, batchSize, firstPos, lastPos, selectedIndex)
		return stableDebugPhaseDecode
	}

	if batchSize != 1 {
		stableDebugPhase.Store(stableDebugPhaseNone)
		return stableDebugPhaseNone
	}
	stableDebugPhase.Store(stableDebugPhaseNone)
	return stableDebugPhaseNone
}

func stableDebugRequestSemantic(semantic string) {
	switch semantic {
	case "prefill_last":
		stableDebugForce.Store(stableDebugPhasePrefill)
	case "decode_first":
		stableDebugForce.Store(stableDebugPhaseDecode)
	default:
		stableDebugForce.Store(stableDebugPhaseNone)
	}
}

func stableDebugActive() bool {
	return stableDebugPhase.Load() != stableDebugPhaseNone
}

func stableDebugSetLayer(layer int) {
	stableDebugLayer.Store(int32(layer))
}

func stableDebugFinish() {
	stableDebugPhase.Store(stableDebugPhaseNone)
}

func stableDebugSemantic() string {
	switch stableDebugPhase.Load() {
	case stableDebugPhasePrefill:
		return "prefill_last"
	case stableDebugPhaseDecode:
		return "decode_first"
	default:
		return ""
	}
}

func stableDebugSelectToken(ctx ml.Context, t ml.Tensor) ml.Tensor {
	if stableDebugPhase.Load() != stableDebugPhasePrefill {
		return t
	}
	shape := t.Shape()
	if len(shape) < 2 {
		return t
	}
	lastDim := len(shape) - 1
	last := t.Dim(lastDim) - 1
	if last < 0 {
		return t
	}
	return t.Slice(ctx, lastDim, last, last+1, 1)
}

func stableDebugTensor(ctx ml.Context, stage string, t ml.Tensor) {
	if !stableDebugActive() || stableDebugLayer.Load() != 0 {
		return
	}
	t = stableDebugSelectToken(ctx, t)
	ctx.Forward(t).Compute(t)
	vals := t.Floats()
	if len(vals) == 0 {
		vals = t.BackendGet()
	}
	if len(vals) == 0 {
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=%s layer=0 semantic=%s shape=%v empty=true\n", stage, stableDebugSemantic(), t.Shape())
		return
	}

	minV, maxV, sum := vals[0], vals[0], float64(0)
	for _, v := range vals {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
		sum += float64(v)
	}
	mean := sum / float64(len(vals))
	var sq float64
	for _, v := range vals {
		d := float64(v) - mean
		sq += d * d
	}
	std := math.Sqrt(sq / float64(len(vals)))
	hash := sha256.New()
	for _, v := range vals {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		hash.Write(b[:])
	}
	sampleN := min(8, len(vals))
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=%s layer=0 semantic=%s shape=%v min=%g max=%g mean=%.8g std=%.8g sha256=%x sample=%v\n",
		stage, stableDebugSemantic(), t.Shape(), minV, maxV, mean, std, hash.Sum(nil), vals[:sampleN])
}

func stableDebugExperts(ctx ml.Context, stage string, ids, probs ml.Tensor) {
	if !stableDebugActive() || stableDebugLayer.Load() != 0 {
		return
	}
	ctx.Forward(ids, probs).Compute(ids, probs)
	raw := ids.Bytes()
	decoded := make([]int32, len(raw)/4)
	for i := range decoded {
		decoded[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	p := probs.Floats()
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=%s layer=0 semantic=%s expert_ids=%v expert_probs=%v\n", stage, stableDebugSemantic(), decoded, p)
}

func stableDebugLogits(ctx ml.Context, stage string, t ml.Tensor, k int) {
	if !stableDebugActive() {
		return
	}
	ctx.Forward(t).Compute(t)
	vals := t.Floats()
	if len(vals) == 0 {
		vals = t.BackendGet()
	}
	top := make([]stableTopKV, len(vals))
	for i, v := range vals {
		top[i] = stableTopKV{Index: i, Value: v}
	}
	sort.Slice(top, func(i, j int) bool { return top[i].Value > top[j].Value })
	if k < len(top) {
		top = top[:k]
	}
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=%s layer=-1 semantic=%s topk=%v\n", stage, stableDebugSemantic(), top)
	if stableDebugPhase.Load() == stableDebugPhaseDecode && len(top) > 0 {
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=first_generated_token layer=-1 semantic=%s id=%d value=%g\n", stableDebugSemantic(), top[0].Index, top[0].Value)
	}
}

func stableDebugTokenIDs(ctx ml.Context, t ml.Tensor) {
	if !stableDebugActive() {
		return
	}
	ctx.Forward(t).Compute(t)
	raw := t.Bytes()
	if len(raw) == 0 {
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=input_tokens layer=-1 ids=[]\n")
		return
	}
	ids := make([]int32, len(raw)/4)
	for i := range ids {
		ids[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	lastID := ids[len(ids)-1]
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=input_tokens layer=-1 semantic=%s ids=%v last_id=%d\n", stableDebugSemantic(), ids, lastID)
}
