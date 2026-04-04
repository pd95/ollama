package gptoss

import (
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
	stableDebugArmed   atomic.Bool
	stableDebugDone    atomic.Bool
	stableDebugLayer   atomic.Int32
)

type stableTopKV struct {
	Index int
	Value float32
}

func stableDebugBegin(batchSeq, batchSize int, firstPos int32, opts *Options) bool {
	if !stableDebugEnabled || stableDebugDone.Load() || batchSeq != 1 || batchSize != 1 || firstPos <= 0 {
		return false
	}
	if !stableDebugArmed.CompareAndSwap(false, true) {
		return false
	}
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=config layer=-1 heads=%d kv_heads=%d head_dim=%d rope_base=%g rope_scale=%g sliding_window=%d experts=%d experts_used=%d\n",
		opts.numHeads, opts.numKVHeads, opts.headDim(), opts.ropeBase, opts.ropeScale, 0, opts.numExperts, opts.numExpertsUsed)
	return true
}

func stableDebugActive() bool {
	return stableDebugArmed.Load()
}

func stableDebugSetLayer(layer int) {
	stableDebugLayer.Store(int32(layer))
}

func stableDebugFinish() {
	if stableDebugArmed.CompareAndSwap(true, false) {
		stableDebugDone.Store(true)
	}
}

func stableDebugTensor(ctx ml.Context, stage string, t ml.Tensor) {
	if !stableDebugActive() || stableDebugLayer.Load() != 0 {
		return
	}
	ctx.Forward(t).Compute(t)
	vals := t.Floats()
	if len(vals) == 0 {
		vals = t.BackendGet()
	}
	if len(vals) == 0 {
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=%s layer=0 shape=%v empty=true\n", stage, t.Shape())
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
	sampleN := min(8, len(vals))
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=%s layer=0 shape=%v min=%g max=%g mean=%.8g std=%.8g sample=%v\n",
		stage, t.Shape(), minV, maxV, mean, std, vals[:sampleN])
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
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=%s layer=0 expert_ids=%v expert_probs=%v\n", stage, decoded, p)
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
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=stable stage=%s layer=-1 topk=%v\n", stage, top)
}
