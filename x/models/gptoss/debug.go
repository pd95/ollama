package gptoss

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"sync/atomic"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

var (
	mlxDebugEnabled = os.Getenv("OLLAMA_GPTOSS_ACT_DEBUG") != ""
	mlxDebugArmed   atomic.Bool
	mlxDebugDone    atomic.Bool
	mlxDebugLayer   atomic.Int32
)

type mlxTopKV struct {
	Index int
	Value float32
}

func mlxDebugBegin(tokens *mlx.Array, caches []cache.Cache, cfg *Config) bool {
	if !mlxDebugEnabled || mlxDebugDone.Load() || tokens == nil || !tokens.Valid() || tokens.NumDims() != 2 {
		return false
	}
	if tokens.Dim(0) != 1 || tokens.Dim(1) != 1 || len(caches) == 0 || caches[0] == nil || caches[0].Offset() <= 0 {
		return false
	}
	if !mlxDebugArmed.CompareAndSwap(false, true) {
		return false
	}
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=config layer=-1 heads=%d kv_heads=%d head_dim=%d rope_base=%g rope_scale=%g sliding_window=%d experts=%d experts_used=%d\n",
		cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim, cfg.RopeTheta, cfg.RopeScalingFactor, cfg.SlidingWindow, cfg.ExpertCount, cfg.ExpertsPerToken)
	return true
}

func mlxDebugActive() bool {
	return mlxDebugArmed.Load()
}

func mlxDebugSetLayer(layer int) {
	mlxDebugLayer.Store(int32(layer))
}

func mlxDebugFinish() {
	if mlxDebugArmed.CompareAndSwap(true, false) {
		mlxDebugDone.Store(true)
	}
}

func mlxDebugTensor(stage string, t *mlx.Array) {
	if !mlxDebugActive() || mlxDebugLayer.Load() != 0 {
		return
	}
	mlx.Eval(t)
	vals := t.Floats()
	if len(vals) == 0 {
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=%s layer=0 shape=%v empty=true\n", stage, t.Dims())
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
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=%s layer=0 shape=%v min=%g max=%g mean=%.8g std=%.8g sha256=%x sample=%v\n",
		stage, t.Dims(), minV, maxV, mean, std, hash.Sum(nil), vals[:sampleN])
}

func mlxDebugExperts(stage string, ids, probs *mlx.Array) {
	if !mlxDebugActive() || mlxDebugLayer.Load() != 0 {
		return
	}
	mlx.Eval(ids, probs)
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=%s layer=0 expert_ids=%v expert_probs=%v\n", stage, ids.Ints(), probs.Floats())
}

func mlxDebugLogits(stage string, t *mlx.Array, k int) {
	if !mlxDebugActive() {
		return
	}
	mlx.Eval(t)
	vals := t.Floats()
	top := make([]mlxTopKV, len(vals))
	for i, v := range vals {
		top[i] = mlxTopKV{Index: i, Value: v}
	}
	sort.Slice(top, func(i, j int) bool { return top[i].Value > top[j].Value })
	if k < len(top) {
		top = top[:k]
	}
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=%s layer=-1 topk=%v\n", stage, top)
}

func mlxDebugTokenIDs(tokens *mlx.Array) {
	if !mlxDebugActive() {
		return
	}
	mlx.Eval(tokens)
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=input_tokens layer=-1 ids=%v\n", tokens.Ints())
}
