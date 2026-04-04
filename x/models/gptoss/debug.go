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
	mlxDebugConfig  atomic.Bool
	mlxDebugPrefill atomic.Bool
	mlxDebugDecode  atomic.Bool
	mlxDebugLayer   atomic.Int32
	mlxDebugPhase   atomic.Int32
	mlxDebugForce   atomic.Int32
)

const (
	mlxDebugPhaseNone int32 = iota
	mlxDebugPhasePrefill
	mlxDebugPhaseDecode
)

type mlxTopKV struct {
	Index int
	Value float32
}

func mlxDebugSelectedIndex(tokenCount int) int {
	if mlxDebugPhase.Load() == mlxDebugPhasePrefill && tokenCount > 0 {
		return tokenCount - 1
	}
	return 0
}

func mlxDebugBegin(tokens *mlx.Array, caches []cache.Cache, cfg *Config) int32 {
	if !mlxDebugEnabled || tokens == nil || !tokens.Valid() {
		mlxDebugPhase.Store(mlxDebugPhaseNone)
		return mlxDebugPhaseNone
	}
	if mlxDebugConfig.CompareAndSwap(false, true) {
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=config layer=-1 heads=%d kv_heads=%d head_dim=%d rope_base=%g rope_scale=%g sliding_window=%d experts=%d experts_used=%d\n",
			cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim, cfg.RopeTheta, cfg.RopeScalingFactor, cfg.SlidingWindow, cfg.ExpertCount, cfg.ExpertsPerToken)
	}

	L := 1
	if tokens.NumDims() > 1 {
		L = tokens.Dim(1)
	}
	cacheOffset := 0
	if len(caches) > 0 && caches[0] != nil {
		cacheOffset = caches[0].Offset()
	}
	selectedIndex := mlxDebugSelectedIndex(L)
	switch mlxDebugForce.Swap(mlxDebugPhaseNone) {
	case mlxDebugPhasePrefill:
		mlxDebugPrefill.Store(true)
		mlxDebugPhase.Store(mlxDebugPhasePrefill)
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=step layer=-1 semantic=prefill_last batch_seq=%d batch_size=1 cache_offset=%d selected_index=%d\n", L, cacheOffset, selectedIndex)
		return mlxDebugPhasePrefill
	case mlxDebugPhaseDecode:
		mlxDebugDecode.Store(true)
		mlxDebugPhase.Store(mlxDebugPhaseDecode)
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=step layer=-1 semantic=decode_first batch_seq=1 batch_size=1 cache_offset=%d selected_index=0\n", cacheOffset)
		return mlxDebugPhaseDecode
	}
	mlxDebugPhase.Store(mlxDebugPhaseNone)
	return mlxDebugPhaseNone
}

func mlxDebugRequestSemantic(semantic string) {
	switch semantic {
	case "prefill_last":
		mlxDebugForce.Store(mlxDebugPhasePrefill)
	case "decode_first":
		mlxDebugForce.Store(mlxDebugPhaseDecode)
	default:
		mlxDebugForce.Store(mlxDebugPhaseNone)
	}
}

func mlxDebugActive() bool {
	return mlxDebugPhase.Load() != mlxDebugPhaseNone
}

func mlxDebugSetLayer(layer int) {
	mlxDebugLayer.Store(int32(layer))
}

func mlxDebugFinish() {
	mlxDebugPhase.Store(mlxDebugPhaseNone)
}

func mlxDebugSemantic() string {
	switch mlxDebugPhase.Load() {
	case mlxDebugPhasePrefill:
		return "prefill_last"
	case mlxDebugPhaseDecode:
		return "decode_first"
	default:
		return ""
	}
}

func mlxDebugSelectToken(t *mlx.Array) *mlx.Array {
	if mlxDebugPhase.Load() != mlxDebugPhasePrefill || t == nil || !t.Valid() {
		return t
	}
	dims := t.Dims()
	switch len(dims) {
	case 3:
		return mlx.SliceStartStop(t, []int32{0, int32(dims[1] - 1), 0}, []int32{int32(dims[0]), int32(dims[1]), int32(dims[2])})
	case 4:
		return mlx.SliceStartStop(t, []int32{0, 0, int32(dims[2] - 1), 0}, []int32{int32(dims[0]), int32(dims[1]), int32(dims[2]), int32(dims[3])})
	default:
		return t
	}
}

func mlxDebugTensor(stage string, t *mlx.Array) {
	if !mlxDebugActive() || mlxDebugLayer.Load() != 0 {
		return
	}
	t = mlxDebugSelectToken(t)
	mlx.Eval(t)
	vals := t.Floats()
	if len(vals) == 0 {
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=%s layer=0 semantic=%s shape=%v empty=true\n", stage, mlxDebugSemantic(), t.Dims())
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
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=%s layer=0 semantic=%s shape=%v min=%g max=%g mean=%.8g std=%.8g sha256=%x sample=%v\n",
		stage, mlxDebugSemantic(), t.Dims(), minV, maxV, mean, std, hash.Sum(nil), vals[:sampleN])
}

func mlxDebugExperts(stage string, ids, probs *mlx.Array) {
	if !mlxDebugActive() || mlxDebugLayer.Load() != 0 {
		return
	}
	mlx.Eval(ids, probs)
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=%s layer=0 semantic=%s expert_ids=%v expert_probs=%v\n", stage, mlxDebugSemantic(), ids.Ints(), probs.Floats())
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
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=%s layer=-1 semantic=%s topk=%v\n", stage, mlxDebugSemantic(), top)
	if mlxDebugPhase.Load() == mlxDebugPhaseDecode && len(top) > 0 {
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=first_generated_token layer=-1 semantic=%s id=%d value=%g\n", mlxDebugSemantic(), top[0].Index, top[0].Value)
	}
}

func mlxDebugTokenIDs(tokens *mlx.Array) {
	if !mlxDebugActive() {
		return
	}
	mlx.Eval(tokens)
	ids := tokens.Ints()
	if len(ids) == 0 {
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=input_tokens layer=-1 semantic=%s ids=[]\n", mlxDebugSemantic())
		return
	}
	selectedIndex := 0
	selectedIDs := ids
	if mlxDebugPhase.Load() == mlxDebugPhasePrefill {
		selectedIndex = len(ids) - 1
		selectedIDs = ids[selectedIndex:]
	}
	lastID := selectedIDs[len(selectedIDs)-1]
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=input_tokens layer=-1 semantic=%s full_count=%d selected_index=%d ids=%v last_id=%d\n",
		mlxDebugSemantic(), len(ids), selectedIndex, selectedIDs, lastID)
}

func mlxDebugMeta(stage string, t *mlx.Array) {
	if !mlxDebugActive() || mlxDebugLayer.Load() != 0 {
		return
	}
	if t == nil {
		fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=%s layer=0 semantic=%s valid=false\n", stage, mlxDebugSemantic())
		return
	}
	fmt.Fprintf(os.Stderr, "GPTOSS_DEBUG path=mlx stage=%s layer=0 semantic=%s valid=%t dtype=%s shape=%v\n",
		stage, mlxDebugSemantic(), t.Valid(), t.DType(), t.Dims())
}
