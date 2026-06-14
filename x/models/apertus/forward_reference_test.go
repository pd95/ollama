package apertus

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime/debug"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func TestForwardReference(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	refPath := firstNonEmpty(os.Getenv("APERTUS_REF_PATH"), os.Getenv("PORTING_APERTUS_REF_PATH"))
	if refPath == "" {
		t.Skip("set APERTUS_REF_PATH or PORTING_APERTUS_REF_PATH to an HF activation safetensors reference")
	}
	modelName := firstNonEmpty(os.Getenv("APERTUS_MODEL_NAME"), os.Getenv("PORTING_APERTUS_MODEL_NAME"), "apertus-mlx")

	thread, err := mlxthread.Start("apertus-forward-reference", func() error {
		if err := mlx.CheckInit(); err != nil {
			return err
		}
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		configureCompileForTest(t)
		return nil
	})
	if err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	defer func() {
		if err := thread.Stop(context.Background(), func() {
			mlx.Sweep()
			mlx.ClearCache()
		}); err != nil {
			t.Fatal(err)
		}
	}()

	if err := thread.Do(context.Background(), func() (err error) {
		defer func() {
			if recovered := recover(); recovered != nil {
				err = fmt.Errorf("panic in forward reference test: %v\n%s", recovered, debug.Stack())
			}
		}()
		runForwardReference(t, refPath, modelName)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestForwardCacheReference(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	refPath := firstNonEmpty(os.Getenv("APERTUS_CACHE_REF_PATH"), os.Getenv("PORTING_APERTUS_CACHE_REF_PATH"))
	if refPath == "" {
		t.Skip("set APERTUS_CACHE_REF_PATH or PORTING_APERTUS_CACHE_REF_PATH to an HF cached decode safetensors reference")
	}
	modelName := firstNonEmpty(os.Getenv("APERTUS_MODEL_NAME"), os.Getenv("PORTING_APERTUS_MODEL_NAME"), "apertus-mlx")

	thread, err := mlxthread.Start("apertus-forward-cache-reference", func() error {
		if err := mlx.CheckInit(); err != nil {
			return err
		}
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		configureCompileForTest(t)
		return nil
	})
	if err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	defer func() {
		if err := thread.Stop(context.Background(), func() {
			mlx.Sweep()
			mlx.ClearCache()
		}); err != nil {
			t.Fatal(err)
		}
	}()

	if err := thread.Do(context.Background(), func() (err error) {
		defer func() {
			if recovered := recover(); recovered != nil {
				err = fmt.Errorf("panic in cache reference test: %v\n%s", recovered, debug.Stack())
			}
		}()
		runForwardCacheReference(t, refPath, modelName)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func runForwardReference(t *testing.T, refPath, modelName string) {
	t.Helper()
	ref := loadReferenceFiltered(t, refPath, map[string]bool{
		"input_ids":                          true,
		"logits":                             true,
		"model.embed_tokens":                 true,
		"model.layers.0":                     true,
		"model.layers.0.attention_layernorm": true,
		"model.layers.0.self_attn":           true,
		"model.layers.0.self_attn.q_proj":    true,
		"model.layers.0.self_attn.q_norm":    true,
		"model.layers.0.self_attn.k_proj":    true,
		"model.layers.0.self_attn.k_norm":    true,
		"model.layers.0.self_attn.v_proj":    true,
		"model.layers.0.mlp.up_proj":         true,
		"model.layers.0.mlp.act_fn":          true,
		"model.layers.7":                     true,
		"model.layers.15":                    true,
		"model.layers.31":                    true,
		"model.norm":                         true,
	})

	inputIDs := ref["input_ids"]
	if inputIDs == nil {
		panic("reference is missing input_ids")
	}
	tokens := inputIDs.AsType(mlx.DTypeInt32)
	mlx.Pin(tokens)
	defer mlx.Unpin(tokens)
	mlx.Eval(tokens)
	dims := tokens.Dims()
	if len(dims) != 2 {
		panic(fmt.Sprintf("input_ids shape = %v, want rank 2", dims))
	}
	B, L := int32(dims[0]), int32(dims[1])

	m := loadImportedModel(t, modelName)
	b := &batch.Batch{
		InputIDs:     tokens,
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{L},
	}
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))

	h := m.EmbedTokens.Forward(tokens)
	compareReference(t, "model.embed_tokens", h, ref["model.embed_tokens"], 0.9999)

	layer0 := m.Layers[0]
	hNorm := layer0.AttentionNorm.Forward(h, m.RMSNormEps)
	compareReference(t, "model.layers.0.attention_layernorm", hNorm, ref["model.layers.0.attention_layernorm"], 0.9999)

	q := layer0.Attention.QProj.Forward(hNorm)
	compareReference(t, "model.layers.0.self_attn.q_proj", q, ref["model.layers.0.self_attn.q_proj"], 0.9999)
	q = mlx.Reshape(q, B, L, m.NumAttentionHeads, m.HeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	q = headRMSNorm(layer0.Attention.QNorm, q, B, m.NumAttentionHeads, L, m.HeadDim, m.RMSNormEps)
	compareReference(t, "model.layers.0.self_attn.q_norm", q, ref["model.layers.0.self_attn.q_norm"], 0.9999)

	k := layer0.Attention.KProj.Forward(hNorm)
	compareReference(t, "model.layers.0.self_attn.k_proj", k, ref["model.layers.0.self_attn.k_proj"], 0.9999)
	k = mlx.Reshape(k, B, L, m.NumKeyValueHeads, m.HeadDim)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	k = headRMSNorm(layer0.Attention.KNorm, k, B, m.NumKeyValueHeads, L, m.HeadDim, m.RMSNormEps)
	compareReference(t, "model.layers.0.self_attn.k_norm", k, ref["model.layers.0.self_attn.k_norm"], 0.9999)

	v := layer0.Attention.VProj.Forward(hNorm)
	compareReference(t, "model.layers.0.self_attn.v_proj", v, ref["model.layers.0.self_attn.v_proj"], 0.9999)

	attnOut := layer0.Attention.Forward(hNorm, b, nil, positions, B, L, m.Config)
	compareReference(t, "model.layers.0.self_attn", attnOut, ref["model.layers.0.self_attn"], 0.999)

	h = mlx.Add(h, attnOut)
	ffnInput := layer0.FFNNorm.Forward(h, m.RMSNormEps)
	mlpUp := layer0.MLP.UpProj.Forward(ffnInput)
	compareReference(t, "model.layers.0.mlp.up_proj", mlpUp, ref["model.layers.0.mlp.up_proj"], 0.999)
	mlpAct := layer0.MLP.Act.Forward(mlpUp)
	compareReference(t, "model.layers.0.mlp.act_fn", mlpAct, ref["model.layers.0.mlp.act_fn"], 0.997)

	h = mlx.Add(h, layer0.MLP.DownProj.Forward(mlpAct))
	compareReference(t, "model.layers.0", h, ref["model.layers.0"], 0.999)

	for i := 1; i < len(m.Layers); i++ {
		h = m.Layers[i].Forward(h, b, nil, positions, B, L, m.Config)
		switch i {
		case 7, 15, 31:
			key := fmt.Sprintf("model.layers.%d", i)
			compareReference(t, key, h, ref[key], 0.999)
		}
	}

	h = m.Norm.Forward(h, m.RMSNormEps)
	compareReference(t, "model.norm", h, ref["model.norm"], 0.997)
	if ref["logits"] != nil {
		logits := mlx.Reshape(m.LMHead.Forward(h), B, L, m.VocabSize)
		compareReference(t, "logits", logits, ref["logits"], 0.997)
		compareLogitArgmax(t, "logits", logits, ref["logits"])
	}
}

func configureCompileForTest(t *testing.T) {
	t.Helper()
	switch strings.ToLower(os.Getenv("PORTING_APERTUS_COMPILE")) {
	case "1", "true", "enable", "enabled", "on":
		t.Log("enabling MLX compile")
		mlx.EnableCompile()
	case "0", "false", "disable", "disabled", "off":
		t.Log("disabling MLX compile")
		mlx.DisableCompile()
	}
}

func runForwardCacheReference(t *testing.T, refPath, modelName string) {
	t.Helper()
	ref := loadReferenceFiltered(t, refPath, map[string]bool{
		"input_ids":                       true,
		"prefill_input_ids":               true,
		"logits":                          true,
		"model.embed_tokens":              true,
		"model.layers.0":                  true,
		"model.layers.0.self_attn":        true,
		"model.layers.0.self_attn.q_norm": true,
		"model.layers.0.self_attn.k_norm": true,
		"model.layers.0.mlp.up_proj":      true,
		"model.layers.0.mlp.act_fn":       true,
		"model.layers.7":                  true,
		"model.layers.15":                 true,
		"model.layers.31":                 true,
		"model.norm":                      true,
	})

	prefillIDs := ref["prefill_input_ids"]
	if prefillIDs == nil {
		panic("reference is missing prefill_input_ids")
	}
	decodeIDs := ref["input_ids"]
	if decodeIDs == nil {
		panic("reference is missing input_ids")
	}
	prefill := prefillIDs.AsType(mlx.DTypeInt32)
	decode := decodeIDs.AsType(mlx.DTypeInt32)
	mlx.Pin(prefill, decode)
	defer mlx.Unpin(prefill, decode)
	mlx.Eval(prefill, decode)

	prefillDims := prefill.Dims()
	decodeDims := decode.Dims()
	if len(prefillDims) != 2 {
		panic(fmt.Sprintf("prefill_input_ids shape = %v, want rank 2", prefillDims))
	}
	if len(decodeDims) != 2 {
		panic(fmt.Sprintf("input_ids shape = %v, want rank 2", decodeDims))
	}
	if prefillDims[0] != 1 || decodeDims[0] != 1 {
		panic(fmt.Sprintf("only batch size 1 is supported, got prefill=%v decode=%v", prefillDims, decodeDims))
	}

	m := loadImportedModel(t, modelName)
	caches := m.NewCaches()

	prefillLen := int32(prefillDims[1])
	m.Forward(&batch.Batch{
		InputIDs:     prefill,
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{prefillLen},
	}, caches)
	mlx.Sweep()

	decodeLen := int32(decodeDims[1])
	decodeBatch := &batch.Batch{
		InputIDs:     decode,
		SeqOffsets:   []int32{prefillLen},
		SeqQueryLens: []int32{decodeLen},
	}
	positions := mlx.FromValues(decodeBatch.SeqOffsets, len(decodeBatch.SeqOffsets))

	h := mlx.Reshape(m.EmbedTokens.Forward(decode), 1, decodeLen, m.HiddenSize)
	compareReference(t, "model.embed_tokens", h, ref["model.embed_tokens"], 0.9999)

	layer0 := m.Layers[0]
	hNorm := layer0.AttentionNorm.Forward(h, m.RMSNormEps)
	q := layer0.Attention.QProj.Forward(hNorm)
	q = mlx.Reshape(q, 1, decodeLen, m.NumAttentionHeads, m.HeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	q = headRMSNorm(layer0.Attention.QNorm, q, 1, m.NumAttentionHeads, decodeLen, m.HeadDim, m.RMSNormEps)
	compareReference(t, "model.layers.0.self_attn.q_norm", q, ref["model.layers.0.self_attn.q_norm"], 0.9999)

	k := layer0.Attention.KProj.Forward(hNorm)
	k = mlx.Reshape(k, 1, decodeLen, m.NumKeyValueHeads, m.HeadDim)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	k = headRMSNorm(layer0.Attention.KNorm, k, 1, m.NumKeyValueHeads, decodeLen, m.HeadDim, m.RMSNormEps)
	compareReference(t, "model.layers.0.self_attn.k_norm", k, ref["model.layers.0.self_attn.k_norm"], 0.9999)

	attnOut := layer0.Attention.Forward(hNorm, decodeBatch, caches[0], positions, 1, decodeLen, m.Config)
	compareReference(t, "model.layers.0.self_attn", attnOut, ref["model.layers.0.self_attn"], 0.999)

	h = mlx.Add(h, attnOut)
	ffnInput := layer0.FFNNorm.Forward(h, m.RMSNormEps)
	mlpUp := layer0.MLP.UpProj.Forward(ffnInput)
	compareReference(t, "model.layers.0.mlp.up_proj", mlpUp, ref["model.layers.0.mlp.up_proj"], 0.999)
	mlpAct := layer0.MLP.Act.Forward(mlpUp)
	compareReference(t, "model.layers.0.mlp.act_fn", mlpAct, ref["model.layers.0.mlp.act_fn"], 0.997)

	h = mlx.Add(h, layer0.MLP.DownProj.Forward(mlpAct))
	compareReference(t, "model.layers.0", h, ref["model.layers.0"], 0.999)

	for i := 1; i < len(m.Layers); i++ {
		h = m.Layers[i].Forward(h, decodeBatch, caches[i], positions, 1, decodeLen, m.Config)
		switch i {
		case 7, 15, 31:
			key := fmt.Sprintf("model.layers.%d", i)
			compareReference(t, key, h, ref[key], 0.999)
		}
	}

	h = m.Norm.Forward(h, m.RMSNormEps)
	compareReference(t, "model.norm", h, ref["model.norm"], 0.997)
	if ref["logits"] != nil {
		logits := mlx.Reshape(m.LMHead.Forward(h), 1, decodeLen, m.VocabSize)
		compareReference(t, "logits", logits, ref["logits"], 0.997)
		compareLogitArgmax(t, "logits", logits, ref["logits"])
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func loadImportedModel(t *testing.T, modelName string) *Model {
	t.Helper()

	root, err := model.Open(modelName)
	if err != nil {
		panic(fmt.Sprintf("open imported model %q: %v", modelName, err))
	}
	defer root.Close()

	bm, err := base.New(root)
	if err != nil {
		panic(fmt.Sprintf("construct model: %v", err))
	}
	m, ok := bm.(*Model)
	if !ok {
		panic(fmt.Sprintf("expected *apertus.Model, got %T", bm))
	}

	tensors := loadTensorsForTest(t, root)
	if err := m.LoadWeights(tensors); err != nil {
		panic(fmt.Sprintf("load weights: %v", err))
	}

	collected := mlx.Collect(m)
	for _, arr := range collected {
		mlx.Pin(arr)
	}
	mlx.Sweep()
	mlx.Eval(collected...)

	return m
}

func loadTensorsForTest(t *testing.T, root *model.Root) map[string]*mlx.Array {
	t.Helper()

	raw := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	for _, layer := range root.Manifest.GetTensorLayers("") {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		for name, arr := range mlx.Load(root.Manifest.BlobPath(layer.Digest)) {
			raw[name] = arr
		}
	}

	scaleBaseNames := make(map[string]bool)
	out := make(map[string]*mlx.Array, len(raw))
	for name, arr := range raw {
		if strings.HasSuffix(name, ".scale") {
			baseName := strings.TrimSuffix(name, ".scale")
			out[baseName+"_scale"] = arr
			scaleBaseNames[baseName] = true
		}
	}
	for name, arr := range raw {
		if strings.HasSuffix(name, ".scale") {
			continue
		}
		if strings.HasSuffix(name, ".bias") && !strings.HasSuffix(name, ".weight_qbias") {
			baseName := strings.TrimSuffix(name, ".bias")
			if scaleBaseNames[baseName] {
				out[baseName+"_qbias"] = arr
				continue
			}
		}
		out[name] = arr
	}
	return out
}

func loadReferenceFiltered(t *testing.T, path string, keep map[string]bool) map[string]*mlx.Array {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		panic(fmt.Sprintf("reference data not available at %s: %v", path, err))
	}

	out := make(map[string]*mlx.Array)
	for name, arr := range mlx.Load(path) {
		if keep[name] {
			out[name] = arr
		}
	}
	if len(out) == 0 {
		panic(fmt.Sprintf("no reference tensors loaded from %s", path))
	}
	for _, arr := range out {
		mlx.Pin(arr)
	}
	mlx.Sweep()
	return out
}

func compareReference(t *testing.T, name string, got, want *mlx.Array, minCos float64) {
	t.Helper()
	if want == nil {
		panic(fmt.Sprintf("reference is missing %q", name))
	}

	wantShape := want.Dims()
	shape := make([]int32, len(wantShape))
	for i, dim := range wantShape {
		shape[i] = int32(dim)
	}
	got = mlx.Reshape(got, shape...)
	got = got.AsType(mlx.DTypeFloat32)
	want = want.AsType(mlx.DTypeFloat32)

	size := 1
	for _, dim := range wantShape {
		size *= dim
	}
	gotFlat := mlx.Reshape(got, int32(size))
	wantFlat := mlx.Reshape(want, int32(size))
	diff := mlx.Sub(gotFlat, wantFlat).Abs()
	dot := mlx.Sum(mlx.Mul(gotFlat, wantFlat), 0, false)
	gotNorm := mlx.Sum(mlx.Mul(gotFlat, gotFlat), 0, false).Sqrt()
	wantNorm := mlx.Sum(mlx.Mul(wantFlat, wantFlat), 0, false).Sqrt()
	cosine := mlx.Div(dot, mlx.Mul(gotNorm, wantNorm))
	maxDiff := diff.MaxAxis(0, false)
	meanDiff := mlx.Mean(diff, 0, false)
	mlx.Eval(cosine, maxDiff, meanDiff)

	cos := scalarFloat(cosine)
	maxDiffValue := scalarFloat(maxDiff)
	meanDiffValue := scalarFloat(meanDiff)
	if maxDiffValue <= 1e-6 {
		t.Logf("%s: shape=%v exact max_diff=%.6g mean_diff=%.6g", name, wantShape, maxDiffValue, meanDiffValue)
		return
	}
	t.Logf("%s: shape=%v cos=%.8f max_diff=%.6g mean_diff=%.6g", name, wantShape, cos, maxDiffValue, meanDiffValue)
	if math.IsNaN(cos) || cos < minCos {
		t.Errorf("%s cosine similarity %.8f below %.8f", name, cos, minCos)
	}
}

func compareLogitArgmax(t *testing.T, name string, got, want *mlx.Array) {
	t.Helper()
	if want == nil {
		panic(fmt.Sprintf("reference is missing %q", name))
	}
	wantShape := want.Dims()
	if len(wantShape) != 3 {
		panic(fmt.Sprintf("%s shape = %v, want [B,L,V]", name, wantShape))
	}
	shape := make([]int32, len(wantShape))
	for i, dim := range wantShape {
		shape[i] = int32(dim)
	}
	got = mlx.Reshape(got, shape...)
	last := wantShape[1] - 1
	gotLast := got.Slice(mlx.Slice(), mlx.Slice(last, last+1), mlx.Slice()).Squeeze(1)
	wantLast := want.Slice(mlx.Slice(), mlx.Slice(last, last+1), mlx.Slice()).Squeeze(1)
	gotID := gotLast.Argmax(-1, false).AsType(mlx.DTypeInt32)
	wantID := wantLast.Argmax(-1, false).AsType(mlx.DTypeInt32)
	mlx.Eval(gotID, wantID)
	gotValue := gotID.Int()
	wantValue := wantID.Int()
	t.Logf("%s last-token argmax got=%d want=%d", name, gotValue, wantValue)
	if gotValue != wantValue {
		t.Errorf("%s last-token argmax = %d, want %d", name, gotValue, wantValue)
	}
}

func scalarFloat(x *mlx.Array) float64 {
	x = x.AsType(mlx.DTypeFloat32)
	mlx.Eval(x)
	values := x.Floats()
	if len(values) == 0 {
		return math.NaN()
	}
	return float64(values[0])
}
