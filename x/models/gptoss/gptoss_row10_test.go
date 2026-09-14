package gptoss

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func row10ValidConfig() Config {
	return Config{
		NumHiddenLayers: 2, HiddenSize: 64, IntermediateSize: 64, NumAttentionHeads: 2,
		NumKeyValueHeads: 1, HeadDim: 32, NumLocalExperts: 2, NumExpertsPerTok: 1,
		SlidingWindow: 128, VocabSize: 256, MaxPositionEmbeddings: 1024,
		RopeTheta: 10000, RMSNormEps: 1e-5, RopeScaling: RopeScaling{Factor: 1, OriginalMaxPositionEmbeddings: 1024},
	}
}

func TestConfigDimensions(t *testing.T) {
	base := row10ValidConfig()
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{name: "valid", mutate: func(*Config) {}},
		{name: "zero scalar", mutate: func(c *Config) { c.HiddenSize = 0 }, wantErr: "hidden_size"},
		{name: "negative scalar", mutate: func(c *Config) { c.NumLocalExperts = -1 }, wantErr: "num_local_experts"},
		{name: "loop ceiling accepted", mutate: func(c *Config) { c.NumHiddenLayers = gptossMaxLoopEntries; c.NumLocalExperts = 1 }},
		{name: "loop ceiling rejected", mutate: func(c *Config) { c.NumHiddenLayers = gptossMaxLoopEntries + 1 }, wantErr: "num_hidden_layers"},
		{name: "dimension ceiling accepted", mutate: func(c *Config) { c.HiddenSize = gptossMaxDimension }},
		{name: "dimension ceiling rejected", mutate: func(c *Config) { c.HiddenSize = gptossMaxDimension + 1 }, wantErr: "hidden_size"},
		{name: "query product accepted", mutate: func(c *Config) { c.NumAttentionHeads = 1 << 12; c.HeadDim = 1 << 12 }, wantErr: ""},
		{name: "query product rejected", mutate: func(c *Config) { c.NumAttentionHeads = 1 << 12; c.HeadDim = (1 << 12) + 1 }, wantErr: "query dimension"},
		{name: "int32 wrapping head product", mutate: func(c *Config) { c.NumAttentionHeads = 1 << 16; c.HeadDim = 1 << 16 }, wantErr: "query dimension"},
		{name: "gate up product rejected", mutate: func(c *Config) { c.IntermediateSize = (gptossMaxDimension / 2) + 1 }, wantErr: "gate/up dimension"},
		{name: "layer expert work rejected", mutate: func(c *Config) { c.NumHiddenLayers = 1025; c.NumLocalExperts = 1024 }, wantErr: "layer/expert work"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := base
			tt.mutate(&cfg)
			_, err := validateGPTOSSConfigDimensions(&cfg)
			if tt.wantErr == "" && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)) {
				t.Fatalf("error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestValidateGPTOSSScalarDimensions(t *testing.T) {
	setters := []struct {
		name string
		set  func(*Config, int32)
	}{
		{"layers", func(c *Config, v int32) { c.NumHiddenLayers = v }},
		{"hidden", func(c *Config, v int32) { c.HiddenSize = v }},
		{"intermediate", func(c *Config, v int32) { c.IntermediateSize = v }},
		{"attention heads", func(c *Config, v int32) { c.NumAttentionHeads = v }},
		{"kv heads", func(c *Config, v int32) { c.NumKeyValueHeads = v }},
		{"head dim", func(c *Config, v int32) { c.HeadDim = v }},
		{"experts", func(c *Config, v int32) { c.NumLocalExperts = v }},
		{"experts per token", func(c *Config, v int32) { c.NumExpertsPerTok = v }},
		{"sliding window", func(c *Config, v int32) { c.SlidingWindow = v }},
		{"vocab", func(c *Config, v int32) { c.VocabSize = v }},
		{"context", func(c *Config, v int32) { c.MaxPositionEmbeddings = v }},
		{"original context", func(c *Config, v int32) { c.RopeScaling.OriginalMaxPositionEmbeddings = v }},
	}
	for _, setter := range setters {
		for _, value := range []int32{0, -1} {
			t.Run(setter.name, func(t *testing.T) {
				cfg := row10ValidConfig()
				setter.set(&cfg, value)
				if _, err := validateGPTOSSConfigDimensions(&cfg); err == nil {
					t.Fatalf("value %d accepted", value)
				}
			})
		}
	}
}

func row10ValidFusedDescriptor() fusedExpertDescriptor {
	return fusedExpertDescriptor{
		present: true, transpose: true, mode: "mxfp4", bits: 4, groupSize: 32,
		weightType: mlx.DTypeUint32, scaleType: mlx.DTypeUint8, biasType: mlx.DTypeBFloat16,
		weightDims: []int{2, 64, 8}, scaleDims: []int{2, 64, 2}, biasDims: []int{2, 64},
	}
}

func TestFusedExpertEligibility(t *testing.T) {
	base := row10ValidFusedDescriptor()
	tests := []struct {
		name   string
		mutate func(*fusedExpertDescriptor)
	}{
		{name: "mode", mutate: func(d *fusedExpertDescriptor) { d.mode = "affine" }},
		{name: "bits", mutate: func(d *fusedExpertDescriptor) { d.bits = 8 }},
		{name: "group", mutate: func(d *fusedExpertDescriptor) { d.groupSize = 64 }},
		{name: "qbias", mutate: func(d *fusedExpertDescriptor) { d.hasQBias = true }},
		{name: "transpose", mutate: func(d *fusedExpertDescriptor) { d.transpose = false }},
		{name: "weight dtype", mutate: func(d *fusedExpertDescriptor) { d.weightType = mlx.DTypeUint8 }},
		{name: "scale dtype", mutate: func(d *fusedExpertDescriptor) { d.scaleType = mlx.DTypeBFloat16 }},
		{name: "bias dtype", mutate: func(d *fusedExpertDescriptor) { d.biasType = mlx.DTypeFloat32 }},
		{name: "weight rank", mutate: func(d *fusedExpertDescriptor) { d.weightDims = []int{2, 64} }},
		{name: "scale rank", mutate: func(d *fusedExpertDescriptor) { d.scaleDims = []int{2, 64} }},
		{name: "bias rank", mutate: func(d *fusedExpertDescriptor) { d.biasDims = []int{128} }},
	}
	if !fusedExpertDescriptorEligible(base) {
		t.Fatal("valid descriptor rejected")
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := base
			tt.mutate(&got)
			if fusedExpertDescriptorEligible(got) {
				t.Fatal("malformed descriptor accepted")
			}
		})
	}
	pairMutations := []struct {
		name   string
		mutate func(*fusedExpertDescriptor)
	}{
		{"mode", func(d *fusedExpertDescriptor) { d.mode = "affine" }},
		{"bits", func(d *fusedExpertDescriptor) { d.bits = 8 }},
		{"group", func(d *fusedExpertDescriptor) { d.groupSize = 64 }},
		{"qbias", func(d *fusedExpertDescriptor) { d.hasQBias = true }},
		{"transpose", func(d *fusedExpertDescriptor) { d.transpose = false }},
		{"shape", func(d *fusedExpertDescriptor) { d.weightDims = []int{2, 63, 8} }},
		{"dtype", func(d *fusedExpertDescriptor) { d.scaleType = mlx.DTypeBFloat16 }},
	}
	for _, side := range []string{"gate", "up"} {
		for _, mutation := range pairMutations {
			t.Run(side+" "+mutation.name, func(t *testing.T) {
				gate, up := base, base
				if side == "gate" {
					mutation.mutate(&gate)
				} else {
					mutation.mutate(&up)
				}
				if fusedExpertDescriptorsPairEligible(gate, up) {
					t.Fatal("asymmetric pair accepted")
				}
			})
		}
	}
}

var row10MLXMu sync.Mutex

func row10WithMLX(t *testing.T, fn func()) {
	t.Helper()
	row10MLXMu.Lock()
	defer row10MLXMu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	thread, err := mlxthread.Start("gptoss-row10-test", func() error {
		if err := mlx.CheckInit(); err != nil {
			return err
		}
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		return nil
	})
	if err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	defer func() { _ = thread.Stop(context.Background(), func() { mlx.Sweep(); mlx.ClearCache() }) }()
	if err := thread.Do(context.Background(), func() error { fn(); return nil }); err != nil {
		t.Fatal(err)
	}
}

func row10NativeExpertTensors(prefix string, experts, out, in int) map[string]*mlx.Array {
	return map[string]*mlx.Array{
		prefix + ".weight":       mlx.Zeros(mlx.DTypeUint32, experts, out, in/8),
		prefix + ".weight_scale": mlx.Zeros(mlx.DTypeUint8, experts, out, in/32),
		prefix + ".bias":         mlx.Zeros(mlx.DTypeBFloat16, experts, out),
	}
}

func TestDirectExpertValidation(t *testing.T) {
	row10WithMLX(t, func() {
		cfg := row10ValidConfig()
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = 32, 4, "mxfp4"
		valid := func(ts map[string]*mlx.Array) error {
			_, err := loadDirectExpertProjection(ts, 0, "p", 64, 64, &cfg)
			return err
		}
		if err := valid(row10NativeExpertTensors("p", 2, 64, 64)); err != nil {
			t.Fatalf("valid native projection rejected: %v", err)
		}
		for _, tt := range []struct {
			name   string
			mutate func(map[string]*mlx.Array)
		}{
			{name: "weight dtype", mutate: func(ts map[string]*mlx.Array) { ts["p.weight"] = mlx.Zeros(mlx.DTypeUint8, 2, 64, 8) }},
			{name: "weight rank", mutate: func(ts map[string]*mlx.Array) { ts["p.weight"] = mlx.Zeros(mlx.DTypeUint32, 2, 512) }},
			{name: "weight experts", mutate: func(ts map[string]*mlx.Array) { ts["p.weight"] = mlx.Zeros(mlx.DTypeUint32, 3, 64, 8) }},
			{name: "weight output", mutate: func(ts map[string]*mlx.Array) { ts["p.weight"] = mlx.Zeros(mlx.DTypeUint32, 2, 63, 8) }},
			{name: "packed axis", mutate: func(ts map[string]*mlx.Array) { ts["p.weight"] = mlx.Zeros(mlx.DTypeUint32, 2, 64, 9) }},
			{name: "scale dtype", mutate: func(ts map[string]*mlx.Array) { ts["p.weight_scale"] = mlx.Zeros(mlx.DTypeBFloat16, 2, 64, 2) }},
			{name: "scale rank", mutate: func(ts map[string]*mlx.Array) { ts["p.weight_scale"] = mlx.Zeros(mlx.DTypeUint8, 2, 128) }},
			{name: "scale experts", mutate: func(ts map[string]*mlx.Array) { ts["p.weight_scale"] = mlx.Zeros(mlx.DTypeUint8, 3, 64, 2) }},
			{name: "scale output", mutate: func(ts map[string]*mlx.Array) { ts["p.weight_scale"] = mlx.Zeros(mlx.DTypeUint8, 2, 63, 2) }},
			{name: "scale groups", mutate: func(ts map[string]*mlx.Array) { ts["p.weight_scale"] = mlx.Zeros(mlx.DTypeUint8, 2, 64, 3) }},
			{name: "bias dtype", mutate: func(ts map[string]*mlx.Array) { ts["p.bias"] = mlx.Zeros(mlx.DTypeFloat32, 2, 64) }},
			{name: "bias rank", mutate: func(ts map[string]*mlx.Array) { ts["p.bias"] = mlx.Zeros(mlx.DTypeBFloat16, 128) }},
			{name: "bias experts", mutate: func(ts map[string]*mlx.Array) { ts["p.bias"] = mlx.Zeros(mlx.DTypeBFloat16, 3, 64) }},
			{name: "bias axis", mutate: func(ts map[string]*mlx.Array) { ts["p.bias"] = mlx.Zeros(mlx.DTypeBFloat16, 2, 63) }},
			{name: "qbias", mutate: func(ts map[string]*mlx.Array) { ts["p.weight_qbias"] = mlx.Zeros(mlx.DTypeUint8, 2, 64, 2) }},
			{name: "missing weight", mutate: func(ts map[string]*mlx.Array) { delete(ts, "p.weight") }},
			{name: "missing bias", mutate: func(ts map[string]*mlx.Array) { delete(ts, "p.bias") }},
			{name: "missing scale", mutate: func(ts map[string]*mlx.Array) { delete(ts, "p.weight_scale") }},
			{name: "orphan scale", mutate: func(ts map[string]*mlx.Array) { delete(ts, "p.weight"); delete(ts, "p.bias") }},
			{name: "orphan qbias", mutate: func(ts map[string]*mlx.Array) { clear(ts); ts["p.weight_qbias"] = mlx.Zeros(mlx.DTypeUint8, 2, 64, 2) }},
		} {
			t.Run(tt.name, func(t *testing.T) {
				ts := row10NativeExpertTensors("p", 2, 64, 64)
				tt.mutate(ts)
				if err := valid(ts); err == nil {
					t.Fatal("malformed projection accepted")
				}
			})
		}
		if err := valid(row10NativeExpertTensors("p", 2, 64, 64)); err != nil {
			t.Fatalf("valid projection rejected after malformed cases: %v", err)
		}
		for _, projection := range []struct {
			prefix      string
			wantOut, in int
		}{
			{"model.layers.0.mlp.experts.gate_proj", 64, 64},
			{"model.layers.0.mlp.experts.up_proj", 64, 64},
			{"model.layers.0.mlp.experts.down_proj", 64, 64},
		} {
			if _, err := loadDirectExpertProjection(row10NativeExpertTensors(projection.prefix, 2, projection.wantOut, projection.in), 0, projection.prefix, projection.wantOut, projection.in, &cfg); err != nil {
				t.Fatalf("valid native %s rejected: %v", projection.prefix, err)
			}
		}
		pairPrefix := "model.layers.0.mlp.experts.gate_up_proj"
		gatePrefix := strings.Replace(pairPrefix, "gate_up_proj", "gate_proj", 1)
		upPrefix := strings.Replace(pairPrefix, "gate_up_proj", "up_proj", 1)
		gateOnly := row10NativeExpertTensors(gatePrefix, 2, 64, 64)
		if _, err := loadDirectExpertPair(gateOnly, 0, pairPrefix, 64, 64, &cfg); err == nil {
			t.Fatal("gate-only native pair accepted")
		}
		upOnly := row10NativeExpertTensors(upPrefix, 2, 64, 64)
		if _, err := loadDirectExpertPair(upOnly, 0, pairPrefix, 64, 64, &cfg); err == nil {
			t.Fatal("up-only native pair accepted")
		}
	})
}

func TestAttentionSinkValidation(t *testing.T) {
	row10WithMLX(t, func() {
		source := mlx.FromValues([]float32{0.25, -0.5}, 2).AsType(mlx.DTypeBFloat16)
		prepared, err := prepareGPTOSSAttentionSinks(0, "sink", source, 2)
		if err != nil {
			t.Fatal(err)
		}
		if source.DType() != mlx.DTypeBFloat16 {
			t.Fatalf("source dtype drifted to %s", source.DType())
		}
		if prepared == source || prepared.DType() != mlx.DTypeBFloat16 {
			t.Fatalf("prepared sinks = %p dtype %s, source = %p", prepared, prepared.DType(), source)
		}
		preparedValues := prepared.AsType(mlx.DTypeFloat32)
		mlx.Eval(preparedValues)
		if got := preparedValues.Floats(); len(got) != 2 || got[0] != 0.25 || got[1] != -0.5 {
			t.Fatalf("prepared materialized values = %v", got)
		}
		preparedIdentity := prepared
		for range 3 {
			if prepared != preparedIdentity {
				t.Fatal("prepared sink identity changed")
			}
			if err := validateGPTOSSAttentionSinkDispatch(mlx.Zeros(mlx.DTypeBFloat16, 1, 2, 1, 4), prepared, 2); err != nil {
				t.Fatalf("prepared dispatch proof: %v", err)
			}
		}
		if _, err := prepareGPTOSSAttentionSinks(0, "sink", mlx.Zeros(mlx.DTypeFloat32, 2), 2); err == nil {
			t.Fatal("F32 checkpoint sink accepted")
		}
		if _, err := prepareGPTOSSAttentionSinks(0, "sink", mlx.Zeros(mlx.DTypeUint8, 2), 2); err == nil {
			t.Fatal("U8 checkpoint sink accepted")
		}
		for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			bad := mlx.FromValues([]float32{value, 0}, 2).AsType(mlx.DTypeBFloat16)
			if _, err := prepareGPTOSSAttentionSinks(0, "sink", bad, 2); err == nil || !strings.Contains(err.Error(), "non-finite") {
				t.Fatalf("non-finite %v error = %v", value, err)
			}
		}
		if err := validateGPTOSSAttentionSinkDispatch(mlx.Zeros(mlx.DTypeFloat32, 1, 2, 1, 4), prepared, 2); err == nil || err.Error() != "prepared sink dtype BF16 must equal actual query dtype F32" {
			t.Fatalf("query/sink dtype mismatch error = %v", err)
		}
		if err := validateGPTOSSAttentionSinkDispatch(mlx.Zeros(mlx.DTypeBFloat16, 1, 2, 4), prepared, 2); err == nil {
			t.Fatal("rank-3 query accepted")
		}
		if err := validateGPTOSSAttentionSinkDispatch(mlx.Zeros(mlx.DTypeBFloat16, 1, 3, 1, 4), prepared, 2); err == nil || err.Error() != "query head axis = 3, want configured heads = 2 (shape [1 3 1 4])" {
			t.Fatalf("query head mismatch error = %v", err)
		}
		if err := validateGPTOSSAttentionSinkDispatch(mlx.Zeros(mlx.DTypeBFloat16, 1, 2, 1, 4), mlx.Zeros(mlx.DTypeBFloat16, 1), 2); err == nil {
			t.Fatal("sink head mismatch accepted")
		}
	})
}

func TestSharedSDPASinkDTypeAndMemoIsolation(t *testing.T) {
	row10WithMLX(t, func() {
		call := func(dtype, sinkType mlx.DType, withSink bool, b *batch.Batch) {
			q := mlx.Zeros(dtype, 1, 1, 1, 4)
			k := mlx.Zeros(dtype, 1, 1, 1, 4)
			v := mlx.Zeros(dtype, 1, 1, 1, 4)
			opts := []nn.SDPAOption{nn.WithKV(k, v, []int32{1})}
			if withSink {
				opts = append(opts, nn.WithSinks(mlx.Zeros(sinkType, 1)))
			}
			out := nn.ScaledDotProductAttention(b, q, 0.5, opts...)
			if out == nil || !out.Valid() {
				t.Fatal("SDPA returned invalid output")
			}
		}
		newBatch := func() *batch.Batch {
			return &batch.Batch{
				InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 1),
				SeqOffsets:   []int32{0},
				SeqQueryLens: []int32{1},
			}
		}
		call(mlx.DTypeFloat32, mlx.DTypeFloat32, true, newBatch())
		call(mlx.DTypeBFloat16, mlx.DTypeBFloat16, true, newBatch())
		for _, mismatch := range []struct{ query, sink mlx.DType }{
			{mlx.DTypeFloat32, mlx.DTypeBFloat16},
			{mlx.DTypeBFloat16, mlx.DTypeFloat32},
		} {
			func() {
				defer func() {
					got := recover()
					if got == nil {
						t.Fatal("query/sink dtype mismatch accepted")
					}
					want := fmt.Sprintf("mlx.FastScaledDotProductAttention: sinks must have shape [heads]=[1] and dtype %s for query [1 1 1 4], got shape [1] dtype %s", mismatch.query, mismatch.sink)
					if fmt.Sprint(got) != want {
						t.Fatalf("panic = %q, want %q", got, want)
					}
				}()
				call(mismatch.query, mismatch.sink, true, newBatch())
			}()
		}
		shared := newBatch()
		call(mlx.DTypeFloat32, mlx.DTypeFloat32, false, shared)
		call(mlx.DTypeFloat32, mlx.DTypeFloat32, true, shared)
		call(mlx.DTypeFloat32, mlx.DTypeFloat32, false, shared)
	})
}
