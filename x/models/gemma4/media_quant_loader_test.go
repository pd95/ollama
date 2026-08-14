package gemma4

import (
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	gemma4metadata "github.com/ollama/ollama/x/models/gemma4/metadata"
	"github.com/ollama/ollama/x/models/nn"
)

func TestGemma4TowerMediaQuantizedLoadWeights(t *testing.T) {
	for _, quantized := range []bool{false, true} {
		name := "dense"
		if quantized {
			name = "mixed_nvfp4"
		}
		t.Run(name, func(t *testing.T) {
			useMLXTestThread(t)
			m, tensors := testGemma4TowerMediaModel(t, quantized)
			if err := m.LoadWeights(tensors); err != nil {
				t.Fatal(err)
			}
			if m.Vision == nil || m.EmbedVision == nil || m.Audio == nil || m.EmbedAudio == nil {
				t.Fatalf("media loaders were suppressed: vision=%v embed_vision=%v audio=%v embed_audio=%v", m.Vision != nil, m.EmbedVision != nil, m.Audio != nil, m.EmbedAudio != nil)
			}

			assertGemma4LinearStorage(t, m.Vision.PatchEmbedder.InputProj, quantized)
			assertGemma4LinearStorage(t, m.EmbedVision.Projection, quantized)
			for _, layer := range m.Vision.Layers {
				for _, linear := range []nn.LinearLayer{
					layer.Attention.QProj.Linear, layer.Attention.KProj.Linear,
					layer.Attention.VProj.Linear, layer.Attention.OProj.Linear,
					layer.MLP.GateProj.Linear, layer.MLP.UpProj.Linear, layer.MLP.DownProj.Linear,
				} {
					assertGemma4LinearStorage(t, linear, quantized)
				}
			}

			assertGemma4LinearStorage(t, m.Audio.InputProj, quantized)
			assertGemma4LinearStorage(t, m.Audio.OutputProj, quantized)
			assertGemma4LinearStorage(t, m.EmbedAudio.Projection, quantized)
			if quantized {
				assertGemma4QuantizedLinearType(t, m.EmbedVision.Projection, "mxfp8")
				assertGemma4QuantizedLinearType(t, m.Audio.OutputProj, "mxfp8")
				assertGemma4QuantizedLinearType(t, m.EmbedAudio.Projection, "mxfp8")
			}
			for _, layer := range m.Audio.Layers {
				for _, linear := range []nn.LinearLayer{
					layer.FeedForward1.Up.Linear, layer.FeedForward1.Down.Linear,
					layer.FeedForward2.Up.Linear, layer.FeedForward2.Down.Linear,
					layer.Attention.Q.Linear, layer.Attention.K.Linear, layer.Attention.V.Linear,
					layer.Attention.Output.Linear, layer.Attention.RelativeK,
					layer.LightConv.Start.Linear, layer.LightConv.End.Linear,
				} {
					assertGemma4LinearStorage(t, linear, quantized)
				}
			}

			if got := m.Vision.PatchEmbedder.PositionEmbeddingTable; got != tensors["model.vision_tower.patch_embedder.position_embedding_table"] {
				t.Fatal("vision position table was not retained as a dense tensor")
			}
			if quantized {
				output, ok := m.Audio.OutputProj.(*nn.QuantizedLinear)
				if !ok || output.Bias != tensors["model.audio_tower.output_proj.bias"] {
					t.Fatal("quantized audio output projection lost its learned bias")
				}
			}
		})
	}
}

func TestGemma4UnifiedMediaQuantizedLoadWeights(t *testing.T) {
	for _, quantized := range []bool{false, true} {
		name := "dense"
		if quantized {
			name = "nvfp4"
		}
		t.Run(name, func(t *testing.T) {
			useMLXTestThread(t)
			m, tensors := testGemma4UnifiedMediaModel(quantized)
			if err := m.LoadWeights(tensors); err != nil {
				t.Fatal(err)
			}
			if m.UnifiedVision == nil || m.EmbedVision == nil || m.EmbedAudio == nil {
				t.Fatalf("unified media loaders were suppressed: vision=%v embed_vision=%v embed_audio=%v", m.UnifiedVision != nil, m.EmbedVision != nil, m.EmbedAudio != nil)
			}
			if m.Vision != nil || m.Audio != nil {
				t.Fatalf("unified model loaded tower encoders: vision=%v audio=%v", m.Vision != nil, m.Audio != nil)
			}

			assertGemma4LinearStorage(t, m.UnifiedVision.PatchDense, quantized)
			assertGemma4LinearStorage(t, m.EmbedVision.Projection, quantized)
			assertGemma4LinearStorage(t, m.EmbedAudio.Projection, false)
			if quantized {
				assertGemma4QuantizedLinearType(t, m.UnifiedVision.PatchDense, "mxfp8")
				assertGemma4QuantizedLinearType(t, m.EmbedVision.Projection, "mxfp8")
			}
			if m.UnifiedVision.PosEmbedding != tensors["model.vision_embedder.pos_embedding"] {
				t.Fatal("unified position embedding was not retained as a dense tensor")
			}
			if quantized {
				patch, ok := m.UnifiedVision.PatchDense.(*nn.QuantizedLinear)
				if !ok || patch.Bias != tensors["model.vision_embedder.patch_dense.bias"] {
					t.Fatal("quantized unified patch projection lost its learned bias")
				}
			}
		})
	}
}

func TestGemma4MediaQuantizedLoadUsesSharedGlobalScale(t *testing.T) {
	useMLXTestThread(t)
	m, tensors := testGemma4UnifiedMediaModel(true)
	globalScale := mlx.FromValue(float32(1))
	tensors["model.vision_embedder.patch_dense.weight.global_scale"] = globalScale
	if err := m.LoadWeights(tensors); err != nil {
		t.Fatal(err)
	}
	linear, ok := m.UnifiedVision.PatchDense.(*nn.QuantizedLinear)
	if !ok {
		t.Fatalf("patch projection type = %T, want *nn.QuantizedLinear", m.UnifiedVision.PatchDense)
	}
	if linear.GlobalScale != globalScale {
		t.Fatal("Gemma4 media projection did not reuse the shared global-scale tensor")
	}
}

func testGemma4TowerMediaModel(t *testing.T, quantized bool) (*Model, map[string]*mlx.Array) {
	t.Helper()
	const hidden = 64
	tensorQuant := make(map[string]*model.TensorQuantInfo)
	tensors := testGemma4BaseTensors(hidden)
	visionConfig := &VisionConfig{
		ModelType: "gemma4_vision", HiddenSize: hidden, IntermediateSize: 128,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: hidden,
		RMSNormEps: 1e-6, PatchSize: 8, PositionEmbeddingSize: 4,
	}
	audioConfig := &AudioConfig{
		ModelType: "gemma4_audio", AttentionChunkSize: 2, AttentionContextLeft: 2,
		AttentionInvalidLogit: -1e9, AttentionLogitCap: 50, ConvKernelSize: 3,
		GradientClipping: 1e10, HiddenSize: hidden, NumAttentionHeads: 1,
		NumHiddenLayers: 1, OutputProjDims: hidden, ResidualWeight: 0.5,
		RMSNormEps: 1e-6, SubsamplingConvChannels: []int32{2, 2},
	}

	putGemma4TestLinear(tensors, tensorQuant, "model.vision_tower.patch_embedder.input_proj", hidden, 8*8*3, quantized)
	tensors["model.vision_tower.patch_embedder.position_embedding_table"] = testGemma4Array(4, hidden)
	for i := range visionConfig.NumHiddenLayers {
		layer := fmt.Sprintf("model.vision_tower.encoder.layers.%d", i)
		for _, spec := range []struct {
			path    string
			out, in int
		}{
			{layer + ".self_attn.q_proj.linear", hidden, hidden},
			{layer + ".self_attn.k_proj.linear", hidden, hidden},
			{layer + ".self_attn.v_proj.linear", hidden, hidden},
			{layer + ".self_attn.o_proj.linear", hidden, hidden},
			{layer + ".mlp.gate_proj.linear", 128, hidden},
			{layer + ".mlp.up_proj.linear", 128, hidden},
			{layer + ".mlp.down_proj.linear", hidden, 128},
		} {
			putGemma4TestLinear(tensors, tensorQuant, spec.path, spec.out, spec.in, quantized)
		}
		for _, suffix := range []string{
			".self_attn.q_norm.weight", ".self_attn.k_norm.weight",
			".input_layernorm.weight", ".post_attention_layernorm.weight",
			".pre_feedforward_layernorm.weight", ".post_feedforward_layernorm.weight",
		} {
			tensors[layer+suffix] = testGemma4Array(hidden)
		}
	}
	putGemma4TestLinear(tensors, tensorQuant, "model.embed_vision.embedding_projection", hidden, hidden, quantized)

	required, err := gemma4metadata.RequiredAudioTensorShapes(audioMetadataConfig(audioConfig, hidden))
	if err != nil {
		t.Fatal(err)
	}
	for name, shape := range required {
		if len(shape) == 2 && strings.HasSuffix(name, ".weight") {
			putGemma4TestLinear(tensors, tensorQuant, strings.TrimSuffix(name, ".weight"), int(shape[0]), int(shape[1]), quantized)
			continue
		}
		tensors[name] = testGemma4Array32(shape...)
	}

	return &Model{
		TextConfig: &TextConfig{
			HiddenSize: hidden, VocabSize: 4, RMSNormEps: 1e-6,
			TensorQuant: tensorQuant,
		},
		VisionConfig:         visionConfig,
		AudioConfig:          audioConfig,
		AudioProcessorConfig: &AudioProcessorConfig{},
	}, tensors
}

func testGemma4UnifiedMediaModel(quantized bool) (*Model, map[string]*mlx.Array) {
	const hidden = 64
	tensorQuant := make(map[string]*model.TensorQuantInfo)
	tensors := testGemma4BaseTensors(hidden)
	visionConfig := &VisionConfig{
		ModelType: "gemma4_unified_vision", MMEmbedDim: hidden, MMPosembSize: 4,
		ModelPatchSize: 8, PatchSize: 8, PoolingKernelSize: 1, RMSNormEps: 1e-6,
	}
	audioConfig := &AudioConfig{
		ModelType: "gemma4_unified_audio", AudioEmbedDim: 640, AudioSamplesPerToken: 640,
		HiddenSize: 640, OutputProjDims: 640, RMSNormEps: 1e-6,
	}
	patchDim := 8 * 8 * 3
	for _, name := range []string{"patch_ln1", "patch_ln2", "pos_norm"} {
		dim := hidden
		if name == "patch_ln1" {
			dim = patchDim
		}
		tensors["model.vision_embedder."+name+".weight"] = testGemma4Array(dim)
		tensors["model.vision_embedder."+name+".bias"] = testGemma4Array(dim)
	}
	tensors["model.vision_embedder.pos_embedding"] = testGemma4Array(4, 2, hidden)
	putGemma4TestLinear(tensors, tensorQuant, "model.vision_embedder.patch_dense", hidden, patchDim, quantized)
	tensors["model.vision_embedder.patch_dense.bias"] = testGemma4Array(hidden)
	putGemma4TestLinear(tensors, tensorQuant, "model.embed_vision.embedding_projection", hidden, hidden, quantized)
	putGemma4TestLinear(tensors, tensorQuant, "model.embed_audio.embedding_projection", hidden, 640, false)

	return &Model{
		TextConfig: &TextConfig{
			HiddenSize: hidden, VocabSize: 4, RMSNormEps: 1e-6,
			TensorQuant: tensorQuant,
		},
		VisionConfig:         visionConfig,
		AudioConfig:          audioConfig,
		AudioProcessorConfig: &AudioProcessorConfig{},
	}, tensors
}

func testGemma4BaseTensors(hidden int) map[string]*mlx.Array {
	return map[string]*mlx.Array{
		"model.embed_tokens.weight": testGemma4Array(4, hidden),
		"model.norm.weight":         testGemma4Array(hidden),
	}
}

func putGemma4TestLinear(tensors map[string]*mlx.Array, tensorQuant map[string]*model.TensorQuantInfo, path string, out, in int, quantized bool) {
	weightName := path + ".weight"
	weight := testGemma4Array(out, in)
	if !quantized {
		tensors[weightName] = weight
		return
	}
	quantType := "nvfp4"
	if strings.Contains(path, ".v_proj") || strings.Contains(path, ".k_proj") || strings.Contains(path, "down_proj") || isGemma4TestMediaBoundaryProjection(path) {
		quantType = "mxfp8"
	}
	groupSize, bits, mode := model.QuantizationParams(quantType)
	packed, scales, qbias := mlx.Quantize(weight, groupSize, bits, mode)
	if qbias != nil {
		mlx.Eval(packed, scales, qbias)
		tensors[weightName+"_qbias"] = qbias
	} else {
		mlx.Eval(packed, scales)
	}
	tensors[weightName] = packed
	tensors[weightName+"_scale"] = scales
	tensorQuant[weightName] = &model.TensorQuantInfo{QuantType: quantType, GroupSize: groupSize}
}

func isGemma4TestMediaBoundaryProjection(path string) bool {
	for _, suffix := range []string{
		"embed_vision.embedding_projection",
		"vision_embedder.patch_dense",
		"audio_tower.output_proj",
		"embed_audio.embedding_projection",
	} {
		if strings.HasSuffix(path, suffix) {
			return true
		}
	}
	return false
}

func testGemma4Array(shape ...int) *mlx.Array {
	if len(shape) == 0 {
		return mlx.FromValue(float32(0.01))
	}
	return mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, shape...), 0.01)
}

func testGemma4Array32(shape ...int32) *mlx.Array {
	dims := make([]int, len(shape))
	for i, dim := range shape {
		dims[i] = int(dim)
	}
	return testGemma4Array(dims...)
}

func assertGemma4LinearStorage(t *testing.T, linear nn.LinearLayer, quantized bool) {
	t.Helper()
	if quantized {
		if _, ok := linear.(*nn.QuantizedLinear); !ok {
			t.Fatalf("linear type = %T, want *nn.QuantizedLinear", linear)
		}
		return
	}
	if _, ok := linear.(*nn.Linear); !ok {
		t.Fatalf("linear type = %T, want *nn.Linear", linear)
	}
}

func assertGemma4QuantizedLinearType(t *testing.T, linear nn.LinearLayer, quantType string) {
	t.Helper()
	quantized, ok := linear.(*nn.QuantizedLinear)
	if !ok {
		t.Fatalf("linear type = %T, want *nn.QuantizedLinear", linear)
	}
	groupSize, bits, mode := model.QuantizationParams(quantType)
	if quantized.GroupSize != groupSize || quantized.Bits != bits || quantized.Mode != mode {
		t.Fatalf("quantization params = (%d, %d, %q), want (%d, %d, %q)",
			quantized.GroupSize, quantized.Bits, quantized.Mode, groupSize, bits, mode)
	}
}
