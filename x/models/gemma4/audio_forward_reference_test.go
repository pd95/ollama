package gemma4

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	runnermodel "github.com/ollama/ollama/x/mlxrunner/model"
	gemma4metadata "github.com/ollama/ollama/x/models/gemma4/metadata"
	"github.com/ollama/ollama/x/models/nn"
)

const (
	gemma4LanguageLogitMaxDiff = 1.25
	gemma4LanguageLogitRMSE    = 0.25
)

func TestAudioForwardReference(t *testing.T) {
	modelDir := os.Getenv("GEMMA4_AUDIO_MODEL_DIR")
	refDir := os.Getenv("GEMMA4_AUDIO_REF_DIR")
	wavPath := os.Getenv("GEMMA4_AUDIO_WAV")
	if modelDir == "" || refDir == "" || wavPath == "" {
		t.Skip("set GEMMA4_AUDIO_MODEL_DIR, GEMMA4_AUDIO_REF_DIR, and GEMMA4_AUDIO_WAV for audio parity")
	}
	skipIfNoMLX(t)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
	}

	configData, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	audioConfig, err := parseAudioConfig(configData)
	if err != nil || audioConfig == nil {
		t.Fatalf("parse audio config: %v", err)
	}
	textConfig, err := parseTextConfig(configData)
	if err != nil {
		t.Fatal(err)
	}
	processorData, err := os.ReadFile(filepath.Join(modelDir, "processor_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	processorConfig, err := parseAudioProcessorConfig(processorData)
	if err != nil {
		t.Fatal(err)
	}
	wav, err := os.ReadFile(wavPath)
	if err != nil {
		t.Fatal(err)
	}
	input, err := preprocessGemma4Audio(context.Background(), wav, processorConfig)
	if err != nil {
		t.Fatal(err)
	}

	source, err := mlx.LoadSafetensorsNative(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Free()
	reference, err := mlx.LoadSafetensorsNative(filepath.Join(refDir, "audio-checkpoints.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer reference.Free()
	if wantFrames := reference.Get("input_features").Dim(1); wantFrames > input.Frames {
		padding := wantFrames - input.Frames
		input.Features = append(input.Features, make([]float32, padding*input.FeatureSize)...)
		input.FeatureMask = append(input.FeatureMask, make([]bool, padding)...)
		input.Frames = wantFrames
	}

	required, err := gemma4metadata.RequiredAudioTensorShapes(audioMetadataConfig(audioConfig, textConfig.HiddenSize))
	if err != nil {
		t.Fatal(err)
	}
	tensors := make(map[string]*mlx.Array, len(required))
	for name := range required {
		tensors[name] = source.Get(name)
		if tensors[name] == nil {
			t.Fatalf("source is missing %s", name)
		}
	}
	embedAudio, err := loadMultimodalEmbedder(tensors, "embed_audio", audioConfig.RMSNormEps, 0, 0, "", nil)
	if err != nil {
		t.Fatal(err)
	}

	features := mlx.FromValues(input.Features, 1, input.Frames, input.FeatureSize)
	compareAudioReference(t, "input_features", features, reference.Get("input_features"), 2e-5, 1e-5)
	compareAudioMaskReference(t, "input_features_mask", input.FeatureMask, reference.Get("input_features_mask"), false)
	var projectedForward *mlx.Array
	if audioConfig.unified() {
		projectedForward = embedAudio.Forward(features)
		compareAudioReference(t, "multimodal_projection", projectedForward, reference.Get("multimodal_projection"), 0.5, 0.05)
	} else {
		audioModel, err := loadAudioModel(tensors, audioConfig, textConfig.HiddenSize, 0, 0, "", nil)
		if err != nil {
			t.Fatal(err)
		}
		valid := append([]bool(nil), input.FeatureMask...)
		x := mlx.ExpandDims(features, -1)
		x, valid = audioModel.Conv0.Forward(x, valid)
		x, valid = audioModel.Conv1.Forward(x, valid)
		compareAudioMaskReference(t, "audio_forward_mask", valid, reference.Get("audio_forward_mask"), true)
		validOutputs := 0
		for _, value := range valid {
			if value {
				validOutputs++
			}
		}
		if validOutputs != input.SoftTokens {
			t.Fatalf("valid audio outputs = %d, want %d", validOutputs, input.SoftTokens)
		}
		x = mlx.Reshape(x, 1, int32(x.Dim(1)), int32(x.Dim(2)*x.Dim(3)))
		x = audioModel.InputProj.Forward(x)
		compareAudioReference(t, "subsample", x, reference.Get("subsample"), 0.5, 0.05)
		for index, layer := range audioModel.Layers {
			x = layer.Forward(x, valid)
			if index == 0 || index == len(audioModel.Layers)-1 {
				name := "layer_0"
				if index != 0 {
					name = "layer_11"
				}
				compareAudioReference(t, name, x, reference.Get(name), 0.5, 0.05)
			}
		}
		x = audioModel.OutputProj.Forward(x)
		compareAudioReference(t, "output_projection", x, reference.Get("output_projection"), 0.5, 0.05)
		x = embedAudio.Forward(x)
		compareAudioReference(t, "multimodal_projection", x, reference.Get("multimodal_projection"), 0.5, 0.05)

		forward := audioModel.Forward(input)
		compareAudioReference(t, "audio_physical_output", forward, reference.Get("audio_physical_output"), 0.5, 0.05)
		projectedForward = embedAudio.Forward(forward)
	}
	inputIDs := reference.Get("input_ids")
	embedWeight := source.Get("model.language_model.embed_tokens.weight")
	if inputIDs == nil || embedWeight == nil {
		t.Fatal("reference input IDs or source token embeddings are missing")
	}
	inputIDs = inputIDs.AsType(mlx.DTypeInt32)
	tokenEmbeddings := mlx.MulScalar(mlx.Take(embedWeight, inputIDs, 0), textConfig.EmbedScale)
	mlx.Eval(inputIDs)
	start, end, err := mediaTokenSpan(intsToInt32(inputIDs.Ints()), textConfig.AudioTokenIDValue)
	if err != nil {
		t.Fatal(err)
	}
	if end-start != input.SoftTokens {
		t.Fatalf("audio token span = %d, want %d", end-start, input.SoftTokens)
	}
	parts := make([]*mlx.Array, 0, 3)
	if start > 0 {
		parts = append(parts, mlx.SliceStartStop(tokenEmbeddings, []int32{0, 0, 0}, []int32{1, int32(start), textConfig.HiddenSize}))
	}
	parts = append(parts, projectedForward)
	if end < inputIDs.Dim(1) {
		parts = append(parts, mlx.SliceStartStop(tokenEmbeddings, []int32{0, int32(end), 0}, []int32{1, int32(inputIDs.Dim(1)), textConfig.HiddenSize}))
	}
	finalEmbeddings := mlx.Concatenate(parts, 1)
	compareAudioReference(t, "final_input_embeddings", finalEmbeddings, reference.Get("final_input_embeddings"), 0.5, 0.05)

	if importedModel := os.Getenv("GEMMA4_AUDIO_MODEL_NAME"); importedModel != "" {
		fullModel := loadImportedGemma4ReferenceModel(t, importedModel)
		// The Python checkpoint records raw model logits before generation-time
		// token suppression. Keep this comparison at the same boundary.
		fullModel.SuppressLogitBias = nil
		pleIDs := intsToInt32(inputIDs.Ints())
		modelBatch := &batch.Batch{
			InputIDs:     inputIDs,
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(len(pleIDs))},
			Media: []batch.MediaItem{{
				Seq: 0, Pos: start, Features: projectedForward,
				Opaque: &gemma4PreparedMedia{Kind: llm.MediaKindAudio, SoftTokens: end - start},
			}},
		}
		hidden, _ := fullModel.Forward(modelBatch, fullModel.NewCaches())
		logits := fullModel.Unembed(hidden)
		last := mlx.SliceStartStop(logits, []int32{0, int32(len(pleIDs) - 1), 0}, []int32{1, int32(len(pleIDs)), textConfig.VocabSize})
		wantLogits := reference.Get("prefill_logits")
		if audioConfig.unified() {
			compareLanguageLogitReference(t, "prefill_logits", last, wantLogits)
		} else {
			compareAudioReference(t, "prefill_logits", last, wantLogits, 0.5, 0.05)
		}
		gotID := last.Argmax(-1, false).AsType(mlx.DTypeInt32)
		wantID := wantLogits.Argmax(-1, false).AsType(mlx.DTypeInt32)
		mlx.Eval(gotID, wantID)
		if got, want := gotID.Int(), wantID.Int(); got != want {
			t.Fatalf("prefill argmax = %d, want %d", got, want)
		} else {
			t.Logf("prefill argmax matched: %d", got)
		}

		compareGreedyDecodeReference(t, fullModel, len(pleIDs), caches, last, gotID, reference)
	}
}

func TestLanguageForwardReference(t *testing.T) {
	modelDir := os.Getenv("GEMMA4_AUDIO_MODEL_DIR")
	refDir := os.Getenv("GEMMA4_AUDIO_REF_DIR")
	importedModel := os.Getenv("GEMMA4_AUDIO_MODEL_NAME")
	if modelDir == "" || refDir == "" || importedModel == "" {
		t.Skip("set GEMMA4_AUDIO_MODEL_DIR, GEMMA4_AUDIO_REF_DIR, and GEMMA4_AUDIO_MODEL_NAME for language parity")
	}
	skipIfNoMLX(t)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
	}

	reference, err := mlx.LoadSafetensorsNative(filepath.Join(refDir, "audio-checkpoints.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer reference.Free()
	inputIDs := reference.Get("input_ids")
	referenceEmbeddings := reference.Get("final_input_embeddings")
	if inputIDs == nil || referenceEmbeddings == nil {
		t.Fatal("reference input IDs or final input embeddings are missing")
	}
	inputIDs = inputIDs.AsType(mlx.DTypeInt32)
	mlx.Eval(inputIDs, referenceEmbeddings)
	model := loadImportedGemma4ReferenceModel(t, importedModel)
	model.SuppressLogitBias = nil
	source, err := mlx.LoadSafetensorsNative(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Free()
	for _, index := range representativeLayerIndices(len(model.Layers)) {
		layer := model.Layers[index]
		weights := map[string]*mlx.Array{
			"self_attn.q_proj": denseReferenceWeight(t, layer.Attention.QProj),
			"self_attn.k_proj": denseReferenceWeight(t, layer.Attention.KProj),
			"self_attn.o_proj": denseReferenceWeight(t, layer.Attention.OProj),
		}
		if layer.Attention.VProj != nil {
			weights["self_attn.v_proj"] = denseReferenceWeight(t, layer.Attention.VProj)
		}
		for name, got := range weights {
			key := fmt.Sprintf("model.language_model.layers.%d.%s.weight", index, name)
			compareAudioReference(t, "source_"+key, got, source.Get(key), 0, 0)
		}
	}

	textOnly := reference.Get("input_features") == nil
	if textOnly {
		nativeEmbeddings := model.TokenEmbeddings(inputIDs)
		compareAudioReference(t, "text_input_embeddings", nativeEmbeddings, referenceEmbeddings, 2e-5, 1e-5)
	}

	pleIDs := intsToInt32(inputIDs.Ints())
	for i, token := range pleIDs {
		if token == model.AudioTokenIDValue || token == model.ImageTokenIDValue {
			pleIDs[i] = 0
		}
	}
	modelBatch := &batch.Batch{
		InputIDs:        inputIDs,
		InputEmbeddings: referenceEmbeddings,
		PLEInputIDs:     mlx.FromValues(pleIDs, 1, len(pleIDs)),
		SeqOffsets:      []int32{0},
		SeqQueryLens:    []int32{int32(len(pleIDs))},
	}
	for name, got := range gemma4FirstLayerReference(model, modelBatch) {
		observeAudioReference(t, name, got, reference.Get(name), 0.5, 0.05)
	}
	compareIndependentGemma4ReferenceLayers(t, model, modelBatch, reference)
	caches := model.NewCaches()
	layers, hidden := forwardGemma4ReferenceLayers(model, modelBatch, caches)
	for index, got := range layers {
		name := fmt.Sprintf("cached_layer_%02d", index)
		want := reference.Get(name)
		if want == nil {
			t.Fatalf("%s tensor is missing", name)
		}
		observeAudioReference(t, name, got, want, 0.5, 0.05)
		observeAudioReference(t, "last_"+name, lastSequenceState(got), lastSequenceState(want), 0.5, 0.05)
		roundtripName := fmt.Sprintf("roundtrip_layer_%02d", index)
		observeAudioReference(t, roundtripName, got, reference.Get(roundtripName), 0.5, 0.05)
	}
	observeAudioReference(t, "cached_final_hidden", hidden, reference.Get("cached_final_hidden"), 0.5, 0.05)
	observeAudioReference(t, "last_cached_final_hidden", lastSequenceState(hidden), lastSequenceState(reference.Get("cached_final_hidden")), 0.5, 0.05)
	observeAudioReference(t, "roundtrip_final_hidden", hidden, reference.Get("roundtrip_final_hidden"), 0.5, 0.05)

	logits := model.Unembed(hidden)
	last := mlx.SliceStartStop(logits,
		[]int32{0, int32(len(pleIDs) - 1), 0},
		[]int32{1, int32(len(pleIDs)), model.VocabSize},
	)
	wantLogits := reference.Get("prefill_logits")
	if textOnly {
		compareLanguageLogitReference(t, "text_prefill_logits", last, wantLogits)
	} else {
		observeAudioReference(t, "exact_embedding_prefill_logits", last, wantLogits, 0.5, 0.05)
	}
	observeAudioReference(t, "roundtrip_prefill_logits", last, reference.Get("roundtrip_prefill_logits"), 0.5, 0.05)
	gotID := last.Argmax(-1, false).AsType(mlx.DTypeInt32)
	wantID := wantLogits.Argmax(-1, false).AsType(mlx.DTypeInt32)
	mlx.Eval(gotID, wantID)
	if got, want := gotID.Int(), wantID.Int(); got != want {
		t.Fatalf("exact-embedding prefill argmax = %d, want %d", got, want)
	}
	t.Logf("exact-embedding prefill argmax matched: %d", gotID.Int())
	compareGreedyDecodeReference(t, model, len(pleIDs), caches, last, gotID, reference)
	referenceHead := model.Unembed(lastSequenceState(reference.Get("cached_final_hidden")))
	compareAudioReference(t, "reference_hidden_lm_head", referenceHead, wantLogits, 0.5, 0.05)

	uncachedLayers, uncachedHidden := forwardGemma4ReferenceLayers(model, modelBatch, nil)
	for index, got := range uncachedLayers {
		name := fmt.Sprintf("cached_layer_%02d", index)
		observeAudioReference(t, "uncached_native_"+name, got, reference.Get(name), 0.5, 0.05)
	}
	observeAudioReference(t, "uncached_native_final_hidden", uncachedHidden, reference.Get("cached_final_hidden"), 0.5, 0.05)
	uncachedLogits := model.Unembed(uncachedHidden)
	uncachedLast := mlx.SliceStartStop(uncachedLogits,
		[]int32{0, int32(len(pleIDs) - 1), 0},
		[]int32{1, int32(len(pleIDs)), model.VocabSize},
	)
	observeAudioReference(t, "uncached_native_prefill_logits", uncachedLast, reference.Get("prefill_logits_uncached"), 0.5, 0.05)
}

func denseReferenceWeight(t *testing.T, layer nn.LinearLayer) *mlx.Array {
	t.Helper()
	linear, ok := layer.(*nn.Linear)
	if !ok {
		t.Fatalf("reference parity requires a dense linear, got %T", layer)
	}
	return linear.Weight
}

func representativeLayerIndices(count int) []int {
	if count <= 0 {
		return nil
	}
	if count == 1 {
		return []int{0}
	}
	indices := []int{0, count / 2, count - 1}
	if count > 6 {
		indices = append(indices, 5)
	}
	slices.Sort(indices)
	return slices.Compact(indices)
}

func compareIndependentGemma4ReferenceLayers(t *testing.T, m *Model, b *batch.Batch, reference *mlx.SafetensorsFile) {
	t.Helper()
	if m.HiddenSizePerLayer > 0 || m.EmbedTokensPerLayer != nil || len(m.KVShareMap) > 0 {
		t.Fatal("independent decoder-layer parity requires a model without PLE or KV sharing")
	}
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	for index, layer := range m.Layers {
		inputName := fmt.Sprintf("layer_input_%02d", index)
		outputName := fmt.Sprintf("cached_layer_%02d", index)
		input := reference.Get(inputName)
		want := reference.Get(outputName)
		if input == nil || want == nil {
			t.Fatalf("independent layer reference is missing %s or %s", inputName, outputName)
		}
		got, _ := layer.Forward(input, b, nil, positions, B, L, m.TextConfig, nil, nil)
		compareAudioReference(t, "independent_"+outputName, got, want, 0.5, 0.05)
	}
}

func compareGreedyDecodeReference(t *testing.T, m *Model, promptLength int, caches []cache.Cache, initialLogits, initialID *mlx.Array, reference *mlx.SafetensorsFile) {
	t.Helper()
	wantTokens := reference.Get("generation_token_ids")
	if wantTokens == nil {
		return
	}
	if wantTokens.NumDims() != 1 {
		t.Fatalf("generation token IDs shape = %v, want [tokens]", wantTokens.Dims())
	}
	wantTokens = wantTokens.AsType(mlx.DTypeInt32)
	mlx.Eval(wantTokens)
	wantValues := wantTokens.Ints()
	wantTopIDs := reference.Get("generation_top_ids")
	wantTopLogprobs := reference.Get("generation_top_logprobs")
	if (wantTopIDs == nil) != (wantTopLogprobs == nil) {
		t.Fatal("generation reference must contain both top IDs and top log probabilities")
	}
	if wantTopIDs != nil {
		if wantTopIDs.NumDims() != 2 || wantTopIDs.Dim(0) != len(wantValues) || wantTopIDs.Dim(1) != 2 {
			t.Fatalf("generation top IDs shape = %v, want [%d 2]", wantTopIDs.Dims(), len(wantValues))
		}
		if !equalIntShape(wantTopIDs.Dims(), wantTopLogprobs.Dims()) {
			t.Fatalf("generation top log probabilities shape = %v, want %v", wantTopLogprobs.Dims(), wantTopIDs.Dims())
		}
		wantTopIDs = wantTopIDs.AsType(mlx.DTypeInt32)
		wantTopLogprobs = wantTopLogprobs.AsType(mlx.DTypeFloat32)
		mlx.Eval(wantTopIDs, wantTopLogprobs)
	}

	currentLogits := initialLogits
	gotToken := initialID.Int()
	for step, wantToken := range wantValues {
		if gotToken != wantToken {
			probeIDs := mlx.FromValues([]int32{int32(gotToken), int32(wantToken)}, 2)
			probeLogits := mlx.Take(currentLogits, probeIDs, 2).AsType(mlx.DTypeFloat32)
			mlx.Eval(probeLogits)
			probeValues := probeLogits.Floats()
			if wantTopIDs != nil && wantTopLogprobs != nil {
				refIDs := wantTopIDs.Ints()
				refValues := wantTopLogprobs.Floats()
				base := step * 2
				t.Fatalf("greedy decode token %d = %d, want %d; native logits got=%g want=%g; reference top2=[%d:%g %d:%g]", step, gotToken, wantToken, probeValues[0], probeValues[1], refIDs[base], refValues[base], refIDs[base+1], refValues[base+1])
			}
			t.Fatalf("greedy decode token %d = %d, want %d; native logits got=%g want=%g", step, gotToken, wantToken, probeValues[0], probeValues[1])
		}
		if step == len(wantValues)-1 {
			break
		}

		inputToken := mlx.FromValues([]int32{int32(gotToken)}, 1, 1)
		hidden := m.Forward(&batch.Batch{
			InputIDs:     inputToken,
			SeqOffsets:   []int32{int32(promptLength + step)},
			SeqQueryLens: []int32{1},
		}, caches)
		currentLogits = m.Unembed(hidden)
		gotID := currentLogits.Argmax(-1, false).AsType(mlx.DTypeInt32)
		mlx.Eval(gotID)
		gotToken = gotID.Int()
	}
	t.Logf("greedy decode matched %d generated tokens", len(wantValues))
}

func lastSequenceState(x *mlx.Array) *mlx.Array {
	return mlx.SliceStartStop(x,
		[]int32{0, int32(x.Dim(1) - 1), 0},
		[]int32{1, int32(x.Dim(1)), int32(x.Dim(2))},
	)
}

func gemma4FirstLayerReference(m *Model, b *batch.Batch) map[string]*mlx.Array {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	layer := m.Layers[0]
	inputNorm := mlx.RMSNormFn(b.InputEmbeddings, layer.InputNormScaled, m.RMSNormEps)
	attentionLayer := layer.Attention
	headDim := m.HeadDim
	qProjection := attentionLayer.QProj.Forward(inputNorm)
	q := mlx.Reshape(qProjection, B, L, m.NumAttentionHeads, headDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	q = mlx.RMSNormFn(q, attentionLayer.QNormScaled, m.RMSNormEps)
	qRope := mlx.RoPEWithFreqs(q, m.SlidingRopeDims, false, m.SlidingRopeBase, 1.0, positions, nil)

	kProjection := attentionLayer.KProj.Forward(inputNorm)
	k := mlx.Reshape(kProjection, B, L, m.NumKeyValueHeads, headDim)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	vProjection := attentionLayer.VProj.Forward(inputNorm)
	v := mlx.Reshape(vProjection, B, L, m.NumKeyValueHeads, headDim)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	k = mlx.RMSNormFn(k, attentionLayer.KNormScaled, m.RMSNormEps)
	kRope := mlx.RoPEWithFreqs(k, m.SlidingRopeDims, false, m.SlidingRopeBase, 1.0, positions, nil)
	v = mlx.RMSNormFn(v, nil, m.RMSNormEps)
	attentionHeads := mlx.FastScaledDotProductAttention(qRope, kRope, v, m.SlidingScale, "causal", nil)
	attentionMerged := mlx.Reshape(mlx.Transpose(attentionHeads, 0, 2, 1, 3), B, L, m.NumAttentionHeads*headDim)
	attention := attentionLayer.OProj.Forward(attentionMerged)
	postAttention := mlx.RMSNormFn(attention, layer.PostAttnNormScaled, m.RMSNormEps)
	attentionResidual := mlx.Add(b.InputEmbeddings, postAttention)
	preFF := mlx.RMSNormFn(attentionResidual, layer.PreFFNormScaled, m.RMSNormEps)
	mlp := layer.MLP.Forward(preFF)
	postFF := mlx.RMSNormFn(mlp, layer.PostFFNormScaled, m.RMSNormEps)
	output := mlx.Add(attentionResidual, postFF)
	if layer.LayerScalar != nil {
		output = mlx.Mul(output, layer.LayerScalar)
	}
	return map[string]*mlx.Array{
		"layer0_input_norm":          inputNorm,
		"layer0_query_projection":    qProjection,
		"layer0_query_norm":          q,
		"layer0_query_rope":          qRope,
		"layer0_key_projection":      kProjection,
		"layer0_key_norm":            k,
		"layer0_key_rope":            kRope,
		"layer0_value_projection":    vProjection,
		"layer0_value_norm":          v,
		"layer0_attention_heads":     attentionHeads,
		"layer0_attention_merged":    attentionMerged,
		"layer0_attention":           attention,
		"layer0_post_attention_norm": postAttention,
		"layer0_attention_residual":  attentionResidual,
		"layer0_pre_ff_norm":         preFF,
		"layer0_mlp":                 mlp,
		"layer0_post_ff_norm":        postFF,
		"layer0_output":              output,
	}
}

func forwardGemma4ReferenceLayers(m *Model, b *batch.Batch, caches []cache.Cache) ([]*mlx.Array, *mlx.Array) {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	h := b.InputEmbeddings
	if h == nil {
		h = m.TokenEmbeddings(b.InputIDs)
	}

	var perLayerInputs *mlx.Array
	if m.HiddenSizePerLayer > 0 && m.EmbedTokensPerLayer != nil {
		pleTokens := b.InputIDs
		if b.PLEInputIDs != nil {
			pleTokens = b.PLEInputIDs
		}
		perLayerInputs = m.computePLEInputs(pleTokens, h)
	}

	var sharedKV map[int32]sharedHistory
	if len(m.KVShareMap) > 0 {
		sharedKV = make(map[int32]sharedHistory)
	}
	layers := make([]*mlx.Array, 0, len(m.Layers))
	for i, layer := range m.Layers {
		var c cache.Cache
		if i < len(caches) {
			c = caches[i]
		}
		var pleInput *mlx.Array
		if perLayerInputs != nil {
			pleInput = sliceLayerDim(perLayerInputs, int32(i), B, L, m.HiddenSizePerLayer)
		}
		var donor *sharedHistory
		if layer.KVShareDonor >= 0 {
			if value, ok := sharedKV[layer.KVShareDonor]; ok {
				donor = &value
			}
		}
		var donorKV *sharedHistory
		h, donorKV = layer.Forward(h, b, c, positions, B, L, m.TextConfig, pleInput, donor)
		if layer.IsDonor && donorKV != nil {
			sharedKV[layer.LayerIdx] = *donorKV
		}
		layers = append(layers, h)
	}
	return layers, mlx.RMSNormFn(h, m.NormScaled, m.RMSNormEps)
}

func loadImportedGemma4ReferenceModel(t *testing.T, name string) *Model {
	t.Helper()
	root, err := runnermodel.Open(name)
	if err != nil {
		t.Fatalf("open imported model %q: %v", name, err)
	}
	defer root.Close()
	bm, err := newModel(root)
	if err != nil {
		t.Fatalf("construct imported model: %v", err)
	}
	tensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	for _, layer := range root.Manifest.GetTensorLayers("") {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		for tensorName, array := range mlx.Load(root.Manifest.BlobPath(layer.Digest)) {
			tensors[tensorName] = array
		}
	}
	for tensorName, array := range tensors {
		if strings.HasSuffix(tensorName, ".scale") {
			tensors[strings.TrimSuffix(tensorName, ".scale")+"_scale"] = array
		}
	}
	if err := bm.LoadWeights(tensors); err != nil {
		t.Fatalf("load imported model weights: %v", err)
	}
	collected := mlx.Collect(bm)
	for _, array := range collected {
		mlx.Pin(array)
	}
	mlx.Eval(collected...)
	m, ok := bm.(*Model)
	if !ok {
		t.Fatalf("imported model type = %T, want *gemma4.Model", bm)
	}
	return m
}

func compareAudioMaskReference(t *testing.T, name string, got []bool, want *mlx.Array, invertWant bool) {
	t.Helper()
	if want == nil {
		t.Fatalf("%s tensor is missing", name)
	}
	want = want.AsType(mlx.DTypeInt32)
	mlx.Eval(want)
	values := want.Ints()
	if len(got) != len(values) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(values))
	}
	for i, value := range values {
		expected := value != 0
		if invertWant {
			expected = !expected
		}
		if got[i] != expected {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], expected)
		}
	}
}

func intsToInt32(values []int) []int32 {
	out := make([]int32, len(values))
	for i, value := range values {
		out[i] = int32(value)
	}
	return out
}

func compareAudioReference(t *testing.T, name string, got, want *mlx.Array, atol, rtol float64) {
	t.Helper()
	stats := measureAudioReference(t, name, got, want, atol, rtol)
	if stats.firstMismatch >= 0 {
		t.Errorf("%s[%d] = %g, want %g (diff %g, tolerance %g); max absolute difference %g, mean absolute difference %g, RMSE %g", name, stats.firstMismatch, stats.gotFirst, stats.wantFirst, stats.firstDiff, stats.firstTolerance, stats.maxDiff, stats.meanAbsDiff, stats.rmse)
		return
	}
	t.Logf("%s matched %d values; max absolute difference %g, mean absolute difference %g, RMSE %g", name, stats.count, stats.maxDiff, stats.meanAbsDiff, stats.rmse)
}

func compareLanguageLogitReference(t *testing.T, name string, got, want *mlx.Array) {
	t.Helper()
	stats := measureAudioReference(t, name, got, want, math.Inf(1), 0)
	if stats.maxDiff > gemma4LanguageLogitMaxDiff || stats.rmse > gemma4LanguageLogitRMSE {
		t.Errorf("%s exceeded language-logit bounds: max absolute difference %g (limit %g), mean absolute difference %g, RMSE %g (limit %g)", name, stats.maxDiff, gemma4LanguageLogitMaxDiff, stats.meanAbsDiff, stats.rmse, gemma4LanguageLogitRMSE)
		return
	}
	t.Logf("%s matched language-logit bounds across %d values; max absolute difference %g, mean absolute difference %g, RMSE %g", name, stats.count, stats.maxDiff, stats.meanAbsDiff, stats.rmse)
}

func observeAudioReference(t *testing.T, name string, got, want *mlx.Array, atol, rtol float64) {
	t.Helper()
	stats := measureAudioReference(t, name, got, want, atol, rtol)
	if stats.firstMismatch >= 0 {
		t.Logf("%s diagnostic exceeded tolerance first at %d (diff %g, tolerance %g); max absolute difference %g, mean absolute difference %g, RMSE %g", name, stats.firstMismatch, stats.firstDiff, stats.firstTolerance, stats.maxDiff, stats.meanAbsDiff, stats.rmse)
		return
	}
	t.Logf("%s diagnostic matched %d values; max absolute difference %g, mean absolute difference %g, RMSE %g", name, stats.count, stats.maxDiff, stats.meanAbsDiff, stats.rmse)
}

type audioReferenceStats struct {
	count                      int
	firstMismatch              int
	gotFirst, wantFirst        float32
	firstDiff, firstTolerance  float64
	maxDiff, meanAbsDiff, rmse float64
}

func measureAudioReference(t *testing.T, name string, got, want *mlx.Array, atol, rtol float64) audioReferenceStats {
	t.Helper()
	if got == nil || want == nil {
		t.Fatalf("%s tensor is missing", name)
	}
	if !equalIntShape(got.Dims(), want.Dims()) {
		t.Fatalf("%s shape = %v, want %v", name, got.Dims(), want.Dims())
	}
	got = got.AsType(mlx.DTypeFloat32)
	want = want.AsType(mlx.DTypeFloat32)
	mlx.Eval(got, want)
	gotValues, wantValues := got.Floats(), want.Floats()
	stats := audioReferenceStats{count: len(wantValues), firstMismatch: -1}
	var sumAbs, sumSquared float64
	for i := range wantValues {
		gotValue, wantValue := float64(gotValues[i]), float64(wantValues[i])
		if math.IsNaN(gotValue) || math.IsInf(gotValue, 0) || math.IsNaN(wantValue) || math.IsInf(wantValue, 0) {
			t.Fatalf("%s[%d] contains a non-finite value: got %g, want %g", name, i, gotValue, wantValue)
		}
		diff := math.Abs(gotValue - wantValue)
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			t.Fatalf("%s[%d] produced a non-finite difference: got %g, want %g", name, i, gotValue, wantValue)
		}
		if diff > stats.maxDiff {
			stats.maxDiff = diff
		}
		sumAbs += diff
		sumSquared += diff * diff
		tolerance := atol + rtol*math.Abs(float64(wantValues[i]))
		if diff > tolerance && stats.firstMismatch < 0 {
			stats.firstMismatch = i
			stats.gotFirst = gotValues[i]
			stats.wantFirst = wantValues[i]
			stats.firstDiff = diff
			stats.firstTolerance = tolerance
		}
	}
	if stats.count > 0 {
		stats.meanAbsDiff = sumAbs / float64(stats.count)
		stats.rmse = math.Sqrt(sumSquared / float64(stats.count))
		if math.IsNaN(stats.meanAbsDiff) || math.IsInf(stats.meanAbsDiff, 0) || math.IsNaN(stats.rmse) || math.IsInf(stats.rmse, 0) {
			t.Fatalf("%s produced non-finite aggregate statistics: mean absolute difference %g, RMSE %g", name, stats.meanAbsDiff, stats.rmse)
		}
	}
	return stats
}
