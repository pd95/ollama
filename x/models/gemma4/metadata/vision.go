// Package metadata contains Gemma 4 model metadata checks that do not depend
// on MLX, allowing import, server, and runtime paths to share one definition.
package metadata

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

const (
	defaultVisionLayers   = 16
	MaxVisionLayers       = 512
	MaxVisionHidden       = 32_768
	MaxVisionIntermediate = 262_144
	MaxVisionHeads        = 512
	MaxVisionHeadDim      = 1_024
	MaxVisionSoftTokens   = 16_384
	MaxImageDimension     = 16_384
	MaxResizePixels       = 1 << 20
	MaxPositionEntries    = 1 << 20
	MaxPositionValues     = 1 << 26
)

type ConfigFile struct {
	Architectures      []string      `json:"architectures"`
	ModelType          string        `json:"model_type"`
	TextConfig         TextConfig    `json:"text_config"`
	VisionConfig       *VisionConfig `json:"vision_config"`
	AudioConfig        *AudioConfig  `json:"audio_config"`
	Quantization       Quantization  `json:"quantization"`
	QuantizationConfig Quantization  `json:"quantization_config"`
}

type TextConfig struct {
	HiddenSize int `json:"hidden_size"`
}

type Quantization struct {
	Bits      int    `json:"bits"`
	GroupSize int    `json:"group_size"`
	Mode      string `json:"mode"`
	Method    string `json:"quant_method"`
}

type VisionConfig struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	HeadDim               int     `json:"head_dim"`
	RMSNormEps            float64 `json:"rms_norm_eps"`
	DefaultOutputLength   int     `json:"default_output_length"`
	PatchSize             int     `json:"patch_size"`
	PositionEmbeddingSize int     `json:"position_embedding_size"`
	PoolingKernelSize     int     `json:"pooling_kernel_size"`
	UseClippedLinears     bool    `json:"use_clipped_linears"`
	Standardize           bool    `json:"standardize"`
	MMEmbedDim            int     `json:"mm_embed_dim"`
	MMPosembSize          int     `json:"mm_posemb_size"`
	ModelPatchSize        int     `json:"model_patch_size"`
	NumSoftTokens         int     `json:"num_soft_tokens"`
	OutputProjDims        int     `json:"output_proj_dims"`
	RopeParameters        struct {
		RopeTheta float64 `json:"rope_theta"`
	} `json:"rope_parameters"`
}

type Architecture struct {
	Unified                bool
	TextHiddenSize         int
	HiddenSize             int
	IntermediateSize       int
	NumHiddenLayers        int
	NumAttentionHeads      int
	NumKeyValueHeads       int
	HeadDim                int
	RMSNormEps             float64
	DefaultOutputLength    int
	PatchSize              int
	PositionEmbeddingSize  int
	PoolingKernelSize      int
	UseClippedLinears      bool
	ClippingBoundsOptional bool
	Standardize            bool
	RopeTheta              float64
	MMEmbedDim             int
	MMPosembSize           int
	ModelPatchSize         int
}

type TensorDescriptor struct {
	Dtype     string
	Shape     []int32
	QuantType string
	GroupSize int
}

func ProjectVisionArchitecture(cfg ConfigFile) (Architecture, error) {
	return projectVisionArchitecture(cfg, true)
}

func projectVisionArchitecture(cfg ConfigFile, requireTextWidth bool) (Architecture, error) {
	if cfg.VisionConfig == nil {
		return Architecture{}, fmt.Errorf("missing vision_config")
	}
	v := *cfg.VisionConfig
	if isUnifiedConfig(cfg) {
		if v.NumSoftTokens == 0 {
			v.NumSoftTokens = 280
		}
		if v.PatchSize == 0 {
			v.PatchSize = 16
		}
		if v.PoolingKernelSize == 0 {
			v.PoolingKernelSize = 3
		}
		if v.ModelPatchSize == 0 {
			v.ModelPatchSize = v.PatchSize * v.PoolingKernelSize
		}
		if v.MMEmbedDim == 0 {
			v.MMEmbedDim = v.OutputProjDims
		}
		if v.MMPosembSize == 0 {
			v.MMPosembSize = 1120
		}
		a := Architecture{
			Unified: true, TextHiddenSize: cfg.TextConfig.HiddenSize,
			DefaultOutputLength: v.NumSoftTokens, PatchSize: v.PatchSize,
			PoolingKernelSize: v.PoolingKernelSize, MMEmbedDim: v.MMEmbedDim,
			MMPosembSize: v.MMPosembSize, ModelPatchSize: v.ModelPatchSize, RMSNormEps: 1e-6,
		}
		if err := validateUnifiedArchitecture(a, requireTextWidth); err != nil {
			return Architecture{}, err
		}
		return a, nil
	}
	if v.HiddenSize == 0 {
		v.HiddenSize = 768
	}
	if v.IntermediateSize == 0 {
		v.IntermediateSize = 3072
	}
	if v.NumHiddenLayers == 0 {
		v.NumHiddenLayers = defaultVisionLayers
	}
	if v.NumAttentionHeads == 0 {
		v.NumAttentionHeads = 12
	}
	if v.NumKeyValueHeads == 0 {
		v.NumKeyValueHeads = v.NumAttentionHeads
	}
	if v.HeadDim == 0 {
		v.HeadDim = 64
	}
	if v.RMSNormEps == 0 {
		v.RMSNormEps = 1e-6
	}
	if v.DefaultOutputLength == 0 {
		v.DefaultOutputLength = 280
	}
	if v.PatchSize == 0 {
		v.PatchSize = 16
	}
	if v.PositionEmbeddingSize == 0 {
		v.PositionEmbeddingSize = 10240
	}
	if v.PoolingKernelSize == 0 {
		v.PoolingKernelSize = 3
	}
	if v.RopeParameters.RopeTheta == 0 {
		v.RopeParameters.RopeTheta = 100
	}

	a := Architecture{
		TextHiddenSize: cfg.TextConfig.HiddenSize,
		HiddenSize:     v.HiddenSize, IntermediateSize: v.IntermediateSize,
		NumHiddenLayers: v.NumHiddenLayers, NumAttentionHeads: v.NumAttentionHeads,
		NumKeyValueHeads: v.NumKeyValueHeads, HeadDim: v.HeadDim,
		RMSNormEps: v.RMSNormEps, DefaultOutputLength: v.DefaultOutputLength,
		PatchSize: v.PatchSize, PositionEmbeddingSize: v.PositionEmbeddingSize,
		PoolingKernelSize: v.PoolingKernelSize, UseClippedLinears: v.UseClippedLinears,
		// Published Gemma 4 towers may declare clippable linears while
		// omitting every bound. Mixed/partial bound sets are never permitted.
		ClippingBoundsOptional: true,
		Standardize:            v.Standardize, RopeTheta: v.RopeParameters.RopeTheta,
	}
	if err := validateArchitecture(a, requireTextWidth); err != nil {
		return Architecture{}, err
	}
	return a, nil
}

func isUnifiedConfig(cfg ConfigFile) bool {
	if strings.EqualFold(cfg.ModelType, "gemma4_unified") || strings.EqualFold(cfg.VisionConfig.ModelType, "gemma4_unified_vision") {
		return true
	}
	for _, architecture := range cfg.Architectures {
		switch strings.ToLower(architecture) {
		case "gemma4unifiedforcausallm", "gemma4unifiedforconditionalgeneration":
			return true
		}
	}
	return false
}

func validateUnifiedArchitecture(a Architecture, requireTextWidth bool) error {
	if a.TextHiddenSize < 0 || a.TextHiddenSize > MaxVisionHidden || (requireTextWidth && a.TextHiddenSize == 0) {
		return fmt.Errorf("invalid Gemma4 text hidden_size %d", a.TextHiddenSize)
	}
	for _, f := range []struct {
		name       string
		value, max int
	}{
		{"mm_embed_dim", a.MMEmbedDim, MaxVisionHidden},
		{"mm_posemb_size", a.MMPosembSize, MaxPositionEntries},
		{"model_patch_size", a.ModelPatchSize, MaxImageDimension},
		{"num_soft_tokens", a.DefaultOutputLength, MaxVisionSoftTokens},
		{"patch_size", a.PatchSize, MaxImageDimension},
		{"pooling_kernel_size", a.PoolingKernelSize, MaxImageDimension},
	} {
		if f.value <= 0 || f.value > f.max {
			return fmt.Errorf("invalid Gemma4 unified vision %s %d (limit %d)", f.name, f.value, f.max)
		}
	}
	expectedPatch, ok := checkedProduct(MaxImageDimension, int64(a.PatchSize), int64(a.PoolingKernelSize))
	if !ok || expectedPatch != int64(a.ModelPatchSize) {
		return fmt.Errorf("unified model_patch_size %d does not match patch_size * pooling_kernel_size (%d)", a.ModelPatchSize, expectedPatch)
	}
	if _, ok := checkedProduct(math.MaxInt32, 3, int64(a.ModelPatchSize), int64(a.ModelPatchSize)); !ok {
		return fmt.Errorf("invalid Gemma4 unified patch dimension")
	}
	if _, ok := checkedProduct(3*MaxResizePixels, int64(a.DefaultOutputLength), int64(a.ModelPatchSize), int64(a.ModelPatchSize), 3); !ok {
		return fmt.Errorf("Gemma4 unified patch allocation exceeds %d values", 3*MaxResizePixels)
	}
	if _, ok := checkedProduct(MaxPositionValues, int64(a.DefaultOutputLength), 2); !ok {
		return fmt.Errorf("Gemma4 unified position allocation exceeds %d values", MaxPositionValues)
	}
	if a.MMPosembSize < a.DefaultOutputLength {
		return fmt.Errorf("Gemma4 unified position table %d is smaller than token count %d", a.MMPosembSize, a.DefaultOutputLength)
	}
	return nil
}

func validateArchitecture(a Architecture, requireTextWidth bool) error {
	if a.TextHiddenSize < 0 || a.TextHiddenSize > MaxVisionHidden || (requireTextWidth && a.TextHiddenSize == 0) {
		return fmt.Errorf("invalid Gemma4 text hidden_size %d", a.TextHiddenSize)
	}
	for _, f := range []struct {
		name       string
		value, max int
	}{
		{"hidden_size", a.HiddenSize, MaxVisionHidden},
		{"intermediate_size", a.IntermediateSize, MaxVisionIntermediate},
		{"num_hidden_layers", a.NumHiddenLayers, MaxVisionLayers},
		{"num_attention_heads", a.NumAttentionHeads, MaxVisionHeads},
		{"num_key_value_heads", a.NumKeyValueHeads, MaxVisionHeads},
		{"head_dim", a.HeadDim, MaxVisionHeadDim},
		{"default_output_length", a.DefaultOutputLength, MaxVisionSoftTokens},
		{"patch_size", a.PatchSize, MaxImageDimension},
		{"position_embedding_size", a.PositionEmbeddingSize, MaxPositionEntries},
		{"pooling_kernel_size", a.PoolingKernelSize, MaxImageDimension},
	} {
		if f.value <= 0 || f.value > f.max {
			return fmt.Errorf("invalid Gemma4 vision %s %d (limit %d)", f.name, f.value, f.max)
		}
	}
	if a.NumKeyValueHeads > a.NumAttentionHeads || a.NumAttentionHeads%a.NumKeyValueHeads != 0 {
		return fmt.Errorf("invalid Gemma4 vision attention heads %d/%d", a.NumAttentionHeads, a.NumKeyValueHeads)
	}
	if p, ok := checkedProduct(int64(MaxVisionHidden), int64(a.NumAttentionHeads), int64(a.HeadDim)); !ok || p != int64(a.HiddenSize) {
		return fmt.Errorf("Gemma4 vision attention width does not match hidden_size %d", a.HiddenSize)
	}
	if !finitePositive(a.RMSNormEps) {
		return fmt.Errorf("invalid Gemma4 vision rms_norm_eps %v", a.RMSNormEps)
	}
	if !finitePositive(float64(float32(a.RMSNormEps))) {
		return fmt.Errorf("invalid Gemma4 vision rms_norm_eps %v after float32 projection", a.RMSNormEps)
	}
	if !finitePositive(a.RopeTheta) {
		return fmt.Errorf("invalid Gemma4 vision rope_theta %v", a.RopeTheta)
	}
	if !finitePositive(float64(float32(a.RopeTheta))) {
		return fmt.Errorf("invalid Gemma4 vision rope_theta %v after float32 projection", a.RopeTheta)
	}
	if _, ok := checkedProduct(MaxResizePixels, int64(a.DefaultOutputLength), int64(a.PoolingKernelSize), int64(a.PoolingKernelSize), int64(a.PatchSize), int64(a.PatchSize)); !ok {
		return fmt.Errorf("Gemma4 vision resize budget exceeds limit %d", MaxResizePixels)
	}
	if _, ok := checkedProduct(MaxPositionValues, int64(a.DefaultOutputLength), int64(a.PoolingKernelSize), int64(a.PoolingKernelSize), int64(a.HeadDim)); !ok {
		return fmt.Errorf("Gemma4 vision position allocation exceeds limit %d", MaxPositionValues)
	}
	patchSide, ok := checkedProduct(MaxPositionEntries, int64(a.DefaultOutputLength), int64(a.PoolingKernelSize))
	if !ok || patchSide > int64(a.PositionEmbeddingSize) {
		return fmt.Errorf("Gemma4 vision patch side %d exceeds position table size %d", patchSide, a.PositionEmbeddingSize)
	}
	return nil
}

func finitePositive(v float64) bool { return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0) }
func checkedProduct(limit int64, values ...int64) (int64, bool) {
	if limit <= 0 {
		return 0, false
	}
	p := int64(1)
	for _, v := range values {
		if v <= 0 || p > limit/v {
			return 0, false
		}
		p *= v
	}
	return p, true
}

// ValidateVisionTensors is the non-executable names-only normalized gate for
// callers that cannot inspect blobs or know the enclosing text width.
// Capability surfaces use the strict descriptor variants below.
func ValidateVisionTensors(cfg ConfigFile, names []string) error {
	a, err := projectVisionArchitecture(cfg, false)
	if err != nil {
		return err
	}
	return validateNames(a, names, installedMode)
}

func ValidateVisionSourceTensors(cfg ConfigFile, names []string) error {
	a, err := projectVisionArchitecture(cfg, false)
	if err != nil {
		return err
	}
	return validateNames(a, names, sourceMode)
}

func ValidateVisionSourceInventory(cfg ConfigFile, tensors map[string]TensorDescriptor) error {
	return validateInventory(cfg, tensors, sourceMode)
}

func ValidateVisionInstalledInventory(cfg ConfigFile, tensors map[string]TensorDescriptor) error {
	return validateInventory(cfg, tensors, installedMode)
}

func ValidateVisionRuntimeInventory(cfg ConfigFile, tensors map[string]TensorDescriptor) error {
	return validateInventory(cfg, tensors, runtimeMode)
}

type inventoryMode int

const (
	sourceMode inventoryMode = iota
	installedMode
	runtimeMode
)

func validateInventory(cfg ConfigFile, tensors map[string]TensorDescriptor, mode inventoryMode) error {
	a, err := ProjectVisionArchitecture(cfg)
	if err != nil {
		return err
	}
	if a.TextHiddenSize <= 0 {
		return fmt.Errorf("missing Gemma4 text hidden_size")
	}
	if a.Unified {
		return validateUnifiedInventory(cfg, a, tensors, mode)
	}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	if err := validateNames(a, names, mode); err != nil {
		return err
	}
	prefix, _ := visionPrefix(names, mode)
	patchDim, _ := checkedProduct(math.MaxInt32, 3, int64(a.PatchSize), int64(a.PatchSize))
	if err := requireLinearDescriptor(tensors, prefix+"vision_tower.patch_embedder.input_proj", []int32{int32(a.HiddenSize), int32(patchDim)}, mode, cfg); err != nil {
		return err
	}
	if err := requireDense(tensors, prefix+"vision_tower.patch_embedder.position_embedding_table", []int32{2, int32(a.PositionEmbeddingSize), int32(a.HiddenSize)}); err != nil {
		return err
	}
	for i := range a.NumHiddenLayers {
		layer := fmt.Sprintf("%svision_tower.encoder.layers.%d", prefix, i)
		for _, p := range []struct {
			suffix string
			shape  []int32
		}{
			{".self_attn.q_proj", []int32{int32(a.HiddenSize), int32(a.HiddenSize)}},
			{".self_attn.k_proj", []int32{int32(a.NumKeyValueHeads * a.HeadDim), int32(a.HiddenSize)}},
			{".self_attn.v_proj", []int32{int32(a.NumKeyValueHeads * a.HeadDim), int32(a.HiddenSize)}},
			{".self_attn.o_proj", []int32{int32(a.HiddenSize), int32(a.HiddenSize)}},
			{".mlp.gate_proj", []int32{int32(a.IntermediateSize), int32(a.HiddenSize)}},
			{".mlp.up_proj", []int32{int32(a.IntermediateSize), int32(a.HiddenSize)}},
			{".mlp.down_proj", []int32{int32(a.HiddenSize), int32(a.IntermediateSize)}},
		} {
			if err := requireLinearDescriptor(tensors, layer+p.suffix, p.shape, mode, cfg); err != nil {
				return err
			}
			if err := validateClipDescriptors(tensors, layer+p.suffix, a.UseClippedLinears, a.ClippingBoundsOptional); err != nil {
				return err
			}
		}
		for _, suffix := range []string{".self_attn.q_norm.weight", ".self_attn.k_norm.weight"} {
			if err := requireDense(tensors, layer+suffix, []int32{int32(a.HeadDim)}); err != nil {
				return err
			}
		}
		for _, suffix := range []string{".input_layernorm.weight", ".post_attention_layernorm.weight", ".pre_feedforward_layernorm.weight", ".post_feedforward_layernorm.weight"} {
			if err := requireDense(tensors, layer+suffix, []int32{int32(a.HiddenSize)}); err != nil {
				return err
			}
		}
	}
	if a.Standardize {
		if err := requireDense(tensors, prefix+"vision_tower.std_bias", []int32{int32(a.HiddenSize)}); err != nil {
			return err
		}
		if err := requireDense(tensors, prefix+"vision_tower.std_scale", []int32{int32(a.HiddenSize)}); err != nil {
			return err
		}
	}
	for _, p := range []string{"embed_vision.embedding_projection", "model.embed_vision.embedding_projection"} {
		if linearPresent(tensors, p, mode) {
			return requireLinearDescriptor(tensors, p, []int32{int32(a.TextHiddenSize), int32(a.HiddenSize)}, mode, cfg)
		}
	}
	return fmt.Errorf("missing embed_vision.embedding_projection weight")
}

func validateNames(a Architecture, names []string, mode inventoryMode) error {
	if a.Unified {
		return validateUnifiedNames(names, mode)
	}
	prefix, ok := visionPrefix(names, mode)
	if !ok {
		return fmt.Errorf("missing vision_tower.patch_embedder.input_proj weight")
	}
	require := func(name string) error {
		if !slices.Contains(names, name) {
			return fmt.Errorf("missing %s", name)
		}
		return nil
	}
	if err := require(prefix + "vision_tower.patch_embedder.position_embedding_table"); err != nil {
		return err
	}
	for i := range a.NumHiddenLayers {
		layer := fmt.Sprintf("%svision_tower.encoder.layers.%d", prefix, i)
		for _, p := range []string{".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj", ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj"} {
			if !linearNamePresent(names, layer+p, mode) {
				return fmt.Errorf("missing %s weight", layer+p)
			}
			if err := validateClipNames(names, layer+p, a.UseClippedLinears, a.ClippingBoundsOptional); err != nil {
				return err
			}
		}
		for _, n := range []string{".self_attn.q_norm.weight", ".self_attn.k_norm.weight", ".input_layernorm.weight", ".post_attention_layernorm.weight", ".pre_feedforward_layernorm.weight", ".post_feedforward_layernorm.weight"} {
			if err := require(layer + n); err != nil {
				return err
			}
		}
	}
	if a.Standardize {
		if err := require(prefix + "vision_tower.std_bias"); err != nil {
			return err
		}
		if err := require(prefix + "vision_tower.std_scale"); err != nil {
			return err
		}
	}
	for _, p := range []string{"embed_vision.embedding_projection", "model.embed_vision.embedding_projection"} {
		if linearNamePresent(names, p, mode) {
			return nil
		}
	}
	return fmt.Errorf("missing embed_vision.embedding_projection weight")
}

func validateUnifiedNames(names []string, mode inventoryMode) error {
	for _, name := range []string{
		"model.vision_embedder.patch_ln1.weight", "model.vision_embedder.patch_ln1.bias",
		"model.vision_embedder.patch_dense.bias", "model.vision_embedder.patch_ln2.weight",
		"model.vision_embedder.patch_ln2.bias", "model.vision_embedder.pos_embedding",
		"model.vision_embedder.pos_norm.weight", "model.vision_embedder.pos_norm.bias",
	} {
		if !slices.Contains(names, name) {
			return fmt.Errorf("missing %s", name)
		}
	}
	for _, path := range []string{"model.vision_embedder.patch_dense", "model.embed_vision.embedding_projection"} {
		if !linearNamePresent(names, path, mode) {
			return fmt.Errorf("missing %s weight", path)
		}
	}
	return nil
}

func validateUnifiedInventory(cfg ConfigFile, a Architecture, tensors map[string]TensorDescriptor, mode inventoryMode) error {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	if err := validateUnifiedNames(names, mode); err != nil {
		return err
	}
	patchDim, _ := checkedProduct(math.MaxInt32, 3, int64(a.ModelPatchSize), int64(a.ModelPatchSize))
	for name, shape := range map[string][]int32{
		"model.vision_embedder.patch_ln1.weight": {int32(patchDim)},
		"model.vision_embedder.patch_ln1.bias":   {int32(patchDim)},
		"model.vision_embedder.patch_dense.bias": {int32(a.MMEmbedDim)},
		"model.vision_embedder.patch_ln2.weight": {int32(a.MMEmbedDim)},
		"model.vision_embedder.patch_ln2.bias":   {int32(a.MMEmbedDim)},
		"model.vision_embedder.pos_embedding":    {int32(a.MMPosembSize), 2, int32(a.MMEmbedDim)},
		"model.vision_embedder.pos_norm.weight":  {int32(a.MMEmbedDim)},
		"model.vision_embedder.pos_norm.bias":    {int32(a.MMEmbedDim)},
	} {
		if err := requireDense(tensors, name, shape); err != nil {
			return err
		}
	}
	if err := requireLinearDescriptor(tensors, "model.vision_embedder.patch_dense", []int32{int32(a.MMEmbedDim), int32(patchDim)}, mode, cfg); err != nil {
		return err
	}
	return requireLinearDescriptor(tensors, "model.embed_vision.embedding_projection", []int32{int32(a.TextHiddenSize), int32(a.MMEmbedDim)}, mode, cfg)
}

func visionPrefix(names []string, mode inventoryMode) (string, bool) {
	for _, p := range []string{"", "model."} {
		if linearNamePresent(names, p+"vision_tower.patch_embedder.input_proj", mode) {
			return p, true
		}
	}
	return "", false
}

func linearNamePresent(names []string, path string, mode inventoryMode) bool {
	for _, base := range []string{path, path + ".linear"} {
		if slices.Contains(names, base+".weight") {
			return true
		}
		if mode == sourceMode && slices.Contains(names, base+".weight_packed") && slices.Contains(names, base+".weight_scale") && slices.Contains(names, base+".weight_global_scale") {
			return true
		}
	}
	return false
}

func linearPresent(t map[string]TensorDescriptor, p string, mode inventoryMode) bool {
	names := make([]string, 0, len(t))
	for n := range t {
		names = append(names, n)
	}
	return linearNamePresent(names, p, mode)
}

func requireDense(t map[string]TensorDescriptor, name string, shape []int32) error {
	d, ok := t[name]
	if !ok {
		return fmt.Errorf("missing %s", name)
	}
	if !isFloat(d.Dtype) {
		return fmt.Errorf("%s dtype %s is not floating point", name, d.Dtype)
	}
	if !slices.Equal(d.Shape, shape) {
		return fmt.Errorf("%s shape %v, want %v", name, d.Shape, shape)
	}
	return nil
}

func requireLinearDescriptor(t map[string]TensorDescriptor, path string, logical []int32, mode inventoryMode, cfg ConfigFile) error {
	for _, base := range []string{path, path + ".linear"} {
		if d, ok := t[base+".weight"]; ok {
			if isFloat(d.Dtype) {
				if hasProducerCompanion(t, base) {
					return fmt.Errorf("cross-producer companions for dense %s.weight", base)
				}
				if !slices.Equal(d.Shape, logical) {
					return fmt.Errorf("%s.weight shape %v, want %v", base, d.Shape, logical)
				}
				return nil
			}
			if mode == sourceMode {
				return validatePackedSource(t, base, d, logical, cfg, false)
			}
			return validateNormalizedQuant(t, base, d, logical, mode, cfg)
		}
		if mode == sourceMode {
			if d, ok := t[base+".weight_packed"]; ok {
				return validatePackedSource(t, base, d, logical, cfg, true)
			}
		}
	}
	return fmt.Errorf("missing %s weight", path)
}

func validatePackedSource(t map[string]TensorDescriptor, base string, weight TensorDescriptor, logical []int32, cfg ConfigFile, compressed bool) error {
	if compressed {
		if !strings.EqualFold(weight.Dtype, "U8") || !packedWeightShape(weight.Shape, logical, 2) {
			return fmt.Errorf("invalid compressed NVFP4 weight %s", base)
		}
		scale, ok := t[base+".weight_scale"]
		if !ok {
			return fmt.Errorf("incomplete packed weight %s.weight_packed", base)
		}
		global, ok := t[base+".weight_global_scale"]
		if !ok {
			return fmt.Errorf("incomplete packed weight %s.weight_packed", base)
		}
		if !packedScaleShape(scale, logical, 16) || !isE4M3(scale.Dtype) {
			return fmt.Errorf("invalid packed scale %s.weight_scale", base)
		}
		if !isScalar(global.Shape) || !strings.EqualFold(global.Dtype, "F32") {
			return fmt.Errorf("invalid packed global scale %s.weight_global_scale", base)
		}
		for _, suffix := range []string{".scales", ".biases", ".weight_scale_2"} {
			if _, cross := t[base+suffix]; cross {
				return fmt.Errorf("cross-producer packed companion %s%s", base, suffix)
			}
		}
		return nil
	}
	if scale, ok := t[base+".scales"]; ok {
		for _, suffix := range []string{".weight_scale", ".weight_global_scale", ".weight_scale_2"} {
			if _, cross := t[base+suffix]; cross {
				return fmt.Errorf("cross-producer packed companion %s%s", base, suffix)
			}
		}
		q := sourceQuant(cfg)
		if q.Bits <= 0 || q.GroupSize <= 0 {
			return fmt.Errorf("missing MLX packed quantization contract for %s", base)
		}
		if !strings.EqualFold(weight.Dtype, "U32") || !packedWeightShape(weight.Shape, logical, int32(32/q.Bits)) || !packedScaleShape(scale, logical, int32(q.GroupSize)) || !isScaleDtype(scale.Dtype) {
			return fmt.Errorf("invalid MLX packed descriptors for %s", base)
		}
		if bias, present := t[base+".biases"]; present && (!isFloat(bias.Dtype) || !slices.Equal(bias.Shape, scale.Shape)) {
			return fmt.Errorf("invalid MLX packed bias for %s", base)
		}
		return nil
	}
	if scale, ok := t[base+".weight_scale"]; ok {
		for _, suffix := range []string{".scales", ".biases", ".weight_global_scale"} {
			if _, cross := t[base+suffix]; cross {
				return fmt.Errorf("cross-producer packed companion %s%s", base, suffix)
			}
		}
		if !strings.EqualFold(weight.Dtype, "U8") || !packedWeightShape(weight.Shape, logical, 2) || !packedScaleShape(scale, logical, 16) || !isE4M3(scale.Dtype) {
			return fmt.Errorf("invalid ModelOpt NVFP4 descriptors for %s", base)
		}
		if g, ok := t[base+".weight_scale_2"]; ok && (!isScalar(g.Shape) || !strings.EqualFold(g.Dtype, "F32")) {
			return fmt.Errorf("invalid ModelOpt global scale %s", base)
		}
		return nil
	}
	return fmt.Errorf("packed source weight %s has no recognized companions", base)
}

func validateNormalizedQuant(t map[string]TensorDescriptor, base string, weight TensorDescriptor, logical []int32, mode inventoryMode, cfg ConfigFile) error {
	for _, suffix := range []string{".scales", ".weight_packed", ".weight_global_scale", ".weight_scale_2"} {
		if _, exists := t[base+suffix]; exists {
			return fmt.Errorf("source-only packed companion %s%s in normalized inventory", base, suffix)
		}
	}
	if mode == installedMode {
		if _, exists := t[base+".weight_scale"]; exists {
			return fmt.Errorf("source-only packed companion %s.weight_scale in installed inventory", base)
		}
	}
	scaleName := base + ".weight.scale"
	biasName := base + ".weight.bias"
	if mode == runtimeMode {
		scaleName = base + ".weight_scale"
		biasName = base + ".weight_qbias"
	}
	scale, ok := t[scaleName]
	if !ok {
		return fmt.Errorf("missing or invalid normalized scale %s", scaleName)
	}
	if strings.EqualFold(scale.Dtype, "U8") {
		if !strings.EqualFold(weight.Dtype, "U32") || !packedWeightShape(weight.Shape, logical, 8) || !packedScaleShape(scale, logical, 16) {
			return fmt.Errorf("invalid normalized NVFP4 descriptors for %s", base)
		}
		if global, present := t[base+".weight.global_scale"]; present && (!isScalar(global.Shape) || !strings.EqualFold(global.Dtype, "F32")) {
			return fmt.Errorf("invalid normalized global scale for %s", base)
		}
		if _, present := t[biasName]; present {
			return fmt.Errorf("unexpected affine bias for normalized NVFP4 %s", base)
		}
		return nil
	}
	bits, groupSize, ok := inferAffineContract(weight, scale, logical)
	q := sourceQuant(cfg)
	if q.Bits > 0 && q.Bits != bits {
		ok = false
	}
	if q.GroupSize > 0 && q.GroupSize != groupSize {
		ok = false
	}
	if weight.GroupSize > 0 && weight.GroupSize != groupSize {
		ok = false
	}
	if !ok || !isFloat(scale.Dtype) || !strings.EqualFold(weight.Dtype, "U32") {
		return fmt.Errorf("invalid normalized affine descriptors for %s", base)
	}
	if bias, present := t[biasName]; present && (!isFloat(bias.Dtype) || !slices.Equal(bias.Shape, scale.Shape)) {
		return fmt.Errorf("invalid normalized affine bias for %s", base)
	}
	if _, present := t[base+".weight.global_scale"]; present {
		return fmt.Errorf("unexpected global scale for normalized affine %s", base)
	}
	return nil
}

func inferAffineContract(weight, scale TensorDescriptor, logical []int32) (bits, groupSize int, ok bool) {
	if len(weight.Shape) != 2 || len(scale.Shape) != 2 || len(logical) != 2 || weight.Shape[0] != logical[0] || scale.Shape[0] != logical[0] || weight.Shape[1] <= 0 || scale.Shape[1] <= 0 {
		return 0, 0, false
	}
	if logical[1]%weight.Shape[1] != 0 || logical[1]%scale.Shape[1] != 0 {
		return 0, 0, false
	}
	perWord := logical[1] / weight.Shape[1]
	if perWord <= 0 || 32%perWord != 0 {
		return 0, 0, false
	}
	bits = int(32 / perWord)
	groupSize = int(logical[1] / scale.Shape[1])
	if (bits != 4 && bits != 8) || groupSize <= 0 {
		return 0, 0, false
	}
	return bits, groupSize, true
}

func sourceQuant(cfg ConfigFile) Quantization {
	if cfg.QuantizationConfig.Bits != 0 {
		return cfg.QuantizationConfig
	}
	return cfg.Quantization
}

func hasProducerCompanion(t map[string]TensorDescriptor, b string) bool {
	for _, s := range []string{".scales", ".weight_scale", ".weight_global_scale", ".weight_scale_2"} {
		if _, ok := t[b+s]; ok {
			return true
		}
	}
	return false
}

func packedWeightShape(got, logical []int32, perWord int32) bool {
	return len(got) == 2 && len(logical) == 2 && got[0] == logical[0] && logical[1]%perWord == 0 && got[1] == logical[1]/perWord
}

func packedScaleShape(d TensorDescriptor, logical []int32, group int32) bool {
	return len(d.Shape) == 2 && len(logical) == 2 && d.Shape[0] == logical[0] && logical[1]%group == 0 && d.Shape[1] == logical[1]/group
}

func isScaleDtype(d string) bool {
	switch strings.ToUpper(d) {
	case "U8", "F8_E4M3", "F8_E4M3FN", "BF16", "F16", "F32":
		return true
	}
	return false
}

func isE4M3(d string) bool {
	switch strings.ToUpper(d) {
	case "F8_E4M3", "F8_E4M3FN":
		return true
	}
	return false
}

func isFloat(d string) bool {
	switch strings.ToUpper(d) {
	case "BF16", "F16", "F32", "BFLOAT16", "FLOAT16", "FLOAT32":
		return true
	}
	return false
}

func isScalar(s []int32) bool { return len(s) == 0 || (len(s) == 1 && s[0] == 1) }

func validateClipNames(names []string, path string, enabled, optional bool) error {
	present := 0
	for _, s := range []string{".input_min", ".input_max", ".output_min", ".output_max"} {
		if slices.Contains(names, path+s) {
			present++
		}
	}
	if present != 0 && present != 4 {
		return fmt.Errorf("incomplete clipping tensors for %s", path)
	}
	if !enabled && present != 0 {
		return fmt.Errorf("unexpected clipping tensors for %s", path)
	}
	if enabled && present == 0 && !optional {
		return fmt.Errorf("missing clipping tensors for %s", path)
	}
	return nil
}

func validateClipDescriptors(t map[string]TensorDescriptor, path string, enabled, optional bool) error {
	names := make([]string, 0, len(t))
	for n := range t {
		names = append(names, n)
	}
	if err := validateClipNames(names, path, enabled, optional); err != nil {
		return err
	}
	for _, s := range []string{".input_min", ".input_max", ".output_min", ".output_max"} {
		if d, ok := t[path+s]; ok {
			if !isFloat(d.Dtype) || !isScalar(d.Shape) {
				return fmt.Errorf("invalid clipping tensor %s%s", path, s)
			}
		}
	}
	return nil
}
