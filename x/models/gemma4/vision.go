package gemma4

// Portions of the Gemma 4 vision preprocessing and embedding flow are adapted
// from MLX-VLM's MIT-licensed Gemma 4 implementation. See
// docs/third-party/mlx-vlm.md for the pinned source revision and license.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	stddraw "image/draw"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"slices"

	xdraw "golang.org/x/image/draw"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	gemma4metadata "github.com/ollama/ollama/x/models/gemma4/metadata"
	"github.com/ollama/ollama/x/models/nn"
)

const (
	defaultGemma4BOIToken   = "<|image>"
	defaultGemma4ImageToken = "<|image|>"
	defaultGemma4EOIToken   = "<image|>"
	defaultGemma4BOAToken   = "<|audio>"
	defaultGemma4AudioToken = "<|audio|>"
	defaultGemma4EOAToken   = "<audio|>"

	maxGemma4ImageBytes           = 32 << 20
	maxGemma4ImageDimension       = 16_384
	maxGemma4ImagePixels          = 64 << 20
	maxGemma4ResizePixels         = 1 << 20
	maxGemma4VisionHiddenSize     = 32_768
	maxGemma4VisionIntermediate   = 262_144
	maxGemma4VisionLayers         = 512
	maxGemma4VisionHeads          = 512
	maxGemma4VisionHeadDim        = 1_024
	maxGemma4VisionSoftTokens     = 16_384
	maxGemma4PositionTableEntries = 1 << 20
	maxGemma4PositionValues       = 1 << 26
	maxIntValue                   = int64(^uint(0) >> 1)
)

type VisionRopeParameters struct {
	RopeTheta float32 `json:"rope_theta"`
	RopeType  string  `json:"rope_type"`
}

type VisionConfig struct {
	Unified               bool                 `json:"-"`
	ModelType             string               `json:"model_type"`
	HiddenSize            int32                `json:"hidden_size"`
	IntermediateSize      int32                `json:"intermediate_size"`
	NumHiddenLayers       int32                `json:"num_hidden_layers"`
	NumAttentionHeads     int32                `json:"num_attention_heads"`
	NumKeyValueHeads      int32                `json:"num_key_value_heads"`
	HeadDim               int32                `json:"head_dim"`
	RMSNormEps            float32              `json:"rms_norm_eps"`
	DefaultOutputLength   int32                `json:"default_output_length"`
	PatchSize             int32                `json:"patch_size"`
	PositionEmbeddingSize int32                `json:"position_embedding_size"`
	PoolingKernelSize     int32                `json:"pooling_kernel_size"`
	UseClippedLinears     bool                 `json:"use_clipped_linears"`
	Standardize           bool                 `json:"standardize"`
	MMEmbedDim            int32                `json:"mm_embed_dim"`
	MMPosembSize          int32                `json:"mm_posemb_size"`
	ModelPatchSize        int32                `json:"model_patch_size"`
	NumSoftTokens         int32                `json:"num_soft_tokens"`
	OutputProjDims        int32                `json:"output_proj_dims"`
	RopeParameters        VisionRopeParameters `json:"rope_parameters"`
}

func (c *VisionConfig) unified() bool {
	return c != nil && c.Unified
}

type gemma4MediaTokens struct {
	BOI   string
	Image string
	EOI   string
	BOA   string
	Audio string
	EOA   string
}

type gemma4ImageInput struct {
	Pixels      []float32
	Width       int
	Height      int
	PatchWidth  int
	PatchHeight int
	SoftTokens  int
	Patches     []float32
	Positions   []int32
}

type gemma4MediaLayout struct {
	ImageSpans [][2]int
}

type gemma4MediaPayload struct {
	Image      *gemma4ImageInput
	Audio      *gemma4AudioInput
	ImageStart int
	ImageEnd   int
	AudioStart int
	AudioEnd   int
}

type ClippableLinear struct {
	Linear    nn.LinearLayer
	InputMin  *mlx.Array
	InputMax  *mlx.Array
	OutputMin *mlx.Array
	OutputMax *mlx.Array
}

type VisionAttention struct {
	QProj *ClippableLinear
	KProj *ClippableLinear
	VProj *ClippableLinear
	OProj *ClippableLinear

	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm
}

type VisionMLP struct {
	GateProj *ClippableLinear
	UpProj   *ClippableLinear
	DownProj *ClippableLinear
}

type VisionLayer struct {
	Attention *VisionAttention
	MLP       *VisionMLP

	InputNorm    *nn.RMSNorm
	PostAttnNorm *nn.RMSNorm
	PreFFNorm    *nn.RMSNorm
	PostFFNorm   *nn.RMSNorm
}

type VisionPatchEmbedder struct {
	InputProj              nn.LinearLayer
	PositionEmbeddingTable *mlx.Array
	PatchSize              int32
	PositionEmbeddingSize  int32
}

type VisionModel struct {
	Config        *VisionConfig
	PatchEmbedder *VisionPatchEmbedder
	Layers        []*VisionLayer
	StdBias       *mlx.Array
	StdScale      *mlx.Array
}

type UnifiedVisionEmbedder struct {
	PatchLN1     *nn.LayerNorm
	PatchDense   nn.LinearLayer
	PatchLN2     *nn.LayerNorm
	PosEmbedding *mlx.Array
	PosNorm      *nn.LayerNorm
	PatchDim     int32
}

type MultimodalEmbedder struct {
	Projection nn.LinearLayer
	Eps        float32
}

type visionPositionArrays struct {
	X       *mlx.Array
	Y       *mlx.Array
	RopeCos *mlx.Array
	RopeSin *mlx.Array
}

func parseVisionConfig(configData []byte) (*VisionConfig, error) {
	var wrapped struct {
		VisionConfig *VisionConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(configData, &wrapped); err != nil {
		return nil, fmt.Errorf("parse vision config: %w", err)
	}
	if wrapped.VisionConfig == nil {
		return nil, nil
	}
	var metadataConfig gemma4metadata.ConfigFile
	if err := json.Unmarshal(configData, &metadataConfig); err != nil {
		return nil, fmt.Errorf("parse vision metadata: %w", err)
	}
	architecture, err := gemma4metadata.ProjectVisionArchitecture(metadataConfig)
	if err != nil {
		return nil, err
	}
	cfg := *wrapped.VisionConfig
	cfg.Unified = architecture.Unified
	cfg.HiddenSize = int32(architecture.HiddenSize)
	cfg.IntermediateSize = int32(architecture.IntermediateSize)
	cfg.NumHiddenLayers = int32(architecture.NumHiddenLayers)
	cfg.NumAttentionHeads = int32(architecture.NumAttentionHeads)
	cfg.NumKeyValueHeads = int32(architecture.NumKeyValueHeads)
	cfg.HeadDim = int32(architecture.HeadDim)
	cfg.RMSNormEps = float32(architecture.RMSNormEps)
	cfg.DefaultOutputLength = int32(architecture.DefaultOutputLength)
	cfg.PatchSize = int32(architecture.PatchSize)
	cfg.PositionEmbeddingSize = int32(architecture.PositionEmbeddingSize)
	cfg.PoolingKernelSize = int32(architecture.PoolingKernelSize)
	cfg.RopeParameters.RopeTheta = float32(architecture.RopeTheta)
	cfg.MMEmbedDim = int32(architecture.MMEmbedDim)
	cfg.MMPosembSize = int32(architecture.MMPosembSize)
	cfg.ModelPatchSize = int32(architecture.ModelPatchSize)
	if architecture.Unified {
		cfg.DefaultOutputLength = int32(architecture.DefaultOutputLength)
		cfg.RMSNormEps = float32(architecture.RMSNormEps)
	}
	return &cfg, nil
}

func validateVisionConfig(cfg *VisionConfig) error {
	if cfg == nil {
		return errors.New("missing Gemma4 vision config")
	}
	// This validates vision-only call sites before the enclosing text model is
	// available. Executable inventory validation supplies the real text width.
	_, err := gemma4metadata.ProjectVisionArchitecture(metadataConfigFromVision(cfg, 1))
	return err
}

func checkedPositiveProduct(limit int64, values ...int64) (int64, bool) {
	if limit <= 0 {
		return 0, false
	}
	product := int64(1)
	for _, value := range values {
		if value <= 0 || product > limit/value {
			return 0, false
		}
		product *= value
	}
	return product, true
}

func defaultGemma4MediaTokens() gemma4MediaTokens {
	return gemma4MediaTokens{
		BOI:   defaultGemma4BOIToken,
		Image: defaultGemma4ImageToken,
		EOI:   defaultGemma4EOIToken,
		BOA:   defaultGemma4BOAToken,
		Audio: defaultGemma4AudioToken,
		EOA:   defaultGemma4EOAToken,
	}
}

func parseGemma4MediaTokens(data []byte, fallback gemma4MediaTokens) gemma4MediaTokens {
	var cfg struct {
		BOIToken   string `json:"boi_token"`
		ImageToken string `json:"image_token"`
		EOIToken   string `json:"eoi_token"`
		BOAToken   string `json:"boa_token"`
		AudioToken string `json:"audio_token"`
		EOAToken   string `json:"eoa_token"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fallback
	}
	if cfg.BOIToken != "" {
		fallback.BOI = cfg.BOIToken
	}
	if cfg.ImageToken != "" {
		fallback.Image = cfg.ImageToken
	}
	if cfg.EOIToken != "" {
		fallback.EOI = cfg.EOIToken
	}
	if cfg.BOAToken != "" {
		fallback.BOA = cfg.BOAToken
	}
	if cfg.AudioToken != "" {
		fallback.Audio = cfg.AudioToken
	}
	if cfg.EOAToken != "" {
		fallback.EOA = cfg.EOAToken
	}
	return fallback
}

func metadataConfigFromVision(cfg *VisionConfig, textHidden int) gemma4metadata.ConfigFile {
	if cfg == nil {
		return gemma4metadata.ConfigFile{}
	}
	v := &gemma4metadata.VisionConfig{
		ModelType:  cfg.ModelType,
		HiddenSize: int(cfg.HiddenSize), IntermediateSize: int(cfg.IntermediateSize), NumHiddenLayers: int(cfg.NumHiddenLayers),
		NumAttentionHeads: int(cfg.NumAttentionHeads), NumKeyValueHeads: int(cfg.NumKeyValueHeads), HeadDim: int(cfg.HeadDim),
		RMSNormEps: float64(cfg.RMSNormEps), DefaultOutputLength: int(cfg.DefaultOutputLength), PatchSize: int(cfg.PatchSize),
		PositionEmbeddingSize: int(cfg.PositionEmbeddingSize), PoolingKernelSize: int(cfg.PoolingKernelSize),
		UseClippedLinears: cfg.UseClippedLinears, Standardize: cfg.Standardize,
		MMEmbedDim: int(cfg.MMEmbedDim), MMPosembSize: int(cfg.MMPosembSize), ModelPatchSize: int(cfg.ModelPatchSize),
		NumSoftTokens: int(cfg.DefaultOutputLength), OutputProjDims: int(cfg.OutputProjDims),
	}
	if cfg.Unified {
		v.ModelType = "gemma4_unified_vision"
	}
	v.RopeParameters.RopeTheta = float64(cfg.RopeParameters.RopeTheta)
	return gemma4metadata.ConfigFile{TextConfig: gemma4metadata.TextConfig{HiddenSize: textHidden}, VisionConfig: v}
}

func validateGemma4VisionWeights(tensors map[string]*mlx.Array, cfg *VisionConfig, textHidden int, tq map[string]*model.TensorQuantInfo) (bool, error) {
	sentinel := firstNonNil(tensors, "vision_tower.patch_embedder.input_proj.weight", "model.vision_tower.patch_embedder.input_proj.weight")
	packedSentinel := firstNonNil(tensors, "vision_tower.patch_embedder.input_proj.weight_packed", "model.vision_tower.patch_embedder.input_proj.weight_packed")
	if cfg != nil && cfg.unified() {
		sentinel = firstNonNil(tensors, "model.vision_embedder.patch_ln1.weight")
		packedSentinel = firstNonNil(tensors, "model.vision_embedder.patch_dense.weight_packed")
	}
	if sentinel == nil {
		if packedSentinel != nil {
			return false, fmt.Errorf("runtime contains source-only Gemma4 vision packed sentinel")
		}
		return false, nil
	}
	descriptors := make(map[string]gemma4metadata.TensorDescriptor, len(tensors))
	for name, tensor := range tensors {
		if tensor != nil {
			shape := make([]int32, tensor.NumDims())
			for i, d := range tensor.Dims() {
				shape[i] = int32(d)
			}
			d := gemma4metadata.TensorDescriptor{Dtype: tensor.DType().String(), Shape: shape}
			if q := tq[name]; q != nil {
				d.QuantType = q.QuantType
				d.GroupSize = q.GroupSize
			}
			descriptors[name] = d
		}
	}
	if err := gemma4metadata.ValidateVisionRuntimeInventory(metadataConfigFromVision(cfg, textHidden), descriptors); err != nil {
		return false, err
	}
	return true, nil
}

func loadUnifiedVisionEmbedder(tensors map[string]*mlx.Array, cfg *VisionConfig, groupSize, bits int, mode string, tq map[string]*model.TensorQuantInfo) (*UnifiedVisionEmbedder, error) {
	const prefix = "model.vision_embedder."
	patchDim, ok := checkedPositiveProduct(math.MaxInt32, int64(cfg.ModelPatchSize), int64(cfg.ModelPatchSize), 3)
	if !ok {
		return nil, errors.New("invalid Gemma4 unified patch dimension")
	}
	linears := model.NewLinearFactory(tensors, groupSize, bits, mode, tq)
	patchDense := linears.Make(prefix + "patch_dense")
	if patchDense == nil {
		return nil, errors.New("missing Gemma4 unified patch projection")
	}
	makeLayerNorm := func(path string, width int) (*nn.LayerNorm, error) {
		weight, bias := tensors[path+".weight"], tensors[path+".bias"]
		if weight == nil || bias == nil || !slices.Equal(weight.Dims(), []int{width}) || !slices.Equal(bias.Dims(), []int{width}) {
			return nil, fmt.Errorf("invalid Gemma4 unified layer norm %s", path)
		}
		return &nn.LayerNorm{Weight: weight, Bias: bias, Eps: 1e-5}, nil
	}
	patchLN1, err := makeLayerNorm(prefix+"patch_ln1", int(patchDim))
	if err != nil {
		return nil, err
	}
	patchLN2, err := makeLayerNorm(prefix+"patch_ln2", int(cfg.MMEmbedDim))
	if err != nil {
		return nil, err
	}
	posNorm, err := makeLayerNorm(prefix+"pos_norm", int(cfg.MMEmbedDim))
	if err != nil {
		return nil, err
	}
	pos := tensors[prefix+"pos_embedding"]
	if pos == nil || !slices.Equal(pos.Dims(), []int{int(cfg.MMPosembSize), 2, int(cfg.MMEmbedDim)}) {
		return nil, errors.New("invalid Gemma4 unified position embedding")
	}
	return &UnifiedVisionEmbedder{PatchLN1: patchLN1, PatchDense: patchDense, PatchLN2: patchLN2, PosEmbedding: pos, PosNorm: posNorm, PatchDim: int32(patchDim)}, nil
}

func resolveVisionPrefix(tensors map[string]*mlx.Array) string {
	if tensors["vision_tower.patch_embedder.input_proj.weight"] != nil {
		return ""
	}
	if tensors["model.vision_tower.patch_embedder.input_proj.weight"] != nil {
		return "model."
	}
	return ""
}

func loadVisionModel(tensors map[string]*mlx.Array, cfg *VisionConfig, groupSize, bits int, mode string, tq map[string]*model.TensorQuantInfo) (*VisionModel, error) {
	if err := validateVisionConfig(cfg); err != nil {
		return nil, err
	}
	prefix := resolveVisionPrefix(tensors)
	linears := model.NewLinearFactory(tensors, groupSize, bits, mode, tq)

	patchProj := linears.Make(prefix + "vision_tower.patch_embedder.input_proj")
	if patchProj == nil {
		return nil, fmt.Errorf("missing vision patch projection")
	}
	posTable := tensors[prefix+"vision_tower.patch_embedder.position_embedding_table"]
	if posTable == nil {
		return nil, fmt.Errorf("missing vision position embedding table")
	}

	v := &VisionModel{
		Config: cfg,
		PatchEmbedder: &VisionPatchEmbedder{
			InputProj:              patchProj,
			PositionEmbeddingTable: posTable,
			PatchSize:              cfg.PatchSize,
			PositionEmbeddingSize:  cfg.PositionEmbeddingSize,
		},
		Layers: make([]*VisionLayer, cfg.NumHiddenLayers),
	}

	for i := range cfg.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("%svision_tower.encoder.layers.%d", prefix, i)
		layer := &VisionLayer{
			Attention: &VisionAttention{
				QProj: makeClippableLinear(tensors, linears, layerPrefix+".self_attn.q_proj", cfg.UseClippedLinears),
				KProj: makeClippableLinear(tensors, linears, layerPrefix+".self_attn.k_proj", cfg.UseClippedLinears),
				VProj: makeClippableLinear(tensors, linears, layerPrefix+".self_attn.v_proj", cfg.UseClippedLinears),
				OProj: makeClippableLinear(tensors, linears, layerPrefix+".self_attn.o_proj", cfg.UseClippedLinears),
				QNorm: nn.NewRMSNorm(tensors[layerPrefix+".self_attn.q_norm.weight"], cfg.RMSNormEps),
				KNorm: nn.NewRMSNorm(tensors[layerPrefix+".self_attn.k_norm.weight"], cfg.RMSNormEps),
			},
			MLP: &VisionMLP{
				GateProj: makeClippableLinear(tensors, linears, layerPrefix+".mlp.gate_proj", cfg.UseClippedLinears),
				UpProj:   makeClippableLinear(tensors, linears, layerPrefix+".mlp.up_proj", cfg.UseClippedLinears),
				DownProj: makeClippableLinear(tensors, linears, layerPrefix+".mlp.down_proj", cfg.UseClippedLinears),
			},
			InputNorm:    nn.NewRMSNorm(tensors[layerPrefix+".input_layernorm.weight"], cfg.RMSNormEps),
			PostAttnNorm: nn.NewRMSNorm(tensors[layerPrefix+".post_attention_layernorm.weight"], cfg.RMSNormEps),
			PreFFNorm:    nn.NewRMSNorm(tensors[layerPrefix+".pre_feedforward_layernorm.weight"], cfg.RMSNormEps),
			PostFFNorm:   nn.NewRMSNorm(tensors[layerPrefix+".post_feedforward_layernorm.weight"], cfg.RMSNormEps),
		}
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			return nil, fmt.Errorf("vision layer %d: missing attention projection", i)
		}
		if layer.Attention.QNorm.Weight == nil || layer.Attention.KNorm.Weight == nil {
			return nil, fmt.Errorf("vision layer %d: missing attention norm", i)
		}
		if layer.MLP.GateProj == nil || layer.MLP.UpProj == nil || layer.MLP.DownProj == nil {
			return nil, fmt.Errorf("vision layer %d: missing mlp projection", i)
		}
		if layer.InputNorm.Weight == nil || layer.PostAttnNorm.Weight == nil || layer.PreFFNorm.Weight == nil || layer.PostFFNorm.Weight == nil {
			return nil, fmt.Errorf("vision layer %d: missing block norm", i)
		}
		v.Layers[i] = layer
	}

	if cfg.Standardize {
		v.StdBias = tensors[prefix+"vision_tower.std_bias"]
		v.StdScale = tensors[prefix+"vision_tower.std_scale"]
		if err := validateVisionStandardizationTensors(cfg, v.StdBias != nil, v.StdScale != nil); err != nil {
			return nil, err
		}
	}

	return v, nil
}

func validateVisionStandardizationTensors(cfg *VisionConfig, hasBias, hasScale bool) error {
	if cfg == nil || !cfg.Standardize {
		return nil
	}
	if !hasBias || !hasScale {
		return fmt.Errorf("missing Gemma4 vision standardization tensors: bias=%t scale=%t", hasBias, hasScale)
	}
	return nil
}

func loadMultimodalEmbedder(tensors map[string]*mlx.Array, path string, eps float32, groupSize, bits int, mode string, tq map[string]*model.TensorQuantInfo) (*MultimodalEmbedder, error) {
	linears := model.NewLinearFactory(tensors, groupSize, bits, mode, tq)
	proj := linears.Make(path + ".embedding_projection")
	if proj == nil {
		proj = linears.Make("model." + path + ".embedding_projection")
	}
	if proj == nil {
		return nil, fmt.Errorf("missing %s embedding projection", path)
	}
	return &MultimodalEmbedder{Projection: proj, Eps: eps}, nil
}

func makeClippableLinear(tensors map[string]*mlx.Array, linears model.LinearFactory, path string, useClip bool) *ClippableLinear {
	linear := linears.Make(path + ".linear")
	if linear == nil {
		linear = linears.Make(path)
	}
	if linear == nil {
		return nil
	}
	out := &ClippableLinear{Linear: linear}
	if useClip {
		out.InputMin = tensors[path+".input_min"]
		out.InputMax = tensors[path+".input_max"]
		out.OutputMin = tensors[path+".output_min"]
		out.OutputMax = tensors[path+".output_max"]
	}
	return out
}

func (l *ClippableLinear) Forward(x *mlx.Array) *mlx.Array {
	if l.InputMin != nil && l.InputMax != nil {
		x = mlx.Clip(x, l.InputMin, l.InputMax)
	}
	out := l.Linear.Forward(x)
	if l.OutputMin != nil && l.OutputMax != nil {
		out = mlx.Clip(out, l.OutputMin, l.OutputMax)
	}
	return out
}

func (m *MultimodalEmbedder) Forward(x *mlx.Array) *mlx.Array {
	return m.Projection.Forward(mlx.RMSNormFn(x, nil, m.Eps))
}

// PrepareMedia implements the runner's ordered media contract. Each image or
// audio segment remains a separate cache-identity item.
func (m *Model) PrepareMedia(ctx context.Context, segments []base.Segment) (*base.PreparedRequest, error) {
	prepared := &base.PreparedRequest{}
	var layout gemma4MediaLayout
	for source, seg := range segments {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if seg.Data == nil {
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
			continue
		}
		start := len(prepared.Tokens)
		var payload gemma4MediaPayload
		var mediaData []float32
		var dims []int
		switch seg.Kind {
		case "image":
			if m.VisionConfig == nil || (m.Vision == nil && m.UnifiedVision == nil) || m.EmbedVision == nil {
				return nil, fmt.Errorf("this model does not support image input")
			}
			img, err := preprocessGemma4Image(ctx, seg.Data, m.VisionConfig, int(m.VisionSoftTokens))
			if err != nil {
				return nil, err
			}
			prepared.Tokens = append(prepared.Tokens, m.BOITokenIDValue)
			payload.ImageStart = len(prepared.Tokens) - start
			for range img.SoftTokens {
				prepared.Tokens = append(prepared.Tokens, m.ImageTokenIDValue)
			}
			payload.ImageEnd = len(prepared.Tokens) - start
			prepared.Tokens = append(prepared.Tokens, m.EOITokenIDValue)
			geom := *img
			mediaData = geom.Pixels
			dims = []int{1, 3, geom.Height, geom.Width}
			geom.Pixels = nil
			if m.UnifiedVision != nil {
				mediaData = geom.Patches
				dims = []int{1, geom.SoftTokens, int(m.UnifiedVision.PatchDim)}
				geom.Patches = nil
				layout.ImageSpans = append(layout.ImageSpans, [2]int{start + payload.ImageStart, start + payload.ImageEnd})
			}
			payload.Image = &geom
		case "audio":
			if m.AudioConfig == nil || m.AudioProcessorConfig == nil || m.Audio == nil || m.EmbedAudio == nil {
				return nil, fmt.Errorf("this model does not support audio input")
			}
			audio, err := preprocessGemma4Audio(ctx, seg.Data, m.AudioProcessorConfig)
			if err != nil {
				return nil, err
			}
			prepared.Tokens = append(prepared.Tokens, m.BOATokenIDValue)
			payload.AudioStart = len(prepared.Tokens) - start
			for range audio.SoftTokens {
				prepared.Tokens = append(prepared.Tokens, m.AudioTokenIDValue)
			}
			payload.AudioEnd = len(prepared.Tokens) - start
			prepared.Tokens = append(prepared.Tokens, m.EOATokenIDValue)
			geom := *audio
			mediaData = geom.Features
			dims = []int{1, geom.Frames, 128}
			geom.Features = nil
			payload.Audio = &geom
		default:
			return nil, fmt.Errorf("gemma4 does not support %s input", seg.Kind)
		}
		item := base.PreparedItem{
			Range:     [2]int{start, len(prepared.Tokens)},
			Source:    source,
			MediaData: mediaData,
			Dims:      dims,
			Opaque:    payload,
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		prepared.Items = append(prepared.Items, item)
	}
	if len(layout.ImageSpans) > 0 {
		prepared.Layout = &layout
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return prepared, nil
}

// EncodeMedia builds one lazy image or audio feature graph from runner-owned
// MediaData.
func (m *Model) EncodeMedia(item *base.PreparedItem, data *mlx.Array) *mlx.Array {
	payload := item.Opaque.(gemma4MediaPayload)
	if payload.Audio != nil {
		features := m.EmbedAudio.Forward(m.Audio.Forward(data, payload.Audio))
		return mlx.Squeeze(features, 0)
	}
	var encoded *mlx.Array
	if m.UnifiedVision != nil {
		patches := mlx.Reshape(data, 1, int32(payload.Image.SoftTokens), m.UnifiedVision.PatchDim)
		encoded = m.UnifiedVision.Forward(patches, payload.Image)
	} else {
		pixels := mlx.Reshape(data, 1, 3, int32(payload.Image.Height), int32(payload.Image.Width))
		encoded = m.Vision.Forward(pixels, payload.Image)
	}
	features := m.EmbedVision.Forward(encoded)
	return mlx.Squeeze(features, 0)
}

func gemma4MediaRun(item batch.MediaItem) (start, end int) {
	payload := item.Opaque.(gemma4MediaPayload)
	if payload.Audio != nil {
		return item.Pos + payload.AudioStart, item.Pos + payload.AudioEnd
	}
	return item.Pos + payload.ImageStart, item.Pos + payload.ImageEnd
}

func (m *Model) scatterMedia(h *mlx.Array, b *batch.Batch) *mlx.Array {
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		start, end := gemma4MediaRun(item)
		base := int(b.SeqOffsets[item.Seq])
		lo := max(start, base)
		hi := min(end, base+int(b.SeqQueryLens[item.Seq]))
		if hi <= lo {
			continue
		}
		features := item.Features.Slice(mlx.Slice(lo-start, hi-start), mlx.Slice())
		features = mlx.Reshape(features.AsType(h.DType()), 1, int32(hi-lo), m.HiddenSize)
		h = h.SliceUpdate(features, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(lo-base, hi-base), mlx.Slice())
	}
	return h
}

// pleTokens masks feature-bearing media tokens from PLE exactly where the
// same prepared items are scattered into the token embeddings.
func gemma4PLETokens(tokens *mlx.Array, b *batch.Batch) *mlx.Array {
	for _, item := range b.Media {
		start, end := gemma4MediaRun(item)
		base := int(b.SeqOffsets[item.Seq])
		lo := max(start, base)
		hi := min(end, base+int(b.SeqQueryLens[item.Seq]))
		if hi <= lo {
			continue
		}
		zeros := mlx.Zeros(mlx.DTypeInt32, 1, hi-lo)
		tokens = tokens.SliceUpdate(zeros, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(lo-base, hi-base))
	}
	return tokens
}

func preprocessGemma4Image(ctx context.Context, data []byte, cfg *VisionConfig, maxSoftTokens int) (*gemma4ImageInput, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateVisionConfig(cfg); err != nil {
		return nil, err
	}
	if err := validateGemma4ImageDataSize(len(data)); err != nil {
		return nil, err
	}
	imageConfig, _, err := decodeGemma4ImageConfig(ctx, data)
	if err != nil {
		return nil, fmt.Errorf("decode Gemma4 image config: %w", err)
	}
	if err := validateGemma4ImageDimensions(imageConfig.Width, imageConfig.Height); err != nil {
		return nil, err
	}
	img, _, err := decodeGemma4Image(ctx, data)
	if err != nil {
		return nil, fmt.Errorf("decode Gemma4 image: %w", err)
	}
	b := img.Bounds()
	width, height := b.Dx(), b.Dy()
	if err := validateGemma4ImageDimensions(width, height); err != nil {
		return nil, err
	}

	patchSize := int(cfg.PatchSize)
	pooling := int(cfg.PoolingKernelSize)
	if patchSize <= 0 || pooling <= 0 {
		return nil, fmt.Errorf("invalid Gemma4 vision patch configuration")
	}
	if maxSoftTokens <= 0 {
		maxSoftTokens = int(cfg.DefaultOutputLength)
	}
	if maxSoftTokens > int(cfg.DefaultOutputLength) || maxSoftTokens > maxGemma4VisionSoftTokens {
		return nil, fmt.Errorf("invalid Gemma4 image soft-token limit %d", maxSoftTokens)
	}
	maxPatches64, ok := checkedPositiveProduct(maxIntValue, int64(maxSoftTokens), int64(pooling), int64(pooling))
	if !ok {
		return nil, errors.New("Gemma4 image patch budget exceeds platform limits")
	}
	maxPatches := int(maxPatches64)
	targetW, targetH, err := gemma4ResizeDimensions(width, height, patchSize, maxPatches, pooling)
	if err != nil {
		return nil, err
	}

	resized := img
	if targetW != width || targetH != height {
		resized, err = resizeGemma4Image(ctx, img, b, targetW, targetH)
		if err != nil {
			return nil, err
		}
	}

	patchW := targetW / patchSize
	patchH := targetH / patchSize
	patchCount, ok := checkedPositiveProduct(maxIntValue, int64(patchW), int64(patchH))
	if !ok {
		return nil, errors.New("Gemma4 image patch count exceeds platform limits")
	}
	if !cfg.unified() {
		if _, ok := checkedPositiveProduct(maxGemma4PositionValues, patchCount, int64(cfg.HeadDim)); !ok {
			return nil, fmt.Errorf("Gemma4 image position allocation exceeds limit %d", maxGemma4PositionValues)
		}
	}
	softTokens := int(patchCount / (int64(pooling) * int64(pooling)))
	if softTokens <= 0 || softTokens > maxSoftTokens {
		return nil, fmt.Errorf("Gemma4 image produced %d soft tokens, limit %d", softTokens, maxSoftTokens)
	}

	input := &gemma4ImageInput{
		Width:       targetW,
		Height:      targetH,
		PatchWidth:  patchW,
		PatchHeight: patchH,
		SoftTokens:  softTokens,
	}
	if cfg.unified() {
		input.Patches, input.Positions, err = imageToUnifiedPatchesContext(ctx, resized, int(cfg.ModelPatchSize))
		if err != nil {
			return nil, err
		}
		expected, ok := checkedPositiveProduct(maxIntValue, int64(softTokens), int64(cfg.ModelPatchSize), int64(cfg.ModelPatchSize), 3)
		if !ok || int64(len(input.Patches)) != expected || len(input.Positions) != softTokens*2 {
			return nil, errors.New("Gemma4 unified patch count does not match soft token count")
		}
	} else {
		input.Pixels, err = imageToCHWFloat32Context(ctx, resized)
		if err != nil {
			return nil, err
		}
	}
	return input, nil
}

func imageToUnifiedPatchesContext(ctx context.Context, img image.Image, patchSize int) ([]float32, []int32, error) {
	if patchSize <= 0 {
		return nil, nil, errors.New("invalid Gemma4 unified model patch size")
	}
	b := img.Bounds()
	width, height := b.Dx(), b.Dy()
	if width%patchSize != 0 || height%patchSize != 0 {
		return nil, nil, fmt.Errorf("Gemma4 unified image dimensions %dx%d are not divisible by patch size %d", width, height, patchSize)
	}
	patchW, patchH := width/patchSize, height/patchSize
	patchDim, ok := checkedPositiveProduct(maxIntValue, int64(patchSize), int64(patchSize), 3)
	if !ok {
		return nil, nil, errors.New("Gemma4 unified patch dimension exceeds platform limits")
	}
	patchCount, ok := checkedPositiveProduct(maxIntValue, int64(patchW), int64(patchH))
	if !ok || patchCount > math.MaxInt32 {
		return nil, nil, errors.New("Gemma4 unified patch count exceeds platform limits")
	}
	values, ok := checkedPositiveProduct(maxIntValue, patchCount, patchDim)
	if !ok {
		return nil, nil, errors.New("Gemma4 unified patch allocation exceeds platform limits")
	}
	positionsCount, ok := checkedPositiveProduct(maxIntValue, patchCount, 2)
	if !ok {
		return nil, nil, errors.New("Gemma4 unified position allocation exceeds platform limits")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	patches := make([]float32, int(values))
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	positions := make([]int32, int(positionsCount))
	for py := range patchH {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		for px := range patchW {
			patch := py*patchW + px
			positions[2*patch], positions[2*patch+1] = int32(px), int32(py)
			offset := patch * int(patchDim)
			for y := range patchSize {
				for x := range patchSize {
					r, g, blue, _ := img.At(b.Min.X+px*patchSize+x, b.Min.Y+py*patchSize+y).RGBA()
					i := offset + (y*patchSize+x)*3
					patches[i], patches[i+1], patches[i+2] = float32(r)/65535, float32(g)/65535, float32(blue)/65535
				}
			}
		}
	}
	return patches, positions, nil
}

type gemma4CancellationPanic struct{ err error }

type cancellableDrawImage struct {
	stddraw.Image
	contextErr func() error
}

func (dst cancellableDrawImage) Set(x, y int, c color.Color) {
	if err := dst.contextErr(); err != nil {
		panic(gemma4CancellationPanic{err: err})
	}
	dst.Image.Set(x, y, c)
}

func resizeGemma4Image(ctx context.Context, src image.Image, bounds image.Rectangle, width, height int) (resized image.Image, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	defer func() {
		if recovered := recover(); recovered != nil {
			if canceled, ok := recovered.(gemma4CancellationPanic); ok {
				resized, err = nil, canceled.err
				return
			}
			panic(recovered)
		}
	}()
	xdraw.CatmullRom.Scale(cancellableDrawImage{Image: dst, contextErr: ctx.Err}, dst.Bounds(), src, bounds, xdraw.Over, nil)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return dst, nil
}

func validateGemma4ImageDataSize(size int) error {
	if size <= 0 {
		return errors.New("Gemma4 image is empty")
	}
	if size > maxGemma4ImageBytes {
		return fmt.Errorf("Gemma4 image is %d bytes, limit %d", size, maxGemma4ImageBytes)
	}
	return nil
}

type contextReader struct {
	contextErr func() error
	r          io.Reader
}

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.contextErr(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

func decodeGemma4ImageConfig(ctx context.Context, data []byte) (image.Config, string, error) {
	r := func() io.Reader { return contextReader{contextErr: ctx.Err, r: bytes.NewReader(data)} }
	cfg, format, err := image.DecodeConfig(r())
	if err != nil && ctx.Err() != nil {
		return image.Config{}, "", ctx.Err()
	}
	return cfg, format, err
}

func decodeGemma4Image(ctx context.Context, data []byte) (image.Image, string, error) {
	r := func() io.Reader { return contextReader{contextErr: ctx.Err, r: bytes.NewReader(data)} }
	img, format, err := image.Decode(r())
	if err != nil && ctx.Err() != nil {
		return nil, "", ctx.Err()
	}
	return img, format, err
}

func validateGemma4ImageDimensions(width, height int) error {
	if width <= 0 || height <= 0 {
		return fmt.Errorf("invalid Gemma4 image dimensions %dx%d", width, height)
	}
	if width > maxGemma4ImageDimension || height > maxGemma4ImageDimension {
		return fmt.Errorf("Gemma4 image dimensions %dx%d exceed limit %dx%d", width, height, maxGemma4ImageDimension, maxGemma4ImageDimension)
	}
	pixels := int64(width) * int64(height)
	if pixels > maxGemma4ImagePixels {
		return fmt.Errorf("Gemma4 image has %d pixels, limit %d", pixels, maxGemma4ImagePixels)
	}
	return nil
}

func gemma4ResizeDimensions(width, height, patchSize, maxPatches, poolingKernelSize int) (int, int, error) {
	if width <= 0 || height <= 0 || patchSize <= 0 || maxPatches <= 0 || poolingKernelSize <= 0 {
		return 0, 0, errors.New("invalid Gemma4 resize parameters")
	}
	if err := validateGemma4ImageDimensions(width, height); err != nil {
		return 0, 0, err
	}
	targetPixels, ok := checkedPositiveProduct(maxGemma4ResizePixels, int64(maxPatches), int64(patchSize), int64(patchSize))
	if !ok {
		return 0, 0, fmt.Errorf("Gemma4 resize target exceeds %d pixels", maxGemma4ResizePixels)
	}
	sourcePixels := int64(height) * int64(width)
	targetPx := float64(targetPixels)
	factor := math.Sqrt(targetPx / float64(sourcePixels))
	sideMult64, ok := checkedPositiveProduct(maxGemma4ImageDimension, int64(poolingKernelSize), int64(patchSize))
	if !ok {
		return 0, 0, errors.New("invalid Gemma4 pooled patch size")
	}
	sideMult := int(sideMult64)

	targetH := int(math.Floor(factor*float64(height)/float64(sideMult))) * sideMult
	targetW := int(math.Floor(factor*float64(width)/float64(sideMult))) * sideMult

	if targetH == 0 && targetW == 0 {
		return 0, 0, errors.New("attempting to resize Gemma4 image to 0 x 0")
	}

	poolingArea, ok := checkedPositiveProduct(maxIntValue, int64(poolingKernelSize), int64(poolingKernelSize))
	if !ok {
		return 0, 0, errors.New("invalid Gemma4 pooling area")
	}
	maxSideLength64, ok := checkedPositiveProduct(maxGemma4ImageDimension, int64(maxPatches)/poolingArea, sideMult64)
	if !ok {
		return 0, 0, errors.New("invalid Gemma4 maximum resize side")
	}
	maxSideLength := int(maxSideLength64)
	if targetH == 0 {
		targetH = sideMult
		targetW = min(int(math.Floor(float64(width)/float64(height)))*sideMult, maxSideLength)
	}
	if targetW == 0 {
		targetW = sideMult
		targetH = min(int(math.Floor(float64(height)/float64(width)))*sideMult, maxSideLength)
	}
	if targetW <= 0 || targetH <= 0 {
		return 0, 0, fmt.Errorf("invalid Gemma4 resize target %dx%d", targetW, targetH)
	}
	if err := validateGemma4ImageDimensions(targetW, targetH); err != nil {
		return 0, 0, fmt.Errorf("invalid Gemma4 resize target: %w", err)
	}
	if int64(targetW)*int64(targetH) > targetPixels {
		return 0, 0, fmt.Errorf("Gemma4 resize target %dx%d exceeds %d-pixel patch budget", targetW, targetH, targetPixels)
	}
	return targetW, targetH, nil
}

func imageToCHWFloat32(img image.Image) ([]float32, error) {
	return imageToCHWFloat32Context(context.Background(), img)
}

func imageToCHWFloat32Context(ctx context.Context, img image.Image) ([]float32, error) {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if err := validateGemma4ImageDimensions(width, height); err != nil {
		return nil, err
	}
	plane64 := int64(height) * int64(width)
	values64 := plane64 * 3
	if values64 <= 0 || values64 > maxIntValue {
		return nil, errors.New("Gemma4 pixel allocation exceeds platform limits")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	plane := int(plane64)
	out := make([]float32, int(values64))
	for y := range height {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for x := range width {
			r, g, blue, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			i := y*width + x
			out[i] = float32(r) / 65535
			out[plane+i] = float32(g) / 65535
			out[2*plane+i] = float32(blue) / 65535
		}
	}
	return out, nil
}

func mediaTokenSpan(tokens []int32, mediaTokenID int32) (int, int, error) {
	start, end := -1, -1
	for i, tok := range tokens {
		if tok != mediaTokenID {
			continue
		}
		if start == -1 {
			start = i
		}
		end = i + 1
	}
	if start == -1 {
		return 0, 0, fmt.Errorf("Gemma4 prompt contains no media token id %d", mediaTokenID)
	}
	for i := start; i < end; i++ {
		if tokens[i] != mediaTokenID {
			return 0, 0, errors.New("Gemma4 media tokens are not contiguous")
		}
	}
	return start, end, nil
}

func (m *VisionModel) Forward(pixels *mlx.Array, img *gemma4ImageInput) *mlx.Array {
	positions := m.positionArrays(img)
	h := m.PatchEmbedder.Forward(pixels, positions)
	for _, layer := range m.Layers {
		h = layer.Forward(h, positions, m.Config)
	}
	h = m.pool(h, int32(img.PatchHeight), int32(img.PatchWidth))
	if m.Config.Standardize {
		h = mlx.Mul(mlx.Sub(h, m.StdBias), m.StdScale)
	}
	return h
}

func (m *UnifiedVisionEmbedder) Forward(patches *mlx.Array, img *gemma4ImageInput) *mlx.Array {
	hidden := m.PatchLN2.Forward(m.PatchDense.Forward(m.PatchLN1.Forward(patches)))
	positions := mlx.FromValues(img.Positions, 1, img.SoftTokens, 2)
	x := mlx.Squeeze(mlx.SliceStartStop(positions, []int32{0, 0, 0}, []int32{1, int32(img.SoftTokens), 1}), 2)
	y := mlx.Squeeze(mlx.SliceStartStop(positions, []int32{0, 0, 1}, []int32{1, int32(img.SoftTokens), 2}), 2)
	tableX := mlx.Squeeze(mlx.SliceStartStop(m.PosEmbedding, []int32{0, 0, 0}, []int32{int32(m.PosEmbedding.Dim(0)), 1, int32(m.PosEmbedding.Dim(2))}), 1)
	tableY := mlx.Squeeze(mlx.SliceStartStop(m.PosEmbedding, []int32{0, 1, 0}, []int32{int32(m.PosEmbedding.Dim(0)), 2, int32(m.PosEmbedding.Dim(2))}), 1)
	pos := mlx.Add(tableX.TakeAxis(x, 0), tableY.TakeAxis(y, 0)).AsType(hidden.DType())
	return m.PosNorm.Forward(mlx.Add(hidden, pos))
}

func (m *VisionModel) positionArrays(img *gemma4ImageInput) visionPositionArrays {
	L := img.PatchHeight * img.PatchWidth
	xs := make([]int32, L)
	ys := make([]int32, L)
	cosVals := make([]float32, L*int(m.Config.HeadDim))
	sinVals := make([]float32, L*int(m.Config.HeadDim))
	idx := 0
	for y := range img.PatchHeight {
		for x := range img.PatchWidth {
			xs[idx] = int32(x)
			ys[idx] = int32(y)
			fillVisionRoPE(idx, x, y, int(m.Config.HeadDim), float64(m.Config.RopeParameters.RopeTheta), cosVals, sinVals)
			idx++
		}
	}
	return visionPositionArrays{
		X:       mlx.FromValues(xs, 1, L),
		Y:       mlx.FromValues(ys, 1, L),
		RopeCos: mlx.FromValues(cosVals, 1, L, 1, int(m.Config.HeadDim)),
		RopeSin: mlx.FromValues(sinVals, 1, L, 1, int(m.Config.HeadDim)),
	}
}

func fillVisionRoPE(row, x, y, headDim int, base float64, cosVals, sinVals []float32) {
	ndim := 2
	channelsPerDim := 2 * (headDim / (2 * ndim))
	offset := row * headDim
	for i := range headDim {
		cosVals[offset+i] = 1
	}
	if channelsPerDim == 0 {
		return
	}
	half := channelsPerDim / 2
	positions := []int{x, y}
	for dim, pos := range positions {
		start := dim * channelsPerDim
		for j := range half {
			timescale := math.Pow(base, float64(2*j)/float64(channelsPerDim))
			angle := float64(pos) / timescale
			c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
			cosVals[offset+start+j] = c
			cosVals[offset+start+half+j] = c
			sinVals[offset+start+j] = s
			sinVals[offset+start+half+j] = s
		}
	}
}

func (p *VisionPatchEmbedder) Forward(pixelValues *mlx.Array, positions visionPositionArrays) *mlx.Array {
	dims := pixelValues.Dims()
	B, C, H, W := int32(dims[0]), int32(dims[1]), int32(dims[2]), int32(dims[3])
	patch := p.PatchSize
	patchH := H / patch
	patchW := W / patch
	patches := mlx.Reshape(pixelValues, B, C, patchH, patch, patchW, patch)
	patches = mlx.Transpose(patches, 0, 2, 4, 3, 5, 1)
	patches = mlx.Reshape(patches, B, patchH*patchW, C*patch*patch)
	patches = mlx.MulScalar(mlx.AddScalar(patches, -0.5), 2)
	hidden := p.InputProj.Forward(patches)
	return mlx.Add(hidden, p.positionEmbeddings(positions).AsType(hidden.DType()))
}

func (p *VisionPatchEmbedder) positionEmbeddings(positions visionPositionArrays) *mlx.Array {
	tableX := mlx.Squeeze(mlx.SliceStartStop(p.PositionEmbeddingTable,
		[]int32{0, 0, 0},
		[]int32{1, p.PositionEmbeddingSize, int32(p.PositionEmbeddingTable.Dim(2))},
	), 0)
	tableY := mlx.Squeeze(mlx.SliceStartStop(p.PositionEmbeddingTable,
		[]int32{1, 0, 0},
		[]int32{2, p.PositionEmbeddingSize, int32(p.PositionEmbeddingTable.Dim(2))},
	), 0)
	return mlx.Add(tableX.TakeAxis(positions.X, 0), tableY.TakeAxis(positions.Y, 0))
}

func (l *VisionLayer) Forward(x *mlx.Array, positions visionPositionArrays, cfg *VisionConfig) *mlx.Array {
	normed := l.InputNorm.Forward(x, cfg.RMSNormEps)
	attn := l.Attention.Forward(normed, positions, cfg)
	attn = l.PostAttnNorm.Forward(attn, cfg.RMSNormEps)
	h := mlx.Add(x, attn)
	normed = l.PreFFNorm.Forward(h, cfg.RMSNormEps)
	mlp := l.MLP.Forward(normed, cfg)
	mlp = l.PostFFNorm.Forward(mlp, cfg.RMSNormEps)
	return mlx.Add(h, mlp)
}

func (a *VisionAttention) Forward(x *mlx.Array, positions visionPositionArrays, cfg *VisionConfig) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	q := a.QProj.Forward(x)
	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	k := a.KProj.Forward(x)
	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v := a.VProj.Forward(x)
	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)

	q = a.QNorm.Forward(q, cfg.RMSNormEps)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)
	v = mlx.RMSNormFn(v, nil, cfg.RMSNormEps)

	q = applyVisionRoPE(q, positions)
	k = applyVisionRoPE(k, positions)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	vb := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, int(B), int(L)),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{L},
	}
	out := nn.ScaledDotProductAttention(vb, q, 1.0, nn.WithKV(k, v, vb.SeqQueryLens))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func applyVisionRoPE(x *mlx.Array, positions visionPositionArrays) *mlx.Array {
	rotated := rotateVisionHalf(x)
	return mlx.Add(mlx.Mul(x, positions.RopeCos.AsType(x.DType())), mlx.Mul(rotated, positions.RopeSin.AsType(x.DType())))
}

func rotateVisionHalf(x *mlx.Array) *mlx.Array {
	dims := x.Dims()
	B, L, H, D := int32(dims[0]), int32(dims[1]), int32(dims[2]), int32(dims[3])
	ndim := int32(2)
	channels := 2 * (D / (2 * ndim))
	if channels == 0 {
		return x
	}
	parts := make([]*mlx.Array, 0, ndim+1)
	for dim := range ndim {
		start := dim * channels
		mid := start + channels/2
		end := start + channels
		x1 := mlx.SliceStartStop(x, []int32{0, 0, 0, start}, []int32{B, L, H, mid})
		x2 := mlx.SliceStartStop(x, []int32{0, 0, 0, mid}, []int32{B, L, H, end})
		parts = append(parts, mlx.Concatenate([]*mlx.Array{mlx.Neg(x2), x1}, -1))
	}
	if tail := channels * ndim; tail < D {
		parts = append(parts, mlx.SliceStartStop(x, []int32{0, 0, 0, tail}, []int32{B, L, H, D}))
	}
	return mlx.Concatenate(parts, -1)
}

func (m *VisionMLP) Forward(x *mlx.Array, cfg *VisionConfig) *mlx.Array {
	gate := m.GateProj.Forward(x)
	up := m.UpProj.Forward(x)
	return m.DownProj.Forward(mlx.GeGLU(gate, up))
}

func (m *VisionModel) pool(hidden *mlx.Array, patchH, patchW int32) *mlx.Array {
	k := m.Config.PoolingKernelSize
	B := int32(hidden.Dim(0))
	D := int32(hidden.Dim(2))
	if k <= 1 {
		return mlx.MulScalar(hidden, float32(math.Sqrt(float64(m.Config.HiddenSize))))
	}
	outH := patchH / k
	outW := patchW / k
	x := mlx.Reshape(hidden, B, patchH, patchW, D)
	x = mlx.Reshape(x, B, outH, k, outW, k, D)
	x = mlx.Mean(x, 4, false)
	x = mlx.Mean(x, 2, false)
	x = mlx.Reshape(x, B, outH*outW, D)
	return mlx.MulScalar(x, float32(math.Sqrt(float64(m.Config.HiddenSize))))
}

var _ base.MediaModel = (*Model)(nil)
