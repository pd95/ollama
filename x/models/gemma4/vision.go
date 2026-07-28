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
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"slices"
	"strings"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/webp"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
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

	maxGemma4ImageBytes     = 32 << 20
	maxGemma4ImageDimension = 16_384
	maxGemma4ImagePixels    = 64 << 20
	maxGemma4ResizePixels   = 1 << 20
	maxIntValue             = int64(^uint(0) >> 1)
)

type VisionRopeParameters struct {
	RopeTheta float32 `json:"rope_theta"`
	RopeType  string  `json:"rope_type"`
}

type VisionConfig struct {
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
	return c != nil && strings.Contains(strings.ToLower(c.ModelType), "unified")
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
	cfg := *wrapped.VisionConfig
	if cfg.HiddenSize == 0 {
		cfg.HiddenSize = 768
	}
	if cfg.IntermediateSize == 0 {
		cfg.IntermediateSize = 3072
	}
	if cfg.NumHiddenLayers == 0 {
		cfg.NumHiddenLayers = 16
	}
	if cfg.NumAttentionHeads == 0 {
		cfg.NumAttentionHeads = 12
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = 64
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.DefaultOutputLength == 0 {
		cfg.DefaultOutputLength = cfg.NumSoftTokens
		if cfg.DefaultOutputLength == 0 {
			cfg.DefaultOutputLength = 280
		}
	}
	if cfg.PatchSize == 0 {
		cfg.PatchSize = 16
	}
	if cfg.PositionEmbeddingSize == 0 {
		cfg.PositionEmbeddingSize = 10240
	}
	if cfg.PoolingKernelSize == 0 {
		cfg.PoolingKernelSize = 3
	}
	if cfg.unified() {
		if cfg.MMEmbedDim == 0 {
			cfg.MMEmbedDim = cfg.OutputProjDims
		}
		if cfg.ModelPatchSize == 0 {
			modelPatchSize := int64(cfg.PatchSize) * int64(cfg.PoolingKernelSize)
			if modelPatchSize <= 0 || modelPatchSize > math.MaxInt32 {
				return nil, fmt.Errorf("invalid unified model patch size %d", modelPatchSize)
			}
			cfg.ModelPatchSize = int32(modelPatchSize)
		}
		if cfg.MMPosembSize == 0 {
			cfg.MMPosembSize = 1120
		}
		expectedModelPatchSize := int64(cfg.PatchSize) * int64(cfg.PoolingKernelSize)
		if int64(cfg.ModelPatchSize) != expectedModelPatchSize {
			return nil, fmt.Errorf("unified model_patch_size %d does not match patch_size * pooling_kernel_size (%d)", cfg.ModelPatchSize, expectedModelPatchSize)
		}
	}
	if cfg.RopeParameters.RopeTheta == 0 {
		cfg.RopeParameters.RopeTheta = 100
	}
	return &cfg, nil
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

func hasCompleteGemma4VisionWeights(tensors map[string]*mlx.Array, cfg *VisionConfig) bool {
	names := make([]string, 0, len(tensors))
	for name, tensor := range tensors {
		if tensor != nil {
			names = append(names, name)
		}
	}
	return gemma4metadata.ValidateVisionTensors(gemma4metadata.ConfigFile{
		VisionConfig: &gemma4metadata.VisionConfig{
			ModelType:       cfg.ModelType,
			NumHiddenLayers: int(cfg.NumHiddenLayers),
			Standardize:     cfg.Standardize,
		},
	}, names) == nil
}

func loadUnifiedVisionEmbedder(tensors map[string]*mlx.Array, cfg *VisionConfig, textHiddenSize int32, groupSize, bits int, mode string, tq map[string]*model.TensorQuantInfo) (*UnifiedVisionEmbedder, error) {
	const prefix = "model.vision_embedder."
	patchDim64 := int64(cfg.ModelPatchSize) * int64(cfg.ModelPatchSize) * 3
	if patchDim64 <= 0 || patchDim64 > math.MaxInt32 {
		return nil, fmt.Errorf("invalid Gemma4 unified patch dimension %d", patchDim64)
	}
	patchDim := int32(patchDim64)
	expected := map[string][]int{
		prefix + "patch_ln1.weight":                      {int(patchDim)},
		prefix + "patch_ln1.bias":                        {int(patchDim)},
		prefix + "patch_dense.weight":                    {int(cfg.MMEmbedDim), int(patchDim)},
		prefix + "patch_dense.bias":                      {int(cfg.MMEmbedDim)},
		prefix + "patch_ln2.weight":                      {int(cfg.MMEmbedDim)},
		prefix + "patch_ln2.bias":                        {int(cfg.MMEmbedDim)},
		prefix + "pos_embedding":                         {int(cfg.MMPosembSize), 2, int(cfg.MMEmbedDim)},
		prefix + "pos_norm.weight":                       {int(cfg.MMEmbedDim)},
		prefix + "pos_norm.bias":                         {int(cfg.MMEmbedDim)},
		"model.embed_vision.embedding_projection.weight": {int(textHiddenSize), int(cfg.MMEmbedDim)},
	}
	for name, shape := range expected {
		array := tensors[name]
		if array == nil {
			return nil, fmt.Errorf("missing Gemma4 unified tensor %s", name)
		}
		if !slices.Equal(array.Dims(), shape) {
			return nil, fmt.Errorf("Gemma4 unified tensor %s shape %v, want %v", name, array.Dims(), shape)
		}
		switch array.DType() {
		case mlx.DTypeBFloat16, mlx.DTypeFloat16, mlx.DTypeFloat32:
		default:
			return nil, fmt.Errorf("Gemma4 unified tensor %s has unsupported dtype %s", name, array.DType())
		}
	}
	linears := model.NewLinearFactory(tensors, groupSize, bits, mode, tq)
	patchDense := linears.Make(prefix + "patch_dense")
	posEmbedding := tensors[prefix+"pos_embedding"]
	if patchDense == nil || posEmbedding == nil {
		return nil, errors.New("missing Gemma4 unified vision projection tensors")
	}
	makeLayerNorm := func(path string) (*nn.LayerNorm, error) {
		weight := tensors[path+".weight"]
		bias := tensors[path+".bias"]
		if weight == nil || bias == nil {
			return nil, fmt.Errorf("missing Gemma4 unified layer norm %s", path)
		}
		return &nn.LayerNorm{Weight: weight, Bias: bias, Eps: 1e-5}, nil
	}
	patchLN1, err := makeLayerNorm(prefix + "patch_ln1")
	if err != nil {
		return nil, err
	}
	patchLN2, err := makeLayerNorm(prefix + "patch_ln2")
	if err != nil {
		return nil, err
	}
	posNorm, err := makeLayerNorm(prefix + "pos_norm")
	if err != nil {
		return nil, err
	}
	return &UnifiedVisionEmbedder{
		PatchLN1: patchLN1, PatchDense: patchDense, PatchLN2: patchLN2,
		PosEmbedding: posEmbedding, PosNorm: posNorm,
		PatchDim: patchDim,
	}, nil
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
		if v.StdBias == nil || v.StdScale == nil {
			return nil, fmt.Errorf("missing vision standardization tensors")
		}
	}

	return v, nil
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

func (m *Model) PrepareMediaPrompt(ctx context.Context, prompt string, media []llm.MediaData) (*batch.PreparedInput, error) {
	if len(media) != 1 {
		return nil, fmt.Errorf("Gemma4 MLX currently supports exactly one image or WAV audio input per request, got %d", len(media))
	}

	marker := fmt.Sprintf("[img-%d]", media[0].ID)
	if strings.Count(prompt, marker) != 1 {
		return nil, fmt.Errorf("expected one %s marker in Gemma4 prompt", marker)
	}

	var payload gemma4MediaPayload
	var sequence string
	var tokenID int32
	var softTokens int
	switch media[0].Kind {
	case llm.MediaKindImage:
		if m.VisionConfig == nil {
			return nil, errors.New("Gemma4 MLX model has no vision_config")
		}
		img, err := preprocessGemma4Image(ctx, media[0].Data, m.VisionConfig, int(m.VisionSoftTokens))
		if err != nil {
			return nil, err
		}
		payload.Image = img
		softTokens = img.SoftTokens
		tokenID = m.ImageTokenIDValue
		sequence = m.mediaTokens.BOI + strings.Repeat(m.mediaTokens.Image, softTokens) + m.mediaTokens.EOI
	case llm.MediaKindAudio:
		if m.AudioConfig == nil || m.AudioProcessorConfig == nil {
			return nil, errors.New("Gemma4 MLX model has no supported audio configuration")
		}
		audio, err := preprocessGemma4Audio(ctx, media[0].Data, m.AudioProcessorConfig)
		if err != nil {
			return nil, err
		}
		payload.Audio = audio
		softTokens = audio.SoftTokens
		tokenID = m.AudioTokenIDValue
		sequence = m.mediaTokens.BOA + strings.Repeat(m.mediaTokens.Audio, softTokens) + m.mediaTokens.EOA
	default:
		return nil, fmt.Errorf("Gemma4 MLX does not support %s inputs", media[0].Kind)
	}

	expanded := strings.Replace(prompt, marker, sequence, 1)
	tokens := m.tok.Encode(expanded, m.tok.AddBOS())
	start, end, err := mediaTokenSpan(tokens, tokenID)
	if err != nil {
		return nil, err
	}
	if got := end - start; got != softTokens {
		return nil, fmt.Errorf("Gemma4 media token count = %d, want %d", got, softTokens)
	}

	pleIDs := append([]int32(nil), tokens...)
	for i := start; i < end; i++ {
		pleIDs[i] = 0
	}
	if payload.Image != nil {
		payload.ImageStart, payload.ImageEnd = start, end
	} else {
		payload.AudioStart, payload.AudioEnd = start, end
	}
	prepared := &batch.PreparedInput{
		Tokens:      tokens,
		PLEInputIDs: pleIDs,
		Payload:     &payload,
	}
	if payload.Image != nil && m.VisionConfig.unified() {
		prepared.BidirectionalSpans = []batch.TokenSpan{{Start: start, End: end}}
	}
	return prepared, nil
}

func (m *Model) PrepareMediaEmbeddings(prepared *batch.PreparedInput) error {
	payload, ok := prepared.Payload.(*gemma4MediaPayload)
	if !ok || payload == nil || payload.Image == nil && payload.Audio == nil {
		return errors.New("invalid Gemma4 media payload")
	}

	tokens := mlx.FromValues(prepared.Tokens, 1, len(prepared.Tokens))
	embeddings := m.TokenEmbeddings(tokens)
	var features *mlx.Array
	var start, end int
	if payload.Image != nil {
		if (m.Vision == nil && m.UnifiedVision == nil) || m.EmbedVision == nil {
			return errors.New("Gemma4 MLX vision weights are not loaded; recreate or pull the model so it includes Gemma 4 vision tensor layers")
		}
		var encoded *mlx.Array
		if m.UnifiedVision != nil {
			encoded = m.UnifiedVision.Forward(payload.Image)
		} else {
			encoded = m.Vision.Forward(payload.Image)
		}
		features = m.EmbedVision.Forward(encoded)
		start, end = payload.ImageStart, payload.ImageEnd
	} else {
		if m.EmbedAudio == nil || !m.AudioConfig.unified() && m.Audio == nil {
			if m.AudioConfig.unified() {
				return errors.New("Gemma4 MLX unified audio projection is not loaded; recreate or pull the model with its audio projection")
			}
			return errors.New("Gemma4 MLX audio weights are not loaded; recreate or pull the model so it includes the complete Gemma 4 audio tower")
		}
		if m.AudioConfig.unified() {
			raw := mlx.FromValues(payload.Audio.Features, 1, payload.Audio.SoftTokens, payload.Audio.FeatureSize)
			features = m.EmbedAudio.Forward(raw)
		} else {
			features = m.EmbedAudio.Forward(m.Audio.Forward(payload.Audio))
		}
		start, end = payload.AudioStart, payload.AudioEnd
	}
	features = features.AsType(embeddings.DType())

	if features.Dim(1) != end-start {
		return fmt.Errorf("Gemma4 media feature length = %d, want %d", features.Dim(1), end-start)
	}

	parts := make([]*mlx.Array, 0, 3)
	if start > 0 {
		parts = append(parts, mlx.SliceStartStop(embeddings,
			[]int32{0, 0, 0},
			[]int32{1, int32(start), int32(embeddings.Dim(2))},
		))
	}
	parts = append(parts, features)
	if end < len(prepared.Tokens) {
		parts = append(parts, mlx.SliceStartStop(embeddings,
			[]int32{0, int32(end), 0},
			[]int32{1, int32(len(prepared.Tokens)), int32(embeddings.Dim(2))},
		))
	}
	prepared.InputEmbeddings = mlx.Concatenate(parts, 1)
	return nil
}

func preprocessGemma4Image(ctx context.Context, data []byte, cfg *VisionConfig, maxSoftTokens int) (*gemma4ImageInput, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(data) > maxGemma4ImageBytes {
		return nil, fmt.Errorf("Gemma4 image is %d bytes, limit %d", len(data), maxGemma4ImageBytes)
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
	maxPatches64 := int64(maxSoftTokens) * int64(pooling) * int64(pooling)
	if maxPatches64 <= 0 || maxPatches64 > maxIntValue {
		return nil, errors.New("invalid Gemma4 image patch budget")
	}
	maxPatches := int(maxPatches64)
	targetW, targetH, err := gemma4ResizeDimensions(width, height, patchSize, maxPatches, pooling)
	if err != nil {
		return nil, err
	}

	resized := img
	if targetW != width || targetH != height {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
		xdraw.CatmullRom.Scale(dst, dst.Bounds(), img, b, xdraw.Over, nil)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resized = dst
	}

	patchW := targetW / patchSize
	patchH := targetH / patchSize
	softTokens64 := (int64(patchW) * int64(patchH)) / (int64(pooling) * int64(pooling))
	if softTokens64 <= 0 || softTokens64 > maxIntValue {
		return nil, errors.New("invalid Gemma4 soft token count")
	}
	softTokens := int(softTokens64)
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
		expectedPatchValues := int64(softTokens) * int64(cfg.ModelPatchSize) * int64(cfg.ModelPatchSize) * 3
		if expectedPatchValues > maxIntValue || int64(len(input.Patches)) != expectedPatchValues {
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
	patchDim64 := int64(patchSize) * int64(patchSize) * 3
	patchCount64 := int64(patchW) * int64(patchH)
	patchValues64 := patchCount64 * patchDim64
	positionValues64 := patchCount64 * 2
	if patchDim64 <= 0 || patchValues64 <= 0 || patchValues64 > maxIntValue || positionValues64 > maxIntValue {
		return nil, nil, errors.New("Gemma4 unified patch allocation exceeds platform limits")
	}
	patchDim := int(patchDim64)
	patches := make([]float32, int(patchValues64))
	positions := make([]int32, int(positionValues64))
	for py := range patchH {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		for px := range patchW {
			patch := py*patchW + px
			positions[2*patch] = int32(px)
			positions[2*patch+1] = int32(py)
			offset := patch * patchDim
			for y := range patchSize {
				for x := range patchSize {
					r, g, blue, _ := img.At(b.Min.X+px*patchSize+x, b.Min.Y+py*patchSize+y).RGBA()
					i := offset + (y*patchSize+x)*3
					patches[i] = float32(r) / 65535
					patches[i+1] = float32(g) / 65535
					patches[i+2] = float32(blue) / 65535
				}
			}
		}
	}
	return patches, positions, nil
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
	if err == nil || !isWebP(data) {
		return cfg, format, err
	}
	cfg, err = webp.DecodeConfig(r())
	return cfg, "webp", err
}

func decodeGemma4Image(ctx context.Context, data []byte) (image.Image, string, error) {
	r := func() io.Reader { return contextReader{contextErr: ctx.Err, r: bytes.NewReader(data)} }
	img, format, err := image.Decode(r())
	if err == nil || !isWebP(data) {
		return img, format, err
	}
	img, err = webp.Decode(r())
	return img, "webp", err
}

func isWebP(data []byte) bool {
	return len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP"
}

func validateGemma4ImageDimensions(width, height int) error {
	if width <= 0 || height <= 0 {
		return fmt.Errorf("invalid Gemma4 image dimensions %dx%d", width, height)
	}
	if width > maxGemma4ImageDimension || height > maxGemma4ImageDimension {
		return fmt.Errorf("Gemma4 image dimensions %dx%d exceed limit %dx%d", width, height, maxGemma4ImageDimension, maxGemma4ImageDimension)
	}
	if int64(width)*int64(height) > maxGemma4ImagePixels {
		return fmt.Errorf("Gemma4 image has %d pixels, limit %d", int64(width)*int64(height), maxGemma4ImagePixels)
	}
	return nil
}

func gemma4ResizeDimensions(width, height, patchSize, maxPatches, poolingKernelSize int) (int, int, error) {
	if width <= 0 || height <= 0 || patchSize <= 0 || maxPatches <= 0 || poolingKernelSize <= 0 {
		return 0, 0, errors.New("invalid Gemma4 resize parameters")
	}
	targetPixels := int64(maxPatches) * int64(patchSize) * int64(patchSize)
	sourcePixels := int64(height) * int64(width)
	// CatmullRom.Scale is not context-aware, so keep every resize below a
	// fixed work bound independent of checkpoint-controlled configuration.
	if targetPixels <= 0 || targetPixels > maxGemma4ResizePixels {
		return 0, 0, fmt.Errorf("Gemma4 resize target has %d pixels, limit %d", targetPixels, maxGemma4ResizePixels)
	}
	targetPx := float64(targetPixels)
	factor := math.Sqrt(targetPx / float64(sourcePixels))
	sideMult64 := int64(poolingKernelSize) * int64(patchSize)
	if sideMult64 <= 0 || sideMult64 > maxGemma4ImageDimension {
		return 0, 0, errors.New("invalid Gemma4 pooled patch size")
	}
	sideMult := int(sideMult64)

	targetH := int(math.Floor(factor*float64(height)/float64(sideMult))) * sideMult
	targetW := int(math.Floor(factor*float64(width)/float64(sideMult))) * sideMult

	if targetH == 0 && targetW == 0 {
		return 0, 0, errors.New("attempting to resize Gemma4 image to 0 x 0")
	}

	maxSideLength64 := (int64(maxPatches) / (int64(poolingKernelSize) * int64(poolingKernelSize))) * sideMult64
	if maxSideLength64 <= 0 || maxSideLength64 > maxGemma4ImageDimension {
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

func imageToCHWFloat32(img image.Image) []float32 {
	pixels, _ := imageToCHWFloat32Context(context.Background(), img)
	return pixels
}

func imageToCHWFloat32Context(ctx context.Context, img image.Image) ([]float32, error) {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	plane64 := int64(height) * int64(width)
	values64 := plane64 * 3
	if plane64 <= 0 || values64 > maxIntValue {
		return nil, errors.New("Gemma4 pixel allocation exceeds platform limits")
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

func imageTokenSpan(tokens []int32, imageTokenID int32) (int, int, error) {
	return mediaTokenSpan(tokens, imageTokenID)
}

func (m *VisionModel) Forward(img *gemma4ImageInput) *mlx.Array {
	pixels := mlx.FromValues(img.Pixels, 1, 3, img.Height, img.Width)
	positions := m.positionArrays(img)
	h := m.PatchEmbedder.Forward(pixels, positions)
	for _, layer := range m.Layers {
		h = layer.Forward(h, positions, m.Config)
	}
	h = m.pool(h, int32(img.PatchHeight), int32(img.PatchWidth))
	if m.Config.Standardize && m.StdBias != nil && m.StdScale != nil {
		h = mlx.Mul(mlx.Sub(h, m.StdBias), m.StdScale)
	}
	return h
}

func (m *UnifiedVisionEmbedder) Forward(img *gemma4ImageInput) *mlx.Array {
	patches := mlx.FromValues(img.Patches, 1, img.SoftTokens, int(m.PatchDim))
	hidden := m.PatchLN1.Forward(patches)
	hidden = m.PatchDense.Forward(hidden)
	hidden = m.PatchLN2.Forward(hidden)

	positions := mlx.FromValues(img.Positions, 1, img.SoftTokens, 2)
	x := mlx.Squeeze(mlx.SliceStartStop(positions, []int32{0, 0, 0}, []int32{1, int32(img.SoftTokens), 1}), 2)
	y := mlx.Squeeze(mlx.SliceStartStop(positions, []int32{0, 0, 1}, []int32{1, int32(img.SoftTokens), 2}), 2)
	tableX := mlx.Squeeze(mlx.SliceStartStop(m.PosEmbedding,
		[]int32{0, 0, 0}, []int32{int32(m.PosEmbedding.Dim(0)), 1, int32(m.PosEmbedding.Dim(2))}), 1)
	tableY := mlx.Squeeze(mlx.SliceStartStop(m.PosEmbedding,
		[]int32{0, 1, 0}, []int32{int32(m.PosEmbedding.Dim(0)), 2, int32(m.PosEmbedding.Dim(2))}), 1)
	positionEmbeddings := mlx.Add(tableX.TakeAxis(x, 0), tableY.TakeAxis(y, 0)).AsType(hidden.DType())
	return m.PosNorm.Forward(mlx.Add(hidden, positionEmbeddings))
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

// Compile-time checks for the media hooks used by x/mlxrunner.
var (
	_ interface {
		PrepareMediaPrompt(context.Context, string, []llm.MediaData) (*batch.PreparedInput, error)
	} = (*Model)(nil)
	_ interface {
		PrepareMediaEmbeddings(*batch.PreparedInput) error
	} = (*Model)(nil)
)
