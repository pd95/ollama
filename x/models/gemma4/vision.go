package gemma4

// Portions of the Gemma 4 vision preprocessing and embedding flow are adapted
// from MLX-VLM's MIT-licensed Gemma 4 implementation.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"strings"

	xdraw "golang.org/x/image/draw"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

const (
	defaultGemma4BOIToken   = "<|image>"
	defaultGemma4ImageToken = "<|image|>"
	defaultGemma4EOIToken   = "<image|>"
)

type VisionRopeParameters struct {
	RopeTheta float32 `json:"rope_theta"`
	RopeType  string  `json:"rope_type"`
}

type VisionConfig struct {
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
	RopeParameters        VisionRopeParameters `json:"rope_parameters"`
}

type gemma4MediaTokens struct {
	BOI   string
	Image string
	EOI   string
}

type gemma4ImageInput struct {
	Pixels      []float32
	Width       int
	Height      int
	PatchWidth  int
	PatchHeight int
	SoftTokens  int
}

type gemma4MediaPayload struct {
	Image      *gemma4ImageInput
	ImageStart int
	ImageEnd   int
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
		cfg.DefaultOutputLength = 280
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
	}
}

func parseGemma4MediaTokens(data []byte, fallback gemma4MediaTokens) gemma4MediaTokens {
	var cfg struct {
		BOIToken   string `json:"boi_token"`
		ImageToken string `json:"image_token"`
		EOIToken   string `json:"eoi_token"`
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
	return fallback
}

func hasGemma4VisionWeights(tensors map[string]*mlx.Array) bool {
	return firstNonNil(tensors,
		"vision_tower.patch_embedder.input_proj.weight",
		"model.vision_tower.patch_embedder.input_proj.weight",
	) != nil
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

func (m *Model) PrepareMediaPrompt(prompt string, media []llm.MediaData) (*batch.PreparedInput, error) {
	if len(media) != 1 {
		return nil, fmt.Errorf("Gemma4 MLX currently supports exactly one image per request, got %d", len(media))
	}
	if media[0].Kind != llm.MediaKindImage {
		return nil, fmt.Errorf("Gemma4 MLX currently supports image inputs only")
	}
	if m.VisionConfig == nil {
		return nil, errors.New("Gemma4 MLX model has no vision_config")
	}

	img, err := preprocessGemma4Image(media[0].Data, m.VisionConfig, int(m.VisionSoftTokens))
	if err != nil {
		return nil, err
	}

	marker := fmt.Sprintf("[img-%d]", media[0].ID)
	if strings.Count(prompt, marker) != 1 {
		return nil, fmt.Errorf("expected one %s marker in Gemma4 prompt", marker)
	}
	sequence := m.mediaTokens.BOI + strings.Repeat(m.mediaTokens.Image, img.SoftTokens) + m.mediaTokens.EOI
	expanded := strings.Replace(prompt, marker, sequence, 1)
	tokens := m.tok.Encode(expanded, m.tok.AddBOS())
	start, end, err := imageTokenSpan(tokens, m.ImageTokenIDValue)
	if err != nil {
		return nil, err
	}
	if got := end - start; got != img.SoftTokens {
		return nil, fmt.Errorf("Gemma4 image token count = %d, want %d", got, img.SoftTokens)
	}

	pleIDs := append([]int32(nil), tokens...)
	for i := start; i < end; i++ {
		pleIDs[i] = 0
	}
	return &batch.PreparedInput{
		Tokens:      tokens,
		PLEInputIDs: pleIDs,
		Payload: &gemma4MediaPayload{
			Image:      img,
			ImageStart: start,
			ImageEnd:   end,
		},
	}, nil
}

func (m *Model) PrepareMediaEmbeddings(prepared *batch.PreparedInput) error {
	payload, ok := prepared.Payload.(*gemma4MediaPayload)
	if !ok || payload == nil || payload.Image == nil {
		return errors.New("invalid Gemma4 media payload")
	}
	if m.Vision == nil || m.EmbedVision == nil {
		return errors.New("Gemma4 MLX vision weights are not loaded; recreate or pull the model so it includes Gemma 4 vision tensor layers")
	}

	tokens := mlx.FromValues(prepared.Tokens, 1, len(prepared.Tokens))
	embeddings := m.TokenEmbeddings(tokens)
	imageFeatures := m.EmbedVision.Forward(m.Vision.Forward(payload.Image))
	imageFeatures = imageFeatures.AsType(embeddings.DType())

	if imageFeatures.Dim(1) != payload.ImageEnd-payload.ImageStart {
		return fmt.Errorf("Gemma4 image feature length = %d, want %d", imageFeatures.Dim(1), payload.ImageEnd-payload.ImageStart)
	}

	parts := make([]*mlx.Array, 0, 3)
	if payload.ImageStart > 0 {
		parts = append(parts, mlx.SliceStartStop(embeddings,
			[]int32{0, 0, 0},
			[]int32{1, int32(payload.ImageStart), int32(embeddings.Dim(2))},
		))
	}
	parts = append(parts, imageFeatures)
	if payload.ImageEnd < len(prepared.Tokens) {
		parts = append(parts, mlx.SliceStartStop(embeddings,
			[]int32{0, int32(payload.ImageEnd), 0},
			[]int32{1, int32(len(prepared.Tokens)), int32(embeddings.Dim(2))},
		))
	}
	prepared.InputEmbeddings = mlx.Concatenate(parts, 1)
	return nil
}

func preprocessGemma4Image(data []byte, cfg *VisionConfig, maxSoftTokens int) (*gemma4ImageInput, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode Gemma4 image: %w", err)
	}
	b := img.Bounds()
	width, height := b.Dx(), b.Dy()
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid Gemma4 image dimensions %dx%d", width, height)
	}

	patchSize := int(cfg.PatchSize)
	pooling := int(cfg.PoolingKernelSize)
	if patchSize <= 0 || pooling <= 0 {
		return nil, fmt.Errorf("invalid Gemma4 vision patch configuration")
	}
	if maxSoftTokens <= 0 {
		maxSoftTokens = int(cfg.DefaultOutputLength)
	}
	maxPatches := maxSoftTokens * pooling * pooling
	targetW, targetH, err := gemma4ResizeDimensions(width, height, patchSize, maxPatches, pooling)
	if err != nil {
		return nil, err
	}

	resized := img
	if targetW != width || targetH != height {
		dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
		xdraw.CatmullRom.Scale(dst, dst.Bounds(), img, b, xdraw.Over, nil)
		resized = dst
	}

	pixels := imageToCHWFloat32(resized)
	patchW := targetW / patchSize
	patchH := targetH / patchSize
	softTokens := (patchW * patchH) / (pooling * pooling)
	if softTokens <= 0 || softTokens > maxSoftTokens {
		return nil, fmt.Errorf("Gemma4 image produced %d soft tokens, limit %d", softTokens, maxSoftTokens)
	}

	return &gemma4ImageInput{
		Pixels:      pixels,
		Width:       targetW,
		Height:      targetH,
		PatchWidth:  patchW,
		PatchHeight: patchH,
		SoftTokens:  softTokens,
	}, nil
}

func gemma4ResizeDimensions(width, height, patchSize, maxPatches, poolingKernelSize int) (int, int, error) {
	targetPx := float64(maxPatches * patchSize * patchSize)
	factor := math.Sqrt(targetPx / float64(height*width))
	sideMult := poolingKernelSize * patchSize

	targetH := int(math.Floor(factor*float64(height)/float64(sideMult))) * sideMult
	targetW := int(math.Floor(factor*float64(width)/float64(sideMult))) * sideMult

	if targetH == 0 && targetW == 0 {
		return 0, 0, errors.New("attempting to resize Gemma4 image to 0 x 0")
	}

	maxSideLength := (maxPatches / (poolingKernelSize * poolingKernelSize)) * sideMult
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
	return targetW, targetH, nil
}

func imageToCHWFloat32(img image.Image) []float32 {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	out := make([]float32, 3*height*width)
	plane := height * width
	for y := range height {
		for x := range width {
			r, g, blue, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			i := y*width + x
			out[i] = float32(r) / 65535
			out[plane+i] = float32(g) / 65535
			out[2*plane+i] = float32(blue) / 65535
		}
	}
	return out
}

func imageTokenSpan(tokens []int32, imageTokenID int32) (int, int, error) {
	start, end := -1, -1
	for i, tok := range tokens {
		if tok != imageTokenID {
			continue
		}
		if start == -1 {
			start = i
		}
		end = i + 1
	}
	if start == -1 {
		return 0, 0, fmt.Errorf("Gemma4 prompt contains no image token id %d", imageTokenID)
	}
	for i := start; i < end; i++ {
		if tokens[i] != imageTokenID {
			return 0, 0, errors.New("Gemma4 image tokens are not contiguous")
		}
	}
	return start, end, nil
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
		PrepareMediaPrompt(string, []llm.MediaData) (*batch.PreparedInput, error)
	} = (*Model)(nil)
	_ interface {
		PrepareMediaEmbeddings(*batch.PreparedInput) error
	} = (*Model)(nil)
)
