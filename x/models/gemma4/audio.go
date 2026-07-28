package gemma4

// Gemma 4 E2B/E4B use a Universal Speech Model Conformer. The unified 12B
// architecture instead projects raw 640-sample waveform blocks directly into
// the language model. Both paths are adapted from the MIT-licensed MLX-VLM
// implementation pinned in docs/third-party/mlx-vlm.md and cross-checked
// against Transformers.

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	gemma4metadata "github.com/ollama/ollama/x/models/gemma4/metadata"
	"github.com/ollama/ollama/x/models/nn"
)

type AudioConfig struct {
	ModelType               string  `json:"model_type"`
	AudioEmbedDim           int32   `json:"audio_embed_dim"`
	AudioSamplesPerToken    int32   `json:"audio_samples_per_token"`
	AttentionChunkSize      int32   `json:"attention_chunk_size"`
	AttentionContextLeft    int32   `json:"attention_context_left"`
	AttentionContextRight   int32   `json:"attention_context_right"`
	AttentionInvalidLogit   float32 `json:"attention_invalid_logits_value"`
	AttentionLogitCap       float32 `json:"attention_logit_cap"`
	ConvKernelSize          int32   `json:"conv_kernel_size"`
	GradientClipping        float32 `json:"gradient_clipping"`
	HiddenSize              int32   `json:"hidden_size"`
	NumAttentionHeads       int32   `json:"num_attention_heads"`
	NumHiddenLayers         int32   `json:"num_hidden_layers"`
	OutputProjDims          int32   `json:"output_proj_dims"`
	ResidualWeight          float32 `json:"residual_weight"`
	RMSNormEps              float32 `json:"rms_norm_eps"`
	SubsamplingConvChannels []int32 `json:"subsampling_conv_channels"`
	UseClippedLinears       bool    `json:"use_clipped_linears"`
}

func parseAudioConfig(configData []byte) (*AudioConfig, error) {
	var wrapped struct {
		AudioConfig *AudioConfig `json:"audio_config"`
	}
	if err := json.Unmarshal(configData, &wrapped); err != nil {
		return nil, fmt.Errorf("parse Gemma4 audio config: %w", err)
	}
	if wrapped.AudioConfig == nil {
		return nil, nil
	}
	cfg := wrapped.AudioConfig
	if cfg.unified() {
		if cfg.AudioEmbedDim != 640 || cfg.AudioSamplesPerToken != 640 || cfg.HiddenSize != 640 ||
			cfg.OutputProjDims != 640 || cfg.RMSNormEps <= 0 {
			return nil, errors.New("invalid Gemma4 unified audio configuration")
		}
		return cfg, nil
	}
	if cfg.ModelType != "gemma4_audio" {
		return nil, fmt.Errorf("unsupported Gemma4 audio model type %q", cfg.ModelType)
	}
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.NumAttentionHeads <= 0 ||
		cfg.HiddenSize%cfg.NumAttentionHeads != 0 || cfg.OutputProjDims <= 0 ||
		cfg.AttentionChunkSize <= 0 || cfg.AttentionContextLeft <= 0 || cfg.AttentionContextRight < 0 ||
		cfg.ConvKernelSize <= 0 || cfg.ConvKernelSize%2 == 0 || len(cfg.SubsamplingConvChannels) != 2 ||
		cfg.SubsamplingConvChannels[0] <= 0 || cfg.SubsamplingConvChannels[1] <= 0 ||
		cfg.RMSNormEps <= 0 || cfg.ResidualWeight <= 0 || cfg.GradientClipping <= 0 ||
		cfg.AttentionLogitCap <= 0 || cfg.AttentionInvalidLogit >= 0 {
		return nil, errors.New("invalid Gemma4 audio configuration")
	}
	return cfg, nil
}

func (c *AudioConfig) unified() bool {
	return c != nil && c.ModelType == "gemma4_unified_audio"
}

type audioConvBlock struct {
	Weight *mlx.Array
	Norm   *nn.LayerNorm
}

type audioFeedForward struct {
	PreNorm, PostNorm *nn.RMSNorm
	Up, Down          *ClippableLinear
	Config            *AudioConfig
}

type audioAttention struct {
	Q, K, V, Output *ClippableLinear
	RelativeK       nn.LinearLayer
	RelativeKDType  mlx.DType
	PerDimScale     *mlx.Array
	Config          *AudioConfig
}

type audioLightConv struct {
	PreNorm, ConvNorm *nn.RMSNorm
	Start, End        *ClippableLinear
	DepthwiseWeight   *mlx.Array
	Config            *AudioConfig
}

type audioConformerBlock struct {
	FeedForward1, FeedForward2 *audioFeedForward
	Attention                  *audioAttention
	LightConv                  *audioLightConv
	PreAttentionNorm           *nn.RMSNorm
	PostAttentionNorm          *nn.RMSNorm
	OutputNorm                 *nn.RMSNorm
	Config                     *AudioConfig
}

type AudioModel struct {
	Conv0, Conv1 *audioConvBlock
	InputProj    nn.LinearLayer
	Layers       []*audioConformerBlock
	OutputProj   nn.LinearLayer
	Config       *AudioConfig
}

func hasCompleteGemma4AudioWeights(tensors map[string]*mlx.Array, cfg *AudioConfig, textHidden int32) bool {
	if cfg == nil {
		return false
	}
	names := make([]string, 0, len(tensors))
	for name, tensor := range tensors {
		if tensor != nil {
			names = append(names, name)
		}
	}
	return gemma4metadata.ValidateAudioTensors(audioMetadataConfig(cfg, textHidden), names) == nil
}

func audioMetadataConfig(cfg *AudioConfig, textHidden int32) gemma4metadata.ConfigFile {
	channels := make([]int, len(cfg.SubsamplingConvChannels))
	for i, channel := range cfg.SubsamplingConvChannels {
		channels[i] = int(channel)
	}
	return gemma4metadata.ConfigFile{
		TextConfig: gemma4metadata.TextConfig{HiddenSize: int(textHidden)},
		AudioConfig: &gemma4metadata.AudioConfig{
			ModelType: cfg.ModelType, AudioEmbedDim: int(cfg.AudioEmbedDim),
			AudioSamplesPerToken: int(cfg.AudioSamplesPerToken),
			AttentionChunkSize:   int(cfg.AttentionChunkSize), AttentionContextLeft: int(cfg.AttentionContextLeft),
			AttentionContextRight: int(cfg.AttentionContextRight), AttentionInvalidLogit: cfg.AttentionInvalidLogit,
			AttentionLogitCap: cfg.AttentionLogitCap, ConvKernelSize: int(cfg.ConvKernelSize),
			GradientClipping: cfg.GradientClipping,
			HiddenSize:       int(cfg.HiddenSize), NumAttentionHeads: int(cfg.NumAttentionHeads),
			NumHiddenLayers: int(cfg.NumHiddenLayers), OutputProjDims: int(cfg.OutputProjDims),
			ResidualWeight: cfg.ResidualWeight, RMSNormEps: cfg.RMSNormEps,
			SubsamplingConvChannels: channels, UseClippedLinears: cfg.UseClippedLinears,
		},
	}
}

func validateGemma4AudioWeights(tensors map[string]*mlx.Array, cfg *AudioConfig, textHidden int32) error {
	required, err := gemma4metadata.RequiredAudioTensorShapes(audioMetadataConfig(cfg, textHidden))
	if err != nil {
		return err
	}
	for name, shape := range required {
		tensor := tensors[name]
		if tensor == nil {
			return fmt.Errorf("missing Gemma4 audio tensor %s", name)
		}
		want := make([]int, len(shape))
		for i, dim := range shape {
			want[i] = int(dim)
		}
		if !equalIntShape(tensor.Dims(), want) {
			return fmt.Errorf("Gemma4 audio tensor %s shape %v, want %v", name, tensor.Dims(), want)
		}
		switch tensor.DType() {
		case mlx.DTypeBFloat16, mlx.DTypeFloat16, mlx.DTypeFloat32:
		default:
			return fmt.Errorf("Gemma4 audio tensor %s has unsupported dtype %s", name, tensor.DType())
		}
	}
	return nil
}

func equalIntShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func loadAudioModel(tensors map[string]*mlx.Array, cfg *AudioConfig, textHidden int32, groupSize, bits int, mode string, tq map[string]*model.TensorQuantInfo) (*AudioModel, error) {
	if err := validateGemma4AudioWeights(tensors, cfg, textHidden); err != nil {
		return nil, err
	}
	const prefix = "model.audio_tower."
	linears := model.NewLinearFactory(tensors, groupSize, bits, mode, tq)
	loadConv := func(path string) (*audioConvBlock, error) {
		weight := tensors[path+".conv.weight"]
		norm := tensors[path+".norm.weight"]
		if weight == nil || norm == nil {
			return nil, fmt.Errorf("missing Gemma4 audio convolution tensors at %s", path)
		}
		// Safetensors uses PyTorch OIHW; MLX Conv2d expects OHWI.
		weight = mlx.Transpose(weight, 0, 2, 3, 1)
		return &audioConvBlock{Weight: weight, Norm: &nn.LayerNorm{Weight: norm, Eps: cfg.RMSNormEps}}, nil
	}
	conv0, err := loadConv(prefix + "subsample_conv_projection.layer0")
	if err != nil {
		return nil, err
	}
	conv1, err := loadConv(prefix + "subsample_conv_projection.layer1")
	if err != nil {
		return nil, err
	}
	inputProj := linears.Make(prefix + "subsample_conv_projection.input_proj_linear")
	outputProj := linears.Make(prefix + "output_proj")
	if inputProj == nil || outputProj == nil {
		return nil, errors.New("missing Gemma4 audio input or output projection")
	}

	out := &AudioModel{Conv0: conv0, Conv1: conv1, InputProj: inputProj, OutputProj: outputProj, Config: cfg}
	out.Layers = make([]*audioConformerBlock, cfg.NumHiddenLayers)
	for i := range cfg.NumHiddenLayers {
		path := fmt.Sprintf("%slayers.%d.", prefix, i)
		makeNorm := func(name string) (*nn.RMSNorm, error) {
			weight := tensors[path+name+".weight"]
			if weight == nil {
				return nil, fmt.Errorf("missing Gemma4 audio tensor %s.weight", path+name)
			}
			return nn.NewRMSNorm(weight, cfg.RMSNormEps), nil
		}
		makeFF := func(name string) (*audioFeedForward, error) {
			pre, err := makeNorm(name + ".pre_layer_norm")
			if err != nil {
				return nil, err
			}
			post, err := makeNorm(name + ".post_layer_norm")
			if err != nil {
				return nil, err
			}
			up := makeClippableLinear(tensors, linears, path+name+".ffw_layer_1", cfg.UseClippedLinears)
			down := makeClippableLinear(tensors, linears, path+name+".ffw_layer_2", cfg.UseClippedLinears)
			if up == nil || down == nil {
				return nil, fmt.Errorf("missing Gemma4 audio feed-forward tensors at %s%s", path, name)
			}
			return &audioFeedForward{PreNorm: pre, PostNorm: post, Up: up, Down: down, Config: cfg}, nil
		}
		ff1, err := makeFF("feed_forward1")
		if err != nil {
			return nil, err
		}
		ff2, err := makeFF("feed_forward2")
		if err != nil {
			return nil, err
		}
		preAttn, err := makeNorm("norm_pre_attn")
		if err != nil {
			return nil, err
		}
		postAttn, err := makeNorm("norm_post_attn")
		if err != nil {
			return nil, err
		}
		outNorm, err := makeNorm("norm_out")
		if err != nil {
			return nil, err
		}
		convPre, err := makeNorm("lconv1d.pre_layer_norm")
		if err != nil {
			return nil, err
		}
		convNorm, err := makeNorm("lconv1d.conv_norm")
		if err != nil {
			return nil, err
		}
		depthwise := tensors[path+"lconv1d.depthwise_conv1d.weight"]
		if depthwise == nil {
			return nil, fmt.Errorf("missing Gemma4 audio depthwise convolution at %s", path)
		}
		// PyTorch [channels, 1, kernel] to MLX [channels, kernel, 1].
		depthwise = mlx.Transpose(depthwise, 0, 2, 1)
		convStart := makeClippableLinear(tensors, linears, path+"lconv1d.linear_start", cfg.UseClippedLinears)
		convEnd := makeClippableLinear(tensors, linears, path+"lconv1d.linear_end", cfg.UseClippedLinears)
		q := makeClippableLinear(tensors, linears, path+"self_attn.q_proj", cfg.UseClippedLinears)
		k := makeClippableLinear(tensors, linears, path+"self_attn.k_proj", cfg.UseClippedLinears)
		v := makeClippableLinear(tensors, linears, path+"self_attn.v_proj", cfg.UseClippedLinears)
		attnOut := makeClippableLinear(tensors, linears, path+"self_attn.post", cfg.UseClippedLinears)
		relativeK := linears.Make(path + "self_attn.relative_k_proj")
		perDimScale := tensors[path+"self_attn.per_dim_scale"]
		if convStart == nil || convEnd == nil || q == nil || k == nil || v == nil || attnOut == nil || relativeK == nil || perDimScale == nil {
			return nil, fmt.Errorf("missing Gemma4 audio attention or convolution tensors at %s", path)
		}
		out.Layers[i] = &audioConformerBlock{
			FeedForward1: ff1, FeedForward2: ff2,
			Attention: &audioAttention{
				Q: q, K: k, V: v, Output: attnOut, RelativeK: relativeK,
				RelativeKDType: tensors[path+"self_attn.relative_k_proj.weight"].DType(),
				PerDimScale:    perDimScale, Config: cfg,
			},
			LightConv:        &audioLightConv{PreNorm: convPre, ConvNorm: convNorm, Start: convStart, End: convEnd, DepthwiseWeight: depthwise, Config: cfg},
			PreAttentionNorm: preAttn, PostAttentionNorm: postAttn, OutputNorm: outNorm, Config: cfg,
		}
	}
	return out, nil
}

func audioValidityArray(valid []bool, rank int) *mlx.Array {
	shape := make([]int, rank)
	shape[0], shape[1] = 1, len(valid)
	for i := 2; i < rank; i++ {
		shape[i] = 1
	}
	values := make([]float32, len(valid))
	for i, ok := range valid {
		if ok {
			values[i] = 1
		}
	}
	return mlx.FromValues(values, shape...)
}

func (b *audioConvBlock) Forward(x *mlx.Array, valid []bool) (*mlx.Array, []bool) {
	x = mlx.Mul(x, audioValidityArray(valid, 4).AsType(x.DType()))
	x = mlx.PadConstant(x, []int{1, 2}, []int{1, 1}, []int{1, 1})
	x = mlx.Conv2d(x, b.Weight, 2, 2, 0, 0, 1, 1, 1)
	x = mlx.ReLU(b.Norm.Forward(x))
	return x, downsampleGemma4AudioMask(valid, 1)
}

func (m *AudioModel) Forward(features *mlx.Array, input *gemma4AudioInput) *mlx.Array {
	x := mlx.Reshape(features, 1, int32(input.Frames), 128)
	valid := append([]bool(nil), input.FeatureMask...)
	x = mlx.ExpandDims(x, -1)
	x, valid = m.Conv0.Forward(x, valid)
	x, valid = m.Conv1.Forward(x, valid)
	x = mlx.Reshape(x, 1, int32(x.Dim(1)), int32(x.Dim(2)*x.Dim(3)))
	x = m.InputProj.Forward(x)
	for _, layer := range m.Layers {
		x = layer.Forward(x, valid)
	}
	x = m.OutputProj.Forward(x)
	x = mlx.Mul(x, audioValidityArray(valid, 3).AsType(x.DType()))
	return mlx.SliceStartStop(x, []int32{0, 0, 0}, []int32{1, int32(input.SoftTokens), int32(x.Dim(2))})
}

func (f *audioFeedForward) Forward(x *mlx.Array) *mlx.Array {
	residual := x
	x = mlx.Clamp(x, -f.Config.GradientClipping, f.Config.GradientClipping)
	x = f.PreNorm.Forward(x, 0)
	x = f.Up.Forward(x)
	x = mlx.Mul(x, mlx.Sigmoid(x)) // SiLU
	x = f.Down.Forward(x)
	x = mlx.Clamp(x, -f.Config.GradientClipping, f.Config.GradientClipping)
	x = f.PostNorm.Forward(x, 0)
	return mlx.Add(residual, mlx.MulScalar(x, f.Config.ResidualWeight))
}

func padAudioTime(x *mlx.Array, left, right int) *mlx.Array {
	if left == 0 && right == 0 {
		return x
	}
	return mlx.PadConstant(x, []int{1}, []int{left}, []int{right})
}

func (a *audioAttention) relativeLogits(q *mlx.Array, blocks, chunk, context int) *mlx.Array {
	cfg := a.Config
	headDim := cfg.HiddenSize / cfg.NumAttentionHeads
	span := cfg.AttentionContextLeft + cfg.AttentionContextRight
	half := cfg.HiddenSize / 2
	values := make([]float32, int(span*cfg.HiddenSize))
	logIncrement := math.Log(10000) / float64(max(half-1, 1))
	for p := range span {
		position := float64(cfg.AttentionContextLeft - 1 - p)
		for d := range half {
			angle := position * math.Exp(-float64(d)*logIncrement)
			values[int(p*cfg.HiddenSize+d)] = float32(math.Sin(angle))
			values[int(p*cfg.HiddenSize+half+d)] = float32(math.Cos(angle))
		}
	}
	position := mlx.FromValues(values, int(span), int(cfg.HiddenSize)).AsType(a.RelativeKDType)
	position = a.RelativeK.Forward(position)
	position = position.AsType(q.DType())
	position = mlx.Reshape(position, span, cfg.NumAttentionHeads, headDim)
	position = mlx.Transpose(position, 1, 2, 0)
	qFlat := mlx.Reshape(q, 1, cfg.NumAttentionHeads, int32(blocks*chunk), headDim)
	term := mlx.Matmul(qFlat, position)
	term = mlx.Reshape(term, 1, cfg.NumAttentionHeads, int32(blocks), int32(chunk), span)
	pad := context + 1 - int(span)
	term = mlx.PadConstant(term, []int{4}, []int{0}, []int{pad})
	term = mlx.Reshape(term, 1, cfg.NumAttentionHeads, int32(blocks), int32(chunk*(context+1)))
	term = mlx.SliceStartStop(term, []int32{0, 0, 0, 0}, []int32{1, cfg.NumAttentionHeads, int32(blocks), int32(chunk * context)})
	return mlx.Reshape(term, 1, cfg.NumAttentionHeads, int32(blocks), int32(chunk), int32(context))
}

func audioAttentionMask(valid []bool, blocks, chunk, context, left, future int) *mlx.Array {
	values := audioAttentionMaskValues(valid, blocks, chunk, context, left, future)
	return mlx.FromValues(values, 1, 1, blocks, chunk, context)
}

func audioAttentionMaskValues(valid []bool, blocks, chunk, context, left, future int) []bool {
	values := make([]bool, blocks*chunk*context)
	for u := range blocks {
		for w := range chunk {
			for c := range context {
				actual := u*chunk + c - left
				ok := c >= w && c <= w+left+future && actual >= 0 && actual < len(valid) && valid[actual]
				values[(u*chunk+w)*context+c] = ok
			}
		}
	}
	return values
}

func (a *audioAttention) Forward(x *mlx.Array, valid []bool) *mlx.Array {
	cfg := a.Config
	length := x.Dim(1)
	heads, headDim := int(cfg.NumAttentionHeads), int(cfg.HiddenSize/cfg.NumAttentionHeads)
	chunk := int(cfg.AttentionChunkSize)
	left, future := int(cfg.AttentionContextLeft-1), int(cfg.AttentionContextRight)
	context := chunk + left + future
	blocks := (length + chunk - 1) / chunk
	q := mlx.Reshape(a.Q.Forward(x).AsType(mlx.DTypeFloat32), 1, int32(length), cfg.NumAttentionHeads, int32(headDim))
	k := mlx.Reshape(a.K.Forward(x).AsType(mlx.DTypeFloat32), 1, int32(length), cfg.NumAttentionHeads, int32(headDim))
	v := mlx.Reshape(a.V.Forward(x).AsType(mlx.DTypeFloat32), 1, int32(length), cfg.NumAttentionHeads, int32(headDim))
	qScale := float32(math.Pow(float64(headDim), -0.5) / math.Log(2))
	q = mlx.Mul(q, mlx.MulScalar(mlx.Softplus(a.PerDimScale).AsType(q.DType()), qScale))
	k = mlx.MulScalar(k, float32(math.Log(1+math.E)/math.Log(2)))
	q = padAudioTime(q, 0, blocks*chunk-length)
	q = mlx.Reshape(q, 1, int32(blocks), int32(chunk), int32(heads), int32(headDim))
	q = mlx.Transpose(q, 0, 3, 1, 2, 4)
	indices := make([]int32, blocks*context)
	for u := range blocks {
		for c := range context {
			indices[u*context+c] = int32(u*chunk + c)
		}
	}
	indicesArray := mlx.FromValues(indices, blocks, context)
	k = mlx.Take(padAudioTime(k, left, future+chunk-1), indicesArray, 1)
	v = mlx.Take(padAudioTime(v, left, future+chunk-1), indicesArray, 1)
	k = mlx.Transpose(k, 0, 3, 1, 4, 2)
	content := mlx.Matmul(q, k)
	logits := mlx.Add(content, a.relativeLogits(q, blocks, chunk, context))
	logits = mlx.MulScalar(mlx.DivScalar(logits, cfg.AttentionLogitCap).Tanh(), cfg.AttentionLogitCap)
	condition := audioAttentionMask(valid, blocks, chunk, context, left, future)
	logits = mlx.Where(condition, logits, mlx.FromValue(cfg.AttentionInvalidLogit))
	probs := mlx.SoftmaxAxis(logits, -1, true)
	v = mlx.Transpose(v, 0, 3, 1, 2, 4)
	result := mlx.Matmul(probs, v)
	result = mlx.Transpose(result, 0, 2, 3, 1, 4)
	result = mlx.Reshape(result, 1, int32(blocks*chunk), cfg.HiddenSize)
	result = mlx.SliceStartStop(result, []int32{0, 0, 0}, []int32{1, int32(length), cfg.HiddenSize})
	return a.Output.Forward(result)
}

func (c *audioLightConv) Forward(x *mlx.Array) *mlx.Array {
	residual := x
	x = c.PreNorm.Forward(x, 0)
	x = mlx.GLU(c.Start.Forward(x))
	x = padAudioTime(x, int(c.Config.ConvKernelSize-1), 0)
	x = mlx.Conv1d(x, c.DepthwiseWeight, nil, 1, 0, 1, c.Config.HiddenSize)
	x = mlx.Clamp(x, -c.Config.GradientClipping, c.Config.GradientClipping)
	x = c.ConvNorm.Forward(x, 0)
	x = mlx.Mul(x, mlx.Sigmoid(x))
	return mlx.Add(c.End.Forward(x), residual)
}

func (b *audioConformerBlock) Forward(x *mlx.Array, valid []bool) *mlx.Array {
	x = b.FeedForward1.Forward(x)
	residual := x
	x = mlx.Clamp(x, -b.Config.GradientClipping, b.Config.GradientClipping)
	x = b.PreAttentionNorm.Forward(x, 0)
	x = b.Attention.Forward(x, valid)
	x = mlx.Clamp(x, -b.Config.GradientClipping, b.Config.GradientClipping)
	x = mlx.Add(residual, b.PostAttentionNorm.Forward(x, 0))
	x = mlx.Mul(x, audioValidityArray(valid, 3).AsType(x.DType()))
	x = b.LightConv.Forward(x)
	x = b.FeedForward2.Forward(x)
	x = mlx.Clamp(x, -b.Config.GradientClipping, b.Config.GradientClipping)
	return b.OutputNorm.Forward(x, 0)
}
