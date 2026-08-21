package apertus

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"slices"
	"strings"

	_ "golang.org/x/image/webp"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const (
	maxApertusImageBytes     = 32 << 20
	maxApertusImageDimension = 16_384
	maxApertusImagePixels    = 64 << 20
	visionCodebookChunk      = 4096
)

type VisionTokenizerConfig struct {
	AttnResolutions   []int32 `json:"attn_resolutions"`
	BaseChannels      int32   `json:"base_channels"`
	ChannelMultiplier []int32 `json:"channel_multiplier"`
	CodebookSize      int32   `json:"codebook_size"`
	EmbedDim          int32   `json:"embed_dim"`
	InChannels        int32   `json:"in_channels"`
	LatentChannels    int32   `json:"latent_channels"`
	NumResBlocks      int32   `json:"num_res_blocks"`
	Resolution        int32   `json:"resolution"`
}

func (c VisionTokenizerConfig) validate() error {
	if c.BaseChannels != 256 || !slices.Equal(c.ChannelMultiplier, []int32{1, 1, 2, 2, 4}) ||
		c.CodebookSize != 131072 || c.EmbedDim != 256 || c.InChannels != 3 || c.LatentChannels != 256 ||
		c.NumResBlocks != 4 || c.Resolution != 256 || !slices.Equal(c.AttnResolutions, []int32{16}) {
		return errors.New("unsupported Apertus 1.5 vision tokenizer configuration")
	}
	return nil
}

type apertureConv2D struct {
	weight *mlx.Array
	bias   *mlx.Array
	stride int32
	pad    int32
}

func newApertureConv2D(tensors map[string]*mlx.Array, path string, stride, pad int32) (*apertureConv2D, error) {
	w, b := tensors[path+".weight"], tensors[path+".bias"]
	if w == nil || b == nil || w.NumDims() != 4 || b.NumDims() != 1 || w.Dim(0) != b.Dim(0) {
		return nil, fmt.Errorf("missing or invalid Apertus vision convolution %s", path)
	}
	return &apertureConv2D{weight: mlx.Transpose(w, 0, 2, 3, 1), bias: b, stride: stride, pad: pad}, nil
}

func (c *apertureConv2D) forward(x *mlx.Array) *mlx.Array {
	y := mlx.Conv2d(x, c.weight, c.stride, c.stride, c.pad, c.pad, 1, 1, 1)
	return mlx.Add(y, mlx.Reshape(c.bias, 1, 1, 1, int32(c.bias.Dim(0))))
}

type apertureGroupNorm struct {
	weight *mlx.Array
	bias   *mlx.Array
}

func newApertureGroupNorm(tensors map[string]*mlx.Array, path string) (*apertureGroupNorm, error) {
	w, b := tensors[path+".weight"], tensors[path+".bias"]
	if w == nil || b == nil || w.NumDims() != 1 || b.NumDims() != 1 || w.Dim(0) != b.Dim(0) || w.Dim(0)%32 != 0 {
		return nil, fmt.Errorf("missing or invalid Apertus vision group norm %s", path)
	}
	return &apertureGroupNorm{weight: w, bias: b}, nil
}

func reduceGroupMean(x *mlx.Array) *mlx.Array {
	x = mlx.Mean(x, 1, true)
	x = mlx.Mean(x, 2, true)
	return mlx.Mean(x, 4, true)
}

func (n *apertureGroupNorm) forward(x *mlx.Array) *mlx.Array {
	d := x.Dims()
	c := int32(d[3])
	g := mlx.Reshape(x, int32(d[0]), int32(d[1]), int32(d[2]), 32, c/32)
	mean := reduceGroupMean(g)
	delta := mlx.Sub(g, mean)
	variance := reduceGroupMean(mlx.Mul(delta, delta))
	normalized := mlx.Div(delta, mlx.Add(variance, mlx.NewScalarArray(float32(1e-6))).Sqrt())
	normalized = mlx.Reshape(normalized, int32(d[0]), int32(d[1]), int32(d[2]), c)
	return mlx.Add(mlx.Mul(normalized, mlx.Reshape(n.weight, 1, 1, 1, c)), mlx.Reshape(n.bias, 1, 1, 1, c))
}

type apertureVisionResBlock struct {
	norm1, norm2 *apertureGroupNorm
	conv1, conv2 *apertureConv2D
	shortcut     *apertureConv2D
}

func newApertureVisionResBlock(tensors map[string]*mlx.Array, path string) (*apertureVisionResBlock, error) {
	n1, err := newApertureGroupNorm(tensors, path+".norm1")
	if err != nil {
		return nil, err
	}
	n2, err := newApertureGroupNorm(tensors, path+".norm2")
	if err != nil {
		return nil, err
	}
	c1, err := newApertureConv2D(tensors, path+".conv1", 1, 1)
	if err != nil {
		return nil, err
	}
	c2, err := newApertureConv2D(tensors, path+".conv2", 1, 1)
	if err != nil {
		return nil, err
	}
	b := &apertureVisionResBlock{norm1: n1, norm2: n2, conv1: c1, conv2: c2}
	if tensors[path+".nin_shortcut.weight"] != nil {
		b.shortcut, err = newApertureConv2D(tensors, path+".nin_shortcut", 1, 0)
		if err != nil {
			return nil, err
		}
	}
	return b, nil
}

func (b *apertureVisionResBlock) forward(x *mlx.Array) *mlx.Array {
	h := b.conv1.forward(mlx.SiLU(b.norm1.forward(x)))
	h = b.conv2.forward(mlx.SiLU(b.norm2.forward(h)))
	residual := x
	if b.shortcut != nil {
		residual = b.shortcut.forward(x)
	}
	return mlx.Add(residual, h)
}

type apertureVisionAttention struct {
	norm         *apertureGroupNorm
	q, k, v, out *apertureConv2D
}

func newApertureVisionAttention(tensors map[string]*mlx.Array, path string) (*apertureVisionAttention, error) {
	n, err := newApertureGroupNorm(tensors, path+".norm")
	if err != nil {
		return nil, err
	}
	makeConv := func(name string) (*apertureConv2D, error) { return newApertureConv2D(tensors, path+"."+name, 1, 0) }
	q, err := makeConv("q")
	if err != nil {
		return nil, err
	}
	k, err := makeConv("k")
	if err != nil {
		return nil, err
	}
	v, err := makeConv("v")
	if err != nil {
		return nil, err
	}
	out, err := makeConv("proj_out")
	if err != nil {
		return nil, err
	}
	return &apertureVisionAttention{norm: n, q: q, k: k, v: v, out: out}, nil
}

func (a *apertureVisionAttention) forward(x *mlx.Array) *mlx.Array {
	h := a.norm.forward(x)
	q, k, v := a.q.forward(h), a.k.forward(h), a.v.forward(h)
	d := q.Dims()
	b, height, width, channels := int32(d[0]), int32(d[1]), int32(d[2]), int32(d[3])
	seq := height * width
	q = mlx.Reshape(q, b, seq, channels)
	k = mlx.Reshape(k, b, seq, channels)
	v = mlx.Reshape(v, b, seq, channels)
	scores := mlx.Mul(mlx.Matmul(q, mlx.Transpose(k, 0, 2, 1)), mlx.NewScalarArray(float32(1/math.Sqrt(float64(channels)))))
	scores = mlx.SoftmaxAxis(scores, 2, true)
	h = mlx.Reshape(mlx.Matmul(scores, v), b, height, width, channels)
	return mlx.Add(x, a.out.forward(h))
}

type apertureVisionLevel struct {
	blocks     []*apertureVisionResBlock
	attn       []*apertureVisionAttention
	downsample *apertureConv2D
}

type VisionTokenizer struct {
	State                      []*mlx.Array
	config                     VisionTokenizerConfig
	convIn, convOut, quantConv *apertureConv2D
	levels                     []*apertureVisionLevel
	mid1, mid2                 *apertureVisionResBlock
	midAttn                    *apertureVisionAttention
	normOut                    *apertureGroupNorm
	codebook                   *mlx.Array
}

func loadVisionTokenizer(tensors map[string]*mlx.Array, cfg VisionTokenizerConfig) (*VisionTokenizer, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	const p = "model.vision_tokenizer."
	convIn, err := newApertureConv2D(tensors, p+"encoder.conv_in", 1, 1)
	if err != nil {
		return nil, err
	}
	convOut, err := newApertureConv2D(tensors, p+"encoder.conv_out", 1, 1)
	if err != nil {
		return nil, err
	}
	quantConv, err := newApertureConv2D(tensors, p+"quant_conv", 1, 0)
	if err != nil {
		return nil, err
	}
	normOut, err := newApertureGroupNorm(tensors, p+"encoder.norm_out")
	if err != nil {
		return nil, err
	}
	v := &VisionTokenizer{config: cfg, convIn: convIn, convOut: convOut, quantConv: quantConv, normOut: normOut}
	resolution := cfg.Resolution
	for level := range cfg.ChannelMultiplier {
		l := &apertureVisionLevel{}
		for block := range int(cfg.NumResBlocks) {
			path := fmt.Sprintf("%sencoder.down.%d.block.%d", p, level, block)
			rb, err := newApertureVisionResBlock(tensors, path)
			if err != nil {
				return nil, err
			}
			l.blocks = append(l.blocks, rb)
			if slices.Contains(cfg.AttnResolutions, resolution) {
				a, err := newApertureVisionAttention(tensors, fmt.Sprintf("%sencoder.down.%d.attn.%d", p, level, block))
				if err != nil {
					return nil, err
				}
				l.attn = append(l.attn, a)
			}
		}
		if level != len(cfg.ChannelMultiplier)-1 {
			l.downsample, err = newApertureConv2D(tensors, fmt.Sprintf("%sencoder.down.%d.downsample.conv", p, level), 2, 0)
			if err != nil {
				return nil, err
			}
			resolution /= 2
		}
		v.levels = append(v.levels, l)
	}
	v.mid1, err = newApertureVisionResBlock(tensors, p+"encoder.mid.block_1")
	if err != nil {
		return nil, err
	}
	v.midAttn, err = newApertureVisionAttention(tensors, p+"encoder.mid.attn_1")
	if err != nil {
		return nil, err
	}
	v.mid2, err = newApertureVisionResBlock(tensors, p+"encoder.mid.block_2")
	if err != nil {
		return nil, err
	}
	v.codebook = tensors[p+"quantize.embedding.weight"]
	if v.codebook == nil || !slices.Equal(v.codebook.Dims(), []int{int(cfg.CodebookSize), int(cfg.EmbedDim)}) {
		return nil, errors.New("missing or invalid Apertus vision codebook")
	}
	for name, tensor := range tensors {
		if strings.HasPrefix(name, p) && tensor != nil {
			v.State = append(v.State, tensor)
		}
	}
	addConv := func(conv *apertureConv2D) {
		if conv != nil {
			v.State = append(v.State, conv.weight)
		}
	}
	addBlock := func(block *apertureVisionResBlock) {
		if block != nil {
			addConv(block.conv1)
			addConv(block.conv2)
			addConv(block.shortcut)
		}
	}
	addAttention := func(attn *apertureVisionAttention) {
		if attn != nil {
			addConv(attn.q)
			addConv(attn.k)
			addConv(attn.v)
			addConv(attn.out)
		}
	}
	addConv(v.convIn)
	addConv(v.convOut)
	addConv(v.quantConv)
	for _, level := range v.levels {
		for _, block := range level.blocks {
			addBlock(block)
		}
		for _, attn := range level.attn {
			addAttention(attn)
		}
		addConv(level.downsample)
	}
	addBlock(v.mid1)
	addAttention(v.midAttn)
	addBlock(v.mid2)
	return v, nil
}

func (v *VisionTokenizer) encode(data *mlx.Array, width, height int) (*mlx.Array, error) {
	return v.encodeStaged(data, width, height, nil)
}

func (v *VisionTokenizer) encodeStaged(data *mlx.Array, width, height int, materialize func(...*mlx.Array)) (*mlx.Array, error) {
	x := mlx.Reshape(data, 1, int32(height), int32(width), 3)
	h := v.convIn.forward(x)
	if materialize != nil {
		materialize(h)
	}
	for _, level := range v.levels {
		for i, block := range level.blocks {
			h = block.forward(h)
			if materialize != nil {
				materialize(h)
			}
			if len(level.attn) > 0 {
				h = level.attn[i].forward(h)
				if materialize != nil {
					materialize(h)
				}
			}
		}
		if level.downsample != nil {
			h = mlx.PadConstant(h, []int{1, 2}, []int{0, 0}, []int{1, 1})
			h = level.downsample.forward(h)
			if materialize != nil {
				materialize(h)
			}
		}
	}
	h = v.mid1.forward(h)
	if materialize != nil {
		materialize(h)
	}
	h = v.midAttn.forward(h)
	if materialize != nil {
		materialize(h)
	}
	h = v.mid2.forward(h)
	if materialize != nil {
		materialize(h)
	}
	h = v.convOut.forward(mlx.SiLU(v.normOut.forward(h)))
	if materialize != nil {
		materialize(h)
	}
	h = v.quantConv.forward(h)
	if materialize != nil {
		materialize(h)
	}
	d := h.Dims()
	count := int32(d[1] * d[2])
	flat := mlx.Reshape(h, count, v.config.EmbedDim)
	if materialize != nil {
		materialize(flat)
	}
	var bestScore, bestID *mlx.Array
	for start := int32(0); start < v.config.CodebookSize; start += visionCodebookChunk {
		end := min(start+visionCodebookChunk, v.config.CodebookSize)
		book := v.codebook.Slice(mlx.Slice(int(start), int(end)), mlx.Slice())
		scores := mlx.Matmul(flat, mlx.Transpose(book, 1, 0))
		chunkScore := scores.MaxAxis(1, false)
		chunkID := mlx.Add(scores.Argmax(1, false).AsType(mlx.DTypeInt32), mlx.NewScalarArray(float32(start)).AsType(mlx.DTypeInt32))
		if bestScore == nil {
			bestScore, bestID = chunkScore, chunkID
		} else {
			better := chunkScore.Greater(bestScore)
			bestScore = mlx.Where(better, chunkScore, bestScore)
			bestID = mlx.Where(better, chunkID, bestID)
		}
		if materialize != nil {
			materialize(bestScore, bestID)
		}
	}
	return mlx.Reshape(bestID, 1, count), nil
}

type apertusImageInput struct {
	pixels                []float32
	width, height         int
	gridWidth, gridHeight int
}

func preprocessApertusImage(ctx context.Context, data []byte) (*apertusImageInput, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(data) == 0 || len(data) > maxApertusImageBytes {
		return nil, fmt.Errorf("Apertus image size %d is invalid", len(data))
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode Apertus image: %w", err)
	}
	b := img.Bounds()
	width, height := b.Dx(), b.Dy()
	if width <= 0 || height <= 0 || width > maxApertusImageDimension || height > maxApertusImageDimension || int64(width)*int64(height) > maxApertusImagePixels {
		return nil, fmt.Errorf("Apertus image dimensions %dx%d are invalid", width, height)
	}
	targetW, targetH := apertusImageSize(width, height)
	pixels := resizeApertusImage(img, b, targetW, targetH)
	for y := range targetH {
		if y&127 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		for x := range targetW {
			i := (y*targetW + x) * 3
			pixels[i] = pixels[i]/127.5 - 1
			pixels[i+1] = pixels[i+1]/127.5 - 1
			pixels[i+2] = pixels[i+2]/127.5 - 1
		}
	}
	return &apertusImageInput{pixels: pixels, width: targetW, height: targetH, gridWidth: targetW / 16, gridHeight: targetH / 16}, nil
}

func apertusImageSize(width, height int) (int, int) {
	targetArea := max(256*256, min(1400*1400, width*height))
	aspect := float64(width) / float64(height)
	targetH := int(math.Sqrt(float64(targetArea) / aspect))
	targetW := int(float64(targetH) * aspect)
	targetH = ((targetH + 8) / 16) * 16
	targetW = ((targetW + 8) / 16) * 16
	targetH = max(targetH, 16)
	targetW = max(targetW, 16)
	return targetW, targetH
}

type apertureResamplePoint struct {
	indices []int
	weights []int64
}

func apertureCubic(x float64) float64 {
	x = math.Abs(x)
	const a = -0.5
	if x < 1 {
		return ((a+2)*x-(a+3))*x*x + 1
	}
	if x < 2 {
		return (((a*x-5*a)*x+8*a)*x - 4*a)
	}
	return 0
}

func apertureResamplePoints(input, output int) ([]apertureResamplePoint, uint) {
	scale := float64(input) / float64(output)
	filterScale := max(scale, 1)
	support := 2 * filterScale
	points := make([]apertureResamplePoint, output)
	floatWeights := make([][]float64, output)
	var maxWeight float64
	for out := range output {
		// Match Torchvision's antialiased uint8 path: its coefficient bounds
		// follow Pillow, then weights are converted to dynamic-precision int16.
		center := (float64(out) + 0.5) * scale
		start := max(int(center-support+0.5), 0)
		end := min(int(center+support+0.5), input)
		var sum float64
		for source := start; source < end; source++ {
			weight := apertureCubic((float64(source) - center + 0.5) / filterScale)
			points[out].indices = append(points[out].indices, source)
			floatWeights[out] = append(floatWeights[out], weight)
			sum += weight
		}
		for i := range floatWeights[out] {
			floatWeights[out][i] /= sum
			maxWeight = max(maxWeight, floatWeights[out][i])
		}
	}
	precision := uint(22)
	for candidate := range 22 {
		if int(0.5+maxWeight*float64(uint64(1)<<uint(candidate+1))) >= 1<<15 {
			precision = uint(candidate)
			break
		}
	}
	for i := range points {
		points[i].weights = make([]int64, len(floatWeights[i]))
		for j, weight := range floatWeights[i] {
			points[i].weights[j] = int64(math.Round(weight * float64(uint64(1)<<precision)))
		}
	}
	return points, precision
}

func apertureResamplePixel(source []float32, stride, offset int, point apertureResamplePoint, precision uint) float32 {
	value := int64(1 << (precision - 1))
	for i, index := range point.indices {
		value += int64(source[index*stride+offset]) * point.weights[i]
	}
	return float32(min(max(value>>precision, 0), 255))
}

func resizeApertusImage(img image.Image, bounds image.Rectangle, width, height int) []float32 {
	sourceWidth, sourceHeight := bounds.Dx(), bounds.Dy()
	source := make([]float32, sourceWidth*sourceHeight*3)
	for y := range sourceHeight {
		for x := range sourceWidth {
			c := color.NRGBAModel.Convert(img.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.NRGBA)
			i := (y*sourceWidth + x) * 3
			source[i], source[i+1], source[i+2] = float32(c.R), float32(c.G), float32(c.B)
		}
	}
	if sourceWidth == width && sourceHeight == height {
		return source
	}
	xPoints, xPrecision := apertureResamplePoints(sourceWidth, width)
	yPoints, yPrecision := apertureResamplePoints(sourceHeight, height)
	horizontal := make([]float32, width*sourceHeight*3)
	for y := range sourceHeight {
		for x, point := range xPoints {
			for channel := range 3 {
				horizontal[(y*width+x)*3+channel] = apertureResamplePixel(source, 3, y*sourceWidth*3+channel, point, xPrecision)
			}
		}
	}
	result := make([]float32, width*height*3)
	for y, point := range yPoints {
		for x := range width {
			for channel := range 3 {
				result[(y*width+x)*3+channel] = apertureResamplePixel(horizontal, width*3, x*3+channel, point, yPrecision)
			}
		}
	}
	return result
}
