package apertus

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"slices"
	"strings"

	xdraw "golang.org/x/image/draw"
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

func (v *VisionTokenizer) encode(pixels []float32, width, height int) (*mlx.Array, error) {
	if len(pixels) != width*height*3 {
		return nil, errors.New("invalid Apertus image pixels")
	}
	x := mlx.Reshape(mlx.FromValues(pixels, len(pixels)), 1, int32(height), int32(width), 3)
	h := v.convIn.forward(x)
	for _, level := range v.levels {
		for i, block := range level.blocks {
			h = block.forward(h)
			if len(level.attn) > 0 {
				h = level.attn[i].forward(h)
			}
		}
		if level.downsample != nil {
			h = mlx.PadConstant(h, []int{1, 2}, []int{0, 0}, []int{1, 1})
			h = level.downsample.forward(h)
		}
		mlx.Eval(h)
	}
	h = v.mid1.forward(h)
	h = v.midAttn.forward(h)
	h = v.mid2.forward(h)
	h = v.convOut.forward(mlx.SiLU(v.normOut.forward(h)))
	h = v.quantConv.forward(h)
	d := h.Dims()
	count := int32(d[1] * d[2])
	flat := mlx.Reshape(h, count, v.config.EmbedDim)
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
		mlx.Eval(bestScore, bestID)
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
	targetArea := max(256*256, min(1400*1400, width*height))
	aspect := float64(width) / float64(height)
	targetH := int(math.Sqrt(float64(targetArea) / aspect))
	targetW := int(float64(targetH) * aspect)
	targetH = ((targetH + 8) / 16) * 16
	targetW = ((targetW + 8) / 16) * 16
	targetH = max(targetH, 16)
	targetW = max(targetW, 16)
	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	xdraw.CatmullRom.Scale(dst, dst.Bounds(), img, b, xdraw.Src, nil)
	pixels := make([]float32, targetW*targetH*3)
	for y := range targetH {
		if y&127 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		for x := range targetW {
			r, g, blue, _ := dst.At(x, y).RGBA()
			i := (y*targetW + x) * 3
			pixels[i] = float32(r>>8)/127.5 - 1
			pixels[i+1] = float32(g>>8)/127.5 - 1
			pixels[i+2] = float32(blue>>8)/127.5 - 1
		}
	}
	return &apertusImageInput{pixels: pixels, width: targetW, height: targetH, gridWidth: targetW / 16, gridHeight: targetH / 16}, nil
}
