package qwen3_5

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"io"
	"math"

	"golang.org/x/image/draw"
)

// Processor bounds from the family's preprocessor config: total pixels are
// clamped to [shortest_edge, longest_edge] with both sides rounded to
// multiples of patch_size * spatial_merge_size.
const (
	visionMinPixels = 65536
	visionMaxPixels = 16777216
)

// preparedImage is the model-private state for one image: the pre-merge
// patch grid and the CPU-computed lookups the encoder consumes, all in the
// block-major merge order the pixel rows use.
type preparedImage struct {
	gridH, gridW int32
	// ropePos is (h, w) per patch for the tower's 2D rotary tables.
	ropePos []int32
	// posIdx/posWt are the four bilinear corners per patch into the learned
	// position table, with their interpolation weights.
	posIdx []int32
	posWt  []float32
}

func (p preparedImage) numPatches() int32 { return p.gridH * p.gridW }

func (p preparedImage) numTokens(merge int32) int32 {
	return p.gridH * p.gridW / (merge * merge)
}

// smartResize ports the reference resize rule: round each side to the patch
// factor, then scale into the pixel budget, preserving aspect ratio.
func smartResize(height, width, factor int32) (int32, int32, error) {
	if height <= 0 || width <= 0 {
		return 0, 0, fmt.Errorf("invalid image size %dx%d", width, height)
	}
	longer, shorter := float64(max(height, width)), float64(min(height, width))
	if longer/shorter > 200 {
		return 0, 0, fmt.Errorf("image aspect ratio %.0f exceeds 200", longer/shorter)
	}

	f := float64(factor)
	// The reference uses Python round (half to even).
	hBar := math.RoundToEven(float64(height)/f) * f
	wBar := math.RoundToEven(float64(width)/f) * f
	if hBar*wBar > visionMaxPixels {
		beta := math.Sqrt(float64(height) * float64(width) / visionMaxPixels)
		hBar = math.Max(f, math.Floor(float64(height)/beta/f)*f)
		wBar = math.Max(f, math.Floor(float64(width)/beta/f)*f)
	} else if hBar*wBar < visionMinPixels {
		beta := math.Sqrt(visionMinPixels / (float64(height) * float64(width)))
		hBar = math.Ceil(float64(height)*beta/f) * f
		wBar = math.Ceil(float64(width)*beta/f) * f
	}
	return int32(hBar), int32(wBar), nil
}

// preprocessImage decodes and prepares one image: aspect-preserving resize,
// (x/255-0.5)/0.5 normalization, and patchification into the tower's
// block-major row layout.
func (m *Model) preprocessImage(ctx context.Context, data []byte) (pixels []float32, prep preparedImage, err error) {
	if err := ctx.Err(); err != nil {
		return nil, preparedImage{}, err
	}
	img, _, err := image.Decode(qwenImageContextReader{check: ctx.Err, r: bytes.NewReader(data)})
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, preparedImage{}, ctxErr
		}
		return nil, preparedImage{}, fmt.Errorf("decode image: %w", err)
	}

	v := m.Vision
	patch, merge, temporal := v.PatchSize, v.SpatialMergeSize, v.TemporalPatchSize
	bounds := img.Bounds()
	img, err = dropAlpha(ctx, img, bounds)
	if err != nil {
		return nil, preparedImage{}, err
	}
	targetH, targetW, err := smartResize(int32(bounds.Dy()), int32(bounds.Dx()), patch*merge)
	if err != nil {
		return nil, preparedImage{}, err
	}

	resized, err := resizeQwenImage(ctx, img, bounds, int(targetW), int(targetH))
	if err != nil {
		return nil, preparedImage{}, err
	}
	gridH, gridW := targetH/patch, targetW/patch
	prep = preparedImage{gridH: gridH, gridW: gridW}
	pixels, err = patchifyImage(ctx, resized, gridH, gridW, patch, merge, temporal)
	if err != nil {
		return nil, preparedImage{}, err
	}
	prep.ropePos, prep.posIdx, prep.posWt, err = visionPatchLookups(ctx, gridH, gridW, merge, v.NumPositionEmbeddings)
	if err != nil {
		return nil, preparedImage{}, err
	}
	if err := ctx.Err(); err != nil {
		return nil, preparedImage{}, err
	}
	return pixels, prep, nil
}

// patchifyImage converts the resized image to the tower's row layout:
// block-major over merge blocks, each row [channel][temporal duplicate]
// [pixel row][pixel column].
func patchifyImage(ctx context.Context, resized *image.RGBA, gridH, gridW, patch, merge, temporal int32) ([]float32, error) {
	n := int(gridH * gridW)
	rowLen := int(3 * temporal * patch * patch)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	pixels := make([]float32, n*rowLen)

	p, t := int(patch), int(temporal)
	patchArea := p * p
	row := 0
	for bh := range gridH / merge {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for bw := range gridW / merge {
			for mh := range merge {
				for mw := range merge {
					gh, gw := bh*merge+mh, bw*merge+mw
					out := pixels[row*rowLen:]
					baseX, baseY := int(gw)*p, int(gh)*p
					for py := range p {
						px0 := resized.PixOffset(baseX, baseY+py)
						for px := range p {
							pix := resized.Pix[px0+px*4:]
							o := py*p + px
							for c := range 3 {
								val := float32(pix[c])/127.5 - 1
								for ti := range t {
									out[(c*t+ti)*patchArea+o] = val
								}
							}
						}
					}
					row++
				}
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return pixels, nil
}

// visionPatchLookups computes, in the block-major token order, each patch's
// (h, w) rotary position and its four bilinear corners into the learned
// side*side position grid with their weights.
func visionPatchLookups(ctx context.Context, gridH, gridW, merge, numPosEmbeddings int32) (ropePos, posIdx []int32, posWt []float32, err error) {
	side := int32(math.Sqrt(float64(numPosEmbeddings)))
	n := int(gridH * gridW)
	if err := ctx.Err(); err != nil {
		return nil, nil, nil, err
	}
	ropePos = make([]int32, 0, 2*n)
	posIdx = make([]int32, 0, 4*n)
	posWt = make([]float32, 0, 4*n)

	// linspace(0, side-1, g) per axis over the pre-merge grid.
	coord := func(i, g int32) float64 {
		if g == 1 {
			return 0
		}
		return float64(i) * float64(side-1) / float64(g-1)
	}

	for bh := range gridH / merge {
		if err := ctx.Err(); err != nil {
			return nil, nil, nil, err
		}
		for bw := range gridW / merge {
			for mh := range merge {
				for mw := range merge {
					gh, gw := bh*merge+mh, bw*merge+mw
					ropePos = append(ropePos, gh, gw)

					hg, wg := coord(gh, gridH), coord(gw, gridW)
					hf, wf := int32(hg), int32(wg)
					hc, wc := min(hf+1, side-1), min(wf+1, side-1)
					hFrac, wFrac := float32(hg-float64(hf)), float32(wg-float64(wf))
					posIdx = append(posIdx, hf*side+wf, hf*side+wc, hc*side+wf, hc*side+wc)
					posWt = append(posWt,
						(1-hFrac)*(1-wFrac), (1-hFrac)*wFrac, hFrac*(1-wFrac), hFrac*wFrac)
				}
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, nil, err
	}
	return ropePos, posIdx, posWt, nil
}

// dropAlpha flattens a non-opaque image to straight RGB: the reference
// drops alpha via RGB conversion before resizing, not by compositing.
func dropAlpha(ctx context.Context, img image.Image, bounds image.Rectangle) (image.Image, error) {
	if o, ok := img.(interface{ Opaque() bool }); ok && o.Opaque() {
		return img, ctx.Err()
	}
	flat := image.NewNRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA)
			c.A = 0xff
			flat.SetNRGBA(x, y, c)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return flat, nil
}

type qwenImageContextReader struct {
	check func() error
	r     io.Reader
}

func (r qwenImageContextReader) Read(p []byte) (int, error) {
	if err := r.check(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

type qwenResizeCancellation struct{ marker byte }

var qwenResizeCanceled = &qwenResizeCancellation{marker: 1}

type qwenCancellableImage struct {
	*image.RGBA
	check func() error
}

func (dst qwenCancellableImage) Set(x, y int, c color.Color) {
	if dst.check() != nil {
		panic(qwenResizeCanceled)
	}
	dst.RGBA.Set(x, y, c)
}

func resizeQwenImage(ctx context.Context, src image.Image, bounds image.Rectangle, width, height int) (resized *image.RGBA, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	defer func() {
		if recovered := recover(); recovered != nil {
			if recovered == qwenResizeCanceled {
				resized, err = nil, ctx.Err()
				return
			}
			panic(recovered)
		}
	}()
	draw.CatmullRom.Scale(qwenCancellableImage{RGBA: dst, check: ctx.Err}, dst.Bounds(), src, bounds, draw.Src, nil)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return dst, nil
}
