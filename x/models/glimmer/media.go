package glimmer

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// The reference processor uses the full 4096-token budget for still images.
// Reducing it here discards source detail and disproportionately hurts OCR.
const maxImageTokens = 4096

var _ base.MediaModel = (*Model)(nil)

// preparedImage is glimmer's model-private media state: the patch grid the
// encoder consumes and the soft-token count of its pixel-shuffled output.
type preparedImage struct {
	gridH        int
	gridW        int
	outputTokens int
}

// PrepareMedia implements base.MediaModel: splice each image segment's
// placeholder expansion — image_start, the patch-token run, image_end — into
// the stream, decoding, resizing, and patchifying the image on the CPU.
func (m *Model) PrepareMedia(ctx context.Context, segments []base.Segment) (*base.PreparedRequest, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	prepared := &base.PreparedRequest{}
	for s, seg := range segments {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if seg.Data == nil {
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
			continue
		}
		if !m.HasVision || m.VisionEncoder == nil {
			return nil, fmt.Errorf("this model does not support %s input", seg.Kind)
		}
		if seg.Kind != "image" {
			return nil, fmt.Errorf("glimmer does not support %s input", seg.Kind)
		}

		patches, geom, err := m.preprocessImage(ctx, seg.Data)
		if err != nil {
			return nil, fmt.Errorf("preprocess image: %w", err)
		}

		start := len(prepared.Tokens)
		prepared.Tokens = append(prepared.Tokens, m.ImageStartTokenID)
		for i := range geom.outputTokens {
			if i%256 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			prepared.Tokens = append(prepared.Tokens, m.PatchTokenID)
		}
		prepared.Tokens = append(prepared.Tokens, m.ImageEndTokenID)

		n := geom.gridH * geom.gridW
		prepared.Items = append(prepared.Items, base.PreparedItem{
			Range:     [2]int{start, len(prepared.Tokens)},
			Source:    s,
			MediaData: patches,
			Dims:      []int{1, n, len(patches) / n},
			Opaque:    geom,
			// Patch rows attend causally in the text stack (there is no
			// relaxed image-span mask), so chunked prefill may split the run.
			Causal: true,
		})
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return prepared, nil
}

// EncodeMedia implements base.MediaModel: run the vision tower over one
// prepared image, returning the lazy [outputTokens, hidden] features.
func (m *Model) EncodeMedia(item *base.PreparedItem, data *mlx.Array) *mlx.Array {
	geom := item.Opaque.(preparedImage)
	return m.encodeVision(data, geom.gridH, geom.gridW)
}

// softRun returns a media item's feature-bearing token range: the expansion
// is image_start + patch*N + image_end, so the run starts one past the splice.
func softRun(item batch.MediaItem) (start, end int) {
	geom := item.Opaque.(preparedImage)
	return item.Pos + 1, item.Pos + 1 + geom.outputTokens
}

// scatterMedia overwrites the patch-token rows this chunk covers with the
// item's projected features.
func (m *Model) scatterMedia(h *mlx.Array, b *batch.Batch) *mlx.Array {
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		start, end := softRun(item)
		off := int(b.SeqOffsets[item.Seq])
		qLo := max(start, off)
		qHi := min(end, off+int(b.SeqQueryLens[item.Seq]))
		if qHi <= qLo {
			continue
		}

		feat := item.Features.Slice(mlx.Slice(qLo-start, qHi-start), mlx.Slice())
		feat = mlx.Reshape(feat.AsType(h.DType()), 1, int32(qHi-qLo), m.HiddenSize)
		h = h.SliceUpdate(feat, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(qLo-off, qHi-off), mlx.Slice())
	}
	return h
}

func computeImageSize(width, height, patchStride, maxTokens int) (targetWidth, targetHeight, tokens int) {
	if width <= 0 || height <= 0 || patchStride <= 0 || maxTokens <= 0 {
		return 0, 0, 0
	}

	patchRows := float64(height) / float64(patchStride)
	patchCols := float64(width) / float64(patchStride)
	ratio := patchCols / patchRows
	if patchRows*patchCols > float64(maxTokens) {
		patchRows = math.Sqrt(float64(maxTokens) / ratio)
		patchCols = patchRows * ratio
	}

	rows := []int{int(math.Floor(patchRows)), int(math.Ceil(patchRows))}
	cols := []int{int(math.Floor(patchCols)), int(math.Ceil(patchCols))}
	bestRows, bestCols := 0, 0
	bestDelta := math.Inf(1)
	seen := make(map[[2]int]bool, 4)
	for _, r := range rows {
		for _, c := range cols {
			pair := [2]int{r, c}
			if seen[pair] || r < 1 || c < 1 || r*c > maxTokens {
				continue
			}
			seen[pair] = true
			delta := math.Abs(float64(r)/float64(c) - float64(height)/float64(width))
			// The reference deduplicates candidates with a Python set, leaving
			// exact ties implementation-dependent. Prefer the larger grid to
			// preserve source detail deterministically.
			if delta < bestDelta || (delta == bestDelta && r*c > bestRows*bestCols) {
				bestRows, bestCols, bestDelta = r, c, delta
			}
		}
	}
	if bestRows == 0 {
		bestRows = max(1, int(math.Round(patchRows)))
		bestCols = max(1, int(math.Round(patchCols)))
	}

	return bestCols * patchStride, bestRows * patchStride, bestRows * bestCols
}

// preprocessImage decodes and prepares one image on the CPU: budgeted
// resize, [-1,1] rescale, and patchify to the tower's layout — one row per
// patch, (temporal, RGB, pixel row, pixel column) within it.
func (m *Model) preprocessImage(ctx context.Context, data []byte) ([]float32, preparedImage, error) {
	if err := ctx.Err(); err != nil {
		return nil, preparedImage{}, err
	}
	src, _, err := image.Decode(glimmerContextReader{check: ctx.Err, r: bytes.NewReader(data)})
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, preparedImage{}, ctxErr
		}
		return nil, preparedImage{}, fmt.Errorf("decode: %w", err)
	}

	bounds := src.Bounds()
	patchStride := int(m.VisionPatchSize * m.VisionDownsampleFactor)
	targetW, targetH, outputTokens := computeImageSize(bounds.Dx(), bounds.Dy(), patchStride, maxImageTokens)
	if targetW == 0 || targetH == 0 || outputTokens == 0 {
		return nil, preparedImage{}, fmt.Errorf("invalid image dimensions %dx%d", bounds.Dx(), bounds.Dy())
	}

	resized, err := resizeGlimmerImage(ctx, src, bounds, targetW, targetH)
	if err != nil {
		return nil, preparedImage{}, err
	}

	patchSize := int(m.VisionPatchSize)
	temporal := int(m.VisionPatchTemporal)
	gridH, gridW := targetH/patchSize, targetW/patchSize
	patchDim := temporal * 3 * patchSize * patchSize
	if err := ctx.Err(); err != nil {
		return nil, preparedImage{}, err
	}
	patches := make([]float32, gridH*gridW*patchDim)

	at := 0
	for patchY := range gridH {
		if err := ctx.Err(); err != nil {
			return nil, preparedImage{}, err
		}
		for patchX := range gridW {
			for range temporal {
				for channel := range 3 {
					for y := range patchSize {
						row := (patchY*patchSize+y)*resized.Stride + patchX*patchSize*4
						for x := range patchSize {
							value := resized.Pix[row+x*4+channel]
							patches[at] = 2*float32(value)/255 - 1
							at++
						}
					}
				}
			}
		}
	}

	if err := ctx.Err(); err != nil {
		return nil, preparedImage{}, err
	}
	return patches, preparedImage{gridH: gridH, gridW: gridW, outputTokens: outputTokens}, nil
}

type glimmerContextReader struct {
	check func() error
	r     io.Reader
}

func (r glimmerContextReader) Read(p []byte) (int, error) {
	if err := r.check(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

type glimmerResizeCancellation struct{ marker byte }

var glimmerResizeCanceled = &glimmerResizeCancellation{marker: 1}

type glimmerCancellableImage struct {
	*image.NRGBA
	check func() error
}

func (dst glimmerCancellableImage) Set(x, y int, c color.Color) {
	if dst.check() != nil {
		panic(glimmerResizeCanceled)
	}
	dst.NRGBA.Set(x, y, c)
}

func resizeGlimmerImage(ctx context.Context, src image.Image, bounds image.Rectangle, width, height int) (resized *image.NRGBA, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	dst := image.NewNRGBA(image.Rect(0, 0, width, height))
	defer func() {
		if recovered := recover(); recovered != nil {
			if recovered == glimmerResizeCanceled {
				resized, err = nil, ctx.Err()
				return
			}
			panic(recovered)
		}
	}()
	draw.CatmullRom.Scale(glimmerCancellableImage{NRGBA: dst, check: ctx.Err}, dst.Bounds(), src, bounds, draw.Src, nil)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return dst, nil
}
