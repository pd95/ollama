package apertus

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

// Small tensors reproduce the retained-intermediate failure without loading a
// model or risking a host-sized allocation. The chunked codebook search and
// residual path must keep their inputs alive while releasing completed stages.
func TestVisionStagedMemoryAndCodes(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const channels, side = 32, 16
		conv := func() *apertureConv2D {
			return &apertureConv2D{weight: tuningValues(channels, 1, 1, channels), bias: tuningValues(channels), stride: 1}
		}
		norm := func() *apertureGroupNorm {
			return &apertureGroupNorm{weight: mlx.AddScalar(tuningValues(channels), 1), bias: tuningValues(channels)}
		}
		block := &apertureVisionResBlock{norm1: norm(), norm2: norm(), conv1: conv(), conv2: conv(), shortcut: conv()}
		attention := &apertureVisionAttention{norm: norm(), q: conv(), k: conv(), v: conv(), out: conv()}
		vision := &VisionTokenizer{
			config: VisionTokenizerConfig{EmbedDim: channels, CodebookSize: 8192},
			convIn: conv(), convOut: conv(), quantConv: conv(), normOut: norm(),
			levels: []*apertureVisionLevel{{blocks: []*apertureVisionResBlock{block, block, block, block}}},
			mid1:   block, mid2: block, midAttn: attention,
			codebook: tuningValues(8192, channels),
		}
		vision.convIn.weight = tuningValues(channels, 1, 1, 3)
		data := tuningValues(1, side, side, 3)
		mlx.Eval(data)
		var want []int32
		mlx.Scoped(func() {
			codes, err := vision.encode(data, side, side)
			if err != nil {
				t.Fatal(err)
			}
			mlx.Eval(codes)
			want = slices.Clone(codes.Ints())
		})
		mlx.DefaultStream().Synchronize()
		baseline := mlx.ActiveMemory()
		const activationBytes = side * side * channels * 4
		// Allow a residual stage and attention scores, but not accumulated
		// stages or a retained codebook chunk matrix.
		const maxLiveBytes = 8*activationBytes + 2*side*side*side*side*4
		for range 2 {
			mlx.Scoped(func() {
				stages := 0
				codes, err := vision.encodeStaged(data, side, side, func(arrays ...*mlx.Array) {
					mlx.Eval(arrays...)
					// Count retained allocations, not Metal completion handlers
					// still retiring an otherwise freed buffer.
					mlx.DefaultStream().Synchronize()
					stages++
					if live := mlx.ActiveMemory() - baseline; live > maxLiveBytes {
						t.Fatalf("stage %d retains %d bytes; bounded working set is %d", stages, live, maxLiveBytes)
					}
				})
				if err != nil {
					t.Fatal(err)
				}
				mlx.Eval(codes)
				if got := codes.Ints(); !slices.Equal(got, want) {
					t.Fatal("staged vision codes differ from the original graph")
				}
			})
			mlx.DefaultStream().Synchronize()
			if live := mlx.ActiveMemory() - baseline; live != 0 {
				t.Fatalf("completed encoder retains %d bytes", live)
			}
		}
	})
}
