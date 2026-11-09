package create

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"slices"

	"github.com/ollama/ollama/fs/safetensors"
)

func originalChatterboxRead(ctx context.Context, t *safetensors.TensorData) ([]byte, error) {
	if t.Size < 0 || t.Size > 512<<20 {
		return nil, fmt.Errorf("original Chatterbox tensor exceeds byte limit")
	}
	return io.ReadAll(io.LimitReader(ReaderWithContext(ctx, t.Reader()), t.Size+1))
}

func originalChatterboxTensorBytes(ctx context.Context, t originalChatterboxTensor) ([]byte, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if t.source == nil {
		return t.derived, nil
	}
	raw, err := originalChatterboxRead(ctx, t.source)
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) != t.source.Size {
		return nil, fmt.Errorf("truncated original tensor")
	}
	if t.operation == "narrow" {
		if len(raw)%8 != 0 {
			return nil, fmt.Errorf("invalid original I64 tensor extent")
		}
		out := make([]byte, len(raw)/2)
		for i := 0; i < len(out); i += 4 {
			n := int64(binary.LittleEndian.Uint64(raw[2*i:]))
			if n < 0 || n > math.MaxInt32 {
				return nil, fmt.Errorf("original Turbo token cannot be narrowed to I32")
			}
			binary.LittleEndian.PutUint32(out[i:], uint32(n))
		}
		return out, nil
	}
	if t.dtype == "I32" {
		for i := 0; i < len(raw); i += 4 {
			if binary.LittleEndian.Uint32(raw[i:]) > math.MaxInt32 {
				return nil, fmt.Errorf("negative original Turbo conditioning integer")
			}
		}
		return raw, nil
	}
	if len(raw)%4 != 0 || t.dtype != "F32" {
		return nil, fmt.Errorf("invalid original floating tensor")
	}
	get := func(data []byte, i int) float32 { return math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:])) }
	put := func(i int, x float32) { binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(x)) }
	finite := func(data []byte) error {
		for i := 0; i < len(data); i += 4 {
			if binary.LittleEndian.Uint32(data[i:])&0x7f800000 == 0x7f800000 {
				return fmt.Errorf("nonfinite original tensor")
			}
		}
		return nil
	}
	if err := finite(raw); err != nil {
		return nil, err
	}
	if t.other != nil {
		other, err := originalChatterboxRead(ctx, t.other)
		if err != nil {
			return nil, err
		}
		if int64(len(other)) != t.other.Size || len(other)%4 != 0 {
			return nil, fmt.Errorf("truncated original paired tensor")
		}
		if err := finite(other); err != nil {
			return nil, err
		}
		switch t.operation {
		case "add":
			if len(other) != len(raw) {
				return nil, fmt.Errorf("original bias pair extents differ")
			}
			for i := range len(raw) / 4 {
				put(i, get(raw, i)+get(other, i))
			}
		case "weight_norm":
			if len(t.source.Shape) < 2 || len(other) != int(t.source.Shape[0])*4 || len(other) == 0 || len(raw)%len(other) != 0 {
				return nil, fmt.Errorf("invalid original weight normalization pair")
			}
			width := len(raw) / len(other)
			for channel := range len(other) / 4 {
				if err := checkContext(ctx); err != nil {
					return nil, err
				}
				// Accumulate the squared F32 input in F64 to avoid a serial
				// reduction's size-dependent loss. Store the norm and all
				// subsequent operations at the original F32 precision.
				var sum float64
				for j := range width {
					v := float64(get(raw, channel*width+j))
					sum += v * v
				}
				norm := float32(math.Sqrt(sum)) + float32(1e-12)
				if math.IsInf(float64(norm), 0) {
					return nil, fmt.Errorf("original weight normalization overflow")
				}
				g := get(other, channel)
				for j := range width {
					i := channel*width + j
					put(i, (g*get(raw, i))/norm)
				}
			}
		default:
			return nil, fmt.Errorf("unsupported original paired operation %q", t.operation)
		}
		if err := finite(raw); err != nil {
			return nil, err
		}
	}
	if len(t.axes) == 0 || slices.Equal(t.axes, chatterboxIdentityAxes(len(t.axes))) {
		return raw, nil
	}
	return originalChatterboxPermute(ctx, raw, t.source.Shape, t.axes)
}

func originalChatterboxPermute(ctx context.Context, raw []byte, shape []int32, axes []int) ([]byte, error) {
	if len(shape) != len(axes) || len(shape) > 4 {
		return nil, fmt.Errorf("invalid original tensor permutation rank")
	}
	strides := make([]int, len(shape))
	count := int64(1)
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] <= 0 || count > (512<<20)/4/int64(shape[i]) {
			return nil, fmt.Errorf("invalid original tensor permutation shape")
		}
		strides[i] = int(count)
		count *= int64(shape[i])
	}
	if count*4 != int64(len(raw)) {
		return nil, fmt.Errorf("invalid original tensor permutation extent")
	}
	seen := map[int]bool{}
	for _, axis := range axes {
		if axis < 0 || axis >= len(shape) || seen[axis] {
			return nil, fmt.Errorf("invalid original tensor permutation axes")
		}
		seen[axis] = true
	}
	out := make([]byte, len(raw))
	for dst := range int(count) {
		if dst%65536 == 0 {
			if err := checkContext(ctx); err != nil {
				return nil, err
			}
		}
		rest, src := dst, 0
		for j := len(axes) - 1; j >= 0; j-- {
			axis := axes[j]
			coord := rest % int(shape[axis])
			rest /= int(shape[axis])
			src += coord * strides[axis]
		}
		copy(out[dst*4:dst*4+4], raw[src*4:src*4+4])
	}
	return out, nil
}

func originalTurboTables() []originalChatterboxTensor {
	position := make([]byte, 9999*512*4)
	put := func(data []byte, i int, x float32) { binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(x)) }
	for row := range 9999 {
		pos := row - 4999
		for k := range 256 {
			freq := math.Float32frombits(originalTurboFrequencyBits[k])
			angle := float32(pos) * freq
			put(position, row*512+2*k, float32(math.Sin(float64(angle))))
			put(position, row*512+2*k+1, float32(math.Cos(float64(angle))))
		}
	}
	// The provider stores positive positions in reverse order followed by
	// negative positions. Negate only sine, preserving the zero row.
	for row := range 9999 {
		for k := range 256 {
			i := (row*512 + 2*k) * 4
			x := math.Float32frombits(binary.LittleEndian.Uint32(position[i:]))
			if row == 4999 {
				x = 0
			} else {
				x = -x
			}
			put(position, row*512+2*k, x)
		}
	}
	window := make([]byte, 16*4)
	for i := range 16 {
		put(window, i, float32(.5-.5*math.Cos(2*math.Pi*float64(i)/16)))
	}
	fade := make([]byte, 960*4)
	for i := range 480 {
		angle := float32(math.Pi) * float32(479-i) / 479
		put(fade, 480+i, (float32(math.Cos(float64(angle)))+1)/2)
	}
	return []originalChatterboxTensor{
		{name: "s3gen.encoder.embed.pos_enc.pe", dtype: "F32", shape: []int32{1, 9999, 512}, derived: position},
		{name: "s3gen.encoder.up_embed.pos_enc.pe", dtype: "F32", shape: []int32{1, 9999, 512}, derived: position},
		{name: "s3gen.mel2wav.stft_window", dtype: "F32", shape: []int32{16}, derived: window},
		{name: "s3gen.trim_fade", dtype: "F32", shape: []int32{960}, derived: fade},
	}
}
