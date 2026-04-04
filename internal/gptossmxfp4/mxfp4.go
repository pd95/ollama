package gptossmxfp4

import (
	"crypto/sha256"
	"fmt"
)

type Packed struct {
	Weight      []byte
	WeightShape []int32
	Scale       []byte
	ScaleShape  []int32
}

type Comparison struct {
	StableShape       []int32
	ExperimentalShape []int32

	StableSHA256       string
	ExperimentalSHA256 string
}

func RewriteBlocks(raw []byte) ([]byte, error) {
	if len(raw)%16 != 0 {
		return nil, fmt.Errorf("raw byte length %d is not divisible by 16", len(raw))
	}
	out := make([]byte, len(raw))
	copy(out, raw)

	var tmp [16]byte
	for i := 0; i < len(out); i += 16 {
		for j := range 8 {
			a, b := out[i+j], out[i+j+8]
			tmp[2*j] = (a & 0x0F) | (b << 4)
			tmp[2*j+1] = (a >> 4) | (b & 0xF0)
		}
		copy(out[i:i+16], tmp[:])
	}
	return out, nil
}

func StableFused(blocksRaw []byte, blockShape []int32, scalesRaw []byte, scaleShape []int32) ([]byte, []int32, error) {
	if len(blockShape) == 0 || blockShape[len(blockShape)-1] != 16 {
		return nil, nil, fmt.Errorf("expected blocks last dim 16, got %v", blockShape)
	}
	if len(scaleShape) != len(blockShape)-1 {
		return nil, nil, fmt.Errorf("expected scales rank %d, got %d", len(blockShape)-1, len(scaleShape))
	}

	rewritten, err := RewriteBlocks(blocksRaw)
	if err != nil {
		return nil, nil, err
	}

	groups := int(blockShape[len(blockShape)-2])
	blockPrefixElems := product(blockShape[:len(blockShape)-2])
	if len(rewritten) != blockPrefixElems*groups*16 {
		return nil, nil, fmt.Errorf("blocks byte length %d does not match shape %v", len(rewritten), blockShape)
	}
	if len(scalesRaw) != product(scaleShape) {
		return nil, nil, fmt.Errorf("scale byte length %d does not match shape %v", len(scalesRaw), scaleShape)
	}

	fused := make([]byte, 0, blockPrefixElems*groups*17)
	for prefix := 0; prefix < blockPrefixElems; prefix++ {
		blockBase := prefix * groups * 16
		scaleBase := prefix * groups
		for g := 0; g < groups; g++ {
			fused = append(fused, scalesRaw[scaleBase+g])
			fused = append(fused, rewritten[blockBase+g*16:blockBase+(g+1)*16]...)
		}
	}

	shape := append([]int32(nil), blockShape[:len(blockShape)-2]...)
	shape = append(shape, blockShape[len(blockShape)-2]*blockShape[len(blockShape)-1]*2)
	return fused, shape, nil
}

func PackExperimental(blocksRaw []byte, blockShape []int32, scalesRaw []byte, scaleShape []int32) (*Packed, error) {
	if len(blockShape) == 0 || blockShape[len(blockShape)-1] != 16 {
		return nil, fmt.Errorf("expected blocks last dim 16, got %v", blockShape)
	}
	rewritten, err := RewriteBlocks(blocksRaw)
	if err != nil {
		return nil, err
	}

	weightShape := append([]int32(nil), blockShape[:len(blockShape)-1]...)
	weightShape[len(weightShape)-1] *= 4
	return &Packed{
		Weight:      rewritten,
		WeightShape: weightShape,
		Scale:       append([]byte(nil), scalesRaw...),
		ScaleShape:  append([]int32(nil), scaleShape...),
	}, nil
}

func ExperimentalToStableFused(p *Packed) ([]byte, []int32, error) {
	if p == nil {
		return nil, nil, fmt.Errorf("nil packed tensor")
	}
	if len(p.WeightShape) == 0 || p.WeightShape[len(p.WeightShape)-1]%4 != 0 {
		return nil, nil, fmt.Errorf("expected packed weight last dim divisible by 4, got %v", p.WeightShape)
	}
	if len(p.ScaleShape) != len(p.WeightShape) {
		return nil, nil, fmt.Errorf("expected scale rank %d, got %d", len(p.WeightShape), len(p.ScaleShape))
	}

	groups := int(p.ScaleShape[len(p.ScaleShape)-1])
	blockPrefixElems := product(p.ScaleShape[:len(p.ScaleShape)-1])
	if len(p.Scale) != blockPrefixElems*groups {
		return nil, nil, fmt.Errorf("scale byte length %d does not match shape %v", len(p.Scale), p.ScaleShape)
	}
	if len(p.Weight) != blockPrefixElems*groups*16 {
		return nil, nil, fmt.Errorf("weight byte length %d does not match shapes %v %v", len(p.Weight), p.WeightShape, p.ScaleShape)
	}

	fused := make([]byte, 0, blockPrefixElems*groups*17)
	for prefix := 0; prefix < blockPrefixElems; prefix++ {
		blockBase := prefix * groups * 16
		scaleBase := prefix * groups
		for g := 0; g < groups; g++ {
			fused = append(fused, p.Scale[scaleBase+g])
			fused = append(fused, p.Weight[blockBase+g*16:blockBase+(g+1)*16]...)
		}
	}

	shape := append([]int32(nil), p.ScaleShape[:len(p.ScaleShape)-1]...)
	shape = append(shape, p.ScaleShape[len(p.ScaleShape)-1]*32)
	return fused, shape, nil
}

func Compare(blocksRaw []byte, blockShape []int32, scalesRaw []byte, scaleShape []int32) (*Comparison, error) {
	stableFused, stableShape, err := StableFused(blocksRaw, blockShape, scalesRaw, scaleShape)
	if err != nil {
		return nil, err
	}
	packed, err := PackExperimental(blocksRaw, blockShape, scalesRaw, scaleShape)
	if err != nil {
		return nil, err
	}
	experimentalFused, experimentalShape, err := ExperimentalToStableFused(packed)
	if err != nil {
		return nil, err
	}

	comp := &Comparison{
		StableShape:        stableShape,
		ExperimentalShape:  experimentalShape,
		StableSHA256:       fmt.Sprintf("%x", sha256.Sum256(stableFused)),
		ExperimentalSHA256: fmt.Sprintf("%x", sha256.Sum256(experimentalFused)),
	}
	return comp, nil
}

func product(shape []int32) int {
	n := 1
	for _, dim := range shape {
		n *= int(dim)
	}
	return n
}
