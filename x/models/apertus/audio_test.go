package apertus

import (
	"context"
	"encoding/binary"
	"math"
	"testing"
)

func TestDecodeApertusExtensibleFloatWAV(t *testing.T) {
	data := make([]byte, 12+8+40+8+8)
	copy(data, "RIFF")
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	copy(data[8:], "WAVE")
	copy(data[12:], "fmt ")
	binary.LittleEndian.PutUint32(data[16:], 40)
	fmtChunk := data[20:60]
	binary.LittleEndian.PutUint16(fmtChunk, 0xfffe)
	binary.LittleEndian.PutUint16(fmtChunk[2:], 1)
	binary.LittleEndian.PutUint32(fmtChunk[4:], 24000)
	binary.LittleEndian.PutUint32(fmtChunk[8:], 96000)
	binary.LittleEndian.PutUint16(fmtChunk[12:], 4)
	binary.LittleEndian.PutUint16(fmtChunk[14:], 32)
	binary.LittleEndian.PutUint16(fmtChunk[16:], 22)
	binary.LittleEndian.PutUint16(fmtChunk[18:], 32)
	copy(fmtChunk[24:], []byte{3, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71})
	copy(data[60:], "data")
	binary.LittleEndian.PutUint32(data[64:], 8)
	binary.LittleEndian.PutUint32(data[68:], math.Float32bits(0.25))
	binary.LittleEndian.PutUint32(data[72:], math.Float32bits(-0.5))

	got, err := decodeApertusWAV(context.Background(), data, 24000)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != 0.25 || got[1] != -0.5 {
		t.Fatalf("decoded samples = %v", got)
	}
}
