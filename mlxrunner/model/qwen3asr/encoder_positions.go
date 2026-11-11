// Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
// SPDX-License-Identifier: Apache-2.0

package qwen3asr

import (
	_ "embed"
	"encoding/binary"
	"math"
)

// The provider constructs this non-learned buffer in float32 on the CPU. Keep
// the qualified values rather than rebuilding them with platform-dependent
// exp/sin/cos rounding. Every supported 100-frame CNN chunk uses at most 13
// positions at width 896. Provenance and digest are in positions.json.
//
//go:embed positions.f32
var encoderPositionData string

// Width1024 is constructed independently by the pinned provider for1.7B and
// ForcedAligner. It must never be substituted with the0.6B width896 table.
//
//go:embed positions1024.f32
var encoderPositionData1024 string

func encoderPositions(steps, width int) []float32 {
	data := encoderPositionData
	if width == 1024 {
		data = encoderPositionData1024
	}
	if (width != 896 && width != 1024) || steps < 1 || steps > 13 {
		panic("unsupported Qwen3-ASR encoder position shape")
	}
	pos := make([]float32, steps*width)
	for i := range pos {
		pos[i] = math.Float32frombits(binary.LittleEndian.Uint32([]byte(data[4*i : 4*i+4])))
	}
	return pos
}
