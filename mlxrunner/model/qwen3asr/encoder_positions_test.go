package qwen3asr

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestQwen3ASRProviderPositions(t *testing.T) {
	// Full buffer integrity is separate from independent phase probes below.
	if len(encoderPositionData) != 13*896*4 {
		t.Fatal("invalid canonical position buffer dimensions")
	}
	metadata, err := os.ReadFile("positions.json")
	if err != nil {
		t.Fatal(err)
	}
	var identity struct {
		SHA string `json:"sha256"`
	}
	if err := json.Unmarshal(metadata, &identity); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256([]byte(encoderPositionData))
	if hex.EncodeToString(digest[:]) != identity.SHA {
		t.Fatal("canonical provider position buffer digest mismatch")
	}
	data, err := os.ReadFile("testdata/provider_positions.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Steps   int     `json:"steps"`
		Width   int     `json:"width"`
		Atol    float64 `json:"atol"`
		Samples []struct {
			Step    int     `json:"step"`
			Channel int     `json:"channel"`
			Value   float64 `json:"value"`
		} `json:"samples"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Steps != 13 || fixture.Width != 896 || fixture.Atol != 1.2e-7 || len(fixture.Samples) != 16 {
		t.Fatal("invalid original-provider position fixture")
	}
	actual := encoderPositions(fixture.Steps, fixture.Width)
	for _, sample := range fixture.Samples {
		// Original provider values, not a copy of the native formula. These
		// phases exercise float32 rounding before exp, including both sin/cos.
		got := float64(actual[sample.Step*fixture.Width+sample.Channel])
		if math.Abs(got-sample.Value) > fixture.Atol {
			t.Errorf("position[%d,%d]=%.12g provider=%.12g", sample.Step, sample.Channel, got, sample.Value)
		}
	}
}
