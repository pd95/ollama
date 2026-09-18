package metadata

import (
	"testing"
)

func testConfig() Config {
	return Config{
		Architectures: []string{"Apertus1p5ForConditionalGeneration"},
		ImageTokenID:  131079, ImageTokenOffset: 131272,
		AudioTokenID: 131085, AudioTokenOffset: 262344,
		Vision: VisionConfig{CodebookSize: 131072, EmbedDim: 256, InChannels: 3, ChannelMultiplier: []int32{1, 1, 2, 2, 4}},
		Audio:  AudioConfig{CodebookSize: 4096, CodebookDim: 512, AudioChannels: 1, SamplingRate: 24000, UpsamplingRatios: []int32{6, 5, 5, 4}},
	}
}

func testDescriptors() map[string]TensorDescriptor {
	d := make(map[string]TensorDescriptor)
	for _, name := range visionRequiredNames() {
		d[name] = TensorDescriptor{Dtype: "F32", Shape: []int32{1}}
	}
	for _, name := range audioRequiredNames() {
		d[name] = TensorDescriptor{Dtype: "F32", Shape: []int32{1}}
	}
	d[AudioCodebook] = TensorDescriptor{Dtype: "F32", Shape: []int32{4096, 512}}
	return d
}

func TestMediaInventoryValidation(t *testing.T) {
	if got := len(VisionRequiredTensorNames()); got != 247 {
		t.Fatalf("vision required tensor count = %d, want 247", got)
	}
	if got := len(AudioRequiredTensorNames()); got != 63 {
		t.Fatalf("audio required tensor count = %d, want 63", got)
	}
	cfg, descriptors := testConfig(), testDescriptors()
	if err := ValidateVisionInventory(cfg, descriptors); err != nil {
		t.Fatal(err)
	}
	if err := ValidateAudioInventory(cfg, descriptors); err != nil {
		t.Fatal(err)
	}

	missing := testDescriptors()
	delete(missing, visionPrefix+"encoder.down.4.attn.2.v.bias")
	if err := ValidateVisionInventory(cfg, missing); err == nil {
		t.Fatal("incomplete vision inventory accepted")
	}
	badCodebook := testDescriptors()
	badCodebook[AudioCodebook] = TensorDescriptor{Dtype: "F32", Shape: []int32{512, 4096}}
	if err := ValidateAudioInventory(cfg, badCodebook); err == nil {
		t.Fatal("transposed audio codebook accepted")
	}
	badCfg := cfg
	badCfg.ImageTokenOffset++
	if err := ValidateVisionInventory(badCfg, descriptors); err == nil {
		t.Fatal("invalid vision config accepted")
	}
}
