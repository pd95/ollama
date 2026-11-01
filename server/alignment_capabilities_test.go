package server

import (
	"errors"
	"os"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
	"github.com/ollama/ollama/types/model"
)

func TestForcedAlignerCapabilities(t *testing.T) {
	m := &Model{Config: model.ConfigV2{
		ModelFormat: "safetensors", ModelFamily: "qwen3_forced_aligner",
		Capabilities: []string{"alignment", "audio"},
	}}
	if err := m.CheckCapabilities(model.CapabilityAlignment, model.CapabilityAudio); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(m.CheckCapabilities(model.CapabilityCompletion), errCapabilityCompletion) {
		t.Fatal("classification head was exposed as autoregressive completion")
	}
	if !model.CapabilityAlignment.IsValid() {
		t.Fatal("alignment manifest capability rejected")
	}
}

func TestForcedAlignerImportedInfo(t *testing.T) {
	name := os.Getenv("OLLAMA_QWEN3_ASR_IMPORTED")
	if name == "" {
		t.Skip("requires imported aligner")
	}
	mf, err := manifest.ParseNamedManifest(model.ParseName(name))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := mf.ReadConfig("config.json")
	if err != nil {
		t.Fatal(err)
	}
	if !qwen3asr.IsForcedAligner(raw) {
		t.Skip("ASR reference is not aligner")
	}
	info, err := GetModelInfo(api.ShowRequest{Model: name, Verbose: true})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(info.Capabilities, model.CapabilityAlignment) || slices.Contains(info.Capabilities, model.CapabilityCompletion) || len(info.Tensors) == 0 {
		t.Fatalf("incorrect native alignment model info: capabilities%v tensors%d", info.Capabilities, len(info.Tensors))
	}
	// Existing verbose inspection omits .bias entries. The full708-tensor
	// import schema is independently checked in create; inspect the task head.
	found := false
	for _, tensor := range info.Tensors {
		if tensor.Name == "thinker.lm_head.weight" {
			found = slices.Equal(tensor.Shape, []uint64{5000, 1024})
		}
	}
	if !found {
		t.Fatal("classification head missing from model inspection")
	}
}
