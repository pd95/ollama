package create

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func writeQwen3TTSSource(t *testing.T, dir string) {
	t.Helper()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3TTSForConditionalGeneration"],"model_type":"qwen3_tts","tts_model_type":"custom_voice","tts_model_size":"0b6","talker_config":{"hidden_size":1024,"max_position_embeddings":32768,"spk_id":{"ryan":3061,"vivian":3065}}}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("talker.text_projection.linear_fc1.weight", "F32", []int32{4, 4}, make([]byte, 64)),
	})
	codec := filepath.Join(dir, "speech_tokenizer")
	if err := os.Mkdir(codec, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string]string{
		"speech_tokenizer/config.json": `{"decoder_config":{"hidden_size":512},"output_sample_rate":24000}`,
		"vocab.json":                   `{"H":0,"i":1,"Hi":2}`,
		"merges.txt":                   "#version: 0.2\nH i\n",
		"tokenizer_config.json":        `{"tokenizer_class":"Qwen2Tokenizer","added_tokens_decoder":{"10":{"content":"<|im_start|>","special":true},"11":{"content":"<|im_end|>","special":true}}}`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	createTestSafetensors(t, filepath.Join(codec, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("decoder.pre_conv.conv.weight", "F32", []int32{4, 3, 4}, make([]byte, 192)),
	})
}

func TestQwen3TTSTransferAssets(t *testing.T) {
	dir := t.TempDir()
	writeQwen3TTSSource(t, dir)
	weights, err := SafetensorsWeightFiles(dir)
	if err != nil || !slices.Contains(weights, "speech_tokenizer/model.safetensors") {
		t.Fatalf("weights = %v, %v", weights, err)
	}
	configs, err := SafetensorsConfigFiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"merges.txt", "vocab.json", "speech_tokenizer/config.json"} {
		if !slices.Contains(configs, name) {
			t.Errorf("missing transfer metadata %s: %v", name, configs)
		}
	}
}

func TestQwen3TTSImportSynthesizesTokenizerAndCodec(t *testing.T) {
	dir := t.TempDir()
	writeQwen3TTSSource(t, dir)
	store := newCaptureStore()
	var info ManifestInfo
	err := Create(t.Context(), "qwen-tts-test", dir, testPipelineOptions(), store, func(_ context.Context, _ string, got ManifestInfo) error {
		info = got
		return nil
	}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"speech_tokenizer.decoder.pre_conv.conv.weight", "speech_tokenizer/config.json", "tokenizer.json"} {
		if store.blobs[name] == nil {
			t.Errorf("missing imported asset %s: %v", name, store.names())
		}
	}
	if !slices.Equal(info.ModelConfig.Capabilities, []string{"speech"}) || !slices.Equal(info.ModelConfig.Voices, []string{"ryan", "vivian"}) {
		t.Errorf("capabilities %v, voices %v", info.ModelConfig.Capabilities, info.ModelConfig.Voices)
	}
	tok, err := tokenizer.LoadFromBytes(store.blobs["tokenizer.json"])
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("<|im_start|>Hi<|im_end|>", false); !slices.Equal(got, []int32{10, 2, 11}) {
		t.Errorf("encoded IDs = %v", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); !os.IsNotExist(err) {
		t.Fatalf("source tokenizer.json was written: %v", err)
	}
}

func TestQwen3TTSRejectsIncompleteSourcesBeforeWriting(t *testing.T) {
	for _, file := range []string{"speech_tokenizer/model.safetensors", "speech_tokenizer/config.json", "vocab.json", "merges.txt", "tokenizer_config.json"} {
		t.Run(file, func(t *testing.T) {
			dir := t.TempDir()
			writeQwen3TTSSource(t, dir)
			if err := os.Remove(filepath.Join(dir, file)); err != nil {
				t.Fatal(err)
			}
			store := newCaptureStore()
			err := Create(t.Context(), "qwen-tts-test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
			if err == nil || !strings.Contains(err.Error(), file) || len(store.blobs) != 0 {
				t.Fatalf("error %v, blobs %v", err, store.names())
			}
		})
	}
}

func TestQwen3TTSRejectsCodecNameCollision(t *testing.T) {
	dir := t.TempDir()
	writeQwen3TTSSource(t, dir)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("speech_tokenizer.decoder.pre_conv.conv.weight", "F32", []int32{4, 3, 4}, make([]byte, 192)),
	})
	store := newCaptureStore()
	err := Create(t.Context(), "qwen-tts-test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "already defines") || len(store.blobs) != 0 {
		t.Fatalf("error %v, blobs %v", err, store.names())
	}
}

func TestQwen3TTSRejectsCollidingAddedTokens(t *testing.T) {
	dir := t.TempDir()
	writeQwen3TTSSource(t, dir)
	data, err := json.Marshal(map[string]any{"tokenizer_class": "Qwen2Tokenizer", "added_tokens_decoder": map[string]any{"0": map[string]any{"content": "other", "special": true}}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	store := newCaptureStore()
	err = Create(t.Context(), "qwen-tts-test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
	if !errors.Is(err, ErrInvalidTokenizer) || len(store.blobs) != 0 {
		t.Fatalf("error %v, blobs %v", err, store.names())
	}
}

func TestQwen3TTSPreservesCodecPrecisionWhenQuantizing(t *testing.T) {
	dir := t.TempDir()
	writeQwen3TTSSource(t, dir)
	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatal(err)
	}
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatal(err)
	}
	shape := []int32{64, 64}
	for _, name := range []string{"speech_tokenizer.decoder.pre_transformer.layers.0.self_attn.q_proj.weight", "speaker_encoder.fc.weight"} {
		if got := policy.quantizationType(name, shape, "int4"); got != "" {
			t.Errorf("%s quantization = %q, want source precision", name, got)
		}
	}
	if got := policy.quantizationType("talker.model.layers.0.self_attn.q_proj.weight", shape, "int4"); got == "" {
		t.Error("talker projection was not eligible for requested quantization")
	}
}

// This integration test reads headers and tokenizer files only. It never loads
// model weights into MLX or writes into the optional local checkpoint.
func TestQwen3TTSReferenceTokenizer(t *testing.T) {
	dir := os.Getenv("OLLAMA_QWEN3_TTS_SOURCE")
	if dir == "" {
		t.Skip("set OLLAMA_QWEN3_TTS_SOURCE to the 0.6B CustomVoice 8-bit snapshot")
	}
	validated, err := qwen3TTSTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadFromBytes(validated.data)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text string  `json:"text"`
		IDs  []int32 `json:"ids"`
	}
	data, err := os.ReadFile("testdata/qwen3tts-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		text := "<|im_start|>assistant\n" + c.Text + "<|im_end|>\n<|im_start|>assistant\n"
		if ids := tok.Encode(text, false); !slices.Equal(ids, c.IDs) {
			t.Errorf("%q: IDs %v, want %v", c.Text, ids, c.IDs)
		}
	}
	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatal(err)
	}
	inv, err = prepareQwen3TTSInventory(inv)
	if err != nil {
		t.Fatal(err)
	}
	class, err := Classify(inv, "")
	if err != nil {
		t.Fatal(err)
	}
	if class.Kind != SourcePrequantized || class.Quantize != "int8" {
		t.Fatalf("unexpected checkpoint classification: %+v", class)
	}
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatal(err)
	}
	specs, err := Plan(inv, class, policy)
	if err != nil {
		t.Fatal(err)
	}
	codecTensors := 0
	for _, spec := range specs {
		if strings.HasPrefix(spec.Name, "speech_tokenizer.") {
			codecTensors++
			if spec.Metadata["quant_type"] != "" {
				t.Errorf("codec inherited talker quantization: %+v", spec.Metadata)
			}
		}
	}
	if codecTensors == 0 {
		t.Fatal("no codec tensors planned")
	}
	t.Logf("matched %d reference prompts; planned %d blobs, %d speech-tokenizer tensors", len(cases), len(specs), codecTensors)
}
