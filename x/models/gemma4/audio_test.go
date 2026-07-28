package gemma4

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

const releasedGemma4AudioConfig = `{
  "audio_config": {
    "attention_chunk_size": 12,
    "attention_context_left": 13,
    "attention_context_right": 0,
    "attention_invalid_logits_value": -1000000000.0,
    "attention_logit_cap": 50.0,
    "conv_kernel_size": 5,
    "gradient_clipping": 10000000000.0,
    "hidden_size": 1024,
    "num_attention_heads": 8,
    "num_hidden_layers": 12,
    "output_proj_dims": 1536,
    "residual_weight": 0.5,
    "rms_norm_eps": 0.000001,
    "subsampling_conv_channels": [128, 32],
    "use_clipped_linears": true
  }
}`

func TestParseReleasedAudioConfig(t *testing.T) {
	cfg, err := parseAudioConfig([]byte(releasedGemma4AudioConfig))
	if err != nil {
		t.Fatal(err)
	}
	if cfg == nil || cfg.HiddenSize != 1024 || cfg.NumHiddenLayers != 12 || cfg.OutputProjDims != 1536 {
		t.Fatalf("audio config = %+v", cfg)
	}
	if cfg.AttentionContextLeft != 13 || cfg.AttentionChunkSize != 12 || cfg.ConvKernelSize != 5 {
		t.Fatalf("audio context config = %+v", cfg)
	}

	if cfg, err := parseAudioConfig([]byte(`{"model_type":"gemma4"}`)); err != nil || cfg != nil {
		t.Fatalf("missing audio config = %+v, %v; want nil, nil", cfg, err)
	}
	bad := strings.Replace(releasedGemma4AudioConfig, `"num_attention_heads": 8`, `"num_attention_heads": 7`, 1)
	if _, err := parseAudioConfig([]byte(bad)); err == nil {
		t.Fatal("non-divisible head count: error = nil")
	}
}

func TestAudioAttentionMaskValues(t *testing.T) {
	valid := []bool{true, true, true, false}
	got := audioAttentionMaskValues(valid, 2, 2, 4, 2, 0)
	allowed := func(block, query, context int) bool {
		return got[(block*2+query)*4+context]
	}
	for _, tt := range []struct {
		block, query, context int
		want                  bool
	}{
		{0, 0, 0, false},
		{0, 0, 2, true},
		{0, 0, 3, false},
		{0, 1, 1, false},
		{0, 1, 3, true},
		{1, 0, 0, true},
		{1, 0, 2, true},
		{1, 0, 3, false},
		{1, 1, 3, false},
	} {
		if value := allowed(tt.block, tt.query, tt.context); value != tt.want {
			t.Errorf("mask[%d,%d,%d] = %v, want %v", tt.block, tt.query, tt.context, value, tt.want)
		}
	}
}

func TestParseGemma4AudioTokens(t *testing.T) {
	tokens := parseGemma4MediaTokens([]byte(`{
      "boa_token":"audio-start","audio_token":"audio-soft","eoa_token":"audio-end"
    }`), defaultGemma4MediaTokens())
	if tokens.BOA != "audio-start" || tokens.Audio != "audio-soft" || tokens.EOA != "audio-end" {
		t.Fatalf("audio tokens = %+v", tokens)
	}
	if tokens.Image != defaultGemma4ImageToken {
		t.Fatalf("image token = %q, want unchanged default", tokens.Image)
	}
}

func TestMediaTokenSpanAudio(t *testing.T) {
	start, end, err := mediaTokenSpan([]int32{1, 258881, 258881, 2}, 258881)
	if err != nil || start != 1 || end != 3 {
		t.Fatalf("mediaTokenSpan() = %d, %d, %v; want 1, 3, nil", start, end, err)
	}
}

func TestPrepareAudioMediaPrompt(t *testing.T) {
	tok := testGemma4AudioTokenizer(t)
	cfg := defaultAudioProcessorConfig()
	frames := make([][]float64, 16000)
	for i := range frames {
		frames[i] = []float64{0.25}
	}
	m := &Model{
		TextConfig:           &TextConfig{AudioTokenIDValue: 1},
		AudioConfig:          &AudioConfig{},
		AudioProcessorConfig: &cfg,
		tok:                  tok,
		mediaTokens:          defaultGemma4MediaTokens(),
	}
	media := []llm.MediaData{{ID: 7, Kind: llm.MediaKindAudio, Data: makeTestWAV(t, 1, 16, 16000, frames)}}
	prepared, err := m.PrepareMediaPrompt(context.Background(), "[img-7]", media)
	if err != nil {
		t.Fatal(err)
	}
	payload, ok := prepared.Payload.(*gemma4MediaPayload)
	if !ok || payload.Audio == nil {
		t.Fatalf("payload = %#v, want audio", prepared.Payload)
	}
	if got := payload.AudioEnd - payload.AudioStart; got != payload.Audio.SoftTokens {
		t.Fatalf("audio span = %d, want %d", got, payload.Audio.SoftTokens)
	}
	for i, id := range prepared.PLEInputIDs {
		if i >= payload.AudioStart && i < payload.AudioEnd {
			if id != 0 {
				t.Fatalf("PLEInputIDs[%d] = %d, want 0", i, id)
			}
		}
	}
	if len(prepared.BidirectionalSpans) != 0 {
		t.Fatalf("audio bidirectional spans = %v, want none", prepared.BidirectionalSpans)
	}
	if _, err := m.PrepareMediaPrompt(context.Background(), "missing", media); err == nil || !strings.Contains(err.Error(), "marker") {
		t.Fatalf("missing marker error = %v", err)
	}
	if _, err := m.PrepareMediaPrompt(context.Background(), "[img-7]", append(media, media[0])); err == nil || !strings.Contains(err.Error(), "exactly one") {
		t.Fatalf("multiple media error = %v", err)
	}
}

func TestPrepareAudioMediaEmbeddingsRejectsMissingWeights(t *testing.T) {
	skipIfNoMLX(t)
	m := &Model{
		TextConfig:  &TextConfig{EmbedScale: 1},
		EmbedTokens: nn.NewEmbedding(mlx.FromValues([]float32{0, 0, 1, 1, 2, 2}, 3, 2)),
	}
	prepared := &batch.PreparedInput{
		Tokens: []int32{0, 1, 2},
		Payload: &gemma4MediaPayload{
			Audio:      &gemma4AudioInput{SoftTokens: 1},
			AudioStart: 1,
			AudioEnd:   2,
		},
	}
	if err := m.PrepareMediaEmbeddings(prepared); err == nil || !strings.Contains(err.Error(), "audio weights are not loaded") {
		t.Fatalf("PrepareMediaEmbeddings() error = %v", err)
	}
}

func testGemma4AudioTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	data := []byte(`{
		"model":{"type":"BPE","vocab":{"<|audio>":0,"<|audio|>":1,"<audio|>":2},"merges":[]},
		"added_tokens":[
			{"id":0,"content":"<|audio>","special":true},
			{"id":1,"content":"<|audio|>","special":true},
			{"id":2,"content":"<audio|>","special":true}
		]
	}`)
	tok, err := tokenizer.LoadFromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}
