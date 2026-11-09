package qwen3tts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestBaseReference(t *testing.T) {
	source, reference := os.Getenv("OLLAMA_QWEN3_TTS_SOURCE"), os.Getenv("OLLAMA_SPEECH_REFERENCE_DIR")
	testBaseReference(t, source, reference)
}

func TestBaseReferenceOfficial(t *testing.T) {
	source, reference := os.Getenv("OLLAMA_QWEN3_TTS_OFFICIAL_SOURCE"), os.Getenv("OLLAMA_SPEECH_REFERENCE_DIR")
	testBaseReference(t, source, reference)
}

func testBaseReference(t *testing.T, source, reference string) {
	t.Helper()
	if source == "" || reference == "" {
		t.Skip("set pinned Base source and references")
	}
	raw, err := os.ReadFile(filepath.Join(source, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := struct {
		Type    string        `json:"tts_model_type"`
		Speaker speakerConfig `json:"speaker_encoder_config"`
	}{Speaker: defaultSpeakerConfig()}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.Type != "base" {
		t.Skip("not a Base checkpoint")
	}
	if err := mlx.CheckInit(); err != nil {
		t.Fatal(err)
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		weights := sourceTensors(t, filepath.Join(source, "model.safetensors"))
		speaker, err := loadSpeaker(cfg.Speaker, weights)
		if err != nil {
			t.Fatal(err)
		}
		want := sourceTensors(t, filepath.Join(reference, "core.safetensors"))
		compare(t, "speaker mel bank", speaker.melBank, want["mel_bank"], 0, 0)
		compare(t, "reference mels", speaker.mels(want["reference_pcm"]), want["mels"], 1e-6, 1e-6)
		x := speaker.initial.forward(mlx.Transpose(want["mels"], 0, 2, 1))
		compare(t, "speaker block 0", x, want["speaker_block_0"], 0, 0)
		for i, b := range speaker.blocks {
			x = b.forward(x)
			compare(t, "speaker residual", x, want["speaker_block_"+string(rune('1'+i))], 0, 0)
		}
		compare(t, "speaker mfa", speaker.features(want["mels"]), want["speaker_mfa"], 0, 0)
		compare(t, "speaker pooling", speaker.statistics(speaker.features(want["mels"])), want["speaker_pool"], 0, 0)
		compare(t, "speaker embedding", speaker.forward(want["mels"]), want["speaker_embedding"], 0, 0)
		codecRaw, err := os.ReadFile(filepath.Join(source, "speech_tokenizer/config.json"))
		if err != nil {
			t.Fatal(err)
		}
		encoder, err := loadEncoder(codecRaw, sourceTensors(t, filepath.Join(source, "speech_tokenizer/model.safetensors")))
		if err != nil {
			t.Fatal(err)
		}
		pcm := mlx.Reshape(want["reference_pcm"], 1, 1, -1)
		compare(t, "seanet", encoder.seanet(pcm), want["seanet"], 0, 0)
		compare(t, "encoder transformer", encoder.transform(want["seanet"]), want["encoder_transformer"], 1e-6, 1e-6)
		compare(t, "encoder downsample", encoderConv(encoder.downsample, want["encoder_transformer"], true), want["encoder_downsample"], 0, 0)
		compare(t, "reference codes fixed features", encoder.quantize(want["encoder_downsample"]), want["reference_codes"], 0, 0)
		compare(t, "reference codes end-to-end", encoder.forward(pcm), want["reference_codes"], 0, 0)
		parsed, err := parseConfig(raw)
		if err != nil {
			t.Fatal(err)
		}
		network, err := loadNetwork(parsed, weights, nil)
		if err != nil {
			t.Fatal(err)
		}
		codec, err := loadDecoder(codecRaw, sourceTensors(t, filepath.Join(source, "speech_tokenizer/model.safetensors")))
		if err != nil {
			t.Fatal(err)
		}
		for _, language := range []string{"en", "de"} {
			v := sourceTensors(t, filepath.Join(reference, "hello-"+language+".safetensors"))
			refIDs := v["reference_ids"].Ints()
			profile := &cloneProfile{speaker: speaker.forward(speaker.mels(want["reference_pcm"])), codes: mlx.Contiguous(mlx.Transpose(want["reference_codes"], 0, 2, 1).AsType(mlx.DTypeInt32), false).Ints(), text: refIDs[3 : len(refIDs)-2]}
			prefix, pad, err := network.prepareICL(v["target_ids"].Ints(), language, profile)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, language+" ICL prefix", prefix, v["input_embeds"], 0, 0)
			compare(t, language+" ICL pad", pad, v["tts_pad_embed"], 0, 0)
			state := newState(len(network.talker.body.layers))
			logits, hidden := network.talker.forward(prefix, state)
			compare(t, language+" ICL talker logits", logits, v["talker_logits"], 0, 0)
			compare(t, language+" ICL talker hidden", hidden, v["talker_hidden"], 0, 0)
			state.close()
			wave := codec.decode(v["full_decode_codes"])
			compare(t, language+" ICL full decode", wave, v["full_audio"], 0, 0)
			cut := len(profile.codes) / 16 * 1920
			compare(t, language+" ICL trimmed decode", mlx.Reshape(wave.Slice(mlx.Slice(), mlx.Slice(cut, wave.Dim(1))), -1), v["audio"], 0, 0)
		}
	})
}
