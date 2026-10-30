package chatterbox

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/llm"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func compare(t *mlxtest.T, label string, got, want *mlx.Array, atol, rtol float64) {
	t.Helper()
	if got == nil || want == nil || !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("%s: incompatible shapes", label)
	}
	g := mlx.Contiguous(got.AsType(mlx.DTypeFloat32), false).Floats()
	w := mlx.Contiguous(want.AsType(mlx.DTypeFloat32), false).Floats()
	maxErr, bad := float64(0), 0
	for i, v := range g {
		err := math.Abs(float64(v - w[i]))
		maxErr = math.Max(maxErr, err)
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || err > atol+rtol*math.Abs(float64(w[i])) {
			bad++
		}
	}
	t.Logf("%s %v: max absolute error %.7g; %d/%d outside tolerance", label, got.Dims(), maxErr, bad, len(g))
	if bad > 0 {
		t.Errorf("%s: numerical mismatch", label)
	}
}

func sourceTensors(t *mlxtest.T, path string) map[string]*mlx.Array {
	t.Helper()
	file, err := mlx.LoadSafetensorsNative(path)
	if err != nil {
		t.Fatal(err)
	}
	file.Free()
	out := map[string]*mlx.Array{}
	for name, value := range mlx.Load(path) {
		out[name] = value
	}
	return out
}

func TestHiFTFinalActivationProfiles(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		for _, tc := range []struct {
			slope    float32
			turbo    bool
			negative float32
		}{{.1, false, -.2}, {.01, false, -.02}, {.1, true, -.02}} {
			h := &hiFT{finalSlope: tc.slope, turbo: tc.turbo, post: conv{weight: mlx.FromValues([]float32{1}, 1, 1, 1), stride: 1, dilation: 1, groups: 1}}
			x := mlx.FromValues([]float32{-2, 0, 2}, 1, 1, 3)
			compare(t, "final activation", h.postSpectrum(x), mlx.FromValues([]float32{tc.negative, 0, 2}, 1, 1, 3), 1e-7, 1e-7)
		}
	})
}

func TestReference(t *testing.T) {
	if name := os.Getenv("OLLAMA_SPEECH_REFERENCE_MODEL"); name == "mlx-community/chatterbox-turbo-4bit" || name == "ResembleAI/chatterbox-turbo" {
		t.Skip("Turbo has separate references")
	}
	source, reference := os.Getenv("OLLAMA_S3_SOURCE"), os.Getenv("OLLAMA_SPEECH_REFERENCE_DIR")
	if source == "" || reference == "" {
		t.Skip("set pinned source and reference directories")
	}
	if err := mlx.CheckInit(); err != nil {
		t.Fatal(err)
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		want := sourceTensors(t, filepath.Join(reference, "core.safetensors"))
		resampled := audio.ResampleKaiserBest(want["reference_pcm"].Floats(), 24000, 16000)
		compare(t, "reference polyphase resampling", mlx.FromValues(resampled, len(resampled)), want["reference_16k"], 1e-7, 1e-7)
		s3, err := loadS3Tokenizer(sourceTensors(t, filepath.Join(source, "model.safetensors")))
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "S3 mel", s3.mels(want["reference_16k"]), want["s3_mels"], 1e-6, 1e-6)
		features := s3.features(want["s3_mels"])
		compare(t, "S3 features", features, want["s3_hidden"], 1e-5, 1e-4)
		compare(t, "S3 fixed features codes", s3.quantize(want["s3_hidden"]), want["s3_tokens"], 0, 0)
		compare(t, "S3 end-to-end codes", s3.quantize(s3.features(s3.mels(want["reference_16k"]))), want["s3_tokens"], 0, 0)
		root := os.Getenv("OLLAMA_CHATTERBOX_SOURCE")
		if root == "" {
			t.Fatal("set pinned Chatterbox source")
		}
		tensors := sourceTensors(t, filepath.Join(root, "model.safetensors"))
		w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, 0, 0, "", nil)}
		voice := loadVoiceEncoder(w)
		if w.err != nil {
			t.Fatal(w.err)
		}
		trimmed := trimVoiceReference(resampled)
		compare(t, "VE trimmed PCM", mlx.FromValues(trimmed, len(trimmed)), want["ve_trimmed_pcm"], 0, 0)
		compare(t, "VE mel", voice.mels(want["ve_trimmed_pcm"]), want["ve_mels"], 1e-6, 1e-6)
		compare(t, "VE partial windows", voice.partials(want["ve_mels"]), want["ve_partials"], 0, 0)
		x := want["ve_partials"]
		for i, l := range voice.layers {
			x = l.forward(x)
			compare(t, "VE LSTM "+string(rune('0'+i)), x, want["ve_layer_"+string(rune('0'+i))], 1e-6, 1e-6)
		}
		compare(t, "VE partial embeddings", voice.forward(want["ve_partials"]), want["ve_partial_embeddings"], 1e-6, 1e-6)
		compare(t, "VE speaker embedding", voice.embedding(want["ve_mels"]), want["t3_speaker"], 1e-6, 1e-6)
		camp := loadCAMP(w)
		if w.err != nil {
			t.Fatal(w.err)
		}
		compare(t, "CAMP fbank", camp.fbank(want["reference_16k"]), want["camp_fbank"], 1e-6, 1e-6)
		x = camp.head(mlx.Transpose(want["camp_features"], 0, 2, 1))
		compare(t, "CAMP head", x, want["camp_head"], 1e-6, 1e-6)
		x = camp.initial(x)
		compare(t, "CAMP initial TDNN", x, want["camp_tdnn"], 1e-6, 1e-6)
		for i := range camp.blocks {
			x = camp.block(x, i)
			compare(t, "CAMP dense block", x, want["camp_block_"+string(rune('0'+i))], 1e-5, 1e-4)
			x = camp.transitions[i].forward(x)
			compare(t, "CAMP transition", x, want["camp_transit_"+string(rune('0'+i))], 1e-5, 1e-4)
		}
		compare(t, "CAMP output", camp.forward(want["camp_features"]), want["camp_output"], 1e-5, 1e-4)
		generator := loadT3(w)
		conformer := loadConformer(w)
		flow := loadFlowDecoder(w)
		hifi := loadHiFT(w)
		if w.err != nil {
			t.Fatal(w.err)
		}
		mel := newReferenceMel()
		compare(t, "S3Gen reference mel", mel.forward(want["reference_pcm"]), want["gen_prompt_feat"], 1e-6, 1e-6)
		candidate := &Model{s3: s3, voice: voice, camp: camp, t3: generator, mel: mel, conformer: conformer, embedding: w.get("s3gen.flow.input_embedding.weight"), projection: w.linear("s3gen.flow.encoder_proj"), speaker: w.linear("s3gen.flow.spk_embed_affine_layer")}
		referenceWAV, err := os.ReadFile(filepath.Join(reference, "reference.wav"))
		if err != nil {
			t.Fatal(err)
		}
		held := mlx.NewScope()
		defer held.Close()
		profile, err := candidate.referenceProfile(context.Background(), llm.SpeechRequest{ReferenceAudio: &llm.ReferenceAudio{Format: "wav", Data: referenceWAV}}, held)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "profile S3 tokens", profile.tokens, want["gen_prompt_token"], 0, 0)
		compare(t, "profile S3 mel", profile.mels, want["gen_prompt_feat"], 1e-6, 1e-6)
		compare(t, "profile CAMP embedding", profile.speaker, want["gen_embedding"], 1e-5, 1e-4)
		compare(t, "profile T3 conditioning", profile.conditioning, want["t3_conditioning"], 1e-6, 1e-6)
		conditioning := generator.conditioning(want["t3_speaker"], want["t3_reference_tokens"], want["t3_emotion"])
		compare(t, "T3 conditioning", conditioning, want["t3_conditioning"], 1e-6, 1e-6)
		raw, err := os.ReadFile(filepath.Join(root, "tokenizer.json"))
		if err != nil {
			t.Fatal(err)
		}
		tok, err := tokenizer.LoadFromBytes(raw)
		if err != nil {
			t.Fatal(err)
		}
		casesRaw, err := os.ReadFile(filepath.Join(reference, "text-cases.json"))
		if err != nil {
			t.Fatal(err)
		}
		var cases []struct {
			Text, Language string
			IDs            []int32
		}
		if err := json.Unmarshal(casesRaw, &cases); err != nil {
			t.Fatal(err)
		}
		for _, c := range cases {
			if ids := tok.Encode(normalizedV3Text(c.Text, c.Language), false); !slices.Equal(ids, c.IDs) {
				t.Errorf("text case %q: got %v, want %v", c.Text, ids, c.IDs)
			}
		}
		for _, language := range []string{"en", "de"} {
			v := sourceTensors(t, filepath.Join(reference, "hello-"+language+".safetensors"))
			text := "Hello."
			if language == "de" {
				text = "Guten Tag."
			}
			ids := tok.Encode(normalizedV3Text(text, language), false)
			compare(t, language+" text IDs", mlx.FromValues(ids, 1, len(ids)), v["text_tokens"], 0, 0)
			prefix := generator.prompt(conditioning, v["padded_text_tokens"])
			compare(t, language+" T3 prompt", prefix, v["input_embeds"], 1e-6, 1e-6)
			state := newT3State()
			hidden := generator.forward(prefix, state)
			compare(t, language+" T3 hidden", hidden, v["t3_hidden"], 1e-5, 1e-4)
			compare(t, language+" T3 logits", generator.head.Forward(hidden.Slice(mlx.Slice(), mlx.Slice(hidden.Dim(1)-1, hidden.Dim(1)), mlx.Slice())), v["t3_logits"], 1e-5, 1e-4)
			first := v["speech_tokens"].AsType(mlx.DTypeInt32).Slice(mlx.Slice(), mlx.Slice(0, 1))
			next := mlx.Add(mlx.Take(generator.speech, first, 0), mlx.Take(generator.speechPosition, mlx.FromValues([]int32{1}, 1, 1), 0))
			next = mlx.Concatenate([]*mlx.Array{next, next}, 0)
			compare(t, language+" next embedding", next, v["next_embedding"], 0, 0)
			nextHidden := generator.forward(next, state)
			compare(t, language+" cached T3 hidden", nextHidden, v["next_hidden"], 1e-5, 1e-4)
			compare(t, language+" cached T3 logits", generator.head.Forward(nextHidden), v["next_logits"], 1e-5, 1e-4)
			state.close()
			codes := v["speech_tokens"].AsType(mlx.DTypeInt32).Ints()
			mu, mask, speaker, condition := candidate.flowInputs(codes, profile)
			compare(t, language+" full flow mu", mu, v["flow_mu"], 1e-5, 1e-4)
			compare(t, language+" full flow mask", mask, v["flow_mask"], 0, 0)
			compare(t, language+" full flow speaker", speaker, v["flow_speaker"], 1e-5, 1e-4)
			compare(t, language+" full flow condition", condition, v["flow_condition"], 1e-6, 1e-6)
			position := conformerPosition(v["flow_encoder_input"].Dim(1))
			compare(t, language+" Conformer positions", position, v["flow_encoder_position"], 0, 0)
			embedded := conformerEmbed(v["flow_encoder_input"], conformer.input, conformer.inputNorm)
			compare(t, language+" Conformer embedding", embedded, v["flow_encoder_embedded"], 1e-6, 1e-6)
			lookahead := conformer.lookahead(embedded)
			compare(t, language+" Conformer lookahead", lookahead, v["flow_encoder_lookahead"], 1e-6, 1e-6)
			encoded := conformer.beforeUp(lookahead, position)
			compare(t, language+" Conformer before up", encoded, v["flow_encoder_before_up"], 1e-5, 1e-4)
			upsampled := conformer.up(encoded)
			compare(t, language+" Conformer upsampled", upsampled, v["flow_encoder_upsampled"], 1e-5, 1e-4)
			upEmbedded := conformerEmbed(mlx.Transpose(upsampled, 0, 2, 1), conformer.upInput, conformer.upNorm)
			compare(t, language+" Conformer up embedding", upEmbedded, v["flow_encoder_up_embedded"], 1e-5, 1e-4)
			upPosition := conformerPosition(upEmbedded.Dim(1))
			compare(t, language+" Conformer up positions", upPosition, v["flow_encoder_up_position"], 0, 0)
			compare(t, language+" Conformer output", conformer.afterUp(upEmbedded, upPosition), v["flow_encoder_output"], 1e-5, 1e-4)
			compare(t, language+" Conformer full path", conformer.forward(v["flow_encoder_input"]), v["flow_encoder_output"], 1e-5, 1e-4)
			compare(t, language+" flow time schedule", flowTimes(10), v["flow_times"], 0, 0)
			duplicate := func(x *mlx.Array) *mlx.Array { return mlx.Concatenate([]*mlx.Array{x, x}, 0) }
			conditional := func(x *mlx.Array) *mlx.Array {
				return mlx.Concatenate([]*mlx.Array{x, mlx.Zeros(x.DType(), x.Dims()...)}, 0)
			}
			velocity := flow.eval(duplicate(v["flow_noise"]), duplicate(v["flow_mask"]), conditional(v["flow_mu"]), mlx.Zeros(mlx.DTypeFloat32, 2), conditional(v["flow_speaker"]), conditional(v["flow_condition"]))
			compare(t, language+" initial flow velocity", velocity, v["flow_initial_velocity"], 1e-5, 1e-4)
			mel := flow.solve(v["flow_noise"], v["flow_times"], v["flow_mu"], v["flow_mask"], v["flow_speaker"], v["flow_condition"])
			compare(t, language+" solved flow mels", mel, v["flow_full_mels"], 1e-5, 1e-4)
			compare(t, language+" HiFT F0", hifi.pitch(v["mels"]), v["f0"], 1e-5, 1e-4)
			compare(t, language+" HiFT fixed-noise harmonics", hifi.harmonics(v["f0"], v["nsf_phase"], v["nsf_standard_noise"]), v["nsf_harmonics"], 1e-6, 1e-6)
			compare(t, language+" HiFT fixed-noise excitation", hifi.excitation(v["f0"], v["nsf_phase"], v["nsf_standard_noise"]), v["vocoder_source"], 1e-6, 1e-6)
			real, imag := hifi.sourceSTFT(v["vocoder_source"])
			compare(t, language+" HiFT source FFT real", real, v["source_stft_real"], 0, 0)
			compare(t, language+" HiFT source FFT imag", imag, v["source_stft_imag"], 0, 0)
			sourceSpectrum := mlx.Concatenate([]*mlx.Array{real, imag}, 1)
			x := mlx.Transpose(hifi.pre.forward(mlx.Transpose(v["mels"], 0, 2, 1)), 0, 2, 1)
			for i := range hifi.ups {
				x = hifi.stage(x, sourceSpectrum, i)
				compare(t, language+" HiFT stage", x, v["vocoder_stage_"+string(rune('0'+i))], 1e-5, 1e-4)
			}
			spectrum := hifi.spectrum(v["mels"], v["vocoder_source"])
			compare(t, language+" HiFT spectrum", spectrum, v["vocoder_spectrum"], 1e-5, 1e-4)
			wave := hifi.inverse(spectrum)
			compare(t, language+" HiFT inverse spectrum", wave, v["audio_before_fade"], 1e-6, 1e-5)
			compare(t, language+" HiFT full fixed-source decode", hifi.decode(v["mels"], v["vocoder_source"]), v["audio_before_fade"], 1e-6, 1e-5)
			compare(t, language+" HiFT fade", fadeHiFT(wave), v["audio"], 1e-6, 1e-5)
		}
	})
}
