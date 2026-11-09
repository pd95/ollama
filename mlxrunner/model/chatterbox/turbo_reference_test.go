package chatterbox

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
)

func TestTurboReference(t *testing.T) {
	if name := os.Getenv("OLLAMA_SPEECH_REFERENCE_MODEL"); name != "mlx-community/chatterbox-turbo-4bit" && name != "ResembleAI/chatterbox-turbo" {
		t.Skip("requires pinned Turbo sources")
	}
	source, reference := os.Getenv("OLLAMA_CHATTERBOX_SOURCE"), os.Getenv("OLLAMA_SPEECH_REFERENCE_DIR")
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		want := sourceTensors(t, filepath.Join(reference, "core.safetensors"))
		tensors := turboTensorAliases(sourceTensors(t, filepath.Join(source, "model.safetensors")))
		w := &weights{tensors: tensors, factory: model.NewLinearFactory(tensors, 64, 4, "affine", nil)}
		gen := loadTurboT3(w)
		encoder := loadConformer(w)
		flow := loadFlowDecoder(w)
		hi := loadHiFT(w)
		flow.mixer = w.linear("s3gen.flow.decoder.estimator.time_embed_mixer")
		encoder.positions = tensors["s3gen.flow.encoder.embed.pos_enc.pe"]
		encoder.upPositions = tensors["s3gen.flow.encoder.up_embed.pos_enc.pe"]
		hi.turbo = true
		hi.window = w.get("s3gen.mel2wav.stft_window")
		if w.err != nil {
			t.Fatal(w.err)
		}
		condition := gen.conditioning(want["default_t3.speaker_emb"], want["default_t3.cond_prompt_speech_tokens"])
		compare(t, "Turbo conditioning", condition, want["conditioning"], 1e-6, 1e-5)
		prompt := gen.prompt(condition, want["text_tokens"])
		compare(t, "Turbo prompt", prompt, want["input_embeds"], 1e-6, 1e-5)
		state := newTurboState()
		defer state.close()
		hidden := gen.forward(prompt, state)
		compare(t, "Turbo prefill hidden", hidden, want["t3_hidden"], 1e-5, 1e-4)
		logits := gen.head.Forward(hidden.Slice(mlx.Slice(), mlx.Slice(hidden.Dim(1)-1, hidden.Dim(1)), mlx.Slice()))
		compare(t, "Turbo prefill logits", logits, want["t3_logits"], 1e-5, 1e-4)
		codes := want["raw_speech_tokens"].Slice(mlx.Slice(), mlx.Slice(0, 1))
		next := gen.speech.Forward(codes)
		compare(t, "Turbo feedback embedding", next, want["next_embedding"], 0, 0)
		next = gen.forward(next, state)
		compare(t, "Turbo cached hidden", next, want["next_hidden"], 1e-5, 1e-4)
		compare(t, "Turbo cached logits", gen.head.Forward(next), want["next_logits"], 1e-5, 1e-4)
		compare(t, "Turbo Conformer", encoder.forward(want["flow_encoder_input"]), want["flow_encoder_output"], 1e-5, 1e-4)
		compare(t, "Turbo mean velocity", flow.meanVelocity(want["flow_noise"], want["flow_mask"], want["flow_mu"], want["flow_times"].Slice(mlx.Slice(0, 1)), want["flow_times"].Slice(mlx.Slice(1, 2)), want["flow_speaker"], want["flow_condition"]), want["flow_initial_velocity"], 1e-5, 1e-4)
		solved, err := flow.solveMean(context.Background(), want["flow_noise"], want["flow_times"], want["flow_mu"], want["flow_mask"], want["flow_speaker"], want["flow_condition"])
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "Turbo solved mels", solved, want["flow_full_mels"], 1e-5, 1e-4)
		compare(t, "Turbo HiFT pitch", hi.pitch(want["mels"]), want["f0"], 1e-5, 1e-4)
		compare(t, "Turbo HiFT excitation", hi.excitation(want["f0"], want["nsf_phase"], want["nsf_standard_noise"]), want["vocoder_source"], 1e-5, 1e-4)
		real, imag := hi.sourceSTFT(want["vocoder_source"])
		compare(t, "Turbo source FFT real", real, want["source_stft_real"], 1e-7, 1e-6)
		compare(t, "Turbo source FFT imag", imag, want["source_stft_imag"], 1e-7, 1e-6)
		spec := mlx.Concatenate([]*mlx.Array{want["source_stft_real"], want["source_stft_imag"]}, 1)
		x := mlx.Transpose(hi.pre.forward(mlx.Transpose(want["mels"], 0, 2, 1)), 0, 2, 1)
		for i := range 3 {
			x = hi.stage(x, spec, i)
			compare(t, "Turbo HiFT stage", x, want["vocoder_stage_"+string(rune('0'+i))], 1e-5, 1e-4)
		}
		compare(t, "Turbo HiFT spectrum", hi.spectrum(want["mels"], want["vocoder_source"]), want["vocoder_spectrum"], 1e-5, 1e-4)
		compare(t, "Turbo inverse fixed spectrum", hi.inverse(want["vocoder_spectrum"]), want["audio_before_fade"], 1e-6, 1e-5)
		compare(t, "Turbo waveform", hi.decode(want["mels"], want["vocoder_source"]), want["audio_before_fade"], 1e-5, 1e-4)
		v3 := sourceTensors(t, filepath.Join(os.Getenv("OLLAMA_CHATTERBOX_REFERENCE_SOURCE"), "model.safetensors"))
		for name, a := range v3 {
			if strings.HasPrefix(name, "s3gen.speaker_encoder.") {
				tensors[name] = a
			}
		}
		candidate := &turboModel{generator: gen, Model: Model{voice: loadVoiceEncoder(w), camp: loadCAMP(w), mel: newReferenceMel()}}
		candidate.voice.epsilon = 1e-8
		candidate.voice.turbo = true
		candidate.mel.turbo = true
		var errS3 error
		candidate.s3, errS3 = loadS3Tokenizer(sourceTensors(t, filepath.Join(os.Getenv("OLLAMA_S3_SOURCE"), "model.safetensors")))
		if errS3 != nil {
			t.Fatal(errS3)
		}
		if want["clone_s3_mels"] != nil {
			// Independent CPU PyTorch STFT/mel reference with original saved assets.
			pcm := want["clone_s3_pcm"]
			padded := mlx.Pad(pcm, []int{0}, []int{200}, []int{200}, mlx.FromValue(float32(0)), "reflect")
			frames := padded.AsStrided([]int{pcm.Size() / 160, 400}, []int{160, 1}, 0)
			power := mlx.RFFTCPU(mlx.Mul(frames, candidate.s3.window), 400, -1).Abs().Power(mlx.FromValue(2))
			compare(t, "original S3 FFT power", power, want["clone_s3_power"], 1e-5, 1e-5)
			mel := mlx.MatmulCPU(candidate.s3.bank, mlx.Transpose(power, 1, 0))
			compare(t, "original S3 raw mel", mel, want["clone_s3_raw_mel"], 1e-6, 1e-5)
			compare(t, "original S3 Torch frontend", candidate.s3.mels(want["clone_s3_pcm"]), want["clone_s3_mels"], 5e-6, 5e-6)
		}
		referenceWAV, err := os.ReadFile(filepath.Join(reference, "reference.wav"))
		if err != nil {
			t.Fatal(err)
		}
		held := mlx.NewScope()
		defer held.Close()
		profile, err := candidate.referenceProfileTurbo(context.Background(), llm.SpeechRequest{ReferenceAudio: &llm.ReferenceAudio{Format: "wav", Data: referenceWAV}}, held)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "Turbo clone S3 tokens", profile.tokens, want["clone_gen_prompt_token"], 0, 0)
		compare(t, "Turbo clone window", candidate.mel.window, want["clone_gen_window"], 0, 0)
		compare(t, "Turbo clone mel bank", candidate.mel.bank, want["clone_gen_bank"], 0, 0)
		pcm := want["clone_pcm"]
		padded := mlx.Pad(pcm, []int{0}, []int{720}, []int{720}, mlx.FromValue(float32(0)), "reflect")
		frames := padded.AsStrided([]int{1 + (padded.Size()-1920)/480, 1920}, []int{480, 1}, 0)
		mag := mlx.RFFTCPU(mlx.Mul(frames, candidate.mel.window), 1920, -1).Abs()
		// The unscaled 1920-point CPU FFT uses a different float32 butterfly
		// implementation from NumPy. Keep its operator bound separate from the
		// unchanged final mel/conditioning tolerances below.
		compare(t, "Turbo clone FFT magnitude", mag, want["clone_gen_magnitudes"], 1e-5, 1e-5)
		compare(t, "Turbo clone VE mels", candidate.voice.mels(want["clone_ve_pcm"]), want["clone_ve_mels"], 1e-6, 1e-5)
		compare(t, "Turbo clone VE partials", candidate.voice.partials(want["clone_ve_mels"]), want["clone_ve_partials"], 0, 0)
		compare(t, "Turbo clone VE fixed mels", candidate.voice.embedding(want["clone_ve_mels"]), want["clone_t3_speaker"], 1e-6, 1e-5)
		compare(t, "Turbo clone reference mels", profile.mels, want["clone_gen_prompt_feat"], 1e-5, 1e-5)
		compare(t, "Turbo clone CAMP embedding", profile.speaker, want["clone_gen_embedding"], 1e-5, 1e-4)
		compare(t, "Turbo clone T3 conditioning", profile.conditioning, want["clone_conditioning"], 1e-5, 1e-4)
	})
}
