package mlxrunner

import (
	"os"
	"os/exec"
	"slices"
	"testing"
)

func TestSpeechRunnerFloat32Isolation(t *testing.T) {
	t.Setenv("MLX_ENABLE_TF32", "1")
	for _, architecture := range []string{
		"Qwen3TTSForConditionalGeneration", "KokoroForSpeechSynthesis",
		"VoxtralTTSForConditionalGeneration", "ChatterboxForConditionalGeneration",
		"ChatterboxTurboForConditionalGeneration",
	} {
		for _, inherited := range []bool{false, true} {
			cmd := &exec.Cmd{Env: []string{"UNCHANGED=value"}}
			if inherited {
				cmd.Env = append(cmd.Env, "MLX_ENABLE_TF32=1")
			}
			if !configureSpeechFloat32(cmd, architecture) {
				t.Fatalf("speech architecture %s did not select full float32", architecture)
			}
			if !slices.Contains(cmd.Env, "MLX_ENABLE_TF32=0") || slices.Contains(cmd.Env, "MLX_ENABLE_TF32=1") || !slices.Contains(cmd.Env, "UNCHANGED=value") {
				t.Errorf("architecture=%s inherited=%v child environment=%v", architecture, inherited, cmd.Env)
			}
			if os.Getenv("MLX_ENABLE_TF32") != "1" {
				t.Fatal("speech child configuration changed the parent environment")
			}
		}
	}
	for _, architecture := range []string{"", "LlamaForCausalLM", "GptOssForCausalLM", "Qwen3ASRForConditionalGeneration"} {
		for _, original := range [][]string{nil, {"UNCHANGED=value", "MLX_ENABLE_TF32=1"}, {"MLX_ENABLE_TF32=0"}} {
			cmd := &exec.Cmd{Env: slices.Clone(original)}
			if configureSpeechFloat32(cmd, architecture) || !slices.Equal(cmd.Env, original) {
				t.Errorf("non-TTS architecture %q changed child environment: %v", architecture, cmd.Env)
			}
		}
	}
}
