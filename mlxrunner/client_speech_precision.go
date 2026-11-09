package mlxrunner

import "os/exec"

// configureSpeechFloat32 selects the precision required by the native speech
// codecs and reference encoders. M5's default TF32 float32 matmuls change codec
// IDs and waveform values. Set the child environment before MLX initializes;
// the daemon and other model runner processes keep their own settings.
func configureSpeechFloat32(cmd *exec.Cmd, architecture string) bool {
	switch architecture {
	case "Qwen3TTSForConditionalGeneration", "KokoroForSpeechSynthesis",
		"VoxtralTTSForConditionalGeneration", "ChatterboxForConditionalGeneration",
		"ChatterboxTurboForConditionalGeneration":
		setEnv(cmd, "MLX_ENABLE_TF32", "0")
		return true
	default:
		return false
	}
}
