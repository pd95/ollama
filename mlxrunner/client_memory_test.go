package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestAutomaticMediaMemoryLimit(t *testing.T) {
	const gib = uint64(1 << 30)
	tests := []struct {
		name   string
		system ml.SystemInfo
		gpus   []ml.DeviceInfo
		want   uint64
	}{
		{name: "unknown", want: 0},
		{name: "16 GiB Metal", system: ml.SystemInfo{TotalMemory: 16 * gib}, gpus: []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Metal"}, TotalMemory: 16 * gib}}, want: 12 * gib},
		{name: "24 GiB Metal", system: ml.SystemInfo{TotalMemory: 24 * gib}, gpus: []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Metal"}, TotalMemory: 24 * gib}}, want: 18 * gib},
		{name: "128 GiB Metal", system: ml.SystemInfo{TotalMemory: 128 * gib}, gpus: []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Metal"}, TotalMemory: 128 * gib}}, want: 96 * gib},
		{name: "dedicated GPU", system: ml.SystemInfo{TotalMemory: 128 * gib}, gpus: []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, TotalMemory: 32 * gib}}, want: 24 * gib},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := automaticMediaMemoryLimit(tt.system, tt.gpus); got != tt.want {
				t.Fatalf("automaticMediaMemoryLimit() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestLowerMediaMemoryLimit(t *testing.T) {
	tests := []struct {
		automatic, configured uint64
		want                  uint64
		ignored               bool
	}{
		{automatic: 100, want: 100},
		{automatic: 100, configured: 80, want: 80},
		{automatic: 100, configured: 100, want: 100},
		{automatic: 100, configured: 120, want: 100, ignored: true},
		{configured: 80, want: 80},
	}
	for _, tt := range tests {
		got, ignored := lowerMediaMemoryLimit(tt.automatic, tt.configured)
		if got != tt.want || ignored != tt.ignored {
			t.Errorf("lowerMediaMemoryLimit(%d, %d) = (%d, %v), want (%d, %v)", tt.automatic, tt.configured, got, ignored, tt.want, tt.ignored)
		}
	}
}
