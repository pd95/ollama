package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
)

func TestQwen3ASRRequestCacheRetention(t *testing.T) {
	for _, tc := range []struct {
		name    string
		model   model.Model
		discard bool
	}{
		{"ASR", &qwen3asr.Model{}, true},
		{"other models", textOnlyModel{}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			env := newTransformerEnv()
			defer env.pc.close()
			r := &Runner{Model: tc.model, cache: env.pc}
			for i := range 16 {
				inputs := []int32{1, 2, 3, int32(i + 100), 5, 6}
				simulateRequest(t, env.pc, inputs, []int32{7, 8})
				walkNodes(env.pc.root, func(n *trieNode) bool {
					for _, snapshot := range n.snapshots {
						if snapshot != nil {
							snapshot.(*fakeSnapshot).materialize(1024)
						}
					}
					return true
				})
				if env.pc.pagedOutBytes == 0 {
					t.Fatal("test did not retain snapshots before cleanup")
				}
				r.finishRequestCache()
				if tc.discard && (env.pc.root != nil || env.pc.pagedOutBytes != 0) {
					t.Fatalf("request %d retained ASR history: %d bytes", i, env.pc.pagedOutBytes)
				}
				if !tc.discard && env.pc.root == nil {
					t.Fatal("another model's cache was discarded")
				}
			}
		})
	}
}
