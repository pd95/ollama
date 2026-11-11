package stream_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	_ "github.com/ollama/ollama/mlxrunner/model/qwen3asr"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

// TestReplayNative is an opt-in eager/F32 native replay and lifetime check. It
// exercises actual inference through fresh per-update encoder and decoder state;
// it makes no claim of vLLM parity or immutable words before the final result.
func TestReplayNative(t *testing.T) {
	inputs, output := os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_INPUTS"), os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_OUTPUT")
	if inputs == "" || output == "" {
		t.Skip("set replay input receipt and output directory")
	}
	intervalMS := 2000
	if value := os.Getenv("OLLAMA_QWEN3_ASR_UPDATE_INTERVAL_MS"); value != "" {
		var err error
		intervalMS, err = strconv.Atoi(value)
		if err != nil || (intervalMS != 500 && intervalMS != 1000 && intervalMS != 2000) {
			t.Fatal("replay update interval must be 500, 1000 or 2000 milliseconds")
		}
	}
	freshSeed := os.Getenv("OLLAMA_QWEN3_ASR_FRESH_SEED_NATIVE") == "1"
	updateSamples := intervalMS * stream.SampleRate / 1000
	packetSamples := min(stream.MaxPacketSamples, updateSamples)
	if err := mlx.CheckInit(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(inputs)
	if err != nil {
		t.Fatal(err)
	}
	var entries []struct {
		Name     string `json:"name"`
		Language string `json:"language"`
		Prepared string `json:"prepared"`
		SHA      string `json:"prepared_sha256"`
	}
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatal(err)
	}
	if len(entries) == 0 || len(entries) > 33 {
		t.Fatal("invalid replay corpus size")
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		root, err := model.Open("qwen3-asr:component")
		if err != nil {
			t.Fatal(err)
		}
		m, err := model.New(root)
		if err != nil {
			t.Fatal(err)
		}
		mlx.ScopedArrays(func() []*mlx.Array {
			weights := map[string]*mlx.Array{}
			for _, layer := range root.Manifest.TensorLayers() {
				path, err := manifest.BlobsPath(layer.Digest)
				if err != nil {
					t.Fatal(err)
				}
				for name, a := range mlx.Load(path) {
					weights[name] = a
				}
			}
			if err := m.LoadWeights(weights); err != nil {
				t.Fatal(err)
			}
			return mlx.Collect(m)
		})
		media := m.(model.MediaModel)
		tok := m.Tokenizer()
		ctx, cancel := context.WithTimeout(context.Background(), 150*time.Second)
		defer cancel()
		var records []map[string]any
		var allocations []int
		seen := map[string]bool{}
		// Frequent-update qualification keeps the fixed English/German/silence
		// cohort bounded. At500ms this is36 prefixes:17 per speech clip plus2
		// for silence, within the independent provider worker's37-prefix bound.
		extended := !freshSeed && os.Getenv("OLLAMA_QWEN3_ASR_REPLAY_CORPUS") == "1" && intervalMS == 2000
		corpus := map[string]bool{"corpus-00": true, "corpus-01": true, "corpus-02": true, "corpus-04": true, "corpus-06": true, "corpus-10": true, "silence": true}
		for _, entry := range entries {
			if freshSeed && !map[string]bool{"corpus-00": true, "corpus-02": true, "corpus-03": true, "silence": true}[entry.Name] || !freshSeed && ((!extended && seen[entry.Language]) || (extended && !corpus[entry.Name])) {
				continue
			}
			seen[entry.Language] = true
			wav, err := os.ReadFile(entry.Prepared)
			if err != nil {
				t.Fatal(err)
			}
			h := sha256.Sum256(wav)
			if hex.EncodeToString(h[:]) != entry.SHA {
				t.Fatal("prepared input SHA mismatch")
			}
			pcm, rate, err := audio.Decode(wav)
			if err != nil || rate != stream.SampleRate {
				t.Fatalf("input rate=%d: %v", rate, err)
			}
			seconds := 8
			if extended {
				seconds = 10
			}
			limit := seconds*stream.SampleRate + 17
			if freshSeed && entry.Name == "corpus-03" {
				limit = 16 * stream.SampleRate
			}
			pcm = pcm[:min(len(pcm), limit)]
			language := map[string]string{"en": "English", "de": "German", "auto": ""}[entry.Language]
			var baseline []stream.Update
			passes := [][]int{{packetSamples}, {1, 319, min(7999, packetSamples), packetSamples, 71}, {packetSamples}}
			if freshSeed {
				passes = [][]int{{len(pcm)}, {len(pcm)}}
			}
			for pass, packets := range passes {
				var steps []map[string]any
				infer := func(ctx context.Context, r stream.Request) (string, error) {
					var text string
					var runErr error
					started := time.Now()
					mlx.Scoped(func() {
						prefix := "<|im_start|>system\n" + r.Config.Context + "<|im_end|>\n<|im_start|>user\n<|audio_start|>"
						suffix := "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
						if r.Config.Language != "" {
							suffix += "language " + r.Config.Language + "<asr_text>"
						}
						suffix += r.Prefix
						prepared, err := m.(interface {
							PrepareMediaContext(context.Context, []model.Segment) (*model.PreparedRequest, error)
						}).PrepareMediaContext(ctx, []model.Segment{
							{Tokens: tok.Encode(prefix, false)}, {Kind: "audio", Data: audio.EncodeWAV(r.Samples, stream.SampleRate)}, {Tokens: tok.Encode(suffix, false)},
						})
						if err != nil {
							runErr = err
							return
						}
						if len(prepared.Items) != 1 {
							runErr = fmt.Errorf("expected one media item")
							return
						}
						item := prepared.Items[0]
						features := media.EncodeMedia(&item, mlx.FromValues(item.MediaData, item.Dims...))
						fixture := fmt.Sprintf("%s-step-%02d.safetensors", entry.Name, len(steps))
						caches := m.NewCaches()
						defer func() {
							for _, c := range caches {
								c.Free()
							}
						}()
						n := len(prepared.Tokens)
						b := &batch.Batch{InputIDs: mlx.FromValues(prepared.Tokens, 1, n), SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(n)}, Media: []batch.MediaItem{{Pos: item.Range[0], Features: features}}}
						var ids []int32
						stopped := false
						for step := range 256 {
							if err := ctx.Err(); err != nil {
								runErr = err
								return
							}
							next := mlx.ScopedEval(func() []*mlx.Array {
								h, _ := m.Forward(b, caches)
								last := h.Slice(mlx.Slice(), mlx.Slice(h.Dim(1)-1, h.Dim(1)), mlx.Slice())
								logits := m.Unembed(last).Reshape(1, 151936)
								if step == 0 && pass == 0 {
									waveform, _, err := audio.Decode(audio.EncodeWAV(r.Samples, stream.SampleRate))
									if err == nil {
										err = mlx.SaveSafetensors(filepath.Join(output, fixture), map[string]*mlx.Array{
											"waveform": mlx.FromValues(waveform, len(waveform)), "ids": mlx.FromValues(prepared.Tokens, 1, n),
											"features": mlx.FromValues(item.MediaData, item.Dims...), "encoder": features, "logits": logits,
										})
									}
									if err != nil {
										runErr = err
									}
								}
								return []*mlx.Array{logits.Argmax(-1, false).AsType(mlx.DTypeInt32)}
							})[0].Ints()[0]
							if runErr != nil {
								return
							}
							if tok.IsEOS(next) {
								stopped = true
								break
							}
							ids = append(ids, next)
							b = &batch.Batch{InputIDs: mlx.FromValues([]int32{next}, 1, 1), SeqOffsets: []int32{int32(n + step)}, SeqQueryLens: []int32{1}}
						}
						if !stopped {
							runErr = fmt.Errorf("replay reached token limit")
							return
						}
						text = tok.Decode(ids)
						steps = append(steps, map[string]any{"audio_end_sample": len(r.Samples), "prefix": r.Prefix, "generated_ids": ids, "continuation": text, "seconds": time.Since(started).Seconds(), "fixture": fixture})
					})
					mlx.DefaultStream().Synchronize()
					mlx.ClearCache()
					return text, runErr
				}
				s, err := stream.New(ctx, stream.Config{Language: language, UpdateSamples: updateSamples}, tok, infer)
				if err != nil {
					t.Fatal(err)
				}
				var updates []stream.Update
				if freshSeed {
					u, err := s.Seed(pcm)
					if err != nil {
						s.Close()
						t.Fatal(err)
					}
					if u == nil || u.AudioEnd != len(pcm) || u.Revision != 1 {
						s.Close()
						t.Fatal("invalid fresh Seed update", u)
					}
					updates = append(updates, *u)
				} else {
					for offset, packet := 0, 0; offset < len(pcm); packet++ {
						n := min(packets[packet%len(packets)], len(pcm)-offset)
						u, err := s.Push(offset, pcm[offset:offset+n])
						if err != nil {
							s.Close()
							t.Fatal(err)
						}
						if u != nil {
							updates = append(updates, *u)
						}
						offset += n
					}
				}
				final, err := s.Finish()
				if err != nil {
					t.Fatal(err)
				}
				updates = append(updates, final)
				if freshSeed && (len(steps) != 1 || steps[0]["prefix"] != "" || final.Revision != 2) {
					t.Fatal("Seed/Finish did not reuse exactly one fresh callback", steps, final)
				}
				if pass == 0 {
					baseline = updates
				} else if !reflect.DeepEqual(baseline, updates) {
					t.Fatal("packetization/session changed native hypotheses")
				}
				allocations = append(allocations, mlx.ActiveMemory())
				records = append(records, map[string]any{"input": entry.Name, "requested_language": language, "update_interval_ms": intervalMS, "prepared_sha256": entry.SHA, "fresh_seed": freshSeed, "seed_audio_samples": len(pcm), "pass": pass, "updates": updates, "steps": steps, "active_bytes": mlx.ActiveMemory()})
				t.Logf("%s pass=%d updates=%d active=%d final=%q", entry.Name, pass, len(updates), mlx.ActiveMemory(), final.Text)
			}
		}
		encoded, err := json.MarshalIndent(records, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(output, "replay.json"), encoded, 0o600); err != nil {
			t.Fatal(err)
		}
		if len(seen) < 3 {
			t.Fatal("replay requires English, German and silence/auto")
		}
		if freshSeed && len(records) != 8 {
			t.Fatal("fresh Seed requires exactly four cases/two fresh sessions each")
		}
		if extended && len(records) != 21 {
			t.Fatal("extended replay requires all seven declared corpus entries")
		}
		low, high := allocations[0], allocations[0]
		for _, v := range allocations {
			low = min(low, v)
			high = max(high, v)
		}
		if high-low > 32<<20 {
			t.Fatalf("post-session allocation range %d exceeds 32 MiB", high-low)
		}
	})
}
