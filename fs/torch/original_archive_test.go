package torch

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
)

// Original asset qualification is opt-in and uses owner-installed checkpoints.
// The supervised workspace harness binds both source and independent oracle
// hashes; no model assets are downloaded or bundled by these tests.
func TestOriginalTensorArchives(t *testing.T) {
	name := os.Getenv("OLLAMA_TORCH_ARCHIVE_CASES")
	if name == "" {
		t.Skip("set OLLAMA_TORCH_ARCHIVE_CASES to the supervised original-asset matrix")
	}
	data, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name, Kind, Archive, Reference, ArchiveSHA256, ReferenceSHA256 string
	}
	if err := json.Unmarshal(data, &cases); err != nil || len(cases) == 0 || len(cases) > 64 {
		t.Fatalf("invalid archive case matrix: %v", err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			verifyFileHash(t, c.Archive, c.ArchiveSHA256)
			verifyFileHash(t, c.Reference, c.ReferenceSHA256)
			source, err := Open(c.Archive)
			if err != nil {
				t.Fatal(err)
			}
			defer source.Close()
			oracle, err := safetensors.OpenForExtraction(c.Reference)
			if err != nil {
				t.Fatal(err)
			}
			defer oracle.Close()
			mapped := make(map[string]*Tensor)
			for key, tensor := range source.Tensors {
				switch c.Kind {
				case "kokoro-model":
					module, rest, ok := strings.Cut(key, ".")
					if !ok || !slices.Contains([]string{"bert", "bert_encoder", "predictor", "decoder", "text_encoder"}, module) {
						t.Fatalf("unexpected original Kokoro tensor %s", key)
					}
					key = module + "." + strings.TrimPrefix(rest, "module.")
				case "kokoro-voice":
					if key != "tensor" {
						t.Fatalf("unexpected Kokoro voice field %s", key)
					}
					key = "voice"
				case "voxtral-voice":
					if key != "tensor" && key != "embedding" {
						t.Fatalf("unexpected Voxtral voice field %s", key)
					}
					key = "embedding"
				case "turbo-conditions":
				default:
					t.Fatalf("unsupported original archive role %s", c.Kind)
				}
				if mapped[key] != nil {
					t.Fatalf("tensor alias collision: %s", key)
				}
				mapped[key] = tensor
			}
			if len(mapped) != len(oracle.ListTensors()) {
				t.Fatalf("tensor count %d, want %d", len(mapped), len(oracle.ListTensors()))
			}
			var total int64
			for _, key := range oracle.ListTensors() {
				want, err := oracle.GetTensor(key)
				if err != nil {
					t.Fatal(err)
				}
				actual := mapped[key]
				if actual == nil || !slices.Equal(actual.Shape, want.Shape) {
					t.Fatalf("%s missing or incompatible shape", key)
				}
				var reader io.Reader = io.NewSectionReader(actual, 0, actual.Size)
				if c.Kind == "turbo-conditions" && actual.Dtype == "I64" && want.Dtype == "I32" {
					data, err := io.ReadAll(reader)
					if err != nil {
						t.Fatal(err)
					}
					converted := make([]byte, len(data)/2)
					for i := range len(data) / 8 {
						n := int64(binary.LittleEndian.Uint64(data[i*8:]))
						if n < 0 || n > 1<<31-1 {
							t.Fatal("original conditioning token out of int32 range")
						}
						binary.LittleEndian.PutUint32(converted[i*4:], uint32(n))
					}
					reader = bytes.NewReader(converted)
				} else if actual.Dtype != want.Dtype || actual.Size != want.Size {
					t.Fatalf("%s incompatible dtype/size: %s/%d vs %s/%d", key, actual.Dtype, actual.Size, want.Dtype, want.Size)
				}
				left, right := sha256.New(), sha256.New()
				n, err := io.Copy(left, reader)
				if err != nil || n != want.Size {
					t.Fatalf("%s native tensor read: %d bytes, %v", key, n, err)
				}
				if _, err := io.Copy(right, want.Reader()); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(left.Sum(nil), right.Sum(nil)) {
					t.Fatalf("%s logical tensor bytes differ", key)
				}
				total += n
			}
			verifyFileHash(t, c.Archive, c.ArchiveSHA256)
			verifyFileHash(t, c.Reference, c.ReferenceSHA256)
			t.Logf("TORCH_ARCHIVE_METRIC %s", fmt.Sprintf(`{"name":%q,"tensors":%d,"bytes":%d,"exact":true}`, c.Name, len(mapped), total))
		})
	}
	t.Log("TORCH_ARCHIVE_AUDIT_COMPLETED")
}

func verifyFileHash(t *testing.T, name, expected string) {
	t.Helper()
	file, err := os.Open(name)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	h := sha256.New()
	if _, err := io.Copy(h, file); err != nil {
		t.Fatal(err)
	}
	if hex.EncodeToString(h.Sum(nil)) != expected {
		t.Fatalf("asset identity changed: %s", name)
	}
}
