package torch

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
)

func TestTorchSaveInteroperability(t *testing.T) {
	want, err := safetensors.OpenForExtraction("testdata/synthetic.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()
	for _, name := range []string{"synthetic-protocol2.pt", "synthetic-protocol4.pt"} {
		t.Run(name, func(t *testing.T) {
			a, err := Open(filepath.Join("testdata", name))
			if err != nil {
				t.Fatal(err)
			}
			defer a.Close()
			if len(a.Tensors) != len(want.ListTensors()) {
				t.Fatalf("tensor count=%d, want %d", len(a.Tensors), len(want.ListTensors()))
			}
			for _, key := range want.ListTensors() {
				expected, err := want.GetTensor(key)
				if err != nil {
					t.Fatal(err)
				}
				actual := a.Tensors[key]
				if actual == nil || actual.Dtype != expected.Dtype || !slices.Equal(actual.Shape, expected.Shape) || actual.Size != expected.Size {
					t.Fatalf("%s metadata mismatch: %+v versus %+v", key, actual, expected)
				}
				data, err := io.ReadAll(io.NewSectionReader(actual, 0, actual.Size))
				if err != nil {
					t.Fatal(err)
				}
				gold, err := io.ReadAll(expected.Reader())
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(data, gold) {
					t.Errorf("%s tensor bytes differ", key)
				}
			}
		})
	}
}

func TestStridedStorageAndPartialReads(t *testing.T) {
	a, err := Open("testdata/synthetic-protocol2.pt")
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	source := a.Tensors["view"]
	view, err := a.newTensor(source.storage, source.offset, []int64{3, 2}, []int64{1, 3})
	if err != nil {
		t.Fatal(err)
	}
	want := make([]byte, 24)
	for i, value := range []float32{2, 5, 3, 6, 4, 7} {
		binary.LittleEndian.PutUint32(want[i*4:], math.Float32bits(value))
	}
	for off := range view.Size {
		for _, length := range []int{0, 1, 3, 7, 24} {
			got := make([]byte, length)
			n, err := view.ReadAt(got, off)
			expected := want[off:min(int64(len(want)), off+int64(length))]
			if !bytes.Equal(got[:n], expected) || (n == length && err != nil) || (n < length && err != io.EOF) {
				t.Fatalf("off=%d length=%d: read %v/%v, want %v", off, length, got[:n], err, expected)
			}
		}
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := view.ReadAt(make([]byte, 4), 0); err == nil {
		t.Fatal("read succeeded after archive close")
	}
}

type testMember struct {
	name   string
	method uint16
	data   []byte
}

func fixtureMembers(t *testing.T) []testMember {
	t.Helper()
	a, err := zip.OpenReader("testdata/synthetic-protocol2.pt")
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	var out []testMember
	for _, member := range a.File {
		r, err := member.Open()
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(r)
		r.Close()
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, testMember{member.Name, member.Method, data})
	}
	return out
}

func writeTestArchive(t *testing.T, members []testMember) string {
	t.Helper()
	name := filepath.Join(t.TempDir(), "fixture.pt")
	f, err := os.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	w := zip.NewWriter(f)
	for _, member := range members {
		r, err := w.CreateHeader(&zip.FileHeader{Name: member.name, Method: member.method})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := r.Write(member.data); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return name
}

func TestRejectUnsupportedOrMalformedArchives(t *testing.T) {
	for _, tc := range []struct {
		name, member, contains string
		data                   []byte
	}{
		{"global", "/data.pkl", "global", []byte("\x80\x02cbuiltins\neval\n.")},
		{"missing-memo", "/data.pkl", "memo", []byte("\x80\x02h\x01.")},
		{"stack", "/data.pkl", "underflow", []byte("\x80\x02R.")},
		{"truncated", "/data.pkl", "truncated", []byte("\x80\x02X\xff\xff\xff\x7f.")},
		{"cycle", "/data.pkl", "cyclic", []byte("\x80\x02}q\x00X\x04\x00\x00\x00selfh\x00s.")},
		{"frame", "/data.pkl", "frame", []byte("\x80\x04\x95\xff\xff\xff\xff\xff\xff\xff\xff.")},
		{"endianness", "/byteorder", "little-endian", []byte("big")},
		{"storage-size", "/data/0", "typed storage", []byte{0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			members := fixtureMembers(t)
			for i := range members {
				if strings.HasSuffix(members[i].name, tc.member) {
					members[i].data = tc.data
				}
			}
			a, err := Open(writeTestArchive(t, members))
			if err == nil {
				a.Close()
				t.Fatal("malformed archive accepted")
			}
			if !strings.Contains(err.Error(), tc.contains) {
				t.Errorf("error %q, want %q", err, tc.contains)
			}
		})
	}
	for _, name := range []string{"duplicate", "path", "compression"} {
		t.Run(name, func(t *testing.T) {
			members := fixtureMembers(t)
			switch name {
			case "duplicate":
				members = append(members, members[0])
			case "path":
				members[0].name = "../data.pkl"
			case "compression":
				members[0].method = zip.Deflate
			}
			a, err := Open(writeTestArchive(t, members))
			if err == nil {
				a.Close()
				t.Fatal("unsupported archive accepted")
			}
		})
	}
}

func TestLazyIntegrityAndFiniteChecks(t *testing.T) {
	for _, nonfinite := range []bool{false, true} {
		members := fixtureMembers(t)
		if nonfinite {
			for i := range members {
				if strings.HasSuffix(members[i].name, "/data/0") {
					binary.LittleEndian.PutUint32(members[i].data, 0x7fc00000)
				}
			}
		}
		name := writeTestArchive(t, members)
		a, err := Open(name)
		if err != nil {
			t.Fatal(err) // Metadata inventory must not read weight data.
		}
		tensor := a.Tensors["f32"]
		if !nonfinite {
			f, err := os.OpenFile(name, os.O_WRONLY, 0)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := f.WriteAt([]byte{1}, tensor.storage.offset); err != nil {
				t.Fatal(err)
			}
			f.Close()
		}
		_, err = tensor.ReadAt(make([]byte, 4), 0)
		a.Close()
		want := "checksum"
		if nonfinite {
			want = "nonfinite"
		}
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("nonfinite=%v: got %v, want %s failure", nonfinite, err, want)
		}
	}
}

func TestTensorExtentBounds(t *testing.T) {
	a, err := Open("testdata/synthetic-protocol2.pt")
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	s := a.Tensors["f32"].storage
	for _, tc := range []struct {
		offset        int64
		shape, stride []int64
	}{
		{-1, []int64{1}, []int64{1}},
		{6, []int64{1}, []int64{1}},
		{0, []int64{2, 4}, []int64{4, 1}},
		{0, []int64{2}, []int64{-1}},
		{0, []int64{1 << 31}, []int64{1}},
		{0, []int64{1 << 30, 1 << 30}, []int64{1 << 30, 1}},
	} {
		if _, err := a.newTensor(s, tc.offset, tc.shape, tc.stride); err == nil {
			t.Errorf("invalid extent accepted: %+v", tc)
		}
	}
}

func TestLogicalTensorInventoryBounds(t *testing.T) {
	// Shared storage must not multiply into an unbounded logical output.
	large := &Tensor{Size: maxArchiveBytes / 2}
	a := &Archive{Tensors: make(map[string]*Tensor)}
	root := &dictionary{items: map[string]any{"a": large, "b": large, "c": large}}
	if _, err := a.export(root, "", make(map[*dictionary]bool), 0); err == nil || !strings.Contains(err.Error(), "logical tensor") {
		t.Fatalf("aliased logical expansion accepted: %v", err)
	}
	// Exactly the tensor-count limit remains valid regardless of where an
	// optional null field occurs in dictionary iteration order.
	root = &dictionary{items: map[string]any{"unused": nil}}
	for i := range maxMembers {
		root.items[fmt.Sprint(i)] = &Tensor{Size: 4}
	}
	a = &Archive{Tensors: make(map[string]*Tensor)}
	if _, err := a.export(root, "", make(map[*dictionary]bool), 0); err != nil || len(a.Tensors) != maxMembers {
		t.Fatalf("exact tensor-count boundary: %d, %v", len(a.Tensors), err)
	}
}

func TestArchiveReadCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	a, err := OpenContext(ctx, "testdata/synthetic-protocol2.pt")
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	view := a.Tensors["view"]
	if _, err := view.ReadAt(make([]byte, 4), 0); err != nil {
		t.Fatal(err)
	}
	cancel()
	if _, err := view.ReadAt(make([]byte, 4), 0); err != context.Canceled {
		t.Fatalf("cached storage read after cancellation: %v", err)
	}
	if _, err := OpenContext(ctx, "testdata/synthetic-protocol2.pt"); err != context.Canceled {
		t.Fatalf("open after cancellation: %v", err)
	}
}

func FuzzTensorMetadata(f *testing.F) {
	archive, err := Open("testdata/synthetic-protocol2.pt")
	if err != nil {
		f.Fatal(err)
	}
	defer archive.Close()
	for _, name := range []string{"synthetic-protocol2.pt", "synthetic-protocol4.pt"} {
		z, err := zip.OpenReader(filepath.Join("testdata", name))
		if err != nil {
			f.Fatal(err)
		}
		for _, member := range z.File {
			if strings.HasSuffix(member.Name, "/data.pkl") {
				data, err := readMember(member)
				if err != nil {
					f.Fatal(err)
				}
				f.Add(data)
			}
		}
		z.Close()
	}
	f.Add([]byte("\x80\x02R."))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			t.Skip()
		}
		a := &Archive{storages: make(map[string]*storage), Tensors: make(map[string]*Tensor)}
		for key, original := range archive.storages {
			copy := *original
			copy.dtype, copy.elements = "", 0
			a.storages[key] = &copy
		}
		value, err := decodePickle(data, a)
		if err == nil {
			_, _ = a.export(value, "", make(map[*dictionary]bool), 0)
		}
	})
}
