// Package torch reads bounded torch.save tensor archives without executing
// Python or pickle callbacks. Only dense tensors and dictionary containers are
// supported; this is not a general pickle or model-code loader.
package torch

import (
	"archive/zip"
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path"
	"slices"
	"strings"
	"sync"
)

const (
	maxArchiveBytes = 512 << 20
	maxStorageBytes = 256 << 20
	maxPickleBytes  = 16 << 20
	maxViewBytes    = 32 << 20
	maxMembers      = 2048
	maxValues       = 100_000
)

// Archive retains the source file for streaming tensor reads. Opening reads
// only ZIP/pickle metadata. The caller must Close it after consuming tensors.
type Archive struct {
	checkCanceled func() error
	file          *os.File
	mu            sync.Mutex
	closed        bool
	storages      map[string]*storage
	viewBytes     int64
	exportValues  int
	tensorBytes   int64
	Root          any
	Tensors       map[string]*Tensor
}

type storage struct {
	member   *zip.File
	offset   int64
	dtype    string
	elements int64
	checked  bool
	data     []byte
}

// Tensor describes one logical tensor. ReadAt exposes contiguous row-major
// bytes, preserving storage offsets and strides from the source archive.
type Tensor struct {
	Dtype      string
	Shape      []int32
	Size       int64
	archive    *Archive
	storage    *storage
	offset     int64 // elements
	strides    []int64
	contiguous bool
}

func Open(filename string) (*Archive, error) {
	return OpenContext(context.Background(), filename)
}

// OpenContext binds subsequent storage validation/reads to ctx. Cancellation
// is checked between bounded storage chunks as well as before opening metadata.
func OpenContext(ctx context.Context, filename string) (_ *Archive, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() > maxArchiveBytes {
		return nil, fmt.Errorf("unsupported PyTorch archive size/type")
	}
	z, err := zip.NewReader(f, info.Size())
	if err != nil {
		return nil, fmt.Errorf("PyTorch requires a ZIP tensor archive: %w", err)
	}
	if len(z.File) == 0 || len(z.File) > maxMembers {
		return nil, fmt.Errorf("PyTorch archive member count exceeds bounds")
	}
	a := &Archive{checkCanceled: ctx.Err, file: f, storages: make(map[string]*storage)}
	root := ""
	members := make(map[string]*zip.File)
	var total uint64
	for _, member := range z.File {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		name := member.Name
		prefix, suffix, ok := strings.Cut(name, "/")
		if !ok || prefix == "" || prefix == "." || prefix == ".." || suffix == "" || path.Clean(name) != name || strings.Contains(name, "\\") || strings.HasPrefix(name, "/") {
			return nil, fmt.Errorf("invalid PyTorch archive member %q", name)
		}
		if root == "" {
			root = prefix
		}
		if prefix != root || member.Flags&1 != 0 || member.Method != zip.Store || !member.Mode().IsRegular() {
			return nil, fmt.Errorf("unsupported PyTorch archive member %q", name)
		}
		if _, exists := members[suffix]; exists {
			return nil, fmt.Errorf("duplicate PyTorch archive member %q", name)
		}
		if member.UncompressedSize64 > maxStorageBytes || member.CompressedSize64 != member.UncompressedSize64 || total > maxArchiveBytes-member.UncompressedSize64 {
			return nil, fmt.Errorf("PyTorch archive expanded size exceeds bounds")
		}
		total += member.UncompressedSize64
		members[suffix] = member
		if key, ok := strings.CutPrefix(suffix, "data/"); ok {
			if key == "" || len(key) > 64 || strings.IndexFunc(key, func(r rune) bool { return r < '0' || r > '9' }) >= 0 {
				return nil, fmt.Errorf("invalid PyTorch storage key %q", key)
			}
			offset, err := member.DataOffset()
			if err != nil || offset < 0 || offset > info.Size()-int64(member.UncompressedSize64) {
				return nil, fmt.Errorf("invalid PyTorch storage extent")
			}
			a.storages[key] = &storage{member: member, offset: offset}
		} else {
			switch suffix {
			case "data.pkl":
				if member.UncompressedSize64 > maxPickleBytes {
					return nil, fmt.Errorf("PyTorch pickle metadata exceeds bounds")
				}
			case "byteorder", "version", ".data/serialization_id", ".format_version", ".storage_alignment":
				if member.UncompressedSize64 > 1024 {
					return nil, fmt.Errorf("PyTorch metadata member exceeds bounds")
				}
			default:
				return nil, fmt.Errorf("unsupported PyTorch archive member %q", name)
			}
		}
	}
	if member := members["byteorder"]; member != nil {
		data, err := readMember(member)
		if err != nil || string(data) != "little" {
			return nil, fmt.Errorf("PyTorch archive requires little-endian storage")
		}
	}
	member := members["data.pkl"]
	if member == nil {
		return nil, fmt.Errorf("PyTorch archive has no data.pkl")
	}
	metadata, err := readMember(member)
	if err != nil {
		return nil, err
	}
	value, err := decodePickle(metadata, a)
	if err != nil {
		return nil, fmt.Errorf("PyTorch tensor metadata: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	a.Tensors = make(map[string]*Tensor)
	a.Root, err = a.export(value, "", make(map[*dictionary]bool), 0)
	if err != nil {
		return nil, err
	}
	if len(a.Tensors) == 0 {
		return nil, fmt.Errorf("PyTorch archive has no supported tensors")
	}
	return a, nil
}

func readMember(member *zip.File) ([]byte, error) {
	r, err := member.Open()
	if err != nil {
		return nil, err
	}
	defer r.Close()
	return io.ReadAll(io.LimitReader(r, maxPickleBytes+1))
}

func (a *Archive) export(value any, name string, active map[*dictionary]bool, depth int) (any, error) {
	a.exportValues++
	if depth > 32 || len(name) > 1024 || a.exportValues > maxValues {
		return nil, fmt.Errorf("PyTorch tensor tree exceeds bounds")
	}
	switch value := value.(type) {
	case *Tensor:
		if len(a.Tensors) >= maxMembers || value.Size > maxArchiveBytes-a.tensorBytes {
			return nil, fmt.Errorf("PyTorch logical tensor inventory exceeds bounds")
		}
		if name == "" {
			name = "tensor"
		}
		if _, exists := a.Tensors[name]; exists {
			return nil, fmt.Errorf("PyTorch tensor namespace collision: %s", name)
		}
		a.Tensors[name] = value
		a.tensorBytes += value.Size
		return value, nil
	case *dictionary:
		if active[value] {
			return nil, fmt.Errorf("cyclic PyTorch tensor dictionary")
		}
		active[value] = true
		defer delete(active, value)
		out := make(map[string]any, len(value.items))
		for key, child := range value.items {
			full := key
			if name != "" {
				full = name + "." + key
			}
			converted, err := a.export(child, full, active, depth+1)
			if err != nil {
				return nil, err
			}
			out[key] = converted
		}
		return out, nil
	case nil:
		return nil, nil
	default:
		return nil, fmt.Errorf("unsupported value in PyTorch tensor dictionary: %T", value)
	}
}

func (a *Archive) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil
	}
	a.closed = true
	return a.file.Close()
}

func dtypeSize(dtype string) int64 {
	switch dtype {
	case "F16", "BF16":
		return 2
	case "F32", "I32":
		return 4
	case "F64", "I64":
		return 8
	case "U8", "BOOL":
		return 1
	default:
		return 0
	}
}

// checkStorage verifies the complete storage CRC and rejects nonfinite floating
// point values. Small strided storages are cached within a per-archive budget;
// contiguous weights remain streaming readers without retained weight copies.
func (a *Archive) checkStorage(s *storage, cache bool) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return fmt.Errorf("PyTorch archive is closed")
	}
	if err := a.checkCanceled(); err != nil {
		return err
	}
	if s.checked && (!cache || s.data != nil) {
		return nil
	}
	size := int64(s.member.UncompressedSize64)
	var data []byte
	if cache {
		if size > maxViewBytes-a.viewBytes {
			return fmt.Errorf("PyTorch strided storage cache exceeds bounds")
		}
		data = make([]byte, size)
	}
	buffer := make([]byte, 32<<10)
	var checksum uint32
	for offset := int64(0); offset < size; {
		if err := a.checkCanceled(); err != nil {
			return err
		}
		next := min(int64(len(buffer)), size-offset)
		chunk := buffer[:next]
		if _, err := a.file.ReadAt(chunk, s.offset+offset); err != nil {
			return err
		}
		checksum = crc32.Update(checksum, crc32.IEEETable, chunk)
		if err := finite(chunk, s.dtype); err != nil {
			return err
		}
		if cache {
			copy(data[offset:], chunk)
		}
		offset += next
	}
	if checksum != s.member.CRC32 {
		return fmt.Errorf("PyTorch storage checksum mismatch")
	}
	s.checked = true
	if cache {
		s.data = data
		a.viewBytes += size
	}
	return nil
}

func finite(data []byte, dtype string) error {
	step := dtypeSize(dtype)
	for i := 0; i < len(data); i += int(step) {
		bad := false
		switch dtype {
		case "F32":
			bad = binary.LittleEndian.Uint32(data[i:])&0x7f800000 == 0x7f800000
		case "F64":
			bad = binary.LittleEndian.Uint64(data[i:])&0x7ff0000000000000 == 0x7ff0000000000000
		case "F16":
			bad = binary.LittleEndian.Uint16(data[i:])&0x7c00 == 0x7c00
		case "BF16":
			bad = binary.LittleEndian.Uint16(data[i:])&0x7f80 == 0x7f80
		}
		if bad {
			return fmt.Errorf("nonfinite PyTorch %s storage", dtype)
		}
	}
	return nil
}

func (t *Tensor) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, fmt.Errorf("negative PyTorch tensor read offset")
	}
	if err := t.archive.checkStorage(t.storage, !t.contiguous); err != nil {
		return 0, err
	}
	if t.contiguous {
		return io.NewSectionReader(t.archive.file, t.storage.offset+t.offset*dtypeSize(t.Dtype), t.Size).ReadAt(p, off)
	}
	if off >= t.Size {
		return 0, io.EOF
	}
	n := min(int64(len(p)), t.Size-off)
	width := dtypeSize(t.Dtype)
	for written := int64(0); written < n; {
		logical, within := (off+written)/width, (off+written)%width
		physical := t.offset
		for axis := len(t.Shape) - 1; axis >= 0; axis-- {
			physical += logical % int64(t.Shape[axis]) * t.strides[axis]
			logical /= int64(t.Shape[axis])
		}
		count := min(width-within, n-written)
		copy(p[written:written+count], t.storage.data[physical*width+within:physical*width+within+count])
		written += count
	}
	if n < int64(len(p)) {
		return int(n), io.EOF
	}
	return int(n), nil
}

func (a *Archive) newTensor(s *storage, offset int64, shape, strides []int64) (*Tensor, error) {
	if len(shape) > 8 || len(shape) != len(strides) || offset < 0 || offset > s.elements {
		return nil, fmt.Errorf("invalid PyTorch tensor shape/offset")
	}
	count, extent, expected := int64(1), offset, int64(1)
	contiguous := true
	dims := make([]int32, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] < 0 || shape[i] > 1<<31-1 || strides[i] < 0 || strides[i] > maxArchiveBytes {
			return nil, fmt.Errorf("invalid PyTorch tensor dimensions/strides")
		}
		dims[i] = int32(shape[i])
		if shape[i] > 1 {
			contiguous = contiguous && strides[i] == expected
			extent += (shape[i] - 1) * strides[i]
		}
		if shape[i] != 0 && count > maxStorageBytes/dtypeSize(s.dtype)/shape[i] {
			return nil, fmt.Errorf("PyTorch tensor size exceeds bounds")
		}
		count *= shape[i]
		if shape[i] != 0 && expected > maxStorageBytes/shape[i] {
			return nil, fmt.Errorf("PyTorch logical shape exceeds bounds")
		}
		expected *= shape[i]
	}
	if count > 0 && extent >= s.elements {
		return nil, fmt.Errorf("PyTorch tensor exceeds storage")
	}
	return &Tensor{Dtype: s.dtype, Shape: dims, Size: count * dtypeSize(s.dtype), archive: a, storage: s, offset: offset, strides: slices.Clone(strides), contiguous: contiguous}, nil
}
