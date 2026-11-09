package torch

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"slices"
	"unicode/utf8"
)

type (
	symbol     string
	tuple      []any
	marker     struct{}
	dictionary struct {
		items   map[string]any
		ordered bool
	}
	list struct{ items []any }
)

type pickle struct {
	data     []byte
	pos      int
	frameEnd int
	stack    []any
	memo     map[uint64]any
	archive  *Archive
	err      error
	tensors  int
}

var storageTypes = map[symbol]string{
	"torch.FloatStorage": "F32", "torch.DoubleStorage": "F64",
	"torch.HalfStorage": "F16", "torch.BFloat16Storage": "BF16",
	"torch.LongStorage": "I64", "torch.IntStorage": "I32",
	"torch.ByteStorage": "U8", "torch.BoolStorage": "BOOL",
}

func decodePickle(data []byte, archive *Archive) (any, error) {
	if len(data) < 3 || data[0] != 0x80 || data[1] < 2 || data[1] > 5 {
		return nil, fmt.Errorf("unsupported pickle protocol")
	}
	p := &pickle{data: data, pos: 2, memo: make(map[uint64]any), archive: archive}
	for operations := 0; operations < 1_000_000 && p.err == nil && p.pos < len(data); operations++ {
		if p.frameEnd == p.pos {
			p.frameEnd = 0
		}
		op := p.uint(1)
		switch op {
		case '.':
			if len(p.stack) != 1 || p.pos != len(data) || (p.frameEnd != 0 && p.pos != p.frameEnd) {
				p.fail("invalid pickle completion")
			} else {
				return p.stack[0], nil
			}
		case '(':
			p.push(marker{})
		case ')':
			p.push(tuple{})
		case 't':
			p.push(tuple(p.marked()))
		case 0x85, 0x86, 0x87:
			p.push(tuple(p.tail(int(op - 0x84))))
		case '}':
			p.push(&dictionary{items: make(map[string]any)})
		case 'd':
			items := p.marked()
			d := &dictionary{items: make(map[string]any)}
			p.setItems(d, items)
			p.push(d)
		case 's':
			items := p.tail(2)
			p.setItems(p.dict(p.top()), items)
		case 'u':
			items := p.marked()
			p.setItems(p.dict(p.top()), items)
		case ']':
			p.push(&list{})
		case 'a', 'e':
			var items []any
			if op == 'a' {
				items = []any{p.pop()}
			} else {
				items = p.marked()
			}
			value, ok := p.top().(*list)
			if !ok || len(value.items)+len(items) > maxValues {
				p.fail("invalid or oversized pickle list")
			} else {
				value.items = append(value.items, items...)
			}
		case 'N':
			p.push(nil)
		case 0x88, 0x89:
			p.push(op == 0x88)
		case 'K':
			p.push(int64(p.uint(1)))
		case 'M':
			p.push(int64(p.uint(2)))
		case 'J':
			p.push(int64(int32(p.uint(4))))
		case 'X':
			p.push(p.text(p.uint(4)))
		case 0x8c:
			p.push(p.text(p.uint(1)))
		case 0x8d:
			p.push(p.text(p.uint(8)))
		case 'c':
			module, name := p.line(), p.line()
			p.push(p.global(module + "." + name))
		case 0x93:
			name, module := p.string(p.pop()), p.string(p.pop())
			p.push(p.global(module + "." + name))
		case 'q':
			p.memoPut(p.uint(1))
		case 'r':
			p.memoPut(p.uint(4))
		case 0x94:
			p.memoPut(uint64(len(p.memo)))
		case 'h', 'j':
			n := 1
			if op == 'j' {
				n = 4
			}
			value, ok := p.memo[p.uint(n)]
			if !ok {
				p.fail("missing pickle memo reference")
			}
			p.push(value)
		case 'Q':
			p.push(p.persistent(p.pop()))
		case 'R':
			args := p.tuple(p.pop())
			fn, ok := p.pop().(symbol)
			if !ok {
				p.fail("pickle reducer is not an allowed symbol")
			} else {
				p.push(p.reduce(fn, args))
			}
		case 'b':
			state := p.dict(p.pop())
			value := p.dict(p.top())
			// OrderedDict state_dict metadata is data, not a __setstate__ call.
			// Accept only PyTorch's known _metadata attribute and discard it.
			if state == nil || value == nil || !value.ordered || len(state.items) != 1 {
				p.fail("unsupported pickle BUILD state")
			} else if _, ok := state.items["_metadata"].(*dictionary); !ok {
				p.fail("unsupported OrderedDict attribute")
			}
		case 0x95:
			if p.frameEnd != 0 {
				p.fail("nested pickle frame")
			}
			n := p.uint(8)
			if n > uint64(len(data)-p.pos) {
				p.fail("pickle frame exceeds metadata")
			} else {
				p.frameEnd = p.pos + int(n)
			}
		default:
			p.fail("unsupported pickle opcode 0x%02x", op)
		}
	}
	if p.err != nil {
		return nil, p.err
	}
	return nil, fmt.Errorf("incomplete or oversized pickle program")
}

func (p *pickle) fail(format string, args ...any) {
	if p.err == nil {
		p.err = fmt.Errorf(format, args...)
	}
}

func (p *pickle) take(n uint64) []byte {
	if n > uint64(len(p.data)-p.pos) || (p.frameEnd != 0 && n > uint64(p.frameEnd-p.pos)) {
		p.fail("truncated pickle value")
		return nil
	}
	out := p.data[p.pos : p.pos+int(n)]
	p.pos += int(n)
	return out
}

func (p *pickle) uint(n int) uint64 {
	data := p.take(uint64(n))
	if len(data) != n {
		return 0
	}
	switch n {
	case 1:
		return uint64(data[0])
	case 2:
		return uint64(binary.LittleEndian.Uint16(data))
	case 4:
		return uint64(binary.LittleEndian.Uint32(data))
	case 8:
		return binary.LittleEndian.Uint64(data)
	default:
		p.fail("unsupported integer width")
		return 0
	}
}

func (p *pickle) text(n uint64) string {
	data := p.take(n)
	if !utf8.Valid(data) {
		p.fail("invalid UTF-8 pickle string")
	}
	return string(data)
}

func (p *pickle) line() string {
	n := bytes.IndexByte(p.data[p.pos:], '\n')
	if n < 0 || n > 256 {
		p.fail("invalid pickle global name")
		return ""
	}
	value := p.text(uint64(n))
	p.take(1)
	return value
}

func (p *pickle) push(value any) {
	if len(p.stack) >= maxValues {
		p.fail("pickle stack exceeds bounds")
		return
	}
	p.stack = append(p.stack, value)
}

func (p *pickle) top() any {
	if len(p.stack) == 0 {
		p.fail("pickle stack underflow")
		return nil
	}
	return p.stack[len(p.stack)-1]
}

func (p *pickle) pop() any {
	value := p.top()
	if len(p.stack) != 0 {
		p.stack = p.stack[:len(p.stack)-1]
	}
	return value
}

func (p *pickle) tail(n int) []any {
	if n > len(p.stack) {
		p.fail("pickle tuple underflow")
		return nil
	}
	items := slices.Clone(p.stack[len(p.stack)-n:])
	p.stack = p.stack[:len(p.stack)-n]
	return items
}

func (p *pickle) marked() []any {
	for i := len(p.stack) - 1; i >= 0; i-- {
		if _, ok := p.stack[i].(marker); ok {
			items := slices.Clone(p.stack[i+1:])
			p.stack = p.stack[:i]
			return items
		}
	}
	p.fail("missing pickle mark")
	return nil
}

func (p *pickle) dict(value any) *dictionary {
	d, ok := value.(*dictionary)
	if !ok {
		p.fail("expected pickle dictionary")
		return nil
	}
	return d
}

func (p *pickle) string(value any) string {
	s, ok := value.(string)
	if !ok {
		p.fail("expected pickle string")
	}
	return s
}

func (p *pickle) tuple(value any) tuple {
	t, ok := value.(tuple)
	if !ok {
		p.fail("expected pickle tuple")
	}
	return t
}

func (p *pickle) setItems(d *dictionary, items []any) {
	if d == nil || len(items)%2 != 0 || len(d.items)+len(items)/2 > maxValues {
		p.fail("invalid or oversized pickle dictionary")
		return
	}
	for i := 0; i < len(items); i += 2 {
		key := p.string(items[i])
		if len(key) > 1024 {
			p.fail("pickle dictionary key exceeds bounds")
		}
		if _, exists := d.items[key]; exists {
			p.fail("duplicate pickle dictionary key %q", key)
		}
		d.items[key] = items[i+1]
	}
}

func (p *pickle) memoPut(key uint64) {
	if key >= maxValues || len(p.memo) >= maxValues {
		p.fail("pickle memo exceeds bounds")
		return
	}
	p.memo[key] = p.top()
}

func (p *pickle) global(name string) symbol {
	s := symbol(name)
	if _, ok := storageTypes[s]; ok {
		return s
	}
	if s != "collections.OrderedDict" && s != "torch._utils._rebuild_tensor_v2" && s != "torch._utils._rebuild_tensor" {
		p.fail("unsupported pickle global %q", name)
	}
	return s
}

func (p *pickle) persistent(value any) *storage {
	args := p.tuple(value)
	if len(args) != 5 || args[0] != "storage" {
		p.fail("unsupported PyTorch persistent reference")
		return nil
	}
	storageType, _ := args[1].(symbol)
	dtype := storageTypes[storageType]
	key, ok := args[2].(string)
	location, locationOK := args[3].(string)
	n, countOK := args[4].(int64)
	s := p.archive.storages[key]
	if dtype == "" || !ok || !locationOK || len(location) > 128 || !countOK || n < 0 || n > maxStorageBytes/dtypeSize(dtype) || s == nil || uint64(n*dtypeSize(dtype)) != s.member.UncompressedSize64 {
		p.fail("invalid PyTorch typed storage")
		return nil
	}
	if s.dtype != "" && (s.dtype != dtype || s.elements != n) {
		p.fail("conflicting PyTorch storage references")
		return nil
	}
	// Device tags are descriptive only. Storage bytes are always read locally;
	// no CUDA/MPS initialization or Python deserializer is invoked.
	s.dtype, s.elements = dtype, n
	return s
}

func (p *pickle) reduce(fn symbol, args tuple) any {
	if fn == "collections.OrderedDict" {
		if len(args) != 0 {
			p.fail("unsupported OrderedDict constructor arguments")
		}
		return &dictionary{items: make(map[string]any), ordered: true}
	}
	want := 6
	if fn == "torch._utils._rebuild_tensor" {
		want = 4
	} else if fn != "torch._utils._rebuild_tensor_v2" {
		p.fail("unsupported pickle reducer %q", fn)
		return nil
	}
	if len(args) != want || p.tensors >= maxMembers {
		p.fail("unsupported PyTorch tensor reducer arguments")
		return nil
	}
	s, _ := args[0].(*storage)
	offset, offsetOK := args[1].(int64)
	shape, strides := p.dimensions(args[2]), p.dimensions(args[3])
	if s == nil || !offsetOK {
		p.fail("invalid PyTorch tensor storage/offset")
		return nil
	}
	if want == 6 {
		_, gradOK := args[4].(bool)
		hooks, hooksOK := args[5].(*dictionary)
		if !gradOK || !hooksOK || len(hooks.items) != 0 {
			p.fail("unsupported PyTorch tensor hooks")
			return nil
		}
	}
	tensor, err := p.archive.newTensor(s, offset, shape, strides)
	if err != nil {
		p.fail("%v", err)
		return nil
	}
	p.tensors++
	return tensor
}

func (p *pickle) dimensions(value any) []int64 {
	args := p.tuple(value)
	if len(args) > 8 {
		p.fail("PyTorch tensor rank exceeds bounds")
		return nil
	}
	out := make([]int64, len(args))
	for i, arg := range args {
		n, ok := arg.(int64)
		if !ok {
			p.fail("invalid PyTorch dimension")
		}
		out[i] = n
	}
	return out
}
