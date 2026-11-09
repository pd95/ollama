package create

import (
	"bytes"
	"compress/gzip"
	"context"
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/fs/torch"
)

// These are geometry/configuration, not trained parameters. Original Turbo
// omits the native config and stores its trained dependencies in S3Gen.
//
//go:embed chatterbox_turbo_original_config.json
var originalTurboConfig []byte

//go:embed chatterbox_turbo_original_shapes.json.gz
var originalTurboShapes []byte

//go:embed chatterbox_v3_original_config.json
var originalV3Config []byte

//go:embed chatterbox_v3_original_shapes.json.gz
var originalV3Shapes []byte

var (
	originalTurboWeightFiles = []string{"t3_turbo_v1.safetensors", "ve.safetensors", "s3gen_meanflow.safetensors", "conds.pt"}
	originalTurboConfigFiles = []string{"tokenizer_config.json", "vocab.json", "merges.txt"}
	originalV3WeightFiles    = []string{"t3_mtl23ls_v3.safetensors", "ve.safetensors", "s3gen_v3.safetensors"}
	originalV3ConfigFiles    = []string{"grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"}
)

type originalChatterboxVariant struct {
	turbo           bool
	weights, assets []string
	config, shapes  []byte
}

func originalTurboSource(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "t3_turbo_v1.safetensors"))
	return err == nil
}

func originalV3Source(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "t3_mtl23ls_v3.safetensors"))
	return err == nil
}

func originalChatterboxSource(dir string) bool {
	return originalTurboSource(dir) || originalV3Source(dir)
}

func originalChatterboxVariantForDir(dir string) (*originalChatterboxVariant, error) {
	turbo, v3 := originalTurboSource(dir), originalV3Source(dir)
	if !turbo && !v3 {
		return nil, nil
	}
	if turbo && v3 {
		return nil, fmt.Errorf("ambiguous original Chatterbox v3/Turbo directory")
	}
	v := &originalChatterboxVariant{weights: originalV3WeightFiles, assets: originalV3ConfigFiles, config: originalV3Config, shapes: originalV3Shapes}
	if turbo {
		v = &originalChatterboxVariant{turbo: true, weights: originalTurboWeightFiles, assets: originalTurboConfigFiles, config: originalTurboConfig, shapes: originalTurboShapes}
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); !os.IsNotExist(err) {
		return nil, fmt.Errorf("original Chatterbox files must not be mixed with a native config.json")
	}
	for _, name := range append(slices.Clone(v.weights), v.assets...) {
		info, err := os.Stat(filepath.Join(dir, name))
		if err != nil || !info.Mode().IsRegular() {
			return nil, fmt.Errorf("original Chatterbox requires regular file %s", name)
		}
		if info.Size() <= 0 || info.Size() > 6<<30 || (filepath.Ext(name) != ".safetensors" && info.Size() > 64<<20) {
			return nil, fmt.Errorf("original Chatterbox file %s exceeds supported size", name)
		}
	}
	return v, nil
}

type originalChatterboxTensor struct {
	name, dtype string
	shape       []int32
	source      *safetensors.TensorData
	other       *safetensors.TensorData
	operation   string
	axes        []int
	derived     []byte
}

func (t originalChatterboxTensor) size() int64 {
	n := int64(4)
	for _, dim := range t.shape {
		n *= int64(dim)
	}
	return n
}

// normalizeOriginalChatterbox creates an owned temporary data tree. Source
// snapshots are read-only, and neither Python nor provider code is executed.
// The returned cleanup is always safe to call, including on failure.
func normalizeOriginalChatterbox(ctx context.Context, dir string) (string, func(), error) {
	noop := func() {}
	variant, err := originalChatterboxVariantForDir(dir)
	if err != nil {
		return "", noop, err
	}
	if variant == nil {
		return dir, noop, nil
	}
	if err := checkContext(ctx); err != nil {
		return "", noop, err
	}
	sources := newSourceFiles(ctx, dir)
	defer sources.close()
	groups, err := planOriginalChatterbox(ctx, sources, *variant)
	if err != nil {
		return "", noop, err
	}
	stage, err := os.MkdirTemp("", "ollama-chatterbox-import-*")
	if err != nil {
		return "", noop, err
	}
	cleanup := func() { _ = os.RemoveAll(stage) }
	fail := func(err error) (string, func(), error) { cleanup(); return "", noop, err }
	for file, tensors := range groups {
		if err := os.MkdirAll(filepath.Dir(filepath.Join(stage, file)), 0o700); err != nil {
			return fail(err)
		}
		if err := writeOriginalChatterboxFile(ctx, filepath.Join(stage, file), tensors); err != nil {
			return fail(fmt.Errorf("normalize original Chatterbox %s: %w", file, err))
		}
	}
	configs := map[string][]byte{
		"config.json":              variant.config,
		"s3_tokenizer/config.json": []byte(`{"model_type":"s3_tokenizer_v2","version":"2.0","sample_rate":16000,"token_rate":25,"codebook_size":6561}`),
	}
	if variant.turbo {
		configs["reference_encoder/config.json"] = []byte(`{"model_type":"chatterbox_campplus","version":"1.0","sample_rate":16000,"mel_bins":80,"embedding_dim":192}`)
	}
	for _, name := range variant.assets {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return fail(err)
		}
		if name == "grapheme_mtl_merged_expanded_v1.json" {
			name = "tokenizer.json"
		}
		configs[name] = data
	}
	for name, data := range configs {
		if err := os.WriteFile(filepath.Join(stage, name), data, 0o600); err != nil {
			return fail(err)
		}
	}
	return stage, cleanup, nil
}

func planOriginalChatterbox(ctx context.Context, sources *sourceFiles, variant originalChatterboxVariant) (map[string][]originalChatterboxTensor, error) {
	r, err := gzip.NewReader(bytes.NewReader(variant.shapes))
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var expected map[string]map[string][]int32
	if err := json.NewDecoder(io.LimitReader(r, 1<<20)).Decode(&expected); err != nil {
		return nil, err
	}
	groups := map[string][]originalChatterboxTensor{}
	seen := map[string]map[string]bool{"main": {}, "camp": {}, "s3": {}}
	add := func(group string, item originalChatterboxTensor) error {
		shape, ok := expected[group][item.name]
		if !ok || !slices.Equal(shape, item.shape) || seen[group][item.name] {
			return fmt.Errorf("unsupported original Chatterbox %s tensor %s shape %v", group, item.name, item.shape)
		}
		seen[group][item.name] = true
		groups[group] = append(groups[group], item)
		return nil
	}
	for _, file := range variant.weights[:3] {
		reader, err := sources.extractor(file)
		if err != nil {
			return nil, err
		}
		names := reader.ListTensors()
		if len(names) == 0 || len(names) > 4096 {
			return nil, fmt.Errorf("unsupported original Chatterbox tensor count in %s", file)
		}
		for _, name := range names {
			if err := checkContext(ctx); err != nil {
				return nil, err
			}
			tensor, err := reader.GetTensor(name)
			if err != nil {
				return nil, err
			}
			if strings.HasSuffix(name, ".num_batches_tracked") {
				if tensor.Dtype != "I64" || len(tensor.Shape) != 0 {
					return nil, fmt.Errorf("invalid original Chatterbox training counter %s", name)
				}
				continue
			}
			if err := validateOriginalChatterboxFloat(tensor); err != nil {
				return nil, err
			}
			group, local := "main", name
			component := "s3gen"
			switch file {
			case variant.weights[0]:
				component = "t3"
			case "ve.safetensors":
				component = "ve"
			default:
				if strings.HasPrefix(name, "speaker_encoder.") {
					group, local = "camp", strings.TrimPrefix(name, "speaker_encoder.")
				} else if strings.HasPrefix(name, "tokenizer.") {
					group, local = "s3", strings.TrimPrefix(name, "tokenizer.")
				}
			}
			item := originalChatterboxTensor{source: tensor, dtype: "F32"}
			if component == "ve" && strings.HasPrefix(local, "lstm.bias_") {
				other := strings.Replace(local, "bias_hh", "bias_ih", 1)
				if other != local {
					if _, err := reader.GetTensor(other); err != nil {
						return nil, fmt.Errorf("incomplete original voice bias pair: %s", local)
					}
					continue
				}
				item.other, err = reader.GetTensor(strings.Replace(local, "bias_ih", "bias_hh", 1))
				item.operation = "add"
			} else if strings.Contains(local, ".parametrizations.weight.original") {
				if strings.HasSuffix(local, ".original1") {
					if _, err := reader.GetTensor(strings.TrimSuffix(name, "1") + "0"); err != nil {
						return nil, fmt.Errorf("incomplete original weight normalization pair: %s", name)
					}
					continue
				}
				if !strings.HasSuffix(local, ".original0") {
					return nil, fmt.Errorf("unsupported original weight normalization %s", name)
				}
				item.other = tensor
				item.source, err = reader.GetTensor(strings.TrimSuffix(name, "0") + "1")
				item.operation = "weight_norm"
				local = strings.TrimSuffix(local, ".parametrizations.weight.original0") + ".weight"
			}
			if err != nil {
				return nil, fmt.Errorf("incomplete original Chatterbox parameter pair %s: %w", name, err)
			}
			if err := validateOriginalChatterboxFloat(item.source); err != nil {
				return nil, err
			}
			if item.other != nil {
				if err := validateOriginalChatterboxFloat(item.other); err != nil {
					return nil, err
				}
				if item.operation == "add" && !slices.Equal(item.source.Shape, item.other.Shape) {
					return nil, fmt.Errorf("original voice bias shapes differ: %s", name)
				}
				if item.operation == "weight_norm" && (len(item.source.Shape) < 2 || item.other.Size != int64(item.source.Shape[0])*4) {
					return nil, fmt.Errorf("unsupported weight normalization shape: %s", name)
				}
				if item.operation == "weight_norm" {
					if len(item.source.Shape) != len(item.other.Shape) || item.source.Shape[0] != item.other.Shape[0] {
						return nil, fmt.Errorf("unsupported weight normalization magnitude: %s", name)
					}
					for _, dim := range item.other.Shape[1:] {
						if dim != 1 {
							return nil, fmt.Errorf("unsupported weight normalization broadcast: %s", name)
						}
					}
				}
			}
			var layout originalChatterboxLayout
			switch group {
			case "camp":
				layout = originalCAMPLayout(local, len(item.source.Shape))
				layout.Name = "s3gen.speaker_encoder." + layout.Name
			case "s3":
				layout = originalChatterboxS3Layout(local, len(item.source.Shape))
				if local == "_mel_filters" {
					layout.Name = "mel_filters"
				}
			default:
				if variant.turbo {
					layout, err = originalTurboTensorLayout(component, local, len(item.source.Shape))
				} else {
					layout, err = originalV3TensorLayout(component, local, len(item.source.Shape))
				}
				if err != nil {
					return nil, err
				}
			}
			item.name, item.axes = layout.Name, layout.Axes
			for _, axis := range item.axes {
				item.shape = append(item.shape, item.source.Shape[axis])
			}
			if err := add(group, item); err != nil {
				return nil, err
			}
		}
	}
	if variant.turbo {
		for _, tensor := range originalTurboTables() {
			if err := add("main", tensor); err != nil {
				return nil, err
			}
		}
	}
	for group := range expected {
		if len(seen[group]) != len(expected[group]) {
			return nil, fmt.Errorf("incomplete original Chatterbox %s inventory: got %d of %d", group, len(seen[group]), len(expected[group]))
		}
	}
	if !variant.turbo {
		return map[string][]originalChatterboxTensor{"model.safetensors": append(groups["main"], groups["camp"]...), "s3_tokenizer/model.safetensors": groups["s3"]}, nil
	}
	conditions, err := originalTurboConditions(ctx, sources)
	if err != nil {
		return nil, err
	}
	return map[string][]originalChatterboxTensor{"model.safetensors": groups["main"], "reference_encoder/model.safetensors": groups["camp"], "s3_tokenizer/model.safetensors": groups["s3"], "conds.safetensors": conditions}, nil
}

func validateOriginalChatterboxFloat(t *safetensors.TensorData) error {
	if t == nil || t.Dtype != "F32" || len(t.Shape) > 4 {
		return fmt.Errorf("original Chatterbox requires dense F32 tensors of rank at most four")
	}
	n := int64(4)
	for _, dim := range t.Shape {
		if dim <= 0 || n > (512<<20)/int64(dim) {
			return fmt.Errorf("unsupported original Chatterbox tensor shape %v", t.Shape)
		}
		n *= int64(dim)
	}
	if n != t.Size || n > 512<<20 {
		return fmt.Errorf("invalid original Chatterbox tensor extent %s", t.Name)
	}
	return nil
}

func originalTurboConditions(ctx context.Context, sources *sourceFiles) ([]originalChatterboxTensor, error) {
	reader, err := sources.extractor("conds.pt")
	if err != nil {
		return nil, err
	}
	archive, ok := reader.(*torchTensorSource)
	if !ok {
		return nil, fmt.Errorf("original Turbo conditioning requires a native tensor archive")
	}
	root, ok := archive.archive.Root.(map[string]any)
	if !ok || len(root) != 2 {
		return nil, fmt.Errorf("unsupported original Turbo conditioning dictionary")
	}
	for _, group := range []string{"t3", "gen"} {
		fields, ok := root[group].(map[string]any)
		if !ok || len(fields) > 10 {
			return nil, fmt.Errorf("unsupported original Turbo %s conditioning", group)
		}
		for _, value := range fields {
			if value != nil {
				if _, ok := value.(*torch.Tensor); !ok {
					return nil, fmt.Errorf("non-tensor original Turbo conditioning")
				}
			}
		}
	}
	shapes := map[string][]int32{"t3.speaker_emb": {1, 256}, "t3.cond_prompt_speech_tokens": {1, 375}, "gen.embedding": {1, 192}, "gen.prompt_feat": {1, 500, 80}, "gen.prompt_token": {1, 250}, "gen.prompt_token_len": {1}, "t3.emotion_adv": {1, 1, 1}}
	var result []originalChatterboxTensor
	for _, name := range reader.ListTensors() {
		if err := checkContext(ctx); err != nil {
			return nil, err
		}
		tensor, err := reader.GetTensor(name)
		if err != nil {
			return nil, err
		}
		if len(tensor.Shape) > 4 || tensor.Size > 8<<20 || tensor.Size < 0 || (tensor.Dtype != "F32" && tensor.Dtype != "I64" && tensor.Dtype != "I32") {
			return nil, fmt.Errorf("unsupported original Turbo conditioning tensor %s", name)
		}
		shape, supported := shapes[name]
		if !supported || !slices.Equal(shape, tensor.Shape) {
			return nil, fmt.Errorf("unsupported original Turbo conditioning name/shape %s", name)
		}
		integer := name == "t3.cond_prompt_speech_tokens" || name == "gen.prompt_token" || name == "gen.prompt_token_len"
		if (integer && tensor.Dtype == "F32") || (!integer && tensor.Dtype != "F32") {
			return nil, fmt.Errorf("unsupported original Turbo conditioning dtype %s", name)
		}
		delete(shapes, name)
		item := originalChatterboxTensor{name: name, shape: slices.Clone(tensor.Shape), source: tensor, dtype: tensor.Dtype}
		if tensor.Dtype == "I64" {
			item.dtype, item.operation = "I32", "narrow"
		}
		result = append(result, item)
	}
	if len(shapes) != 0 {
		return nil, fmt.Errorf("incomplete original Turbo conditioning tensors")
	}
	return result, nil
}

func writeOriginalChatterboxFile(ctx context.Context, name string, tensors []originalChatterboxTensor) error {
	slices.SortFunc(tensors, func(a, b originalChatterboxTensor) int { return strings.Compare(a.name, b.name) })
	header := map[string]any{}
	var offset int64
	for _, t := range tensors {
		header[t.name] = map[string]any{"dtype": t.dtype, "shape": t.shape, "data_offsets": []int64{offset, offset + t.size()}}
		offset += t.size()
	}
	data, err := json.Marshal(header)
	if err != nil {
		return err
	}
	data = append(data, bytes.Repeat([]byte(" "), (8-len(data)%8)%8)...)
	f, err := os.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, uint64(len(data))); err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		return err
	}
	for _, t := range tensors {
		raw, err := originalChatterboxTensorBytes(ctx, t)
		if err != nil {
			return fmt.Errorf("%s: %w", t.name, err)
		}
		if int64(len(raw)) != t.size() {
			return fmt.Errorf("wrong normalized extent %s", t.name)
		}
		if _, err := f.Write(raw); err != nil {
			return err
		}
	}
	return f.Close()
}
