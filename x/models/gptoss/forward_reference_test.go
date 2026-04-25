package gptoss

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func TestForwardReferenceShortEmbeddings(t *testing.T) {
	testForwardReferenceEmbeddings(t, "short")
}

func TestForwardReferenceCacheDecodeEmbeddings(t *testing.T) {
	testForwardReferenceEmbeddings(t, "cache")
}

func testForwardReferenceEmbeddings(t *testing.T, caseName string) {
	t.Helper()
	modelDir := envDirOrSkip(t, "GPTOSS_MODEL_DIR")
	refDir := envDirOrSkip(t, "GPTOSS_REF_DIR")
	skipIfNoMLX(t)

	ref := loadGPTOSSForwardReference(t, refDir, caseName)
	defer ref.Close()

	model := loadGPTOSSEmbeddingReferenceModel(t, modelDir)
	defer model.Close()

	inputIDs := flattenReferenceInputIDs(t, ref.Manifest.InputIDs)
	tokens := mlx.FromValues(inputIDs, len(ref.Manifest.InputIDs), len(ref.Manifest.InputIDs[0]))
	assertReferenceInputIDs(t, ref, inputIDs)

	got := model.Model.EmbedTokens.Forward(tokens)
	want := ref.Tensor(t, "model.embed_tokens")
	compareForwardReferenceArrays(t, "model.embed_tokens", got, want, 0)
}

type gptossEmbeddingReferenceModel struct {
	Model *Model
	file  *mlx.SafetensorsFile
}

func (m *gptossEmbeddingReferenceModel) Close() {
	if m != nil && m.file != nil {
		m.file.Free()
	}
}

func loadGPTOSSEmbeddingReferenceModel(t *testing.T, modelDir string) *gptossEmbeddingReferenceModel {
	t.Helper()

	configData, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		t.Skipf("GPTOSS_MODEL_DIR is missing config.json: %v", err)
	}
	cfg, err := parseConfig(configData)
	if err != nil {
		t.Fatalf("parse GPTOSS_MODEL_DIR config.json: %v", err)
	}

	file, weight := loadHFSafetensor(t, modelDir, "model.embed_tokens.weight")
	if err := validateTensorShape("model.embed_tokens.weight", weight, []int{int(cfg.VocabSize), int(cfg.HiddenSize)}, "vocab_size x hidden_size"); err != nil {
		file.Free()
		t.Fatalf("validate embedding tensor: %v", err)
	}

	return &gptossEmbeddingReferenceModel{
		Model: &Model{
			Config:      &cfg,
			EmbedTokens: nn.NewEmbedding(weight),
		},
		file: file,
	}
}

func loadHFSafetensor(t *testing.T, modelDir, tensorName string) (*mlx.SafetensorsFile, *mlx.Array) {
	t.Helper()

	indexPath := filepath.Join(modelDir, "model.safetensors.index.json")
	data, err := os.ReadFile(indexPath)
	if err != nil {
		t.Skipf("GPTOSS_MODEL_DIR is missing model.safetensors.index.json: %v", err)
	}

	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		t.Fatalf("parse %s: %v", indexPath, err)
	}

	fileName, ok := index.WeightMap[tensorName]
	if !ok {
		t.Skipf("GPTOSS_MODEL_DIR index does not contain %q", tensorName)
	}

	path := filepath.Join(modelDir, fileName)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("GPTOSS_MODEL_DIR is missing %s for %q: %v", fileName, tensorName, err)
	}

	file, err := mlx.LoadSafetensorsNative(path)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	tensor := file.Get(tensorName)
	if tensor == nil {
		file.Free()
		t.Fatalf("safetensors file %s did not contain %q", path, tensorName)
	}

	return file, tensor
}

type gptossForwardReference struct {
	Manifest gptossReferenceManifest
	file     *mlx.SafetensorsFile
}

type gptossReferenceManifest struct {
	InputIDs []gptossReferenceTokenRow `json:"input_ids"`
	Tensors  map[string]struct {
		DType string `json:"dtype"`
		Shape []int  `json:"shape"`
	} `json:"tensors"`
}

type gptossReferenceTokenRow []int32

func (r *gptossForwardReference) Close() {
	if r != nil && r.file != nil {
		r.file.Free()
	}
}

func (r *gptossForwardReference) Tensor(t *testing.T, name string) *mlx.Array {
	t.Helper()
	if _, ok := r.Manifest.Tensors[name]; !ok {
		t.Fatalf("reference manifest does not declare tensor %q", name)
	}
	tensor := r.file.Get(name)
	if tensor == nil {
		t.Fatalf("reference safetensors does not contain %q", name)
	}
	return tensor
}

func loadGPTOSSForwardReference(t *testing.T, refDir, caseName string) *gptossForwardReference {
	t.Helper()

	caseDir := filepath.Join(refDir, caseName)
	manifestPath := filepath.Join(caseDir, "activations.safetensors.manifest.json")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Skipf("GPTOSS_REF_DIR is missing %s reference manifest: %v", caseName, err)
	}

	var manifest gptossReferenceManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("parse %s: %v", manifestPath, err)
	}
	if len(manifest.InputIDs) == 0 || len(manifest.InputIDs[0]) == 0 {
		t.Fatalf("%s reference manifest has empty input_ids", caseName)
	}

	refPath := filepath.Join(caseDir, "activations.safetensors")
	if _, err := os.Stat(refPath); err != nil {
		t.Skipf("GPTOSS_REF_DIR is missing %s reference safetensors: %v", caseName, err)
	}
	file, err := mlx.LoadSafetensorsNative(refPath)
	if err != nil {
		t.Fatalf("load %s: %v", refPath, err)
	}

	return &gptossForwardReference{
		Manifest: manifest,
		file:     file,
	}
}

func flattenReferenceInputIDs(t *testing.T, rows []gptossReferenceTokenRow) []int32 {
	t.Helper()
	if len(rows) == 0 {
		t.Fatal("reference input_ids has no rows")
	}
	width := len(rows[0])
	if width == 0 {
		t.Fatal("reference input_ids has no columns")
	}

	out := make([]int32, 0, len(rows)*width)
	for rowIndex, row := range rows {
		if len(row) != width {
			t.Fatalf("reference input_ids row %d width = %d, want %d", rowIndex, len(row), width)
		}
		out = append(out, row...)
	}
	return out
}

func assertReferenceInputIDs(t *testing.T, ref *gptossForwardReference, want []int32) {
	t.Helper()

	gotTensor := ref.Tensor(t, "input_ids")
	got := materializedInts(gotTensor)
	wantInts := make([]int, len(want))
	for i, token := range want {
		wantInts[i] = int(token)
	}
	if !slices.Equal(got, wantInts) {
		t.Fatalf("reference input_ids tensor = %v, want manifest ids %v", got, wantInts)
	}
}

func materializedInts(a *mlx.Array) []int {
	if a == nil {
		return nil
	}
	cloned := a.Clone()
	mlx.Eval(cloned)
	return cloned.Ints()
}

func compareForwardReferenceArrays(t *testing.T, name string, got, want *mlx.Array, absTol float64) {
	t.Helper()

	if got == nil || !got.Valid() {
		t.Fatalf("%s got tensor is invalid", name)
	}
	if want == nil || !want.Valid() {
		t.Fatalf("%s reference tensor is invalid", name)
	}
	if !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("%s dims = %v, want %v", name, got.Dims(), want.Dims())
	}

	gotVals := materializedFloats(got.AsType(mlx.DTypeFloat32))
	wantVals := materializedFloats(want.AsType(mlx.DTypeFloat32))
	if len(gotVals) != len(wantVals) {
		t.Fatalf("%s length = %d, want %d", name, len(gotVals), len(wantVals))
	}

	var maxDiff float64
	maxIndex := -1
	for i := range wantVals {
		diff := math.Abs(float64(gotVals[i] - wantVals[i]))
		if diff > maxDiff {
			maxDiff = diff
			maxIndex = i
		}
		if diff > absTol {
			t.Fatalf("%s[%d] = %v, want %v (diff %v, max tol %v)", name, i, gotVals[i], wantVals[i], diff, absTol)
		}
	}
	t.Logf("%s matched %d values exactly; max diff %v at flat index %d", name, len(wantVals), maxDiff, maxIndex)
}

func envDirOrSkip(t *testing.T, name string) string {
	t.Helper()
	value := os.Getenv(name)
	if value == "" {
		t.Skipf("%s not set; set %s to a local GPT-OSS model/reference directory", name, name)
	}
	candidates, err := envDirCandidates(value)
	if err != nil {
		t.Fatalf("resolve %s=%q: %v", name, value, err)
	}
	var statErr error
	for _, candidate := range candidates {
		info, err := os.Stat(candidate)
		if err != nil {
			statErr = err
			continue
		}
		if !info.IsDir() {
			t.Skipf("%s=%s resolved to %s, which is not a directory", name, value, candidate)
		}
		return candidate
	}
	t.Skipf("%s=%s does not exist; checked %v: %v", name, value, candidates, statErr)
	return ""
}

func envDirCandidates(value string) ([]string, error) {
	if filepath.IsAbs(value) {
		return []string{value}, nil
	}

	cwdAbs, absErr := filepath.Abs(value)
	if absErr != nil {
		return nil, absErr
	}

	root := findGoModRoot()
	if root == "" {
		return []string{cwdAbs}, nil
	}
	rootAbs := filepath.Clean(filepath.Join(root, value))
	candidates := []string{cwdAbs}
	if rootAbs != cwdAbs {
		candidates = append(candidates, rootAbs)
	}

	if workdir := os.Getenv("WORKDIR"); workdir != "" {
		workdirAbs := filepath.Clean(filepath.Join(workdir, value))
		if !slices.Contains(candidates, workdirAbs) {
			candidates = append(candidates, workdirAbs)
		}
	}

	return candidates, nil
}

func findGoModRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}
