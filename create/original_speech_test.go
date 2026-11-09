package create

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

type speechFixtureTensor struct {
	dtype string
	shape []int32
	data  []byte
}

// Small model-schema fixtures use the torch.save data representation. The
// fs/torch tests separately bind decoding to genuine torch.save files and full
// owner-installed original archives; this helper never invokes Python.
func writeSpeechArchive(t *testing.T, name string, root any) {
	t.Helper()
	pickle := []byte{0x80, 2}
	var storages [][]byte
	text := func(value string) {
		pickle = append(pickle, 'X')
		pickle = binary.LittleEndian.AppendUint32(pickle, uint32(len(value)))
		pickle = append(pickle, value...)
	}
	integer := func(value int32) {
		pickle = append(pickle, 'J')
		pickle = binary.LittleEndian.AppendUint32(pickle, uint32(value))
	}
	tuple := func(values []int32) {
		pickle = append(pickle, '(')
		for _, value := range values {
			integer(value)
		}
		pickle = append(pickle, 't')
	}
	var encode func(any)
	encode = func(value any) {
		switch value := value.(type) {
		case map[string]any:
			pickle = append(pickle, '}')
			keys := make([]string, 0, len(value))
			for key := range value {
				keys = append(keys, key)
			}
			slices.Sort(keys)
			for _, key := range keys {
				text(key)
				encode(value[key])
				pickle = append(pickle, 's')
			}
		case speechFixtureTensor:
			kind, width := "FloatStorage", 4
			if value.dtype == "BF16" {
				kind, width = "BFloat16Storage", 2
			} else if value.dtype != "F32" {
				t.Fatalf("unsupported test storage dtype %s", value.dtype)
			}
			pickle = append(pickle, []byte("ctorch._utils\n_rebuild_tensor_v2\n((")...)
			text("storage")
			pickle = append(pickle, []byte("ctorch\n"+kind+"\n")...)
			text(strconv.Itoa(len(storages)))
			text("cuda:0")
			integer(int32(len(value.data) / width))
			pickle = append(pickle, 't', 'Q')
			integer(0)
			tuple(value.shape)
			strides := make([]int32, len(value.shape))
			stride := int32(1)
			for i := len(strides) - 1; i >= 0; i-- {
				strides[i] = stride
				stride *= value.shape[i]
			}
			tuple(strides)
			pickle = append(pickle, []byte("\x89ccollections\nOrderedDict\n)RtR")...)
			storages = append(storages, value.data)
		case nil:
			pickle = append(pickle, 'N')
		default:
			t.Fatalf("unsupported test archive value %T", value)
		}
	}
	encode(root)
	pickle = append(pickle, '.')
	f, err := os.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	w := zip.NewWriter(f)
	members := map[string][]byte{"data.pkl": pickle, "byteorder": []byte("little"), "version": []byte("3\n")}
	for i, data := range storages {
		members["data/"+strconv.Itoa(i)] = data
	}
	for key, data := range members {
		member, err := w.CreateHeader(&zip.FileHeader{Name: "archive/" + key, Method: zip.Store})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := member.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestOriginalKokoroCreatesIdenticalTensorBlobs(t *testing.T) {
	native, prepared := t.TempDir(), t.TempDir()
	config, err := os.ReadFile("testdata/kokoro-config.json")
	if err != nil {
		t.Fatal(err)
	}
	modelRoot := make(map[string]any)
	var canonical []*st.TensorData
	for _, module := range []string{"bert", "bert_encoder", "predictor", "decoder", "text_encoder"} {
		key := "weight"
		shape := []int32{2}
		data := []byte{0, 0, 0, 62, 0, 0, 128, 190}
		if module == "bert" {
			key, shape, data = "embeddings.word_embeddings.weight", []int32{178, 4}, make([]byte, 178*4*4)
		}
		modelRoot[module] = map[string]any{"module." + key: speechFixtureTensor{"F32", shape, data}}
		canonical = append(canonical, st.NewTensorDataFromBytes(module+"."+key, "F32", shape, data))
	}
	for _, dir := range []string{native, prepared} {
		if err := os.WriteFile(filepath.Join(dir, "config.json"), config, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(filepath.Join(dir, "voices"), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	writeSpeechArchive(t, filepath.Join(native, "kokoro-v1_0.pth"), modelRoot)
	createTestSafetensors(t, filepath.Join(prepared, "kokoro-v1_0.safetensors"), canonical)
	voice := speechFixtureTensor{"F32", []int32{510, 1, 256}, make([]byte, 510*256*4)}
	for _, name := range []string{"af_heart", "bf_emma", "zf_ignored"} {
		writeSpeechArchive(t, filepath.Join(native, "voices", name+".pt"), voice)
		createTestSafetensors(t, filepath.Join(prepared, "voices", name+".safetensors"), []*st.TensorData{st.NewTensorDataFromBytes("voice", voice.dtype, voice.shape, voice.data)})
	}
	if !IsSafetensorsModelDir(native) {
		t.Fatal("original Kokoro source not detected")
	}
	files, err := SafetensorsWeightFiles(native)
	if err != nil || !slices.Equal(files, []string{"kokoro-v1_0.pth", "voices/af_heart.pt", "voices/bf_emma.pt"}) {
		t.Fatalf("wrong original transfer files: %v %v", files, err)
	}
	stores := []*captureStore{newCaptureStore(), newCaptureStore()}
	for i, dir := range []string{native, prepared} {
		err := Create(t.Context(), "original-kokoro", dir, testPipelineOptions(), stores[i], func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(stores[0].blobs) != len(stores[1].blobs) {
		t.Fatalf("different output inventories: %d/%d", len(stores[0].blobs), len(stores[1].blobs))
	}
	for name, expected := range stores[1].blobs {
		if !bytes.Equal(stores[0].blobs[name], expected) {
			t.Errorf("native/prepared blob differs: %s", name)
		}
	}
	after, err := os.ReadFile(filepath.Join(native, "config.json"))
	if err != nil || !bytes.Equal(after, config) {
		t.Fatal("source config changed")
	}
	if _, err := os.Stat(filepath.Join(native, "kokoro-v1_0.safetensors")); !os.IsNotExist(err) {
		t.Fatal("creation wrote a conversion into original source")
	}
}

func TestOriginalSpeechAmbiguousSourcesFail(t *testing.T) {
	dir := t.TempDir()
	writeKokoroSource(t, dir)
	writeSpeechArchive(t, filepath.Join(dir, "kokoro-v1_0.pth"), speechFixtureTensor{"F32", []int32{1}, make([]byte, 4)})
	if _, err := SafetensorsWeightFiles(dir); err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("ambiguous model accepted: %v", err)
	}
	if err := os.Remove(filepath.Join(dir, "kokoro-v1_0.pth")); err != nil {
		t.Fatal(err)
	}
	writeSpeechArchive(t, filepath.Join(dir, "voices/af_heart.pt"), speechFixtureTensor{"F32", []int32{510, 1, 256}, make([]byte, 510*256*4)})
	if _, err := SafetensorsWeightFiles(dir); err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("ambiguous voice accepted: %v", err)
	}
}

func TestOriginalVoxtralVoiceSelectionAndTensorRead(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "voice_embedding"), 0o700); err != nil {
		t.Fatal(err)
	}
	voice := speechFixtureTensor{"BF16", []int32{2, 3072}, make([]byte, 2*3072*2)}
	copy(voice.data, []byte{0xc0, 0x3f}) // BF16 1.5, preserving the source bytes.
	for _, name := range voxtralEnglishGermanVoices {
		writeSpeechArchive(t, filepath.Join(dir, "voice_embedding", name+".pt"), voice)
	}
	files, err := voxtralVoiceFiles(dir)
	if err != nil || len(files) != len(voxtralEnglishGermanVoices) {
		t.Fatalf("original voice discovery: %v %v", files, err)
	}
	source, err := openTensorSource(t.Context(), filepath.Join(dir, files[0]))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	tensor, err := source.GetTensor("tensor")
	if err != nil || tensor.Dtype != "BF16" || !slices.Equal(tensor.Shape, voice.shape) {
		t.Fatalf("original voice metadata: %v %v", tensor, err)
	}
	data, err := io.ReadAll(tensor.Reader())
	if err != nil || !bytes.Equal(data, voice.data) {
		t.Fatalf("original BF16 voice bytes differ: %v", err)
	}
	createTestSafetensors(t, filepath.Join(dir, strings.TrimSuffix(files[0], ".pt")+".safetensors"), []*st.TensorData{st.NewTensorDataFromBytes("embedding", voice.dtype, voice.shape, voice.data)})
	if _, err := voxtralVoiceFiles(dir); err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("ambiguous Voxtral voice accepted: %v", err)
	}
}
