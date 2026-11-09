package create

import (
	"compress/gzip"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestOriginalTurboCompleteLearnedLayout(t *testing.T) {
	f, err := os.Open("testdata/chatterbox-turbo-original-layout.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	var fixture struct {
		Provider string            `json:"provider"`
		Revision string            `json:"revision"`
		Cases    []json.RawMessage `json:"cases"`
	}
	if err := json.NewDecoder(r).Decode(&fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Provider != "ResembleAI/chatterbox-turbo" || fixture.Revision != "749d1c1a46eb10492095d68fbcf55691ccf137cd" || len(fixture.Cases) != 1680 {
		t.Fatalf("unexpected original checkpoint metadata: %s %s %d", fixture.Provider, fixture.Revision, len(fixture.Cases))
	}
	seen := map[string]bool{}
	for _, raw := range fixture.Cases {
		var row []json.RawMessage
		if err := json.Unmarshal(raw, &row); err != nil || len(row) != 5 {
			t.Fatalf("invalid fixture row: %s", raw)
		}
		var group, name, want string
		var rank int
		var axes []int
		for i, target := range []any{&group, &name, &rank, &want, &axes} {
			if err := json.Unmarshal(row[i], target); err != nil {
				t.Fatal(err)
			}
		}
		got, err := originalTurboTensorLayout(group, name, rank)
		if err != nil || got.Name != want || !reflect.DeepEqual(got.Axes, axes) {
			t.Errorf("%s/%s: got %+v, %v; want %s axes %v", group, name, got, err, want, axes)
		}
		if seen[got.Name] {
			t.Errorf("output collision at %s", got.Name)
		}
		seen[got.Name] = true
	}
}

func TestOriginalChatterboxLayoutBoundaries(t *testing.T) {
	for _, test := range []struct {
		group, name string
		rank        int
	}{
		{"unknown", "weight", 2},
		{"t3", "", 2},
		{"t3", "weight", -1},
		{"s3gen", "weight", 5},
		{"t3", "tfmr.h.0.attn.c_attn.weight", 3},
		{"ve", "lstm.weight_ih_l3", 2},
		{"ve", "lstm.weight_ih_l0_reverse", 2},
	} {
		if got, err := originalTurboTensorLayout(test.group, test.name, test.rank); err == nil {
			t.Errorf("unexpected acceptance: %+v => %+v", test, got)
		}
	}
	// A square kernel still needs its axes swapped. Rank, rather than
	// equality of dimension lengths, determines the original storage layout.
	got, err := originalTurboTensorLayout("s3gen", "mel2wav.ups.0.weight", 3)
	if err != nil || got.Name != "s3gen.mel2wav.ups.0.conv.weight" || !reflect.DeepEqual(got.Axes, []int{1, 2, 0}) {
		t.Fatalf("transpose convolution: %+v, %v", got, err)
	}
}

func TestOriginalChatterboxSharedDependencies(t *testing.T) {
	for _, test := range []struct {
		name, want string
		rank       int
		axes       []int
	}{
		{"xvector.block1.tdnnd2.nonlinear1.batchnorm.weight", "blocks.0.layers.1.nonlinear1.0.weight", 1, []int{0}},
		{"xvector.transit2.nonlinear.batchnorm.running_var", "transits.1.nonlinear.0.running_var", 1, []int{0}},
		{"xvector.out_nonlinear.batchnorm.bias", "out_nonlinear.0.bias", 1, []int{0}},
		{"head.layer1.0.shortcut.0.weight", "head.layer1.0.shortcut.0.weight", 4, []int{0, 2, 3, 1}},
		{"xvector.tdnn.linear.weight", "tdnn.linear.weight", 3, []int{0, 2, 1}},
	} {
		got := originalCAMPLayout(test.name, test.rank)
		if got.Name != test.want || !reflect.DeepEqual(got.Axes, test.axes) {
			t.Errorf("CAMP %s => %+v; want %s %v", test.name, got, test.want, test.axes)
		}
	}
	for _, test := range []struct {
		name, want string
		rank       int
		axes       []int
	}{
		{"encoder.conv1.weight", "encoder.conv1.weight", 3, []int{0, 2, 1}},
		{"encoder.blocks.0.mlp.2.weight", "encoder.blocks.0.mlp.layers.2.weight", 2, []int{0, 1}},
		{"quantizer._codebook.project_in.weight", "quantizer.fsq_codebook.project_in.weight", 2, []int{0, 1}},
		{"_mel_filters", "_mel_filters", 2, []int{0, 1}},
	} {
		got := originalChatterboxS3Layout(test.name, test.rank)
		if got.Name != test.want || !reflect.DeepEqual(got.Axes, test.axes) {
			t.Errorf("S3 %s => %+v; want %s %v", test.name, got, test.want, test.axes)
		}
	}
}
