package mlxrunner

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/tokenizer"
)

type mediaRejectModel struct{}

func (mediaRejectModel) Forward(*batch.Batch, []cache.Cache) *mlx.Array { return nil }
func (mediaRejectModel) Unembed(*mlx.Array) *mlx.Array                  { return nil }
func (mediaRejectModel) NumLayers() int                                 { return 0 }
func (mediaRejectModel) Tokenizer() *tokenizer.Tokenizer                { return nil }
func (mediaRejectModel) MaxContextLength() int                          { return 128 }
func (mediaRejectModel) LoadWeights(map[string]*mlx.Array) error        { return nil }

type mediaPrepareModel struct {
	mediaRejectModel
	gotPrompt string
	gotMedia  []llm.MediaData
	tokens    []int32
}

type mediaPrefillModel struct {
	mediaRejectModel
	forwards int
	chunks   []int
	offsets  []int32
	t        *testing.T
}

func (m *mediaPrefillModel) Forward(b *batch.Batch, _ []cache.Cache) *mlx.Array {
	m.t.Helper()
	if b.InputEmbeddings == nil || !b.InputEmbeddings.Valid() {
		m.t.Fatal("prefill input embeddings are invalid")
	}
	m.forwards++
	m.chunks = append(m.chunks, b.InputIDs.Dim(1))
	m.offsets = append(m.offsets, b.SeqOffsets[0])
	return b.InputEmbeddings
}

func (m *mediaPrepareModel) PrepareMediaPrompt(_ context.Context, prompt string, media []llm.MediaData) (*batch.PreparedInput, error) {
	m.gotPrompt = prompt
	m.gotMedia = media
	tokens := m.tokens
	if len(tokens) == 0 {
		tokens = []int32{10, 20, 30}
	}
	return &batch.PreparedInput{
		Tokens:  tokens,
		Payload: "prepared",
	}, nil
}

func TestPrepareRejectsMediaWithoutModelPreparer(t *testing.T) {
	runner := Runner{
		Model:         mediaRejectModel{},
		contextLength: 128,
	}
	request := &Request{
		CompletionRequest: CompletionRequest{
			Prompt: "describe [img-0]",
			Media:  []llm.MediaData{llm.NewMediaData(0, []byte("image"))},
		},
	}

	err := runner.Prepare(request)
	if err == nil {
		t.Fatal("Prepare() error = nil, want media rejection")
	}
	if !strings.Contains(err.Error(), "does not support media inputs") {
		t.Fatalf("Prepare() error = %q, want media rejection", err)
	}
	if len(request.Tokens) != 0 {
		t.Fatalf("Prepare() populated %d tokens for rejected media request", len(request.Tokens))
	}
}

func TestPrepareUsesMediaPromptPreparer(t *testing.T) {
	model := &mediaPrepareModel{}
	runner := Runner{
		Model:         model,
		contextLength: 128,
	}
	media := []llm.MediaData{llm.NewMediaData(7, []byte("image"))}
	request := &Request{
		CompletionRequest: CompletionRequest{
			Prompt: "describe [img-0]",
			Media:  media,
			Options: api.Options{
				NumPredict: 4,
			},
		},
	}

	if err := runner.Prepare(request); err != nil {
		t.Fatal(err)
	}
	if model.gotPrompt != "describe [img-0]" {
		t.Fatalf("prompt = %q", model.gotPrompt)
	}
	if len(model.gotMedia) != 1 || model.gotMedia[0].ID != 7 {
		t.Fatalf("media = %#v", model.gotMedia)
	}
	if got, want := request.Tokens, []int32{10, 20, 30}; !equalInt32s(got, want) {
		t.Fatalf("tokens = %v, want %v", got, want)
	}
	if request.Prepared == nil || request.Prepared.Payload != "prepared" {
		t.Fatalf("prepared = %#v", request.Prepared)
	}
}

func TestPrepareRejectsExpandedMultiMediaContext(t *testing.T) {
	model := &mediaPrepareModel{tokens: []int32{1, 2, 3, 4}}
	runner := Runner{Model: model, contextLength: 4}
	request := &Request{CompletionRequest: CompletionRequest{
		Prompt: "compare [img-0] and [img-1]",
		Media: []llm.MediaData{
			{ID: 0, Kind: llm.MediaKindImage, Data: []byte("first")},
			{ID: 1, Kind: llm.MediaKindAudio, Data: []byte("second")},
		},
	}}
	err := runner.Prepare(request)
	if err == nil || !strings.Contains(err.Error(), "input length (4 tokens)") {
		t.Fatalf("Prepare() error = %v, want exact expanded context rejection", err)
	}
	if len(model.gotMedia) != 2 {
		t.Fatalf("media passed to preparer = %d, want 2", len(model.gotMedia))
	}
}

func TestClientCompletionForwardsMedia(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	seen := make(chan CompletionRequest, 1)
	server := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req CompletionRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("decode request: %v", err)
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			seen <- req
			if err := json.NewEncoder(w).Encode(CompletionResponse{Done: true}); err != nil {
				t.Errorf("encode response: %v", err)
			}
		}),
	}
	go server.Serve(listener)
	defer server.Shutdown(context.Background())

	client := &Client{
		port:   listener.Addr().(*net.TCPAddr).Port,
		client: http.DefaultClient,
	}
	err = client.Completion(context.Background(), llm.CompletionRequest{
		Prompt:  "describe [img-0]",
		Media:   []llm.MediaData{llm.NewMediaData(7, []byte("image-bytes"))},
		Options: &api.Options{Temperature: 0},
	}, func(llm.CompletionResponse) {})
	if err != nil {
		t.Fatal(err)
	}

	got := <-seen
	if got.Prompt != "describe [img-0]" {
		t.Fatalf("Prompt = %q", got.Prompt)
	}
	if len(got.Media) != 1 {
		t.Fatalf("Media length = %d, want 1", len(got.Media))
	}
	if got.Media[0].ID != 7 || string(got.Media[0].Data) != "image-bytes" {
		t.Fatalf("Media[0] = %#v, want id 7 image bytes", got.Media[0])
	}
}

func TestMediaInputEmbeddingsSurviveChunkedPrefill(t *testing.T) {
	skipIfNoMLX(t)

	const tokens = 2050
	values := make([]float32, tokens)
	embeddings := mlx.FromValues(values, 1, tokens, 1)
	defer mlx.Sweep()

	model := &mediaPrefillModel{t: t}
	inputIDs := make([]int32, tokens)
	runner := Runner{Model: model}
	session := &cacheSession{
		inputs:          inputIDs,
		remaining:       inputIDs,
		inputEmbeddings: embeddings,
	}
	if _, _, _, err := runner.prefill(context.Background(), session, nil); err != nil {
		t.Fatal(err)
	}
	if model.forwards != 2 {
		t.Fatalf("forward calls = %d, want 2 prefill chunks", model.forwards)
	}
}

func TestPrefillSeedSurvivesPipelineSweep(t *testing.T) {
	skipIfNoMLX(t)

	seed := mlx.FromValues([]int32{7}, 1)
	mlx.Pin(seed)
	defer mlx.Unpin(seed)
	mlx.Sweep()

	if got := seed.Dims(); !slices.Equal(got, []int{1}) {
		t.Fatalf("seed dimensions after sweep = %v, want [1]", got)
	}
	if got := seed.ExpandDims(-1).Dims(); !slices.Equal(got, []int{1, 1}) {
		t.Fatalf("expanded seed dimensions = %v, want [1 1]", got)
	}
}

func TestBidirectionalMediaPrefillChunksAroundCompleteSpan(t *testing.T) {
	skipIfNoMLX(t)

	const tokens = 2050
	embeddings := mlx.FromValues(make([]float32, tokens), 1, tokens, 1)
	defer mlx.Sweep()
	model := &mediaPrefillModel{t: t}
	inputIDs := make([]int32, tokens)
	runner := Runner{Model: model}
	session := &cacheSession{
		inputs:             inputIDs,
		remaining:          inputIDs,
		inputEmbeddings:    embeddings,
		bidirectionalSpans: []batch.TokenSpan{{Start: 10, End: 290}},
	}
	if _, _, _, err := runner.prefill(context.Background(), session, nil); err != nil {
		t.Fatal(err)
	}
	if got, want := model.chunks, []int{10, 280, 1759}; !slices.Equal(got, want) {
		t.Fatalf("prefill chunk sizes = %v, want %v", got, want)
	}
	if got, want := model.offsets, []int32{0, 10, 290}; !slices.Equal(got, want) {
		t.Fatalf("prefill offsets = %v, want %v", got, want)
	}
}

func TestBidirectionalMediaPrefillRejectsOversizedSpan(t *testing.T) {
	span := batch.TokenSpan{Start: 1, End: prefillChunkSize() + 2}
	err := validateBidirectionalSpans([]batch.TokenSpan{span}, span.End+1, prefillChunkSize())
	if err == nil || !strings.Contains(err.Error(), "exceeds prefill limit") {
		t.Fatalf("validateBidirectionalSpans() error = %v, want prefill limit", err)
	}
}

func TestBidirectionalMediaPrefillKeepsLongTextChunked(t *testing.T) {
	const total = 5000
	spans := []batch.TokenSpan{{Start: 3000, End: 3280}}
	var chunks []int
	for position := 0; position < total-1; {
		n, err := nextPrefillChunk(position, total-position-1, prefillChunkSize(), spans)
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, n)
		position += n
	}
	if want := []int{2048, 952, 280, 1719}; !slices.Equal(chunks, want) {
		t.Fatalf("prefill chunk sizes = %v, want %v", chunks, want)
	}
}

func TestBidirectionalMediaPrefillChunksAroundMultipleSpans(t *testing.T) {
	const total = 5000
	spans := []batch.TokenSpan{{Start: 10, End: 290}, {Start: 400, End: 680}}
	var chunks []int
	for position := 0; position < total-1; {
		n, err := nextPrefillChunk(position, total-position-1, prefillChunkSize(), spans)
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, n)
		position += n
	}
	if want := []int{10, 280, 110, 280, 2048, 2048, 223}; !slices.Equal(chunks, want) {
		t.Fatalf("prefill chunk sizes = %v, want %v", chunks, want)
	}
	if err := validateBidirectionalSpans(spans, total, prefillChunkSize()); err != nil {
		t.Fatalf("validateBidirectionalSpans() error = %v", err)
	}
}

func equalInt32s(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
