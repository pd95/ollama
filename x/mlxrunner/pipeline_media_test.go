package mlxrunner

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
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
}

func (m *mediaPrepareModel) PrepareMediaPrompt(prompt string, media []llm.MediaData) (*batch.PreparedInput, error) {
	m.gotPrompt = prompt
	m.gotMedia = media
	return &batch.PreparedInput{
		Tokens:      []int32{10, 20, 30},
		PLEInputIDs: []int32{10, 0, 30},
		Payload:     "prepared",
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
