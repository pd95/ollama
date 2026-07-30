package batch

import "github.com/ollama/ollama/x/mlxrunner/mlx"

// Batch is the per-forward-pass input handed to a model.
type Batch struct {
	// InputIDs is the input token IDs for this forward pass, shape (B, L).
	InputIDs *mlx.Array

	// InputEmbeddings optionally replaces token embedding lookup for this
	// forward pass, shape (B, L, hidden). The token IDs are still present for
	// masks, positions, sampling history, and model-specific side inputs.
	InputEmbeddings *mlx.Array

	// PLEInputIDs optionally supplies token IDs for per-layer embeddings when
	// InputEmbeddings contains multimodal replacements.
	PLEInputIDs *mlx.Array

	// BidirectionalSpans contains absolute token ranges that may attend in
	// both directions during prefill. Model implementations decide which
	// layer types consume these spans.
	BidirectionalSpans []TokenSpan

	// SeqOffsets gives each row's current position within its sequence —
	// where the chunk in InputIDs starts. Length equals the batch dimension
	// of InputIDs.
	SeqOffsets []int32

	// SeqQueryLens is each row's real query length in this forward. Values
	// less than L mean the row's tail is padding that must be masked out.
	// Length equals the batch dimension of InputIDs.
	SeqQueryLens []int32

	// Hidden is the target hidden state a draft model fuses with its input
	// embedding for this step. It is nil for ordinary forward passes.
	Hidden *mlx.Array

	// Memo is per-forward memoization used to cache results, such as masks,
	// which are often the same across layers.
	Memo Memo
}

type TokenSpan struct {
	Start int
	End   int
}

type Memo struct {
	entries map[any]any
}

// PreparedInput carries prompt preparation results across the runner boundary.
// Media-specific implementations may attach CPU-side metadata in Payload during
// Prepare, then materialize InputEmbeddings on the MLX worker thread.
type PreparedInput struct {
	Tokens             []int32
	PLEInputIDs        []int32
	Payload            any
	InputEmbeddings    *mlx.Array
	BidirectionalSpans []TokenSpan
}

// Get returns the memoized value for key and true if present, or nil
// and false otherwise.
func (m *Memo) Get(key any) (any, bool) {
	v, ok := m.entries[key]
	return v, ok
}

// Put stores value under key, allocating on first use.
func (m *Memo) Put(key, value any) {
	if m.entries == nil {
		m.entries = map[any]any{}
	}
	m.entries[key] = value
}
