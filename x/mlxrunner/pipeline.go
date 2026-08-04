package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/tokenizer"
)

type mediaPromptPreparer interface {
	PrepareMediaPrompt(ctx context.Context, prompt string, media []llm.MediaData) (*batch.PreparedInput, error)
}

type mediaEmbeddingPreparer interface {
	PrepareMediaEmbeddings(ctx context.Context, prepared *batch.PreparedInput) error
}

func prefillChunkSize() int {
	return 2 << 10
}

// Prepare tokenizes the prompt and validates it against the model's
// context length. It is safe to call from any goroutine. On success it
// populates request.Tokens and adjusts request.Options.NumPredict.
func (r *Runner) Prepare(request *Request) error {
	return r.PrepareContext(context.Background(), request)
}

// PrepareContext prepares a request while allowing media decoding and
// preprocessing to stop when the caller disconnects.
func (r *Runner) PrepareContext(ctx context.Context, request *Request) error {
	if r.Model == nil {
		return errors.New("model not loaded")
	}

	var tokens []int32
	if len(request.Media) > 0 {
		preparer, ok := r.Model.(mediaPromptPreparer)
		if !ok {
			return fmt.Errorf("MLX runner does not support media inputs for this model")
		}
		prepared, err := preparer.PrepareMediaPrompt(ctx, request.Prompt, request.Media)
		if err != nil {
			return err
		}
		if prepared == nil {
			return errors.New("media prompt preparation returned nil")
		}
		request.Prepared = prepared
		tokens = prepared.Tokens
	} else {
		tokens = r.Tokenizer.Encode(request.Prompt, r.Tokenizer.AddBOS())
	}
	if len(tokens) == 0 {
		return errors.New("empty prompt")
	}

	if len(tokens) >= r.contextLength {
		return fmt.Errorf("input length (%d tokens) exceeds the model's maximum context length (%d tokens)", len(tokens), r.contextLength)
	}

	// Cap generation to stay within the model's context length
	maxGenerate := r.contextLength - len(tokens)
	if request.Options.NumPredict <= 0 {
		request.Options.NumPredict = maxGenerate
	} else {
		request.Options.NumPredict = min(request.Options.NumPredict, maxGenerate)
	}

	request.Tokens = tokens
	return nil
}

// The runner serializes requests today so we just use a fixed slot ID.
const pipelineSlot = 0

func (r *Runner) TextGenerationPipeline(ctx context.Context, request Request) error {
	mlx.ResetPeakMemory()

	defer func() {
		r.Sampler.Remove(pipelineSlot)
		mlx.Sweep()
		mlx.ClearCache()

		if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
			mlx.LogArrays()
			r.cache.dumpTree()
		}
		slog.Info("peak memory", "size", mlx.PrettyBytes(mlx.PeakMemory()))
	}()

	inputs := request.Tokens
	var embeddings *mlx.Array
	var pleInputIDs []int32
	cacheable := len(request.Media) == 0

	if len(request.Media) > 0 {
		preparer, ok := r.Model.(mediaEmbeddingPreparer)
		if !ok {
			return fmt.Errorf("MLX runner does not support media embeddings for this model")
		}
		if request.Prepared == nil {
			return errors.New("missing prepared media input")
		}
		if err := preparer.PrepareMediaEmbeddings(ctx, request.Prepared); err != nil {
			return err
		}
		embeddings = request.Prepared.InputEmbeddings
		if embeddings == nil {
			return errors.New("media prompt preparation did not produce input embeddings")
		}
		pleInputIDs = request.Prepared.PLEInputIDs
	}

	var session *cacheSession
	if cacheable {
		session = r.cache.begin(inputs)
	} else {
		session = r.cache.beginEphemeral(r.Model, inputs)
	}
	session.inputEmbeddings = embeddings
	session.pleInputIDs = pleInputIDs
	if request.Prepared != nil {
		session.bidirectionalSpans = request.Prepared.BidirectionalSpans
	}
	defer session.close()
	caches := session.caches

	// Built before prefill so a drafter with draft caches follows the prompt
	// through prefill alongside the target.
	var spec *speculationSession
	if cacheable {
		spec = r.spec.open(request, caches)
	}
	defer spec.close()

	seed, position, promptEval, err := r.prefill(ctx, session, spec)
	if err == nil {
		// prefill returns the decode seed as an MLX array. Keep it alive across
		// the sweep below and for any decoder graph that still references it.
		mlx.Pin(seed)
		defer mlx.Unpin(seed)
	}
	session.inputEmbeddings = nil
	if request.Prepared != nil {
		request.Prepared.InputEmbeddings = nil
	}
	mlx.Sweep()
	if err != nil {
		return err
	}

	// Register the sampler after prefill completes.
	r.Sampler.Add(pipelineSlot, request.SamplerOpts, inputs)

	var d decoder
	if spec != nil {
		d = spec.decoder(seed, position)
	} else {
		d = r.pipelinedDecoder(nil, caches, seed.ExpandDims(-1), position)
	}
	defer d.close()
	return r.decode(ctx, request, session, d, promptEval)
}

func pinInputEmbeddings(embeddings *mlx.Array) func() {
	mlx.Pin(embeddings)
	return func() { mlx.Unpin(embeddings) }
}

// prefill evaluates the prompt in chunks, leaving one token for decode to
// seed from, and schedules the prompt's periodic snapshots. It returns the
// seed token, the resume position, and the prompt-evaluation duration.
func (r *Runner) prefill(ctx context.Context, session *cacheSession, spec *speculationSession) (*mlx.Array, int, time.Duration, error) {
	if session.inputEmbeddings != nil {
		release := pinInputEmbeddings(session.inputEmbeddings)
		defer release()
	}
	start := time.Now()
	inputs := session.inputs
	tokens := session.remaining
	caches := session.caches
	prefillChunk := prefillChunkSize()
	if err := validateBidirectionalSpans(session.bidirectionalSpans, len(inputs), prefillChunk); err != nil {
		return nil, 0, 0, err
	}

	// Request periodic snapshots during prefill and near the end of the
	// prompt so that long prompts can be partially restored and
	// thinking/generation can be retried without full reprocessing.
	const snapshotInterval = 8192
	var snapshotOffsets []int
	for offset := snapshotInterval; offset < len(inputs); offset += snapshotInterval {
		snapshotOffsets = append(snapshotOffsets, offset)
	}

	const preThinking = 4
	if end := len(inputs) - preThinking; end > 0 {
		snapshotOffsets = append(snapshotOffsets, end)
	}

	materializeCaches := func() {
		state := make([]*mlx.Array, 0, 2*len(caches))
		for _, c := range caches {
			state = append(state, c.State()...)
		}
		if len(state) == 0 {
			return
		}
		mlx.Eval(state...)
	}

	session.schedulePrefillSnapshots(snapshotOffsets)

	total, processed := len(tokens), 0
	position := len(inputs) - len(tokens)
	for total-processed > 1 {
		if err := ctx.Err(); err != nil {
			return nil, 0, 0, err
		}

		n, err := nextPrefillChunk(position, total-processed-1, prefillChunk, session.bidirectionalSpans)
		if err != nil {
			return nil, 0, 0, err
		}

		chunkIDs := mlx.FromValues(tokens[processed:processed+n], 1, n)
		b := &batch.Batch{
			InputIDs:           chunkIDs,
			SeqOffsets:         []int32{int32(position)},
			SeqQueryLens:       []int32{int32(n)},
			BidirectionalSpans: session.bidirectionalSpans,
		}
		if session.inputEmbeddings != nil {
			b.InputEmbeddings = mlx.SliceStartStop(session.inputEmbeddings,
				[]int32{0, int32(position), 0},
				[]int32{1, int32(position + n), int32(session.inputEmbeddings.Dim(2))},
			)
		}
		if len(session.pleInputIDs) > 0 {
			b.PLEInputIDs = mlx.FromValues(session.pleInputIDs[position:position+n], 1, n)
		}
		hidden := r.Model.Forward(b, caches)
		spec.committed(chunkIDs, hidden, position)
		mlx.Sweep()
		materializeCaches()
		processed += n
		position += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)
		logutil.TraceContext(ctx, "mlx prompt forward", "processed", processed, "total", total, "tokens", n, "memory", mlx.Memory{})

		mlx.ClearCache()
	}

	// Settle before attaching: snapshots attach only at offsets every cache
	// has crossed, and the draft caches stay a pair short of the target
	// until the seed completes the frontier pair.
	seed := mlx.FromValues(tokens[processed:], 1)
	spec.settle(seed)
	session.attachPrefillSnapshots()

	return seed, position, time.Since(start), nil
}

func validateBidirectionalSpans(spans []batch.TokenSpan, sequenceLength, maxSpan int) error {
	previousEnd := 0
	for _, span := range spans {
		if span.Start < previousEnd || span.Start < 0 || span.End <= span.Start || span.End >= sequenceLength {
			return fmt.Errorf("invalid bidirectional token span [%d,%d) for sequence length %d", span.Start, span.End, sequenceLength)
		}
		if span.End-span.Start > maxSpan {
			return fmt.Errorf("bidirectional token span length %d exceeds prefill limit %d", span.End-span.Start, maxSpan)
		}
		previousEnd = span.End
	}
	return nil
}

func nextPrefillChunk(position, available, normalChunk int, spans []batch.TokenSpan) (int, error) {
	n := min(available, normalChunk)
	for _, span := range spans {
		switch {
		case position < span.Start:
			return min(n, span.Start-position), nil
		case position == span.Start:
			spanLength := span.End - span.Start
			if spanLength > available {
				return 0, fmt.Errorf("bidirectional token span [%d,%d) cannot be processed as one prefill chunk", span.Start, span.End)
			}
			return spanLength, nil
		case position < span.End:
			return 0, fmt.Errorf("prefill position %d starts inside bidirectional token span [%d,%d)", position, span.Start, span.End)
		}
	}
	return n, nil
}

// A decoder produces each run of tokens to emit, owning its own dispatch and
// synchronization; the decode loop owns the budget, emission, and
// cancellation. next may return none while its first tokens are in flight.
type decoder interface {
	next(remaining int) ([]sampler.Result, error)

	// drain ends production, returning any results sampled but never
	// delivered through next and the position the next forward would have
	// taken; the decoder remains closeable.
	drain() ([]sampler.Result, int)

	close()
}

// decode drives either decoder and owns where generation stops — at an EOS
// or the NumPredict budget. Every produced token is recorded so the caches
// never rest ahead of session.outputs; tokens past the stop are recorded but
// not streamed or counted.
func (r *Runner) decode(ctx context.Context, request Request, session *cacheSession, d decoder, promptEval time.Duration) error {
	// A sampled-but-undelivered result is still a produced token; record it.
	defer func() {
		results, _ := d.drain()
		for _, res := range results {
			session.outputs = append(session.outputs, int32(res.Token.Int()))
		}
	}()

	detok := detokenizer{
		tokenizer:       r.Tokenizer,
		wantLogprobs:    request.SamplerOpts.Logprobs,
		wantTopLogprobs: request.SamplerOpts.TopLogprobs,
	}

	final := CompletionResponse{Done: true, PromptEvalCount: len(request.Tokens), DoneReason: 1}
	final.PromptEvalDuration = promptEval
	now := time.Now()

	// Release MLX's cached free buffers every clearCacheInterval tokens so the
	// allocator's pool does not grow unbounded over a long generation.
	const clearCacheInterval = 256

	generated := 0
	for generated < request.Options.NumPredict {
		if err := ctx.Err(); err != nil {
			return err
		}

		results, err := d.next(request.Options.NumPredict - generated)
		if err != nil {
			return err
		}

		// Record the whole run before streaming any of it: a cancelled
		// stream returns early and must not leave the caches ahead of
		// session.outputs.
		done := false
		stream := len(results)
		for i, res := range results {
			// Int evaluates the array before reading it; a raw data read
			// on a lazy array races its evaluation and returns garbage.
			id := int32(res.Token.Int())
			session.outputs = append(session.outputs, id)
			if done {
				continue
			}
			if r.Tokenizer.IsEOS(id) {
				final.DoneReason = 0
				done = true
				stream = i
				continue
			}
			generated++
			if generated >= request.Options.NumPredict {
				done = true
				stream = i + 1
			}
		}

		for _, res := range results[:stream] {
			resp, ok := detok.detokenize(res)
			if !ok {
				continue
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case request.Responses <- resp:
			}
		}

		if done {
			break
		}

		if generated%clearCacheInterval == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalCount = generated
	final.EvalDuration = time.Since(now)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

// pipelinedDecoder decodes one token per call, one call ahead of emission:
// the next token's chain is dispatched before the returned one is
// synchronized, so the device runs ahead of host emission.
type pipelinedDecoder struct {
	r *Runner
	// spec, when non-nil, receives every forwarded token and settles its
	// drafter at close, keeping a non-drafting session's draft KV level.
	spec     *speculationSession
	caches   []cache.Cache
	position int
	sample   sampler.Result // in flight: sampled, not yet forwarded
}

func (r *Runner) pipelinedDecoder(spec *speculationSession, caches []cache.Cache, seed *mlx.Array, position int) *pipelinedDecoder {
	t := &pipelinedDecoder{r: r, spec: spec, caches: caches, position: position}
	t.sample = t.dispatch(seed)
	return t
}

// dispatch builds one forward-and-sample chain without reading the token's
// value, so it is in flight before the previous token is synchronized.
func (t *pipelinedDecoder) dispatch(token *mlx.Array) sampler.Result {
	r := t.r
	hidden := r.Model.Forward(&batch.Batch{
		InputIDs:     token,
		SeqOffsets:   []int32{int32(t.position)},
		SeqQueryLens: []int32{int32(token.Dim(1))},
	}, t.caches)
	t.spec.committed(token, hidden, t.position)
	t.position += token.Dim(1)
	logits := r.Model.Unembed(hidden)
	next := r.Sampler.Sample([]int{pipelineSlot}, logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1))
	mlx.Pin(next.Arrays()...)
	mlx.Sweep()
	mlx.AsyncEval(next.Arrays()...)
	return next
}

func (t *pipelinedDecoder) next(int) ([]sampler.Result, error) {
	out := t.sample
	t.sample = t.dispatch(out.Token.ExpandDims(-1))
	mlx.Unpin(out.Arrays()...)
	return []sampler.Result{out}, nil
}

// drain ends production: it returns the in-flight sample (sampled but never
// forwarded) and the position its forward would have taken. The decoder
// keeps the sample for close.
func (t *pipelinedDecoder) drain() ([]sampler.Result, int) {
	return []sampler.Result{t.sample}, t.position
}

func (t *pipelinedDecoder) close() {
	// The in-flight sample's forward was never dispatched; its report settles
	// the drafter level with the caches' resting offset.
	t.spec.settle(t.sample.Token)
	mlx.Unpin(t.sample.Arrays()...)
}

// detokenizer serializes sampled tokens into response chunks, holding bytes
// whose UTF-8 sequence hasn't completed yet and the logprobs that belong
// with those bytes so Content and Logprobs stay aligned when a chunk does
// flush.
type detokenizer struct {
	tokenizer       *tokenizer.Tokenizer
	buf             bytes.Buffer
	logprobs        []llm.Logprob
	wantLogprobs    bool
	wantTopLogprobs int
}

func (d *detokenizer) detokenize(res sampler.Result) (CompletionResponse, bool) {
	output := int32(res.Token.Int())
	d.buf.WriteString(d.tokenizer.Decode([]int32{output}))
	d.logprobs = append(d.logprobs, buildLogprob(res, d.wantLogprobs, d.wantTopLogprobs, d.tokenizer.Decode)...)

	content := flushValidUTF8Prefix(&d.buf)
	if content == "" {
		return CompletionResponse{}, false
	}
	resp := CompletionResponse{Content: content, Logprobs: d.logprobs}
	d.logprobs = nil
	return resp, true
}

// buildLogprob converts the sampler's logprob tensors into the wire-format
// llm.Logprob entries the caller wants. The sampler populates its logprob
// tensors whenever any registered slot requested them, so the caller must
// gate emission on its own request config (wantLogprobs / wantTopLogprobs)
// rather than on whether the tensors happen to be non-nil.
func buildLogprob(sample sampler.Result, wantLogprobs bool, wantTopLogprobs int, decode func([]int32) string) []llm.Logprob {
	if !wantLogprobs || sample.Logprob == nil {
		return nil
	}
	tok := func(id int32) string { return decode([]int32{id}) }

	out := llm.Logprob{
		TokenLogprob: llm.TokenLogprob{
			Token:   tok(int32(sample.Token.Int())),
			Logprob: float64(sample.Logprob.Floats()[0]),
		},
	}

	if wantTopLogprobs > 0 && sample.TopTokens != nil {
		ids := sample.TopTokens.Ints()
		vals := sample.TopLogprobs.Floats()
		pairs := make([]llm.TokenLogprob, len(ids))
		for i, id := range ids {
			pairs[i] = llm.TokenLogprob{
				Token:   tok(int32(id)),
				Logprob: float64(vals[i]),
			}
		}
		// The sampler emits the top maxK across registered slots via
		// Argpartition, which leaves entries unsorted.
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].Logprob > pairs[j].Logprob
		})
		if wantTopLogprobs < len(pairs) {
			pairs = pairs[:wantTopLogprobs]
		}
		out.TopLogprobs = pairs
	}
	return []llm.Logprob{out}
}
