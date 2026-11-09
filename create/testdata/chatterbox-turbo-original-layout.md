# Original Turbo layout fixture

`chatterbox-turbo-original-layout.json.gz` contains metadata only, with no
learned weights. It records all 1,680 learned main-model tensor names and axis
permutations for `ResembleAI/chatterbox-turbo` revision
`749d1c1a46eb10492095d68fbcf55691ccf137cd`. Four derived positional/window/fade
tables are excluded.

The oracle is the audited original F32 import: 1,598 copy, transpose or LSTM
bias-sum outputs and 82 folded weight-normalization pairs. Its provenance
SHA-256 and original file SHA-256 values are embedded in the JSON. Each case
contains `[component, original_name, rank, output_name, axes]`. For a folded
pair, `original_name` ends in `.weight`; for combined LSTM biases it names
`bias_ih`. Pair completeness and arithmetic require separate checks.

The fixture was serialized as compact JSON, sorted by output name, then
compressed with gzip timestamp zero. Its SHA-256 is
`b8e771394795d5532d0c0b35fc900be69e1023db387dc9e675bf406e476f2f41`.
The complete layout test checks names, permutations, checkpoint identity,
case count and output collisions. It does not qualify numerical conversion,
derived tables, dependency tensors or end-to-end synthesis.
