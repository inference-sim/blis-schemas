# Vocabularies

<p class="lede">Every enumerated key in a document accepts only the values listed here; an
unrecognized value is an error. The kernel's own enumerations are listed at the end. The
lists are generated from the Go constants, so they are the values the code accepts.</p>

These vocabularies are fixed by the schema and do not change with an engine release.
Values that do, such as backend names and cache dtypes, are listed with the
[rules pack](rules.md) for each release.

## Coefficients

### Units

The unit of a coefficient's `value`.

--8<-- "docs/generated/enum/vocab.Unit.md"

### Methods

How a coefficient's value was obtained. This is the field that keeps a measurement
distinguishable from an estimate, so it is never inferred.

--8<-- "docs/generated/enum/vocab.Method.md"

`measured`
:   From a measurement. The entry should cite it; the schema does not require that.

`literature`
:   From a published result that was not independently reproduced.

`vendor_spec`
:   A datasheet figure: nominal, not achieved.

`copied`
:   Carried over from another scope, which `copied_from` names.

`assumed`
:   A placeholder, or an estimate reached by reasoning. A careful estimate is still assumed;
    the reasoning goes in `rationale`.

`not_charged`
:   Deliberately zero, so that "no cost" can be told apart from "cost unknown".

`measured`, `literature` and `vendor_spec` count as evidenced. That is a statement about
the basis of a number, not its quality: a vendor figure is evidenced but nominal.

### Scope keys

The dimensions along which a coefficient's validity is bounded. A coefficient's `scope`
uses these as its keys.

--8<-- "docs/generated/enum/vocab.ScopeKey.md"

### Source kinds

--8<-- "docs/generated/enum/vocab.SourceKind.md"

### Source roles

--8<-- "docs/generated/enum/vocab.SourceRole.md"

## Catalog facts

### Provenances

Where a chip's or fabric's figures come from. `vendor_spec` is a datasheet figure;
`derived` is one computed from datasheet figures by a step the datasheet does not state.

--8<-- "docs/generated/enum/vocab.Provenance.md"

## Model graphs

### Primitives

A node's `op`.

--8<-- "docs/generated/enum/model.Op.md"

### Dtypes

A graph's `weight_dtype`, and a node's `weight_dtype` and `state_dtype`. Only formats that
change a byte count or a peak rate appear. `nvfp4`, `mxfp4` and `int4` are three different
4-bit formats, not spellings of one.

--8<-- "docs/generated/enum/model.DType.md"

### Attention kinds

An `Attention` node's `kind`. `mla` and `sparse_mla` store one latent vector per token,
which is why their `n_kv` is 1. `sparse_mla` is attention by a sparse kernel over an
explicit set of positions, bounded by a top-k (`index_topk`), a compressed stream
(`compress_ratio`), a sliding window (`window`), or a combination: DeepSeek-V4.1's
window-only layers run the same sparse FlashMLA kernel as its compressed ones.

--8<-- "docs/generated/enum/model.AttentionKind.md"

### Recurrent kinds

A `RecurrentUpdate` node's `recurrent_kind`.

--8<-- "docs/generated/enum/model.RecurrentKind.md"

### Emit conditions

A node's `emit`. `""`, the empty value, means the node always exists, and is what an
omitted `emit` reads as.

--8<-- "docs/generated/enum/model.EmitCondition.md"

`tensor_parallel`
:   Present when the tensor-parallel width exceeds 1.

`expert_parallel`
:   Present when the expert-parallel width exceeds 1.

`tensor_parallel_unless_sp_moe`
:   Present when the tensor-parallel width exceeds 1 and the engine has not made the MoE
    input sequence-parallel. Under sequence-parallel MoE the reduction is replaced by a
    reduce-scatter and all-gather pair.

`sequence_parallel`
:   Present when the model's own sequence parallelism is on, as the rules pack decides it
    for `global.architecture`. DeepSeek-V4.1 then shards the residual stream by token and
    all-gathers each attention input, then reduce-scatters its output.

`tensor_parallel_unless_sequence_parallel`
:   Present when the tensor-parallel width exceeds 1 and the model's own sequence
    parallelism is off: the all-reduces sequence parallelism replaces.

The last two may also sit on a non-collective node, for a node whose layout sequence
parallelism changes and which the graph therefore states once for each mode.

A model that decides its own sequence parallelism never consults the engine-wide rule
`tensor_parallel_unless_sp_moe` names, so its decoder-layer reductions, the MoE's
included, use `tensor_parallel_unless_sequence_parallel`. DeepSeek-V4.1 passes its own flag
to its MoE, and vLLM's FusedMoE then skips the final tensor-parallel all-reduce. The
0.31.0 rules pack reports the engine-wide condition on such a model as an error.

The condition holds on DeepSeek-V4's and V4.1's mega-MoE backend too. That backend requires
expert parallelism, so it runs without the model's sequence parallelism only at tensor
parallelism one or pipeline parallelism above one. There its routed experts combine over
the expert-parallel group with no tensor-parallel all-reduce, and its shared expert, which
fuses into the kernel only under sequence parallelism or tensor parallelism one, runs
tensor-parallel and all-reduces its own output. So the layer still reduces one
hidden-width tensor across tensor-parallel ranks, as on the default backend.

### Modalities

A graph's `modality`. An omitted `modality` means `text_only`.

--8<-- "docs/generated/enum/model.Modality.md"

### Tensor-parallel layouts

A node's `tp`. An omitted `tp` leaves the layout to the cost model's inference. A `GEMM`
accepts `column`, `row` and `replicated`; an `Elementwise` or `Attention` node accepts
`sharded` and `replicated`. No other primitive accepts one.

--8<-- "docs/generated/enum/model.TPLayout.md"

`column`
:   The output width `n` splits across ranks.

`row`
:   The input width `k` splits across ranks. The partial sums are reduced by a collective
    the graph states separately.

`sharded`
:   The work splits across ranks with the heads or columns it operates on.

`replicated`
:   Every rank holds the whole weight and does the whole node for every token it holds.

### Cache kinds

A cache's `kind`, which decides how it grows.

--8<-- "docs/generated/enum/model.CacheKind.md"

`kv`
:   Keys and values, or a latent: one entry per `tokens_per_entry` context tokens.

`index`
:   A sparse-attention indexer's key cache. It grows like `kv`.

`window`
:   A sliding-window ring, bounded by `window` tokens per request.

`state`
:   Fixed per-request state, `entries` entries for the request's lifetime.

### Gather residency

A `Gather` node's `residency`, and the engine setting that can change it, its
`residency_control`. An omitted `residency` means `device`; an omitted
`residency_control` means no setting moves the table.

--8<-- "docs/generated/enum/model.Residency.md"

--8<-- "docs/generated/enum/model.ResidencyControl.md"

`engram_cpu_offload`
:   vLLM's Engram `cpu_offload`. A deployment sets it with
    `engine.engram_config.cpu_offload`. The graph states the residency the engine gives the
    table when the deployment leaves it alone, which is a release fact: under 0.31.0 it is
    `host`, and that pack's `engram-residency-default` rule checks it.

### Gather shards

A `Gather` node's `shard`. An omitted `shard` means `replicated`.

--8<-- "docs/generated/enum/model.GatherShard.md"

`rows`
:   The rows split evenly across ranks, and each token's row is read on the rank that holds
    it. vLLM's vocabulary embedding works this way.

`heads`
:   The table is `rows_per_token` sub-tables, one per hash head, split across S shards.
    Each shard holds ceil(`rows_per_token` / S) of them and reads one row from each for
    every token. S is the tensor-parallel width, or that times the node's data-parallel
    replicas when they split the table rather than share one host copy, which the
    deployment's `engram_config` decides. DeepSeek-V4.1's Engram works this way.

### Context work

An `Elementwise` node's `context.per`, `context.phase` and `context.decode_extent`.

--8<-- "docs/generated/enum/model.ContextPer.md"

`query_token`
:   Once per scheduled token, each over the context.

`request`
:   Once per request per step, over its context, however many of its tokens are scheduled.

--8<-- "docs/generated/enum/model.Phase.md"

An omitted `phase` means every step; `prefill` means only steps that schedule the
request's prompt tokens.

--8<-- "docs/generated/enum/model.Extent.md"

An omitted `decode_extent` means the request's context; `max_model_len` means the engine's
`max_model_len`, for a kernel launched over a buffer padded to it.

### Drafting modes

A speculator's `drafting`. An omitted `drafting` means autoregressive.

--8<-- "docs/generated/enum/model.Drafting.md"

`parallel_block`
:   The draft stack runs once per step over `num_spec_tokens` positions per request, and
    each position predicts one draft token. DeepSeek-V4.1's DSpark drafts this way.

## Deployments

### Pool roles

--8<-- "docs/generated/enum/deployment.Role.md"

## Traces

### Trace modes

A trace header's `mode`: how the trace was produced.

--8<-- "docs/generated/enum/workload.Mode.md"

### Time units

A trace header's `time_unit`. Several members name the same unit, because the tools that
write traces spell it differently.

--8<-- "docs/generated/enum/workload.TimeUnit.md"

## Kernel

### Resources

What a `StepEstimate` attributes cost to.

--8<-- "docs/generated/enum/kernel.Resource.md"

### Directions

A tier transfer's direction.

--8<-- "docs/generated/enum/kernel.Direction.md"

Source: [`vocab`](https://github.com/inference-sim/blis-schemas/tree/main/vocab), and the
`spec/` and `kernel` packages.
