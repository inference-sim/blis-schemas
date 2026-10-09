# Kernel interface

<p class="lede"><code>kernel.Kernel</code> is the interface a cost model implements and a
simulator calls. One value prices one pool of one deployment: how much memory a rank
holds, how long a forward pass over a batch takes, and what transfers and host work cost.
blis-latency-kernel is the implementation in use.</p>

Every method is a pure function of its arguments and the configuration the kernel was
constructed from: the same inputs give the same outputs, nothing is mutated, and calls are
safe from several goroutines at once. There is no `Init` method; a kernel comes from its
implementation's constructor, already resolved. Why the interface is drawn this way is the
subject of [Where the kernel stops](../design/kernel-boundary.md).

## Methods

| Method | Returns |
|---|---|
| **Memory** | |
| `FixedBytes() MemoryBreakdown` | Per-rank memory that does not depend on the requests: weights, activation peak, captured CUDA graphs, redundant experts, communication buffers. |
| `SequenceFixedBytes() int64` | Per-sequence memory that does not grow with length, such as recurrent state. Zero for a model with attention only. |
| `SequenceVariableBytes(tokens int) int64` | Per-sequence memory for a given token count, quantized to the engine's page size. |
| **Step time** | |
| `StepTime(Batch) StepEstimate` | The time of one forward pass over a batch. |
| **Transfers** | |
| `TierTime(tier, dir, bytes, queueDepth) time.Duration` | One transfer to or from an offload tier. |
| `PDTransferTime(tokens, from, to Placement) time.Duration` | Moving one request's KV between pools. |
| **Host work** | |
| `AdmissionOverhead(promptTokens int) time.Duration` | CPU work before a request can be scheduled, such as tokenization. Excludes queueing. |
| `OutputTokenOverhead() time.Duration` | Host work per emitted token, such as detokenization and streaming. |
| `CompletionOverhead() time.Duration` | Fixed work when a request finishes. Counts toward end-to-end latency, not time to first token. |
| **Provenance** | |
| `Provenance() []CoefficientOrigin` | Every coefficient the kernel used, with its set, method and scope. |
| `Resolved() Resolution` | What the kernel resolved, including requests it overrode. |
| `Deployment() deployment.Pool` | The pool it prices, as the document stated it. |

`Deployment()` returns the request and `Resolved()` the resolution. For the parallel
widths or the all-reduce backend, read `Resolved()`. For settings that resolution does not
change, such as `max_num_seqs` and `block_size`, read the pool. The returned pool is a
shallow copy; treat it as read-only.

## Step estimates

A step is priced as a sequence of stages, one per layer, since each layer needs the
previous layer's output. Within a stage, work on different resources can overlap.

`StepEstimate` reports two bounds and a diagnosis:

`Overlap`
:   Perfect overlap within each stage: the slowest resource in each stage, summed over the
    stages.

`NoOverlap`
:   No overlap anywhere: every resource's work, summed.

`Bottleneck`
:   The resource that did the most work over the step, which answers "what should be
    relieved". It is not the resource that set `Overlap`; since `Overlap` sums per-stage
    maxima, no single resource need account for it.

`PerResource`
:   Each resource's total, for a caller that needs to see why.

A configuration whose conclusion changes between the two bounds needs a measurement rather
than a prediction.

The resources are:

--8<-- "docs/generated/enum/kernel.Resource.md"

A collective appears under `sm` when the resolved backend is an SM-consuming kernel and
under a link otherwise. The resolved backend decides this, not the requested one.

## Batches

A `Batch` is what the simulator supplies for one step.

| Field | Meaning |
|---|---|
| `Reqs []ReqShape` | One entry per request in the batch. |
| `DecodeThreshold int` | Engine configuration that selects the attention kernel for each request. |
| `SMBudget int` | SMs available this step: the chip's count less any held by a concurrent offload fetch. |

Each `ReqShape` carries the counts an engine's own scheduler reads:

| Field | Meaning |
|---|---|
| `Scheduled` | Tokens scheduled this step. Under speculative decoding a decoding request schedules the draft length plus one, so a count above 1 does not imply prefill. |
| `Computed` | Tokens already computed, as of batch formation. |
| `PromptLen` | The prompt length. With `Computed`, it tells the last chunk of a prefill split across steps apart from a decode. |
| `CachedTokens` | The prefix-cache hit, already subtracted from `Scheduled`. A count, not a rate. |

`Batch.Tokens()` sums `Scheduled`. `Batch.UniformDecode(width)` reports whether every
request schedules exactly `width` tokens. Some engine behavior, such as when dual-batch
overlap engages, uses one threshold for such a batch and another for any other.

## Resolution

A deployment states requests. The kernel's constructor decides what runs, and
`Resolution` reports it.

| Field | Meaning |
|---|---|
| `TensorParallelWidth`, `DataParallelWidth`, `ExpertParallelWidth` | The resolved layout. The expert width is `tp × pcp × dp` with expert parallelism on, and 1 with it off. |
| `AllReduceBackend` | The all-reduce that will run, which may differ from the request. |
| `AsyncScheduling` | The resolved value of the tri-state request. |
| `CascadeAttention` | Whether cascade attention is available after every condition is applied. |
| `SequenceParallelMoE` | Whether the MoE input is made sequence-parallel, replacing an all-reduce with a reduce-scatter and all-gather. |
| `Overrides []Override` | Each request the layout could not honor: the field, what was requested, what was resolved, and why. |

A width of zero means unset, not "no parallelism". `TensorParallel()`, `DataParallel()`
and `ExpertParallel()` return the widths with a floor of one, for a reader that may see a
zero value.

## Other types

| Type | Meaning |
|---|---|
| `MemoryBreakdown` | `Weights`, `ActivationPeak`, `CUDAGraph`, `EPLBRedundant`, `CommBuffers`, in bytes; `Total()` sums them. |
| `Direction` | `to_tier` (eviction) or `from_tier` (fetch). Priced separately because flash reads and writes differ. |
| `Placement` | A rank's `Node`, `Rack` and `Pool` (the pool's role). Comparing two placements chooses the link a transfer uses. |
| `CoefficientOrigin` | One row of the provenance trail: entry `Name`, `Set`, `Method`, `Scope`. |

Source: [`kernel`](https://github.com/inference-sim/blis-schemas/tree/main/kernel).
