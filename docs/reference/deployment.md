# Deployment

<p class="lede">The choices made against a scenario: how its cluster is divided into pools,
each pool's parallel layout and engine settings, the KV offload tiers, and how KV moves
from prefill to decode. A deployment is validated with its scenario, because most of what
makes a layout feasible depends on the cluster.</p>

```yaml
--8<-- "docs/examples/deployment-disaggregated.yaml"
```

## Keys

<!-- fields: deployment.Deployment -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `Deployment`. |
| `name` | string | yes | The deployment's identity. |
| `pools` | list | yes | At least one pool, below. |
| `offload` | mapping | no | KV offload tiers below HBM. |
| `pd_transfer` | mapping | if any pool is `prefill` | How KV moves from prefill to decode. |

## Pools

<!-- fields: deployment.Pool -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `role` | string | yes | `colocated`, `prefill` or `decode`. |
| `nodes` | integer | yes | Nodes in the pool, at least 1. |
| `parallel` | mapping | yes | The parallel layout of each engine in the pool. |
| `engine` | mapping | no | Engine settings for the pool. Each omitted setting takes the engine's default. |

A pool may run several engines with the same layout and settings. Its `nodes` is the
pool's whole extent; one engine may occupy only part of it.

Pools of role `prefill` and `decode` come together or not at all, and neither may appear
beside a `colocated` pool.

### `parallel`

<!-- fields: deployment.Parallelism -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `tp` | integer | yes | Tensor-parallel width, at least 1. |
| `pp` | integer | yes | Pipeline-parallel width, at least 1. |
| `dp` | integer | yes | Data-parallel width, at least 1. |
| `dp_local` | integer | no | Data-parallel replicas resident on one node. Must divide `gpus_per_node` and not exceed `dp`. Zero means the launch decides. |
| `pcp` | integer | no | Prefill-context-parallel width. Zero or 1 means off. |
| `dcp` | integer | no | Decode-context-parallel width. Reuses the tensor-parallel ranks; see below. Zero or 1 means off. |
| `enable_expert_parallel` | boolean | no | Shard MoE experts across ranks. Default false. |

There is no key for the expert-parallel width. It is derived: `tp × pcp × dp` when expert
parallelism is enabled, and 1 otherwise. A key for it would let a document state a width
no engine would build. [Parallel layouts](../design/layouts.md) explains this and the
checks below in more depth.

### `engine`

Every engine key is optional. Some are requests the engine may not honor:
`disable_custom_all_reduce`, `allreduce_backend`, `async_scheduling`,
`disable_cascade_attn` and the decode-context keys. A cost model reports what it resolved
in its [`Resolution`](kernel.md#resolution). Which values each string key accepts depends
on the engine release and is checked by its [rules pack](rules.md).

Communication:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `all2all_backend` | string | no | How a mixture-of-experts (MoE) layer sends tokens to their experts and back. |
| `disable_custom_all_reduce` | boolean | no | Disable the custom all-reduce GPU kernel, which runs on the streaming multiprocessors (SMs) rather than the network. Omitted means false, so the kernel is used where the layout allows. |
| `allreduce_backend` | string | no | `custom` or `nccl`, stated directly instead of through `disable_custom_all_reduce`. Must not contradict it. |
| `dbo` | mapping | no | Dual-batch overlap, below. |

Scheduling and memory:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `block_size` | integer | no | KV block size in tokens. |
| `max_num_batched_tokens` | integer | no | Token budget per step. |
| `max_num_seqs` | integer | no | Sequence cap per step. |
| `max_model_len` | integer | no | Context length. |
| `gpu_memory_utilization` | number | no | Fraction of GPU memory the engine may use, in [0, 1]. Zero means the engine's default. |
| `scheduling_policy` | string | no | `fcfs` or `priority` in vLLM 0.29.0. |
| `async_scheduling` | boolean | no | Omitted means the engine decides through its own disqualifying conditions. |
| `cudagraph_mode` | string | no | CUDA graph capture strategy. |

Weights and caches:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `quantization` | string | no | Weight format the engine serves, where it differs from the checkpoint's. Omitted means the checkpoint's own format. |
| `cache_dtype` | string | no | KV cache format. |
| `enable_prefix_caching` | boolean | no | Omitted means true, vLLM's default. `false` describes a deployment without prefix caching. |
| `mamba_cache_dtype` | string | no | Format of a recurrent layer's state. |
| `mamba_ssm_cache_dtype` | string | no | Format of the SSM state alone, where it differs. |
| `mamba_cache_mode` | string | no | Whether recurrent state is fixed per sequence or grows with context. |
| `disable_cascade_attn` | boolean | no | Cascade attention reads a prefix shared by a batch once rather than once per request. Omitted means `true`, so it is disabled unless this is set to `false`. |

Speculation and experts:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `speculative` | mapping | no | Speculative decoding, below. |
| `eplb` | mapping | no | Expert-parallel load balancing, below. |

Decode-context parallelism. These have no effect when `dcp` is 0 or 1:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `dcp_comm_backend` | string | no | The collectives a decode step runs: `ag_rs` (the default) or `a2a` in vLLM 0.29.0. |
| `dcp_q_replicate` | boolean | no | Replicate the query projection of multi-head latent attention (MLA) on every rank, to skip a query all-gather. |
| `cp_kv_cache_interleave_size` | integer | no | Consecutive tokens one rank holds before the next takes over. Zero means unstated, which is a different request from 1. |

#### `engine.dbo`

<!-- fields: deployment.DBO -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `enable_dbo` | boolean | no | Split a batch so one half's communication overlaps the other's compute. |
| `dbo_decode_token_threshold` | integer | no | Batch size in tokens at or above which a uniform-decode batch is split. |
| `dbo_prefill_token_threshold` | integer | no | Batch size in tokens at or above which any other batch is split. |

#### `engine.eplb`

<!-- fields: deployment.EPLB -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `enable_eplb` | boolean | no | Enable expert-parallel load balancing. |
| `num_redundant_experts` | integer | no | Extra expert replicas. With load balancing on, the expert-parallel width must divide the experts plus these. |
| `window_size` | integer | no | Load-balancing window. |
| `step_interval` | integer | no | Steps between rebalances. |

#### `engine.speculative`

<!-- fields: deployment.Speculative -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `method` | string | yes | The draft method. Which names are valid depends on the engine release. |
| `num_spec_tokens` | integer | yes | Draft tokens per step, at least 1. |

## Offload

<!-- fields: deployment.Offload -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `tiers` | list | yes | At least one tier, below. |
| `eviction_policy` | string | no | KV offload cache policy. Omitted means `lru`. |
| `prefetch_depth` | integer | no | How many steps ahead a fetch is issued. |
| `connector` | string | no | The KV connector that implements offload. |
| `spec` | string | no | The offloading spec, such as `CPUOffloadingSpec`. |

### `offload.tiers`

<!-- fields: deployment.Tier -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `tier` | string | yes | The storage class this tier uses. If the scenario's cluster lists `storage`, it must be one of those. |
| `bytes` | integer | yes | Capacity of the tier in bytes, at least 1. |

## PD transfer

<!-- fields: deployment.PDTransfer -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `connector` | string | no | The KV connector that moves KV from prefill to decode. |

There is no bandwidth key. Transfer bandwidth comes from the scenario's fabric and a
registry coefficient, so it cannot be set here.

## Checks

### On its own

A deployment is invalid if

- `kind` is not `Deployment`, `name` is empty, or `pools` is empty;
- a pool's `role` is not recognized, or its `nodes` is less than 1;
- `tp`, `pp` or `dp` is less than 1, `pcp` or `dcp` is negative, `dp_local` is negative,
  or `dp_local` exceeds `dp`;
- `dcp` does not partition the group it reuses. With `pcp` at most 1, `tp` must be
  divisible by `dcp`. With `pcp` above 1, `dcp` must be 1, `pcp` or `tp × pcp`;
- a `prefill` pool has no `decode` pool, a `decode` pool has no `prefill` pool, or a
  `colocated` pool appears beside either;
- a `prefill` pool is present and `pd_transfer` is absent;
- an engine count (`block_size`, `max_num_batched_tokens`, `max_num_seqs`, `max_model_len`,
  `cp_kv_cache_interleave_size`) is negative;
- `gpu_memory_utilization` is not finite or lies outside [0, 1];
- `disable_custom_all_reduce` is true while `allreduce_backend` is `custom`;
- an enabled `dbo` has a negative threshold, or an enabled `eplb` a negative
  `num_redundant_experts`;
- `speculative` is present with an empty `method` or with `num_spec_tokens` below 1;
- `offload` is present with no tiers, a tier is unnamed or named twice, a tier's `bytes`
  is below 1, or `prefetch_depth` is negative.

It draws a warning if expert parallelism is enabled but the derived width is 1, or if
`allreduce_backend` is `nccl` while `disable_custom_all_reduce` is explicitly `false`.

### Against its scenario's cluster

These run when the bundle holds both documents. A deployment without its scenario is
itself reported, because these checks would otherwise be skipped.

- The pools' `nodes` must sum to the cluster's `nodes`. A deployment lays out the whole
  cluster.
- A pool's `dp_local`, when set, must divide `gpus_per_node`.
- An engine needs one GPU per rank, `pp × tp × pcp × dp` in all, and they must fit in the
  pool's nodes. Decode-context and expert parallelism add no ranks.
- With `dp_local` set, the local replicas must fit on one node in the way the engine places
  them. The check refuses only layouts that no launch could place. In the rare case that
  its bounded search cannot decide, it reports a warning that the layout's fit is
  unverified.
- Each offload tier must be one of the cluster's `storage` classes, if the cluster lists any.

Engine-release checks, such as whether a backend name exists or whether prefill-context
parallelism may be combined with data parallelism, are in the [rules pack](rules.md).

Source: [`spec/deployment`](https://github.com/inference-sim/blis-schemas/tree/main/spec/deployment).
