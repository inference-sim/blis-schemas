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
| `router` | mapping | no | The inputs of the llm-d router that routes the deployment's requests. Absent: the router's built-in configuration over every pod. |

## Pools

<!-- fields: deployment.Pool -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `role` | string | yes | `colocated`, `prefill` or `decode`. |
| `nodes` | integer | yes | Nodes in the pool, at least 1. |
| `parallel` | mapping | yes | The parallel layout of each engine in the pool. |
| `engine` | mapping | no | Engine settings for the pool. Each omitted setting takes the engine's default. |
| `labels` | mapping | no | Kubernetes labels on the pool's model-server pods, as its pod template sets them. The router selects and filters endpoints by them. `llm-d.ai/role` is derived from `role` (`colocated` -> `prefill-decode`, `prefill` -> `prefill`, `decode` -> `decode`); given here, it must have that value. |

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
| `data_parallel_load_balancing` | string | no | How the data-parallel ranks are exposed, which decides what a router chooses between. One of the [load-balancing modes](vocabularies.md#data-parallel-load-balancing). Omitted, `internal`. `hybrid` and `external` need `dp` > 1. The rules pack refuses `external` on a model without experts and `hybrid` without `dp_local`, and names the mode that runs when the engine resolves `hybrid` to another. `--api-server-count` is not part of it: it sets how many frontend processes share an endpoint, not which ranks a router can reach. |
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
| `long_prefill_token_threshold` | integer | no | The most tokens of one prefill scheduled in a step, so a long prompt cannot take the whole token budget from the decodes beside it. Not negative; omitted or 0 leaves prefills uncapped. The engine refuses one above `max_model_len`. 0.31.0 waives it for a request alone in the scheduler; 0.29.0 does not. |
| `max_model_len` | integer or `auto` | no | Context length. `auto` (or `-1`) asks the engine to fit the window to the KV memory it has left, which is a different request from omitting the key: omitted, the engine uses the model's whole window and refuses to start if it does not fit. |
| `gpu_memory_utilization` | number | no | Fraction of GPU memory the engine may use, in [0, 1]. Zero means the engine's default. |
| `scheduling_policy` | string | no | `fcfs` or `priority` in vLLM 0.29.0. |
| `async_scheduling` | boolean | no | Omitted means the engine decides through its own disqualifying conditions. |
| `frontend` | string | no | The API server that renders and tokenizes each prompt: one of the [frontends](vocabularies.md#frontends). Omitted, `python`. `rust` is vLLM's vllm-rs frontend, selected by the environment variable `VLLM_USE_RUST_FRONTEND=1`, not a flag. It changes how long a request waits for admission, per prompt token. |
| `cudagraph_mode` | string | no | CUDA graph capture strategy. |
| `max_cudagraph_capture_size` | integer | no | `compilation_config.max_cudagraph_capture_size`: the largest step, in tokens, a CUDA graph is captured for; a larger step runs eager. At least 1. Omitted, the release default, min(`max_num_seqs` x decode query length x 2, 512), or 1024 on SM100. The engine caps it at `max_num_batched_tokens` and truncates it to the largest size it generates. `cudagraph_mode: NONE`, or `--enforce-eager`, which has no field here, captures nothing. |
| `cudagraph_capture_sizes` | list of integers | no | `compilation_config.cudagraph_capture_sizes`: the exact step sizes to capture, each at least 1, not empty. Sizes above `max_num_batched_tokens` are dropped. With `max_cudagraph_capture_size` too, that must be the largest kept, or the engine refuses the pair. |
| `decoder_replay_cudagraph_capture_sizes` | list of integers | no | `compilation_config.decoder_replay_cudagraph_capture_sizes`, the decoder-replay graph sizes of a YOCO model such as DeepSeek-V4.1 (new in vLLM 0.32.0). Each at least 1. Omitted, generated from the sliding window. |
| `custom_ops` | list of strings | no | `compilation_config.custom_ops`: which ops run as custom kernels, `all`, `none`, or `+name` / `-name` entries. No entry may be empty. Omitted, the engine default. |
| `long_prefill_token_threshold_adaptive` | boolean | no | Floor the `long_prefill_token_threshold` cap at a fair share of the token budget (`max_num_batched_tokens` over the queued and running requests). Omitted means false. |
| `max_num_scheduled_tokens` | integer | no | The most tokens the scheduler issues in a step, where the model appends tokens to a batch, as speculative decoding does. Not negative; omitted or 0 uses `max_num_batched_tokens`. |
| `max_num_active_seqs` | integer | no | Caps the requests admitted to running without shrinking `max_num_seqs`. Not negative, and not above `max_num_seqs`; omitted or 0 means `max_num_seqs`. |
| `watermark` | number | no | Fraction of KV blocks kept free when admitting waiting or preempted requests, in [0, 1). Omitted means 0, the engine default. |
| `prefill_schedule_interval` | integer | no | Admit new prefills only once every N engine steps, aligned across data-parallel ranks. Not negative; omitted or 0 means the engine default, 1 (no throttling). |
| `stream_interval` | integer | no | Tokens the API server buffers before streaming them to the client. It changes when a client sees tokens, not what the scheduler does. Not negative; omitted or 0 means 1. |
| `disable_hybrid_kv_cache_manager` | boolean | no | Give every attention layer the same KV size even when they differ (full and sliding-window). Omitted lets the engine decide. |
| `kv_cache_memory_bytes` | integer | no | The KV cache's size per GPU in bytes, instead of deriving it from `gpu_memory_utilization`. Positive. |
| `prefix_match_unit` | integer | no | The finest token boundary a prefix-cache hit lands on; may be finer than the block size where every group's block size divides by it. Not negative; omitted or 0 means the block size. |
| `prefix_cache_retention_interval` | integer | no | Token interval between retained sliding-window and Mamba prefix-cache checkpoints; 0 keeps only semantic checkpoints. Not negative. |
| `kv_events` | mapping | no | KV-event publishing, below. |
| `max_loras` | integer | no | The most distinct LoRA adapters in one batch; a request whose adapter does not fit waits for a later step while those behind it run. Not negative; omitted or 0 means 1. |
| `max_cpu_loras` | integer | no | LoRA adapters kept loaded in CPU memory (an LRU set). Not below `max_loras`; omitted or 0 means `max_loras`. |

Weights and caches:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `quantization` | string | no | Weight format the engine serves, where it differs from the checkpoint's. Omitted means the checkpoint's own format. |
| `moe_backend` | string | no | The expert-kernel backend, `--moe-backend`. Omitted means `auto`. A mega-MoE backend is one of the conditions under which DeepSeek-V4.1 turns on its own sequence parallelism. |
| `cache_dtype` | string | no | KV cache format. |
| `enable_prefix_caching` | boolean | no | Omitted means true, vLLM's default. `false` describes a deployment without prefix caching. |
| `swa_bounded_replay` | boolean | no | DeepSeek-V4.1 SWA bounded replay (`--swa-bounded-replay` / `--no-swa-bounded-replay`). `true` keeps the sliding-window cache out of prefix caching and recomputes a hit's last sliding window; `false` lets the window cache take part in prefix caching (retained window tails). Omitted, the engine's resolution: on in vLLM 0.31.0 under the V2 model runner, without prefill context parallelism and not on ROCm. Only DeepSeek-V4.1 reads it. |
| `mamba_cache_dtype` | string | no | Format of a recurrent layer's state. |
| `mamba_ssm_cache_dtype` | string | no | Format of the SSM state alone, where it differs. |
| `mamba_cache_mode` | string | no | Whether recurrent state is fixed per sequence or grows with context. |
| `disable_cascade_attn` | boolean | no | Cascade attention reads a prefix shared by a batch once rather than once per request. Omitted means `true`, so it is disabled unless this is set to `false`. |
| `use_trtllm_attention` | boolean | no | `attention_config.use_trtllm_attention`: use or skip FlashInfer's TRT-LLM attention backend. Omitted auto-detects. |
| `mla_prefill_backend` | string | no | `attention_config.mla_prefill_backend`: `FLASH_ATTN`, `FLASHINFER` or `TRTLLM_RAGGED` in vLLM 0.32.0. Omitted selects automatically. |
| `linear_backend` | string | no | `kernel_config.linear_backend`, the GEMM backend of linear layers. Omitted is `auto`. |
| `env` | mapping | no | Engine environment variables the launch set that change what the engine runs, such as `VLLM_FLASHINFER_ALLREDUCE_BACKEND`, name to value. Each name must be an environment-variable name. A consumer refuses, or discloses as an assumption, every key it cannot honour. |

Speculation and experts:

<!-- fields: deployment.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `speculative` | mapping | no | Speculative decoding, below. |
| `eplb` | mapping | no | Expert-parallel load balancing, below. |
| `engram_config` | mapping | no | Where DeepSeek-V4.1's Engram tables live, below. Decides the residency of every model `Gather` whose `residency_control` is `engram_cpu_offload`. |

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
| `dbo_decode_token_threshold` | integer | no | Batch size in tokens at or above which a uniform-decode batch is split. Omitted takes the engine's default. With `enable_dbo` on, vLLM 0.31.0 refuses a stated value below the two microbatches, 0 or 1. Not negative. |
| `dbo_prefill_token_threshold` | integer | no | Batch size in tokens at or above which any other batch is split. As for the decode threshold. |

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
| `draft_model` | string | no | The catalog model whose graph is the drafter, where the drafter is a separate checkpoint, as an EAGLE3 head is (vLLM's `speculative_config.model`). Omitted, the target graph's own speculator. Not blank. |
| `kv_cache_dtype` | string | no | The drafter's KV cache format where it differs from the target's. |
| `draft_sample_method` | string | no | `greedy` or `probabilistic`. Recorded: acceptance is supplied, so it does not change which drafts are accepted. |
| `rejection_sample_method` | string | no | `standard`, `synthetic` or `block`. Recorded, as `draft_sample_method`. |
| `enable_adaptive_verification` | boolean | no | Size the verification budget from per-request confidence (DSpark only). Omitted means false. |

#### `engine.engram_config`

The cost-bearing keys of vLLM's `--engram-config`. All three are optional and take the engine's
default when omitted. The rules pack for 0.31.0 refuses the block on a model with no Engram
lookups.

<!-- fields: deployment.EngramConfig -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `cpu_offload` | boolean | no | Keep the tables in pinned host memory, read over the host link. Omitted means true. `false` puts them in GPU memory. |
| `dp_shared_memory` | boolean | no | Keep one host copy per node, shared by its data-parallel replicas. Omitted lets the engine decide: it enables it when `cpu_offload` is on, `use_thp` is off, `dp` exceeds 1 and elastic EP is off. Without sharing, the node's co-located replicas split each table between them instead. |
| `use_thp` | boolean | no | Back a per-rank host table with transparent huge pages. Refused with `dp_shared_memory`. |

#### `engine.kv_events`

vLLM's `--kv-events-config`: whether the engine publishes the changes to the set of KV blocks
its prefix cache can serve. A router's precise prefix-cache scorer consumes the stream.

<!-- fields: deployment.KVEvents -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `enable_kv_cache_events` | boolean | yes | Publish KV-cache events. |
| `publisher` | string | no | `null` or `zmq`. Omitted, the engine's resolution: `zmq` when events are enabled. |
| `topic` | string | no | The topic consumers subscribe to. |

## Offload

<!-- fields: deployment.Offload -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `tiers` | list | yes | At least one tier, below. |
| `eviction_policy` | string | no | KV offload cache policy. Omitted means `lru`. |
| `prefetch_depth` | integer | no | How many steps ahead a fetch is issued. |
| `connector` | string | no | The KV connector that implements offload. |
| `spec` | string | no | The offloading spec, such as `CPUOffloadingSpec`. |
| `connector_options` | mapping | no | The connector's own options, below. |

### `offload.connector_options`

The options vLLM's `kv_transfer_config.kv_connector_extra_config` gives the offload connector.
Each is optional and takes the connector's default when omitted; a misspelled one fails to load.

<!-- fields: deployment.ConnectorOptions -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `load_async` | boolean | no | Load a hit asynchronously (MooncakeStoreConnector; default true). |
| `lookup_async` | boolean | no | Look a prefix up without blocking the scheduler (MooncakeStoreConnector; default false). |
| `enable_offload` | boolean | no | Write KV to the store as well as read it (Mooncake; default false). |
| `enable_cross_layers_blocks` | boolean | no | Store one block across all layers rather than per layer (MooncakeStoreConnector). |
| `lazy_offload` | boolean | no | Defer a CPU write until the block is evicted (SimpleCPUOffloadConnector; default false). |
| `blocks_per_chunk` | integer | no | GPU blocks per offload chunk (OffloadingConnector). Not negative; omitted or 0 is the connector's default. |
| `kv_load_failure_policy` | string | no | What a failed KV load does: `recompute` the failed blocks, or `fail` the request (the default). |

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


## `router`

<!-- fields: deployment.Router -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `epp_flags` | list | no | The router's (EPP's) command-line flags, verbatim, one list entry per flag (`--name=value`). They select the pool with `--pool-name` or `--endpoint-selector`. A Helm chart's `epp.flags` map is rendered to this list (`name: value` becomes `--name=value`). |
| `endpoint_picker_config` | string | no | The `EndpointPickerConfig` file, as the router reads it with `--config-file`; a blank path is an error. Naming the configuration here and in `epp_flags` is an error. |
| `manifests` | list | no | Kubernetes manifest files. The router reads their `InferencePool`, `InferenceObjective` and `InferenceModelRewrite` objects; other kinds are ignored. Objectives and rewrites take effect only for an `InferencePool` (`--pool-name`); with a standalone pool (`--endpoint-selector`) the router does not reconcile them, so a consumer may refuse them there. |

Paths are relative to the directory of the file that holds the deployment. The block names
the router's own inputs and nothing else: scorers, weights, admission and flow control are
configured in those files, exactly as for a real deployment, so the same files configure the
real router and a simulated one. There is no deployment key for a routing decision. The router here is the Endpoint Picker. The
multimodal Coordinator pipeline is a separate llm-d component and is not named by this block; a
simulator that cannot host it refuses it at startup.

These checks are syntactic. A deployment that passes them is well formed, not necessarily
routable by a given consumer: a simulator refuses, at its startup and with the reason, router
flags, plugins or manifests it cannot run faithfully.

`pools[].labels` are a different kind of input from the three above. In a real deployment the
labels live in each pool's pod template; a deployment here has no pod spec, so the labels restate
them, so that an `InferencePool` selector or a label filter has the pods' labels to match.

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
- an engine count (`block_size`, `max_num_batched_tokens`, `max_num_seqs`,
  `cp_kv_cache_interleave_size`) is negative, or `max_model_len` is below `-1`. A
  `max_model_len` that is not a whole number, `auto`, or a number with a `k`/`m`/`g`/`t`
  or `K`/`M`/`G`/`T` suffix fails to load, and so does one that comes to zero or is too
  large to hold;
- `gpu_memory_utilization` is not finite or lies outside [0, 1];
- `max_cudagraph_capture_size` is stated below 1, or `cudagraph_capture_sizes` is stated
  empty or lists a size below 1;
- `disable_custom_all_reduce` is true while `allreduce_backend` is `custom`;
- `dbo` states a negative threshold, or an enabled `eplb` a negative
  `num_redundant_experts`;
- `speculative` is present with an empty `method` or with `num_spec_tokens` below 1;
- `max_num_scheduled_tokens`, `max_num_active_seqs`, `prefill_schedule_interval`,
  `stream_interval`, `prefix_match_unit`, `max_loras` or `max_cpu_loras` is negative,
  `max_num_active_seqs` exceeds `max_num_seqs`, or `max_cpu_loras` is below `max_loras`;
- `watermark` is not finite or lies outside [0, 1), `kv_cache_memory_bytes` is stated below
  1, or `prefix_cache_retention_interval` is negative;
- `kv_events.publisher` is not `null` or `zmq`, a `custom_ops` entry is empty, a
  `decoder_replay_cudagraph_capture_sizes` entry is below 1, or an `env` name is not an
  environment-variable name;
- `speculative.draft_model` is blank, or `speculative.draft_sample_method` or
  `rejection_sample_method` is not one of vLLM's values;
- `offload` is present with no tiers, a tier is unnamed or named twice, a tier's `bytes`
  is below 1, or `prefetch_depth` is negative;
- `offload.connector_options.blocks_per_chunk` is negative, or its `kv_load_failure_policy`
  is not `recompute` or `fail`.

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
