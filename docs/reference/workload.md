# Workload

<p class="lede">Traffic comes at two fidelities. A <strong>shape</strong> describes it as
distributions and lives in the catalog. A <strong>trace</strong> is a captured file of real
requests. A scenario's <code>workload</code> key binds exactly one of the two.</p>

## Binding

The value of a scenario's `workload` key.

```yaml
workload:
  shape: chatbot
```

<!-- fields: workload.Binding -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `shape` | string | one of the two | A workload shape under `workloads/` in the catalog. |
| `trace` | mapping | one of the two | A reference to a captured trace, below. |

A binding with both keys, or with neither, is invalid. To pose a question with no traffic,
omit `workload` from the scenario instead.

## Trace

```yaml
--8<-- "docs/examples/trace-binding.yaml:workload"
```

The rows of a trace can number in the millions, so they stay in the file that `data`
names and never appear in the scenario. The scenario holds the path, an optional digest
and row count, and the small header that says how to read the file.

<!-- fields: workload.TraceRef -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `data` | string | yes | Path to the per-request data CSV. |
| `sha256` | string | no | Hex digest of the data file, 64 characters. |
| `rows` | integer | no | Expected number of rows in the data file. Not negative. |
| `header` | mapping | yes | The trace header, below. |

### `header`

<!-- fields: workload.TraceHeader -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `trace_version` | integer | yes | The trace format version the data file was written in. At least 1. |
| `time_unit` | string | yes | Unit of the file's timestamp columns. One of the [time units](vocabularies.md#time-units). |
| `mode` | string | yes | How the trace was produced. One of the [trace modes](vocabularies.md#trace-modes). |
| `workload_seed` | integer | no | The workload generator's seed, when one was recorded. A seed of 0 is a real seed, distinct from absence. |
| `server` | mapping | no | Configuration of the server the trace came from, below. |
| `goodput_slo_targets` | mapping | no | Latency targets per SLO class, keyed by class name, below. Class names are free. |

### `header.server`

Every key is optional. The server is the one that produced the trace, which need not be
the deployment being simulated.

<!-- fields: workload.TraceServer -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `type` | string | no | The serving engine, such as `vllm`. |
| `model` | string | no | The model it served. |
| `tensor_parallel` | integer | no | Tensor-parallel width. Not negative. |
| `max_num_seqs` | integer | no | Sequence cap. Not negative. |
| `block_size` | integer | no | KV block size. Not negative. |
| `gpu_memory_utilization` | number | no | Fraction of GPU memory, in [0, 1]. Zero means unrecorded. |
| `max_model_len` | integer | no | Context length. Not negative. |

### `header.goodput_slo_targets.<class>`

Thresholds in milliseconds. Zero on a dimension leaves it unconstrained, but a class must
constrain at least one.

<!-- fields: workload.SLODimTargets -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `ttft_ms` | number | no | Time to first token. Not negative. |
| `itl_ms` | number | no | Inter-token latency. Not negative. |
| `e2e_ms` | number | no | End-to-end latency. Not negative. |

## Shape

A catalog file under `workloads/`, named by its file; a `name` key in it is rejected. This is
the catalog's `chatbot`:

```yaml title="workloads/chatbot.yaml"
--8<-- "testdata/workloads/chatbot.yaml"
```

<!-- fields: workload.Shape -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `prefix_tokens` | integer | no | Prefix length shared across requests. Zero means none. |
| `prompt` | mapping | yes | Prompt length distribution, below. |
| `output` | mapping | yes | Output length distribution, below. |

### `prompt` and `output`

<!-- fields: workload.Distribution -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `tokens` | integer | yes | Mean length in tokens. Positive. |
| `tokens_stdev` | integer | no | Standard deviation. |
| `tokens_min` | integer | no | Lower bound. Zero means unset. |
| `tokens_max` | integer | no | Upper bound. Zero means unset. |

A file may also carry `_comment` and `_comment_*` keys, at the top level and inside
`prompt` and `output`. They are discarded when the file is loaded.

## Checks

A binding is invalid if it sets both `shape` and `trace`, or neither. A trace reference is
invalid if

- `data` is empty, `rows` is negative, or `sha256` is present and is not 64 hex digits;
- `trace_version` is less than 1;
- `time_unit` or `mode` is empty or not in its vocabulary;
- a server count is negative, or `gpu_memory_utilization` is not finite or lies outside
  [0, 1];
- an SLO class name is empty, a target is negative or not finite, or a class sets no target.

A shape is invalid if

- `prefix_tokens` is negative, or longer than the mean prompt;
- a mean `tokens` is less than 1;
- `tokens_stdev` or `tokens_min` is negative;
- `tokens_min` exceeds `tokens_max`, or the mean lies outside [`tokens_min`, `tokens_max`]
  where those are set.

A shape built in Go must also have a name; one loaded from a file always does, since the
loader takes it from the file name.

Source: [`spec/workload`](https://github.com/inference-sim/blis-schemas/tree/main/spec/workload).
