# blis-schemas

Go schemas and validators for the BLIS cost model: what a model is, what hardware
can do, what a coefficient claims, what a deployment specifies, what a measured run
recorded, and the interface a cost model implements.

This repository holds schemas and validation only. The data lives elsewhere —
[blis-catalog](https://github.com/inference-sim/blis-catalog) owns declared facts,
[blis-registry](https://github.com/inference-sim/blis-registry) owns learned
coefficients — and keeping the shapes separate from the contents is what lets either
side be validated in CI without the other being vendored in.

## The two layers

Validation is split in two, and the split is the main design decision here.

**Field validation** asks whether a document is structurally what it claims to be:
are the counts positive, does the arithmetic close, is the enumerated value one the
schema knows, is the graph acyclic. These answers are the same in every engine
release, so they live with the types in `spec/` and change only when a schema does.

**Rule validation** asks whether a document describes a deployment a *particular*
engine would run. Backend names appear and disappear between releases; defaults flip;
a feature gains a new disqualifying condition. Compiling those into the types would
mean the schema churns whenever an engine ships, and would silently misjudge a
scenario pinned to an older version. So a rules pack is data about one version,
registered under it in `rules/`.

A scenario names its engine version. Validation selects the matching pack. A version
with no pack is reported as unvalidated rather than assumed fine.

```go
rep := blisschemas.Validate(blisschemas.Bundle{
    Scenario: s, Deployment: d, Model: g, Chip: c, Coefficients: sets,
})
if !rep.OK() {
    for _, p := range rep.Problems() {
        log.Println(p) // field problems first, then rule problems
    }
}
```

`Report` keeps the layers apart, because they need different responses. A field
problem is a malformed document and always the author's to fix. A rule problem may
mean the document is right and its declared engine version is wrong, or that a rule
needs extending for a release it has not seen.

Rules do not run when field validation fails. A rule reading a malformed document
produces findings that are artifacts of the malformation, and a reader cannot tell
those from real ones.

## Layout

```
vocab/              closed vocabularies: units, methods, scope keys, provenance
spec/model/         a model as a DAG of cost primitives
spec/hardware/      chip, fabric, storage device — three schemas, not one
spec/coefficient/   coefficient sets, mirroring blis-registry's contract
spec/scenario/      the immutable problem: model, cluster inventory, workload + refs
spec/deployment/    the mutable config: pools, parallelism, engine knobs, offload, PD
spec/workload/      the "what traffic" binding: a distributional shape or a trace ref
spec/evaluation/    a measured run, for scoring a prediction against
kernel/             the interface a cost model implements
rules/              the version-scoped rule mechanism
rules/v0_29/        one release's rules and constants
internal/validate/  the accumulating, located problem list every validator shares
```

## A scenario's workload: a shape or a trace

A `Scenario` names its traffic in one `workload` slot, which is a sum type: it binds
**either** a distributional shape (by catalog name) **or** a reference to a concrete
captured trace. Exactly one arm is set; a scenario that poses a capacity question with no
traffic omits the slot entirely.

```yaml
# distributional arm — name a catalog workload shape
workload:
  shape: chatbot
```

```yaml
# concrete arm — reference an external TraceV2 by path; the bulk per-request rows
# (which can run to millions) stay in the file, never inlined in the document
workload:
  trace:
    data: traces/agentic-run.csv   # path to the bulk per-request data CSV
    sha256: <64-hex>               # optional integrity digest of that file
    rows: 1048576                  # optional expected row COUNT (not the rows)
    header:                        # the small, bounded trace header metadata
      trace_version: 3
      time_unit: microseconds      # one of a closed set (us/microseconds/ms/s/ns…)
      mode: real                   # real | generated | replayed
      seed: 0                      # workload RNG seed, when one was recorded
      server:                      # provenance of the server that produced the trace
        type: vllm
        tensor_parallel: 8
        gpu_memory_utilization: 0.9
      slo_targets:                 # per-class TTFT/ITL/E2E thresholds, in ms
        critical: {ttft_ms: 500, itl_ms: 50, e2e_ms: 30000}
```

A shape's prefix is one scalar shared length; a trace's prefix is a per-request tree in
the referenced rows, which is why the two are different fidelities of one question rather
than one field. The operating point — the load level (`rate` XOR `concurrency`) — is
deliberately **not** here: it is a run-level sweep axis recorded on a run's result, not a
property of the immutable problem.

## Why the model schema is not a vendor configuration

`blis-catalog` already commits each model's `config.json` verbatim, and that file
speaks one provider's dialect: `hidden_size` here, `d_model` there; a state-space
layer named by `mamba_expand` in one repository and `ssm_cfg` in another.
Re-spelling those fields would tie this schema to whichever vendor defined a model
first, and would still not say what the GPU does.

A `ModelGraph` says what the GPU does. Each node is one cost primitive carrying the
shape parameters that primitive prices. Two rules keep it honest:

- **A node exists when it launches GPU work**, not when a config field exists. An
  architecture-specific scalar that replaces a softmax scale rather than adding an
  operation changes no node.
- **Shapes are derived the way the engine derives them.** Head dimension is hidden
  size over head count, computed, because many configurations never state it.

Deriving a graph from a vendor config is a separate mechanical step. Its output is
committed with a digest of the source, so a drift between the two is detectable
rather than silent.

## Three schemas for hardware

A chip's peak rates are a datasheet fact about a die. A fabric is a property of the
cluster — the same chip runs behind InfiniBand in one deployment and RoCE in another.
Storage tiers are a third thing again.

The separation matters: cross-node collective cost turns on the ratio of
intra-node to inter-node bandwidth, and that ratio belongs to a *pairing* rather than
to either side. `hardware.IntraToInterRatio(chip, fabric)` is a function for that
reason, and `blis-catalog` makes the same split.

## Evolving this repository

- **A new engine release** is a new package under `rules/`, registered in
  `register.go`, with its own `source_test.go` and its own CI job pinned to the new
  tag. Nothing in `spec/` or `kernel/` changes. Editing an existing pack would
  retroactively change what a scenario pinned to that version means.
- **A new vocabulary term** is a deliberate edit to `vocab/`, and its test asserts
  the membership so the addition cannot be accidental.
- **A new primitive or field** changes `spec/`, which is the one case where a schema
  version bump is warranted.

## Testing

```sh
go test ./...                                    # schemas, validators, rules
VLLM_SOURCE=/path/to/vllm go test -run MatchSource ./rules/...
```

The second command is the one that matters for a rules pack. Its constants are a
hand transcription of one engine release's configuration surface, and hand
transcription is where the errors were: an early draft carried four of seventeen
cache dtypes, four of five graph modes, a backend enum for a setting the engine
declares as a boolean, and a field name the engine does not use. Three of those would
have rejected valid deployments.

`rules/v0_29/source_test.go` re-derives each constant from a checkout and fails on
any divergence in either direction — a value the pack accepts that the engine does
not, or the reverse. Without `VLLM_SOURCE` those tests skip and say which constants
went unchecked, because a skipped test reads like a passing one in a summary line.
CI runs them in a separate job against the tagged release.

Every validator has a negative suite. That is not thoroughness for its own sake: a
validator that cannot fail proves nothing, and a check that silently matches nothing
is indistinguishable from a passing one. `coverage_test.go` goes further and asserts
that the schemas express model architectures and deployment shapes taken from
published serving reports rather than written to fit the schema.
