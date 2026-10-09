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
which is why their `n_kv` is 1.

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

### Modalities

A graph's `modality`. An omitted `modality` means `text_only`.

--8<-- "docs/generated/enum/model.Modality.md"

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
