# Command line

<p class="lede">Two commands, one for each data repository. Each takes the root of a
checkout, validates every committed document in it, and exits with a code a CI step can
read.</p>

```sh
go run github.com/inference-sim/blis-schemas/cmd/validate-catalog@<version>  <catalog root>
go run github.com/inference-sim/blis-schemas/cmd/validate-registry@<version> <registry root>
```

Both write every problem to standard error. `validate-catalog` writes to standard output a
heading for each kind of file and one line for each artifact that validates: each file, or
each storage class in `devices/storage.yaml`. `validate-registry` writes one line per set to
standard output, `FAILED` for a set that does not validate. Each ends with a summary line:
`all N ... validate` on standard output, or `N of M ... failed` on standard error.
Warnings are printed and do not fail the run. Both run the field layer only: these
documents name no engine version, so no rules pack applies.

## validate-catalog

| Path | Validated as |
|---|---|
| `models/*/graph.yaml` | a [model graph](model.md), whose `name` must equal the directory. A model directory without one is skipped. |
| `models/*/config.json` | required in every model directory; must be a non-empty JSON object. |
| `models/*/model.yaml` | required in every model directory; a [model identity](model.md#model-identity), whose `name` must equal the directory. |
| `hardware/*.yaml` | a [chip](hardware.md#chip) |
| `networks/*.yaml` | a [fabric](hardware.md#fabric) |
| `devices/storage.yaml` | [storage classes](hardware.md#storage-classes). Optional. |
| `workloads/*.yaml` | a [workload shape](workload.md#shape) |

Only files ending in `.yaml` are read in the flat directories. A path is treated as a
catalog if it contains at least one of `models/`, `hardware/`, `networks/`, `workloads/` and
`devices/`.

## validate-registry

Every file under `coefficients/` whose extension is `.yaml` or `.yml`, in any letter case
and at any depth, is validated as a [coefficient set](coefficients.md). Symbolic links to directories are
not followed. A scenario can refer only to a set at the top level of `coefficients/` with a
`.yaml` extension, by its file name; a set elsewhere validates but cannot be named.

## Exit codes

| Code | validate-catalog | validate-registry |
|---|---|---|
| `0` | every artifact validated | every set validated |
| `1` | an artifact failed to load or validate | a set failed to load or validate |
| `2` | wrong arguments; the root does not exist; none of the five directories is present; or nothing was found to validate | wrong arguments; the root does not exist; `coefficients/` could not be read; or it holds no sets |

Source: [`cmd/validate-catalog`](https://github.com/inference-sim/blis-schemas/tree/main/cmd/validate-catalog),
[`cmd/validate-registry`](https://github.com/inference-sim/blis-schemas/tree/main/cmd/validate-registry).
