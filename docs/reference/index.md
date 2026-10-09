# Reference

<p class="lede">One page per document kind, plus the kernel interface, the closed
vocabularies, the rules packs and the two commands. Each document page opens with an
example, then lists every key the document may contain, then every check its validator
makes.</p>

The key tables are tested against the Go types: a key the type declares and the table
omits, or a key the table lists and the type does not declare, fails the build. The
vocabularies and the rules pack listings are generated from the code. For anything finer
than these pages, read the source; each page links to it.

| Page | Documents | Package |
|---|---|---|
| [Scenario](scenario.md) | `Scenario` | [`spec/scenario`](https://github.com/inference-sim/blis-schemas/tree/main/spec/scenario) |
| [Deployment](deployment.md) | `Deployment` | [`spec/deployment`](https://github.com/inference-sim/blis-schemas/tree/main/spec/deployment) |
| [Workload](workload.md) | a scenario's `workload`, a catalog workload shape | [`spec/workload`](https://github.com/inference-sim/blis-schemas/tree/main/spec/workload) |
| [Model graph](model.md) | `ModelGraph`, a model's identity manifest | [`spec/model`](https://github.com/inference-sim/blis-schemas/tree/main/spec/model) |
| [Hardware](hardware.md) | chip, fabric, storage classes | [`spec/hardware`](https://github.com/inference-sim/blis-schemas/tree/main/spec/hardware) |
| [Coefficient set](coefficients.md) | `CoefficientSet` | [`spec/coefficient`](https://github.com/inference-sim/blis-schemas/tree/main/spec/coefficient) |
| [Evaluation run](evaluation.md) | `EvaluationRun` | [`spec/evaluation`](https://github.com/inference-sim/blis-schemas/tree/main/spec/evaluation) |
| [Kernel interface](kernel.md) | the `Kernel` a cost model implements | [`kernel`](https://github.com/inference-sim/blis-schemas/tree/main/kernel) |
| [Vocabularies](vocabularies.md) | units, methods, scope keys, dtypes, and the rest | [`vocab`](https://github.com/inference-sim/blis-schemas/tree/main/vocab) and `spec/` |
| [Rules packs](rules.md) | the checks for each engine version | [`rules`](https://github.com/inference-sim/blis-schemas/tree/main/rules) |
| [Command line](cli.md) | `validate-catalog`, `validate-registry` | [`cmd`](https://github.com/inference-sim/blis-schemas/tree/main/cmd) |

In the key tables, **Type** is the YAML type. *Required* means that if the key is absent,
the validator reports an error or the loader fails. A key marked *no* may be omitted, and
the table says what omitting it means.
