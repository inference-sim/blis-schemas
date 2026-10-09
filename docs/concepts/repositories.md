# The BLIS repositories

<p class="lede">BLIS is five repositories, each owning one kind of thing. blis-schemas is
the one that holds no data and runs no simulation. It defines the documents the other four
exchange, and the checks those documents must pass.</p>

--8<-- "docs/figures/repositories.html"

| Repository | Owns | What it takes from blis-schemas |
|---|---|---|
| [blis-catalog](https://github.com/inference-sim/blis-catalog) | Declared facts: model configurations and graphs, chips, fabrics, storage classes, workload shapes | Its CI runs `validate-catalog` over every committed file |
| [blis-registry](https://github.com/inference-sim/blis-registry) | Coefficient sets: measured, cited or assumed values, each with its method, sources and scope | Its CI runs `validate-registry` over every committed set |
| [blis-latency-kernel](https://github.com/inference-sim/blis-latency-kernel) | The cost model: memory and step time for one pool of a deployment | Imports the document types and loaders, implements `kernel.Kernel`, and validates its inputs before it constructs a kernel |
| [inference-sim](https://github.com/inference-sim/inference-sim) | The discrete-event simulator: queues, batching, admission, preemption | In pull request #1851, calls a kernel each simulated step through the `kernel` package's types |
| **blis-schemas** | The shapes of all of the above, and the checks on them | — |

## Shapes apart from contents

The catalog and the registry contain no Go. They are YAML and JSON, edited by people who
read datasheets and fit curves. The kernel and the simulator are Go, edited by people who
write cost models and schedulers. If the schema for a chip lived in the kernel, every
change to the catalog would need a kernel change to stay readable, and the catalog could
not check its own files without the kernel present.

The shapes therefore live here. A catalog contributor checks a new chip file against this
module without cloning the kernel. A kernel author reads the same file through the same
type, so the two cannot disagree about what `MemoryGiB` means.

## What each repository does with it

**blis-catalog** runs `validate-catalog` in CI against its own checkout, pinned to a
release of this module. The command loads every model graph, model identity, vendor
`config.json`, chip, fabric, storage class and workload shape, and fails the pull request
if any is malformed. The catalog keeps its own derivation checks: that each committed
`graph.yaml` re-derives from its vendor `config.json`, and the deriver's test suite.
blis-schemas does not cover derivation.

**blis-registry** checks out a pinned release of this module in CI and runs
`validate-registry` over every set. Each set goes through the loader and validator a Go
consumer uses, so a set the kernel could not load fails on the pull request that
introduces it. The registry's own tests check something else: that each committed value
re-derives from public data.

**blis-latency-kernel** depends on blis-schemas and on nothing else in BLIS. It loads
documents with this module's loaders, implements the `Kernel` interface defined here, and
runs `blisschemas.Validate` over its inputs before it constructs a kernel. A field-layer
error stops construction; rule-layer problems do not.

**inference-sim** takes its latency model from the kernel in pull request
[#1851](https://github.com/inference-sim/inference-sim/pull/1851), which is open. In that
branch, the `sim/kernelmodel` package adapts a kernel to the simulator's latency interface
and imports this module's `kernel` and `spec/deployment` packages for the types that cross
the boundary. It computes no cost of its own. On inference-sim's main branch, the
simulator imports neither the kernel nor this module.

## Dependencies run one way

blis-schemas imports nothing from the other repositories. Its only dependency is
`gopkg.in/yaml.v3`. The kernel imports blis-schemas, and the simulator in #1851 imports
both. Nothing imports in the other direction.

## A pinned copy of the catalog

`testdata/` in this repository is a pinned copy of the catalog, made by hand. The tests
load and validate every file in it, so a change to a schema is tested against real catalog
files rather than ones written to fit. Because the copy is fixed, the catalog's `main`
branch can move without breaking this repository's build. The copy is refreshed in a
separate commit that can be reviewed on its own. It is test input, not a second source:
catalog data is edited only in the catalog.

The registry has no pinned copy yet. CI parses a live checkout of it instead.

## Consumers pin releases

Every consumer pins a release tag of this module, and moving to a newer one is a one-line
change the consumer makes when it is ready. A release that changes what a document may
contain is therefore adopted by each repository in its own time. The pins are in each
repository's CI workflow and `go.mod`.
