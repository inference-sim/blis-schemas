package blisschemas_test

// The examples in this file are the Go snippets the documentation shows. They run under
// `go test`, and their Output comments are checked, so a snippet on the site cannot show a
// call that no longer compiles or a message the code no longer prints. The `--8<--` markers
// delimit the part each page includes.

import (
	"fmt"
	"os"
	"path/filepath"

	blisschemas "github.com/inference-sim/blis-schemas"
)

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// ExampleValidate validates a scenario and a deployment together with the catalog
// documents they name.
func ExampleValidate() {
	// --8<-- [start:validate]
	s := must(blisschemas.LoadScenario("docs/examples/scenario.yaml"))
	d := must(blisschemas.LoadDeployment("docs/examples/deployment.yaml"))
	g := must(blisschemas.LoadModelGraph("testdata/models/llama-3.1-8b-instruct/graph.yaml"))
	c := must(blisschemas.LoadChip("testdata/hardware/h100.yaml"))

	rep := blisschemas.Validate(blisschemas.Bundle{
		Scenario: s, Deployment: d, Model: g, Chip: c,
	})
	for _, p := range rep.Problems() {
		fmt.Println(p)
	}
	fmt.Println("ok:", rep.OK(), "rules applied:", rep.RulesApplied)
	// --8<-- [end:validate]

	// Output:
	// ok: true rules applied: 0.29.0
}

// ExampleValidate_problems shows what a failing bundle reports: every problem at once,
// each at the path of the field that is wrong.
func ExampleValidate_problems() {
	s := must(blisschemas.LoadScenario("docs/examples/scenario.yaml"))
	d := must(blisschemas.LoadDeployment("docs/examples/deployment.yaml"))
	// --8<-- [start:problems]
	pool := &d.Pools[0]
	pool.Parallel.TP = 8  // 8 x 4 data-parallel ranks needs 32 GPUs
	pool.Parallel.DCP = 3 // does not divide tp
	pool.Engine.GPUMemoryUtilization = 1.5

	rep := blisschemas.Validate(blisschemas.Bundle{Scenario: s, Deployment: d})
	for _, p := range rep.Problems() {
		fmt.Println(p)
	}
	// --8<-- [end:problems]

	// Output:
	// error: deployment.pools[0].engine.gpu_memory_utilization: must lie in [0, 1] (0 means unset), got 1.5
	// error: deployment.pools[0].parallel: needs 32 GPUs (pp 1 x tp 8 x pcp 1 x dp 4, one per rank) but the pool's 1 node(s) of 8 GPUs provide 8
	// error: deployment.pools[0].parallel.dcp: decode-context parallelism reuses the tensor-parallel ranks when prefill-context parallelism is off, so tp 8 must be divisible by dcp 3
}

// ExampleValidate_rules shows a rule problem: the documents are well formed, but the engine
// version they name would refuse them.
func ExampleValidate_rules() {
	s := must(blisschemas.LoadScenario("docs/examples/scenario.yaml"))
	d := must(blisschemas.LoadDeployment("docs/examples/deployment.yaml"))
	// --8<-- [start:rules]
	d.Pools[0].Engine.CacheDType = "fp8_e4m3fn" // not a name vLLM 0.29.0 accepts

	rep := blisschemas.Validate(blisschemas.Bundle{Scenario: s, Deployment: d})
	fmt.Println("field layer ok:", rep.Field.OK())
	for _, p := range rep.Rule.All() {
		fmt.Println(p)
	}
	// --8<-- [end:rules]

	// Output:
	// field layer ok: true
	// error [enum-values-known]: : pools[0].engine.cache_dtype: "fp8_e4m3fn" is not accepted by engine 0.29.0
}

// ExampleValidateAgainstCatalog resolves a scenario's by-name references against a catalog
// checkout and a registry checkout.
func ExampleValidateAgainstCatalog() {
	// A stand-in registry: resolution checks only that each named set's file exists.
	registry := must(os.MkdirTemp("", "registry"))
	defer os.RemoveAll(registry)
	must(0, os.MkdirAll(filepath.Join(registry, "coefficients"), 0o755))
	for _, name := range []string{"cost-model-primitives", "cost-model-collectives",
		"cost-model-host-overheads", "cost-model-attention", "cost-model-recurrent"} {
		must(0, os.WriteFile(filepath.Join(registry, "coefficients", name+".yaml"), nil, 0o644))
	}

	// --8<-- [start:resolve]
	s := must(blisschemas.LoadScenario("docs/examples/scenario.yaml"))
	s.Cluster.Hardware = "nvidia-h100" // the catalog calls it h100

	rep := blisschemas.ValidateAgainstCatalog(
		blisschemas.Bundle{Scenario: s}, "testdata", registry)
	for _, p := range rep.Problems() {
		fmt.Println(p)
	}
	// --8<-- [end:resolve]

	// Output:
	// error: references.hardware: "nvidia-h100": looked for hardware/nvidia-h100.yaml; the catalog has [a100-80 a100-sxm b200 b300 gb200-nvl72 h100 h200 l40s]
}
