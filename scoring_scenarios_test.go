package blisschemas_test

import (
	"path/filepath"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
)

// TestEveryScoringScenarioLoadsAndValidates checks that the real scoring-scenario files
// load and pass immutable-problem (Scenario) validation. It is deliberately
// Scenario-scoped: those files describe the problem, and after the Scenario/Deployment
// split the tunable layout is a separate document this corpus does not carry. Deployment
// field-and-rule validation is covered against synthetic layouts in
// TestCorpusDeploymentsAreExpressible (coverage_test.go), the full-bundle TestValidBundlePasses
// (schemas_test.go), and the spec/deployment and rules/v0_29 unit tests — so no deployment
// coverage is silently lost here, it simply lives where deployment documents exist.
func TestEveryScoringScenarioLoadsAndValidates(t *testing.T) {
	dir := "/Users/sri/Documents/Projects/blis-latency-kernel/testdata"
	paths, _ := filepath.Glob(filepath.Join(dir, "*.yaml"))
	if len(paths) == 0 {
		t.Skip("no scenarios")
	}
	for _, p := range paths {
		s, err := schemas.LoadScenario(p)
		if err != nil {
			t.Errorf("%s: %v", filepath.Base(p), err)
			continue
		}
		for _, pr := range schemas.Validate(schemas.Bundle{Scenario: s}).Problems() {
			t.Errorf("%s: %s", filepath.Base(p), pr.String())
		}
		t.Logf("%-34s model=%-34s hw=%s nodes=%d gpus_per_node=%d", s.Name, s.Model,
			s.Cluster.Hardware, s.Cluster.Nodes, s.Cluster.GPUsPerNode)
	}
}
