package blisschemas_test

import (
	"path/filepath"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
)

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
		t.Logf("%-34s model=%-34s hw=%s tp=%d dp=%d ep=%d", s.Name, s.Model,
			s.Hardware, s.Pools[0].Parallel.TP, s.Pools[0].Parallel.DP,
			s.Pools[0].Parallel.ExpertParallelWidth())
	}
}
