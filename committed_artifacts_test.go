package blisschemas_test

import (
	"os"
	"path/filepath"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/internal/validate"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
)

// The catalog and registry as they stand on disk must load AND validate through the
// published entry points. This is the check that matters: a schema that accepts
// hand-written examples but not the committed artifacts is not a schema for those
// artifacts.
//
// Loading and validating are separate phases, mirroring the package's own split —
// decodeStrict checks document shape, Validate checks vocabulary and arithmetic. A test
// that only loaded would miss every vocabulary error, which is the larger class.
//
// The paths are absolute because these repositories are siblings rather than vendored
// dependencies. Each test skips when its repository is absent, so the suite still runs
// in a checkout that has only this one.
const (
	catalogRoot  = "/Users/sri/Documents/Projects/blis-catalog"
	registryRoot = "/Users/sri/Documents/Projects/blis-registry"
)

// yamlFiles lists the YAML documents in a directory, skipping the test if the
// directory is absent.
func yamlFiles(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Skipf("not present: %s", dir)
	}
	var out []string
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".yaml" {
			out = append(out, filepath.Join(dir, e.Name()))
		}
	}
	return out
}

// report fails the test on every error and counts every warning.
//
// The split is the point. An error means the document cannot be used as it stands. A
// warning means a reader should look — several pre-existing sets assert `measured`
// without citing a source — and failing on those would make this test unable to
// distinguish a broken artifact from a known, accepted weakness.
func report(t *testing.T, label string, rep schemas.Report) (warnings int) {
	t.Helper()
	for _, p := range rep.Problems() {
		if p.Severity == validate.SeverityError {
			t.Errorf("%s: %s", label, p.String())
			continue
		}
		warnings++
		t.Logf("%s: %s", label, p.String())
	}
	return warnings
}

func TestEveryCommittedCoefficientSetLoadsAndValidates(t *testing.T) {
	paths := yamlFiles(t, filepath.Join(registryRoot, "coefficients"))
	coefficients, warnings := 0, 0
	for _, path := range paths {
		name := filepath.Base(path)
		set, err := schemas.LoadCoefficientSet(path)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if len(set.Coefficients) == 0 {
			t.Errorf("%s: loaded but holds no coefficients", name)
		}
		for _, c := range set.Coefficients {
			// The nested-entry YAML form puts the name in the key, so a loader bug
			// yields entries that are structurally fine and anonymous.
			if c.Name == "" {
				t.Errorf("%s: an entry loaded with an empty name", name)
			}
		}
		coefficients += len(set.Coefficients)
		warnings += report(t, name, schemas.Validate(schemas.Bundle{
			Coefficients: []*coefficient.Set{set},
		}))
	}
	t.Logf("loaded and validated %d coefficients across %d sets", coefficients,
		len(paths))
	// The registry holds over 300 entries, most of them generated per SKU. A sharp
	// drop means a set stopped loading rather than that entries were removed.
	if coefficients < 300 {
		t.Errorf("only %d coefficients loaded; the registry holds over 300",
			coefficients)
	}
	// Every warning today comes from trained-physics.yaml, a pre-existing set of
	// BLIS's own fitted constants that asserts `measured` without citing a source.
	// None of the sets this project generates raises one, and pinning the count means
	// a new unsourced claim in any set fails this test rather than blending in.
	const knownWarnings = 11
	if warnings != knownWarnings {
		t.Errorf("registry raises %d validation warnings, expected %d; if a set "+
			"gained or lost an unsourced claim, change this count deliberately",
			warnings, knownWarnings)
	}
}

func TestEveryCommittedChipLoadsAndValidates(t *testing.T) {
	for _, path := range yamlFiles(t, filepath.Join(catalogRoot, "hardware")) {
		name := filepath.Base(path)
		chip, err := schemas.LoadChip(path)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		// Both counts are read by the cost model: the Triton-fetch derate is a
		// fraction of SMCount, and whether a collective group leaves the node is
		// decided against GPUsPerNode. A chip missing either cannot price an offload
		// run or a multi-node one, so a zero here is a defect and not a default.
		if chip.SMCount <= 0 {
			t.Errorf("%s: SMCount did not load (got %d)", name, chip.SMCount)
		}
		if chip.GPUsPerNode <= 0 {
			t.Errorf("%s: GPUsPerNode did not load (got %d)", name, chip.GPUsPerNode)
		}
		if w := report(t, name, schemas.Validate(schemas.Bundle{Chip: chip})); w > 0 {
			t.Errorf("%s: a chip is a declared fact and should raise no warning", name)
		}
	}
}

func TestEveryDerivedModelGraphLoadsAndValidates(t *testing.T) {
	models, err := os.ReadDir(filepath.Join(catalogRoot, "models"))
	if err != nil {
		t.Skipf("not present: %s", catalogRoot)
	}
	found := 0
	for _, m := range models {
		path := filepath.Join(catalogRoot, "models", m.Name(), "graph.yaml")
		if _, err := os.Stat(path); err != nil {
			continue
		}
		g, err := schemas.LoadModelGraph(path)
		if err != nil {
			t.Errorf("%s: %v", m.Name(), err)
			continue
		}
		found++
		report(t, m.Name(), schemas.Validate(schemas.Bundle{Model: g}))
	}
	t.Logf("validated %d derived model graphs", found)
	// The graphs are generated, so this asserts that the generator and the schema
	// still agree across the whole catalog rather than on one example.
	if found < 24 {
		t.Errorf("found %d graphs; the catalog holds at least 24", found)
	}
}
