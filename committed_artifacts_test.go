package blisschemas_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/internal/validate"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/model"
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
// The artifacts are VENDORED under testdata/, not read from a sibling checkout.
//
// They used to be read from absolute paths in one developer's home directory, which
// meant every test here skipped everywhere else -- CI included -- and the only check
// that loads every ModelGraph through the published entry point ran nowhere.
//
// Vendoring rather than checking out the real repositories is the deliberate choice.
// blis-catalog and blis-registry move independently of this schema, so a test pinned
// to whatever their main branch holds today fails for reasons that have nothing to do
// with a schema change. The registry's unsourced-claim count demonstrated this
// concretely while these tests were being fixed: it is 11 against registry main and 0
// against a working tree that had dropped the set responsible, so a single expected
// value cannot be correct for both. A snapshot makes "what does this schema accept"
// a question with one answer per schema commit, and makes changing that answer a
// reviewable diff rather than a build that breaks on someone else's merge.
//
// testdata/refresh.sh re-copies them and records the source revisions in
// testdata/{catalog,registry}/REVISION.
const (
	catalogRoot  = "testdata/catalog"
	registryRoot = "testdata/registry"
)

// yamlFiles lists the YAML documents in a directory. Every directory it is called with
// is vendored in this repository, so an absent or empty one is a broken checkout rather
// than a reason to skip.
func yamlFiles(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("%s: %v (run testdata/refresh.sh)", dir, err)
	}
	var out []string
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".yaml" {
			out = append(out, filepath.Join(dir, e.Name()))
		}
	}
	if len(out) == 0 {
		t.Fatalf("%s holds no YAML documents (run testdata/refresh.sh)", dir)
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
	// Unsourced claims in the vendored snapshot. Warnings used to come entirely from
	// trained-physics.yaml, a set of BLIS's own fitted constants asserting `measured`
	// without citing a source; blis-registry dropped it, so the snapshot now holds
	// none and every remaining set cites its evidence.
	//
	// Pinned rather than removed, because the pin is the point: a new unsourced claim
	// must fail this test rather than blend in. Because the artifacts are vendored,
	// this number changes only when testdata/refresh.sh is run and the diff reviewed
	// -- it cannot go stale behind another repository's merge, which is what happened
	// when these tests read a sibling checkout.
	const knownWarnings = 0
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
		// SMCount is read by the cost model -- the Triton-fetch derate is a fraction
		// of it -- so a zero here is a defect and not a default.
		if chip.SMCount <= 0 {
			t.Errorf("%s: SMCount did not load (got %d)", name, chip.SMCount)
		}
		// GPUsPerNode is deliberately NOT required. It is a property of the deployment
		// rather than of the silicon -- the same part ships on baseboards of several
		// widths -- and the value the cost model reads comes from the scenario's
		// cluster block. The catalog removed it from every chip except GB200-NVL72,
		// where a tray boundary IS a hardware fact; this assertion went stale then.
		//
		// What remains a schema concern is the relationship: a rack tier is expressed
		// as a multiple of the node tier, so stating one without the other is a defect.
		if chip.GPUsPerRack > 0 && chip.GPUsPerNode <= 0 {
			t.Errorf("%s: GPUsPerRack %d with no GPUsPerNode; the rack tier is "+
				"expressed as a multiple of the node tier", name, chip.GPUsPerRack)
		}
		if w := report(t, name, schemas.Validate(schemas.Bundle{Chip: chip})); w > 0 {
			t.Errorf("%s: a chip is a declared fact and should raise no warning", name)
		}
	}
}

func TestEveryDerivedModelGraphLoadsAndValidates(t *testing.T) {
	models, err := os.ReadDir(filepath.Join(catalogRoot, "models"))
	if err != nil {
		t.Fatalf("%s/models: %v (run testdata/refresh.sh)", catalogRoot, err)
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

// TestVendoredFixturesRecordTheirProvenance keeps the snapshot honest.
//
// A vendored fixture's risk is that nobody knows how old it is. The REVISION files say
// which upstream commit each was taken from, so a reviewer can tell whether a schema
// change was tested against current data, and testdata/refresh.sh can reproduce it.
// Without them the snapshot is an undated copy and "the schema accepts the committed
// artifacts" slowly stops meaning anything.
func TestVendoredFixturesRecordTheirProvenance(t *testing.T) {
	for _, dir := range []string{catalogRoot, registryRoot} {
		path := filepath.Join(dir, "REVISION")
		b, err := os.ReadFile(path)
		if err != nil {
			t.Errorf("%s: %v (run testdata/refresh.sh)", path, err)
			continue
		}
		rev := strings.TrimSpace(string(b))
		if len(rev) != 40 {
			t.Errorf("%s: %q is not a full git revision", path, rev)
			continue
		}
		t.Logf("%s vendored from %s", dir, rev[:12])
	}
}

// TestTheVendoredCatalogCoversTheSchemasBreadth guards against a snapshot that has
// drifted into triviality: a handful of graphs would pass every test above while
// exercising almost none of the schema. The counts are lower bounds, so adding models
// upstream never breaks this -- only a refresh that LOSES coverage does.
func TestTheVendoredCatalogCoversTheSchemasBreadth(t *testing.T) {
	kinds := map[model.AttentionKind]int{}
	recurrent := map[model.RecurrentKind]int{}
	dtypes := map[model.DType]int{}
	speculators := 0

	models, err := os.ReadDir(filepath.Join(catalogRoot, "models"))
	if err != nil {
		t.Fatalf("%s/models: %v", catalogRoot, err)
	}
	for _, m := range models {
		g, err := schemas.LoadModelGraph(
			filepath.Join(catalogRoot, "models", m.Name(), "graph.yaml"))
		if err != nil {
			continue
		}
		dtypes[g.Global.WeightDType]++
		if g.Speculator != nil {
			speculators++
		}
		for _, lk := range g.LayerKinds {
			for _, n := range lk.Nodes {
				if n.AttentionKind != "" {
					kinds[n.AttentionKind]++
				}
				if n.RecurrentKind != "" {
					recurrent[n.RecurrentKind]++
				}
				if n.WeightDType != "" {
					dtypes[n.WeightDType]++
				}
			}
		}
	}

	// Every attention kind the schema defines must appear, or a change to one of them
	// is unexercised by this snapshot.
	for _, k := range []model.AttentionKind{
		model.AttentionGQA, model.AttentionMLA, model.AttentionSparseMLA,
		model.AttentionSWA,
	} {
		if kinds[k] == 0 {
			t.Errorf("no vendored graph uses attention kind %q", k)
		}
	}
	if len(recurrent) < 3 {
		t.Errorf("only %d recurrent kinds vendored (%v); the catalog ships mamba2, "+
			"gdn and kda", len(recurrent), recurrent)
	}
	if len(dtypes) < 5 {
		t.Errorf("only %d weight dtypes vendored (%v)", len(dtypes), dtypes)
	}
	if speculators < 10 {
		t.Errorf("only %d vendored graphs carry a speculator; the catalog has 12",
			speculators)
	}
	t.Logf("vendored coverage: %d attention kinds, %d recurrent kinds, %d dtypes, "+
		"%d speculators", len(kinds), len(recurrent), len(dtypes), speculators)
}
