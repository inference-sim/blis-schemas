package blisschemas

// The documentation under docs/ makes two kinds of claim this file checks. Each example
// document must load and validate without a single finding, so a reader who copies one
// starts from something correct. And each field table must list exactly the YAML keys its
// Go type declares — no key missing, none invented — so a field added to a type cannot
// ship undocumented and a renamed one cannot linger on the site.

import (
	"bufio"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/evaluation"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
	"github.com/inference-sim/blis-schemas/spec/workload"
)

// fixture returns a loaded document, panicking on a load error: every path here is committed,
// so a failure is a broken checkout rather than a finding.
func fixture[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func requireSilent(t *testing.T, name string, rep Report) {
	t.Helper()
	for _, p := range rep.Problems() {
		t.Errorf("%s: %s", name, p)
	}
}

func TestDocExamplesValidate(t *testing.T) {
	ex := func(name string) string { return filepath.Join("docs", "examples", name) }

	t.Run("colocated", func(t *testing.T) {
		requireSilent(t, "colocated", Validate(Bundle{
			Scenario:   fixture(LoadScenario(ex("scenario.yaml"))),
			Deployment: fixture(LoadDeployment(ex("deployment.yaml"))),
			Model:      fixture(LoadModelGraph("testdata/models/llama-3.1-8b-instruct/graph.yaml")),
			Chip:       fixture(LoadChip("testdata/hardware/h100.yaml")),
		}))
	})
	t.Run("disaggregated", func(t *testing.T) {
		requireSilent(t, "disaggregated", Validate(Bundle{
			Scenario:   fixture(LoadScenario(ex("scenario-disaggregated.yaml"))),
			Deployment: fixture(LoadDeployment(ex("deployment-disaggregated.yaml"))),
			Model:      fixture(LoadModelGraph("testdata/models/deepseek-v3/graph.yaml")),
			Chip:       fixture(LoadChip("testdata/hardware/h200.yaml")),
			Fabric:     fixture(LoadFabric("testdata/networks/ib-400g.yaml")),
			Devices:    fixture(LoadStorageDevices("testdata/devices/storage.yaml")),
		}))
	})
	t.Run("coefficient set", func(t *testing.T) {
		requireSilent(t, "coefficient set", Validate(Bundle{
			Coefficients: []*coefficient.Set{fixture(LoadCoefficientSet(ex("coefficient-set.yaml")))},
		}))
	})
	t.Run("evaluation run", func(t *testing.T) {
		requireSilent(t, "evaluation run", Validate(Bundle{
			Evaluation: fixture(LoadEvaluationRun(ex("evaluation-run.yaml"))),
		}))
	})
	t.Run("trace", func(t *testing.T) {
		requireSilent(t, "trace", Validate(Bundle{
			Scenario: fixture(LoadScenario(ex("trace-binding.yaml"))),
		}))
	})
	// Every file in docs/examples/ must be one of the cases above, so an example added
	// later cannot be shown on a page without ever being validated.
	covered := map[string]bool{
		"scenario.yaml": true, "deployment.yaml": true,
		"scenario-disaggregated.yaml": true, "deployment-disaggregated.yaml": true,
		"coefficient-set.yaml": true, "evaluation-run.yaml": true, "trace-binding.yaml": true,
	}
	all, err := filepath.Glob(ex("*"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range all {
		if !covered[filepath.Base(path)] {
			t.Errorf("%s is not validated by TestDocExamplesValidate; add a case for it", path)
		}
	}

	// The by-name references the examples make must resolve against the vendored catalog,
	// or an example would name a chip or a workload the reader cannot find.
	for _, name := range []string{"scenario.yaml", "scenario-disaggregated.yaml", "trace-binding.yaml"} {
		s := fixture(LoadScenario(ex(name)))
		rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, "")
		for _, p := range rep.Problems() {
			if strings.HasPrefix(p.Path, "references.coefficients") {
				continue // the registry is not vendored
			}
			t.Errorf("%s: %s", name, p)
		}
	}
}

// documented maps each field-table marker to the type whose keys the table must list, and
// the keys deliberately left out of it. A key is left out only when the file never carries
// it: an entry's name is its map key, and a storage tier's name is its key in the file.
var documented = map[string]struct {
	typ  reflect.Type
	omit []string
}{
	"scenario.Scenario":      {typ: reflect.TypeOf(scenario.Scenario{})},
	"scenario.Cluster":       {typ: reflect.TypeOf(scenario.Cluster{})},
	"workload.Binding":       {typ: reflect.TypeOf(workload.Binding{})},
	"workload.TraceRef":      {typ: reflect.TypeOf(workload.TraceRef{})},
	"workload.TraceHeader":   {typ: reflect.TypeOf(workload.TraceHeader{})},
	"workload.TraceServer":   {typ: reflect.TypeOf(workload.TraceServer{})},
	"workload.SLODimTargets": {typ: reflect.TypeOf(workload.SLODimTargets{})},
	"workload.Shape":         {typ: reflect.TypeOf(workload.Shape{})},
	"workload.Distribution":  {typ: reflect.TypeOf(workload.Distribution{})},
	"deployment.Deployment":  {typ: reflect.TypeOf(deployment.Deployment{})},
	"deployment.Pool":        {typ: reflect.TypeOf(deployment.Pool{})},
	"deployment.Parallelism": {typ: reflect.TypeOf(deployment.Parallelism{})},
	"deployment.Engine":      {typ: reflect.TypeOf(deployment.Engine{})},
	"deployment.DBO":         {typ: reflect.TypeOf(deployment.DBO{})},
	"deployment.EPLB":        {typ: reflect.TypeOf(deployment.EPLB{})},
	"deployment.Speculative": {typ: reflect.TypeOf(deployment.Speculative{})},
	"deployment.Offload":     {typ: reflect.TypeOf(deployment.Offload{})},
	"deployment.Tier":        {typ: reflect.TypeOf(deployment.Tier{})},
	"deployment.PDTransfer":  {typ: reflect.TypeOf(deployment.PDTransfer{})},
	"model.Graph":            {typ: reflect.TypeOf(model.Graph{})},
	"model.Derivation":       {typ: reflect.TypeOf(model.Derivation{})},
	"model.GlobalShape":      {typ: reflect.TypeOf(model.GlobalShape{})},
	"model.LayerKind":        {typ: reflect.TypeOf(model.LayerKind{})},
	"model.Stack":            {typ: reflect.TypeOf(model.Stack{})},
	"model.Speculator":       {typ: reflect.TypeOf(model.Speculator{})},
	"model.Node":             {typ: reflect.TypeOf(model.Node{})},
	"model.Identity":         {typ: reflect.TypeOf(model.Identity{})},
	"model.Source":           {typ: reflect.TypeOf(model.Source{})},
	"hardware.Chip":          {typ: reflect.TypeOf(hardware.Chip{})},
	"hardware.Fabric":        {typ: reflect.TypeOf(hardware.Fabric{})},
	"hardware.StorageDevice": {typ: reflect.TypeOf(hardware.StorageDevice{}), omit: []string{"name"}},
	"coefficient.Set":        {typ: reflect.TypeOf(coefficient.Set{})},
	"coefficient.Entry":      {typ: reflect.TypeOf(coefficient.Entry{}), omit: []string{"name"}},
	"coefficient.Source":     {typ: reflect.TypeOf(coefficient.Source{})},
	"coefficient.Scope":      {typ: reflect.TypeOf(coefficient.Scope{})},
	"evaluation.Run":         {typ: reflect.TypeOf(evaluation.Run{})},
	"evaluation.Point":       {typ: reflect.TypeOf(evaluation.Point{})},
}

var (
	fieldsMarker = regexp.MustCompile(`^<!-- fields: (\S+) -->$`)
	firstCode    = regexp.MustCompile("`([^`]+)`")
)

func TestDocFieldTablesMatchTypes(t *testing.T) {
	tables := map[string]map[string]bool{} // marker -> keys listed
	paths, err := filepath.Glob(filepath.Join("docs", "reference", "*.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		current := ""
		inTable := false // whether the current marker's table has begun
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if m := fieldsMarker.FindStringSubmatch(line); m != nil {
				current = m[1]
				if _, ok := documented[current]; !ok {
					t.Errorf("%s: marker names %q, which documented does not list", path, current)
				}
				if tables[current] == nil {
					tables[current] = map[string]bool{}
				}
				inTable = false
				continue
			}
			if current == "" {
				continue
			}
			if line == "" && !inTable {
				continue // between the marker and its table
			}
			if !strings.HasPrefix(line, "|") {
				current = ""
				continue
			}
			inTable = true
			cells := strings.Split(strings.Trim(line, "|"), "|")
			m := firstCode.FindStringSubmatch(cells[0])
			if m == nil {
				continue // the header row or the separator
			}
			tables[current][m[1]] = true
		}
		f.Close()
		if err := sc.Err(); err != nil {
			t.Fatal(err)
		}
	}

	names := make([]string, 0, len(documented))
	for n := range documented {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, name := range names {
		d := documented[name]
		got, ok := tables[name]
		if !ok {
			t.Errorf("%s: no field table in docs/reference/ (add one after a `<!-- fields: %s -->` line)", name, name)
			continue
		}
		want := yamlKeys(d.typ)
		for _, k := range d.omit {
			delete(want, k)
		}
		for k := range want {
			if !got[k] {
				t.Errorf("%s: key %q is missing from its field table", name, k)
			}
		}
		for k := range got {
			if !want[k] {
				t.Errorf("%s: field table lists %q, which the type does not declare", name, k)
			}
		}
	}
}

// yamlKeys returns the keys a type reads from YAML: every exported field's tag name,
// skipping fields tagged "-", flattening ",inline" fields, and using the lowercased Go name
// where a field has no tag.
func yamlKeys(t reflect.Type) map[string]bool {
	out := map[string]bool{}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := f.Tag.Get("yaml")
		if tag == "-" || !f.IsExported() {
			continue
		}
		if strings.Contains(tag, ",inline") {
			for k := range yamlKeys(f.Type) {
				out[k] = true
			}
			continue
		}
		name := strings.Split(tag, ",")[0]
		if name == "" {
			name = strings.ToLower(t.Field(i).Name)
		}
		out[name] = true
	}
	return out
}

// TestDocEveryDocumentTypeHasATable walks every struct type reachable from the document
// roots and requires each to be listed in documented, so a nested type added to a schema
// cannot escape TestDocFieldTablesMatchTypes by never being named there.
func TestDocEveryDocumentTypeHasATable(t *testing.T) {
	roots := []reflect.Type{
		reflect.TypeOf(scenario.Scenario{}), reflect.TypeOf(deployment.Deployment{}),
		reflect.TypeOf(model.Graph{}), reflect.TypeOf(model.Identity{}),
		reflect.TypeOf(hardware.Chip{}), reflect.TypeOf(hardware.Fabric{}),
		reflect.TypeOf(hardware.StorageDevice{}), reflect.TypeOf(coefficient.Set{}),
		reflect.TypeOf(workload.Shape{}), reflect.TypeOf(evaluation.Run{}),
	}
	// Types whose wire form is not a mapping, so they have no key table.
	notMappings := map[reflect.Type]bool{reflect.TypeOf(coefficient.Interval{}): true}

	have := map[reflect.Type]bool{}
	for _, d := range documented {
		have[d.typ] = true
	}
	seen := map[reflect.Type]bool{}
	var walk func(reflect.Type)
	walk = func(ty reflect.Type) {
		for ty.Kind() == reflect.Pointer || ty.Kind() == reflect.Slice ||
			ty.Kind() == reflect.Array || ty.Kind() == reflect.Map {
			ty = ty.Elem()
		}
		if ty.Kind() != reflect.Struct || seen[ty] ||
			!strings.HasPrefix(ty.PkgPath(), "github.com/inference-sim/blis-schemas/spec/") {
			return
		}
		seen[ty] = true
		if !have[ty] && !notMappings[ty] {
			t.Errorf("%s is reachable from a document but has no entry in documented and no key table", ty)
		}
		for i := 0; i < ty.NumField(); i++ {
			if f := ty.Field(i); f.IsExported() && f.Tag.Get("yaml") != "-" {
				walk(f.Type)
			}
		}
	}
	for _, r := range roots {
		walk(r)
	}
}
