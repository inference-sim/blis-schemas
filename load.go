package blisschemas

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/evaluation"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
	"github.com/inference-sim/blis-schemas/spec/workload"
)

// Loading is separate from validation on purpose. A parse failure and a validation
// failure need different responses — the first is a malformed file, the second a
// well-formed document that says something wrong — and a combined call would report
// them through one channel.
//
// Every loader rejects unknown fields. A misspelled key that parsed silently would
// leave a scenario that validates while omitting the setting its author intended,
// which is the failure mode hardest to notice: the document looks right and the
// estimate is wrong for a reason nothing reports.

// LoadScenario reads a scenario document. It does not validate; call Validate.
func LoadScenario(path string) (*scenario.Scenario, error) {
	var s scenario.Scenario
	if err := decodeStrict(path, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// LoadDeployment reads a deployment document — the tunable configuration applied to a
// scenario. It does not validate; call Validate.
func LoadDeployment(path string) (*deployment.Deployment, error) {
	var d deployment.Deployment
	if err := decodeStrict(path, &d); err != nil {
		return nil, err
	}
	return &d, nil
}

// LoadModelGraph reads a derived model graph.
func LoadModelGraph(path string) (*model.Graph, error) {
	var g model.Graph
	if err := decodeStrict(path, &g); err != nil {
		return nil, err
	}
	return &g, nil
}

// LoadChip reads a chip descriptor from blis-catalog's hardware namespace. The file
// carries no name — identity is the filename — so the loader supplies it.
func LoadChip(path string) (*hardware.Chip, error) {
	var c hardware.Chip
	if err := decodeStrict(path, &c); err != nil {
		return nil, err
	}
	if c.Name == "" {
		c.Name = stem(path)
	}
	return &c, nil
}

// LoadFabric reads a fabric descriptor from blis-catalog's networks namespace, naming
// it from the filename as LoadChip does.
func LoadFabric(path string) (*hardware.Fabric, error) {
	var f hardware.Fabric
	if err := decodeStrict(path, &f); err != nil {
		return nil, err
	}
	if f.Name == "" {
		f.Name = stem(path)
	}
	return &f, nil
}

// LoadCoefficientSet reads a coefficient set from blis-registry.
func LoadCoefficientSet(path string) (*coefficient.Set, error) {
	var s coefficient.Set
	if err := decodeStrict(path, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// LoadWorkload reads a traffic shape from blis-catalog's workloads namespace.
func LoadWorkload(path string) (*workload.Shape, error) {
	var w workload.Shape
	if err := decodeStrict(path, &w); err != nil {
		return nil, err
	}
	return &w, nil
}

// LoadEvaluationRun reads a measured run.
func LoadEvaluationRun(path string) (*evaluation.Run, error) {
	var r evaluation.Run
	if err := decodeStrict(path, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

// LoadStorageDevices reads the offload-tier classes from blis-catalog's devices
// namespace.
//
// The file is a mapping of tier name to device facts rather than a list, because a tier
// is referred to by name from a deployment's offload block. So this loader is not a plain
// decodeStrict: it walks the mapping and stamps each device's Name from its key, which
// is the only place that key survives into the loaded value.
//
// Sorted by name on return, so two loads of one file yield the same order and a caller
// comparing kernels built from it is comparing configuration rather than map iteration.
func LoadStorageDevices(path string) ([]*hardware.StorageDevice, error) {
	var byName map[string]*hardware.StorageDevice
	if err := decodeStrict(path, &byName); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(byName))
	for name := range byName {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]*hardware.StorageDevice, 0, len(names))
	for _, name := range names {
		d := byName[name]
		if d == nil {
			return nil, fmt.Errorf("%s: tier %q has no device facts", path, name)
		}
		// The key is the tier's identity; a name inside the value would be a second
		// source of truth for it, so the type declares none and it is set here.
		d.Name = name
		out = append(out, d)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: declares no storage tier", path)
	}
	return out, nil
}

// decodeStrict reads one YAML document into dst, rejecting any field the target type
// does not declare. The error names the file, because a caller loading a directory
// needs to know which one failed.
func decodeStrict(path string, dst any) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("opening %s: %w", path, err)
	}
	defer f.Close()

	dec := yaml.NewDecoder(f)
	dec.KnownFields(true)
	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("decoding %s: %w", path, err)
	}
	return nil
}

// stem returns a file's base name without its extension, which is how the catalog
// names a chip, a fabric or a workload.
func stem(path string) string {
	base := filepath.Base(path)
	return strings.TrimSuffix(base, filepath.Ext(base))
}
