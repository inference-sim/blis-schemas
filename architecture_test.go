package blisschemas

import (
	"os/exec"
	"strings"
	"testing"
)

// These tests guard the repository's structure rather than its contents. The two
// validation layers stay separate only if the dependency direction enforces it, and
// a dependency added in a hurry is exactly the kind of change a reviewer misses.

// TestSpecDoesNotDependOnRules is the layering invariant. If a schema package could
// import a rules pack, a field-level check could quietly become version-specific,
// and the separation this repository is organized around would exist only in the
// documentation.
func TestSpecDoesNotDependOnRules(t *testing.T) {
	const mod = "github.com/inference-sim/blis-schemas/"
	for _, pkg := range []string{
		"./spec/model", "./spec/hardware", "./spec/coefficient",
		"./spec/scenario", "./spec/deployment", "./spec/workload",
		"./spec/evaluation", "./spec/simresult", "./vocab", "./internal/validate",
	} {
		out, err := exec.Command("go", "list", "-deps", pkg).Output()
		if err != nil {
			t.Fatalf("go list -deps %s: %v", pkg, err)
		}
		for _, dep := range strings.Split(string(out), "\n") {
			if !strings.HasPrefix(dep, mod) {
				continue
			}
			rel := strings.TrimPrefix(dep, mod)
			if strings.HasPrefix(rel, "rules") {
				t.Errorf("%s depends on %s; field validation must not know about engine versions",
					pkg, rel)
			}
		}
	}
}

// TestKernelImportsOnlySpec checks the interface's DIRECT imports. It describes what
// a cost model computes, so it needs the document types and must not reach for a
// rules pack or the top-level validator. Transitive dependencies are not checked
// here: spec/ legitimately imports the problem list, and forbidding that
// transitively would forbid spec/ itself.
func TestKernelImportsOnlySpec(t *testing.T) {
	const mod = "github.com/inference-sim/blis-schemas/"
	out, err := exec.Command("go", "list", "-f", "{{join .Imports \"\\n\"}}", "./kernel").Output()
	if err != nil {
		t.Fatalf("go list ./kernel: %v", err)
	}
	for _, dep := range strings.Split(string(out), "\n") {
		if !strings.HasPrefix(dep, mod) {
			continue
		}
		rel := strings.TrimPrefix(dep, mod)
		if !strings.HasPrefix(rel, "spec/") && !strings.HasPrefix(rel, "vocab") {
			t.Errorf("kernel imports %s directly; it should need only document types", rel)
		}
	}
}

// TestRulesPackagesAreVersioned: a pack lives under a version-named directory, so
// adding a release cannot accidentally edit an existing one.
func TestRulesPackagesAreVersioned(t *testing.T) {
	out, err := exec.Command("go", "list", "./rules/...").Output()
	if err != nil {
		t.Fatalf("go list ./rules/...: %v", err)
	}
	const mod = "github.com/inference-sim/blis-schemas/"
	for _, pkg := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		rel := strings.TrimPrefix(pkg, mod)
		if rel == "rules" {
			continue // the mechanism itself
		}
		if !strings.HasPrefix(rel, "rules/v") {
			t.Errorf("%s is under rules/ but is not version-named; a pack must be scoped to a release", rel)
		}
	}
}
