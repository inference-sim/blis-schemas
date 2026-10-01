package rules

import (
	"testing"

	"github.com/inference-sim/blis-schemas/internal/validate"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

func TestRegisterRejectsAVersionlessPack(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("registering a pack with no version should panic")
		}
	}()
	Register(&Pack{})
}

func TestRegisterRejectsADuplicate(t *testing.T) {
	Register(&Pack{Version: "test-dup-1"})
	defer func() {
		if recover() == nil {
			t.Error("registering two packs for one version should panic; otherwise one silently wins by import order")
		}
	}()
	Register(&Pack{Version: "test-dup-1"})
}

func TestLookupAndVersions(t *testing.T) {
	Register(&Pack{Version: "test-lookup-1"})
	if Lookup("test-lookup-1") == nil {
		t.Fatal("a registered pack was not found")
	}
	if Lookup("test-absent") != nil {
		t.Error("an unregistered version returned a pack")
	}
	found := false
	for _, v := range Versions() {
		if v == "test-lookup-1" {
			found = true
		}
	}
	if !found {
		t.Error("Versions() omits a registered pack")
	}
}

// TestApplyReportsAnUnknownVersion is the property that keeps a missing pack from
// looking like a clean run.
func TestApplyReportsAnUnknownVersion(t *testing.T) {
	p := Apply(Input{Scenario: &scenario.Scenario{EngineVersion: "does-not-exist"}})
	if len(p.All()) == 0 {
		t.Fatal("an unknown version produced no finding")
	}
	// A warning, not an error: the document may be perfectly valid.
	if !p.OK() {
		t.Error("an unknown version should warn rather than fail")
	}
	if p.All()[0].Rule != "engine-version-known" {
		t.Errorf("finding should be attributed to a rule, got %q", p.All()[0].Rule)
	}
}

func TestApplyWithNoScenario(t *testing.T) {
	if p := Apply(Input{}); p.OK() {
		t.Error("validating nothing should report a problem rather than pass")
	}
}

// TestRulesRunAndAttribute checks that Apply runs a pack's rules and that each
// finding carries its rule name, which is what lets a CI job waive one rule.
func TestRulesRunAndAttribute(t *testing.T) {
	Register(&Pack{Version: "test-apply-1", Rules: []Rule{{
		Name:    "always-fires",
		Because: "exercises the plumbing",
		Check: func(_ Input, out *validate.Problems) {
			out.RuleErrorf("always-fires", "fired")
		},
	}}})
	p := Apply(Input{Scenario: &scenario.Scenario{EngineVersion: "test-apply-1"}})
	if p.OK() {
		t.Fatal("the rule did not fire")
	}
	if p.Errors()[0].Rule != "always-fires" {
		t.Errorf("finding rule = %q, want always-fires", p.Errors()[0].Rule)
	}
}

func TestPackNamesAreSorted(t *testing.T) {
	p := &Pack{Version: "test-names-1", Rules: []Rule{
		{Name: "zebra"}, {Name: "alpha"}, {Name: "middle"},
	}}
	got := p.Names()
	want := []string{"alpha", "middle", "zebra"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Names()[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}
