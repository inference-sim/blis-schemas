package blisschemas

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/internal/registry"
	"github.com/inference-sim/blis-schemas/spec/acceptance"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/workload"
)

func alValue(k int, al float64) *acceptance.Value {
	return &acceptance.Value{NumSpecTokens: k, AcceptanceLength: &al}
}

// speculating returns the shared valid bundle with its pool drafting K tokens.
func speculating(k int) Bundle {
	b := bundle()
	b.Deployment.Pools[0].Engine.Speculative = &deployment.Speculative{Method: "mtp", NumSpecTokens: k}
	return b
}

// An inline acceptance names the K it was stated for, and a deployment speculating
// another K refuses it wherever it was written.
func TestAcceptanceKMustMatchTheDeployment(t *testing.T) {
	trace := func(v *acceptance.Value) *workload.Binding {
		return &workload.Binding{Trace: &workload.TraceRef{Data: "t.csv",
			Header:     workload.TraceHeader{Version: 3, TimeUnit: "us", Mode: workload.ModeReal},
			Acceptance: v}}
	}
	cases := []struct {
		name   string
		mutate func(*Bundle, *acceptance.Value)
		path   string
	}{
		{"scenario pin", func(b *Bundle, v *acceptance.Value) {
			b.Scenario.Acceptance = &acceptance.Pin{Value: v}
		}, "scenario.acceptance.value.num_spec_tokens"},
		{"scenario trace", func(b *Bundle, v *acceptance.Value) {
			b.Scenario.Workload = trace(v)
		}, "scenario.workload.trace.acceptance.num_spec_tokens"},
		{"workload shape", func(b *Bundle, v *acceptance.Value) {
			b.Workload = &workload.Shape{Name: "chatbot",
				Prompt: workload.Distribution{Mean: 256}, Output: workload.Distribution{Mean: 256},
				Acceptance: v}
		}, "workload.acceptance.num_spec_tokens"},
	}
	for _, c := range cases {
		t.Run(c.name+"/matching K", func(t *testing.T) {
			b := speculating(3)
			c.mutate(&b, alValue(3, 2.8))
			if rep := Validate(b); !rep.OK() {
				t.Fatalf("a value at the deployment's K was rejected:\n%v", rep.Problems())
			}
		})
		t.Run(c.name+"/other K", func(t *testing.T) {
			b := speculating(5)
			c.mutate(&b, alValue(3, 2.8))
			rep := Validate(b)
			for _, p := range rep.Field.Errors() {
				if p.Path == c.path && strings.Contains(p.Message, "does not carry across K") {
					return
				}
			}
			t.Errorf("no K refusal at %s:\n%v", c.path, rep.Problems())
		})
		t.Run(c.name+"/no speculation warns", func(t *testing.T) {
			b := bundle()
			c.mutate(&b, alValue(3, 2.8))
			rep := Validate(b)
			if !rep.OK() {
				t.Fatalf("an inert value failed validation:\n%v", rep.Problems())
			}
			for _, p := range rep.Field.All() {
				if p.Path == c.path && p.Severity == "warning" {
					return
				}
			}
			t.Errorf("no warning at %s:\n%v", c.path, rep.Problems())
		})
	}
}

func TestLoadAcceptanceProfileSet(t *testing.T) {
	path := write(t, "a.yaml", `
kind: AcceptanceProfileSet
name: s
profiles:
  - model: deepseek-v4.1-flash
    speculative_method: dspark
    num_spec_tokens: 2
    thinking: off
    category: coding
    acceptance_length: 2.62
    per_position_rates: [0.95, 0.67]
    method: measured
    provenance:
      source: golden.yaml
      hardware: b300
      engine_image: vllm/vllm-openai:deepseekv41-flash-0909
      sampling: {temperature: 1.0}
`)
	s, err := LoadAcceptanceProfileSet(path)
	if err != nil {
		t.Fatal(err)
	}
	if rep := Validate(Bundle{AcceptanceProfiles: []*acceptance.ProfileSet{s}}); !rep.OK() {
		t.Fatalf("a valid set failed validation:\n%v", rep.Problems())
	}
	pr := s.Profiles[0]
	if pr.Thinking != acceptance.ThinkingOff || pr.AcceptanceLength != 2.62 ||
		len(pr.PerPositionRates) != 2 || *pr.Provenance.Sampling.Temperature != 1 {
		t.Errorf("profile did not parse: %+v", pr)
	}

	typo := write(t, "b.yaml", "kind: AcceptanceProfileSet\nname: s\nprofiles:\n  - model: m\n    acceptance_lenght: 2\n")
	if _, err := LoadAcceptanceProfileSet(typo); err == nil {
		t.Error("a misspelled profile key loaded silently")
	}
}

func TestResolveAcceptanceProfilesAgainstRegistry(t *testing.T) {
	reg := t.TempDir()
	if err := os.MkdirAll(filepath.Join(reg, "acceptance"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(reg, "acceptance", "inferencex-golden-al.yaml"),
		[]byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := scenarioRef()
	s.Acceptance = &acceptance.Pin{Profiles: "inferencex-golden-al"}
	for _, p := range ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, reg).Problems() {
		if strings.HasPrefix(p.Path, "references.acceptance") {
			t.Errorf("an existing set did not resolve: %s", p)
		}
	}
	s.Acceptance.Profiles = "not-a-set"
	rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, reg)
	msg := rep.Field.Error()
	if !strings.Contains(msg, "references.acceptance.profiles") ||
		!strings.Contains(msg, "acceptance/not-a-set.yaml; the registry has [inferencex-golden-al]") {
		t.Errorf("a missing acceptance set was not reported against the registry:\n%s", msg)
	}
}

// TestLoadRegistryAcceptanceFilesIfPresent loads and validates every committed acceptance
// set in a registry checkout, as TestLoadRegistryFilesIfPresent does coefficient sets.
func TestLoadRegistryAcceptanceFilesIfPresent(t *testing.T) {
	root := os.Getenv("BLIS_REGISTRY")
	if root == "" {
		t.Skip("BLIS_REGISTRY is unset: the committed acceptance sets were NOT loaded or validated")
	}
	paths, err := registry.AcceptancePaths(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Skipf("BLIS_REGISTRY=%s has no acceptance sets", root)
	}
	for _, path := range paths {
		s, err := LoadAcceptanceProfileSet(path)
		if err != nil {
			t.Errorf("%s: %v", filepath.Base(path), err)
			continue
		}
		if p := s.Validate(); !p.OK() {
			t.Errorf("%s failed validation:\n%s", filepath.Base(path), p.Error())
		}
	}
}
