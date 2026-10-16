package deployment

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The frontend is python or rust, python when unstated; anything else is refused.
func TestFrontendValidation(t *testing.T) {
	for _, c := range []struct {
		f  Frontend
		ok bool
	}{{"", true}, {FrontendPython, true}, {FrontendRust, true}, {"grpc", false}} {
		p := (&Deployment{Kind: "Deployment", Name: "d", Pools: []Pool{{
			Role: RoleColocated, Nodes: 1, Parallel: Parallelism{TP: 1, PP: 1, DP: 1},
			Engine: Engine{Frontend: c.f}}}}).Validate()
		if p.OK() != c.ok {
			t.Errorf("%q: ok=%v, want %v\n%s", c.f, p.OK(), c.ok, p.Error())
		}
		if !c.ok && !strings.Contains(p.Error(), "engine.frontend") {
			t.Errorf("%q: the refusal does not name engine.frontend: %s", c.f, p.Error())
		}
	}
	if Frontend("").Effective() != FrontendPython {
		t.Error("an unstated frontend is not python")
	}
}

// Load, marshal, load: each frontend comes back as itself, and the default is omitted.
func TestFrontendRoundTrips(t *testing.T) {
	for _, f := range []Frontend{"", FrontendPython, FrontendRust} {
		var in Engine
		if err := yaml.Unmarshal([]byte("{frontend: "+string(f)+"}"), &in); err != nil {
			t.Fatal(err)
		}
		out, err := yaml.Marshal(in)
		if err != nil {
			t.Fatal(err)
		}
		var back Engine
		if err := yaml.Unmarshal(out, &back); err != nil || back.Frontend != f {
			t.Fatalf("%q round-tripped through %q to %q (%v)", f, out, back.Frontend, err)
		}
		if f == "" && strings.Contains(string(out), "frontend") {
			t.Fatalf("the default was written: %s", out)
		}
	}
}
