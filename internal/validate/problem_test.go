package validate

import (
	"math"
	"testing"
)

// TestFiniteFieldRejectsNonFinite pins the shared non-finite primitive every numeric
// schema relies on. The cases are chosen for the IEEE-754 subtlety that motivates the
// helper: NaN and +Inf both pass a `<= 0` guard, so a sign check alone would let them
// through; a finite value (including a legitimate negative or zero) must not be flagged
// here, since finiteness and sign are separate checks.
func TestFiniteFieldRejectsNonFinite(t *testing.T) {
	nonFinite := map[string]float64{
		"NaN":  math.NaN(),
		"+Inf": math.Inf(1),
		"-Inf": math.Inf(-1),
	}
	for name, x := range nonFinite {
		p := &Problems{}
		p.FiniteField("f", x)
		if p.OK() {
			t.Errorf("%s should be rejected as non-finite", name)
		}
	}
	finite := map[string]float64{"zero": 0, "negative": -3.5, "positive": 7.2}
	for name, x := range finite {
		p := &Problems{}
		p.FiniteField("f", x)
		if !p.OK() {
			t.Errorf("a finite %s value should not be flagged by the finiteness check", name)
		}
	}
}

func TestSeveritySeparatesFailureFromNotice(t *testing.T) {
	p := &Problems{}
	p.Warnf("a notice")
	if !p.OK() {
		t.Error("a warning alone should not fail a check")
	}
	p.Errorf("a failure")
	if p.OK() {
		t.Error("an error should fail a check")
	}
	if len(p.Errors()) != 1 {
		t.Errorf("Errors() returned %d, want 1", len(p.Errors()))
	}
	if len(p.All()) != 2 {
		t.Errorf("All() returned %d, want 2", len(p.All()))
	}
}

func TestFindingsAreSortedForStableDiffs(t *testing.T) {
	p := &Problems{}
	p.Field("zebra", "last")
	p.Field("alpha", "first")
	p.Field("middle", "second")
	got := p.All()
	for i, want := range []string{"alpha", "middle", "zebra"} {
		if got[i].Path != want {
			t.Errorf("[%d] path = %q, want %q", i, got[i].Path, want)
		}
	}
}

func TestMergePrefixesPaths(t *testing.T) {
	inner := &Problems{}
	inner.Field("tp", "must be positive")
	outer := &Problems{}
	outer.Merge("scenario", inner)
	got := outer.All()
	if len(got) != 1 {
		t.Fatalf("merged %d findings, want 1", len(got))
	}
	if got[0].Path != "scenario.tp" {
		t.Errorf("path = %q, want scenario.tp", got[0].Path)
	}
}

func TestRuleAttributionAppearsInTheMessage(t *testing.T) {
	p := &Problems{}
	p.RuleErrorf("my-rule", "something specific")
	s := p.All()[0].String()
	for _, want := range []string{"error", "my-rule", "something specific"} {
		if !contains(s, want) {
			t.Errorf("rendered finding %q omits %q", s, want)
		}
	}
	// A field-level finding carries no rule name, which is how a reader tells the
	// two layers apart in one log.
	q := &Problems{}
	q.Errorf("plain")
	if contains(q.All()[0].String(), "[") {
		t.Errorf("a field finding should carry no rule attribution: %q", q.All()[0].String())
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
