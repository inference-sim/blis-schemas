package coefficient

import (
	"testing"

	"github.com/inference-sim/blis-schemas/vocab"
)

func validSet() *Set {
	return &Set{
		Kind: "CoefficientSet", Name: "cost-model-primitives-h200",
		Coefficients: []Entry{
			{Name: "gemm_eps_max_bf16", Value: 0.72,
				Units: vocab.UnitDimensionless, Method: vocab.MethodMeasured,
				Fitted: true, Scope: Scope{Hardware: []string{"h200"}},
				Sources: []Source{{Kind: vocab.SourceModel, Cite: "operator table",
					Role: vocab.RolePrimary}}},
			{Name: "gemm_m_half_bf16", Value: 96, Units: vocab.UnitTokens,
				Method: vocab.MethodMeasured, Fitted: true,
				Scope: Scope{Hardware: []string{"h200"}},
				Sources: []Source{{Kind: vocab.SourceModel, Cite: "same fit",
					Role: vocab.RolePrimary}}},
			{Name: "host_output_token", Value: 45.9,
				Units: vocab.UnitMicrosecondsPerToken, Method: vocab.MethodAssumed,
				Scope:     Scope{Hardware: []string{"h200"}},
				Rationale: "scaled from a related fit; see the realization document"},
		},
	}
}

func TestValidSetPasses(t *testing.T) {
	if p := validSet().Validate(); !p.OK() {
		t.Fatalf("a valid set was rejected:\n%s", p.Error())
	}
}

func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Set)
	}{
		{"wrong kind", func(s *Set) { s.Kind = "Coefficients" }},
		{"no name", func(s *Set) { s.Name = "" }},
		{"no entries", func(s *Set) { s.Coefficients = nil }},
		{"duplicate entry name", func(s *Set) {
			s.Coefficients = append(s.Coefficients, s.Coefficients[0])
		}},
		{"unknown unit", func(s *Set) { s.Coefficients[0].Units = "parsecs" }},
		{"unknown method", func(s *Set) { s.Coefficients[0].Method = "divined" }},
		{"empty scope", func(s *Set) { s.Coefficients[0].Scope = Scope{} }},
		{"not_charged with a value", func(s *Set) {
			s.Coefficients[0].Method = vocab.MethodNotCharged
		}},
		{"copied without a source scope", func(s *Set) {
			s.Coefficients[0].Method = vocab.MethodCopied
		}},
		{"unknown source kind", func(s *Set) {
			s.Coefficients[0].Sources[0].Kind = "rumour"
		}},
		{"source without a citation", func(s *Set) {
			s.Coefficients[0].Sources[0].Cite = ""
		}},
		{"inverted interval", func(s *Set) {
			s.Coefficients[0].CI95 = &Interval{Low: 1, High: 0}
		}},
		{"value outside its interval", func(s *Set) {
			s.Coefficients[0].CI95 = &Interval{Low: 0.8, High: 0.9}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := validSet()
			tc.mutate(s)
			if p := s.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

// TestAssumedWithoutRationaleWarns: usable but unweighable, so a warning rather
// than an error.
func TestAssumedWithoutRationaleWarns(t *testing.T) {
	s := validSet()
	s.Coefficients[2].Rationale = ""
	p := s.Validate()
	if !p.OK() {
		t.Errorf("a missing rationale should warn, not fail:\n%s", p.Error())
	}
	if len(p.All()) == 0 {
		t.Error("expected a warning about the missing rationale")
	}
}

func TestEvidencedWithoutSourceWarns(t *testing.T) {
	s := validSet()
	s.Coefficients[0].Sources = nil
	p := s.Validate()
	if !p.OK() {
		t.Errorf("an uncited measurement should warn, not fail:\n%s", p.Error())
	}
	if len(p.All()) == 0 {
		t.Error("expected a warning about the missing citation")
	}
}

func TestScopeEmpty(t *testing.T) {
	if !(Scope{}).Empty() {
		t.Error("a zero scope should report empty")
	}
	if (Scope{TP: []int{8}}).Empty() {
		t.Error("a scope with one dimension should not report empty")
	}
}
