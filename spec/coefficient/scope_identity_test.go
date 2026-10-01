package coefficient

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/vocab"
)

func entry(name string, hw []string, v float64) Entry {
	return Entry{
		Name: name, Value: v, Units: vocab.UnitDimensionless,
		Method: vocab.MethodMeasured, Scope: Scope{Hardware: hw},
		Sources: []Source{{Kind: vocab.SourceModel, Cite: "x", Role: vocab.RolePrimary}},
	}
}

func set(es ...Entry) Set {
	return Set{Kind: "CoefficientSet", Name: "t", Coefficients: es}
}

func hasDuplicate(s Set) bool {
	for _, p := range s.Validate().All() {
		if strings.Contains(p.Message, "duplicate") {
			return true
		}
	}
	return false
}

func TestOneNameAtTwoScopesIsNotADuplicate(t *testing.T) {
	s := set(entry("gemm_eps_max_bf16", []string{"h100"}, 0.903),
		entry("gemm_eps_max_bf16", []string{"h200"}, 0.890))
	if hasDuplicate(s) {
		t.Error("two scopes under one name were rejected; a resolver picks by scope")
	}
}

func TestOneNameAtOneScopeIsADuplicate(t *testing.T) {
	s := set(entry("gemm_eps_max_bf16", []string{"h200"}, 0.890),
		entry("gemm_eps_max_bf16", []string{"h200"}, 0.72))
	if !hasDuplicate(s) {
		t.Error("two entries a resolver cannot choose between were accepted")
	}
}

func TestScopeIdentityIgnoresListOrder(t *testing.T) {
	s := set(entry("hbm_derate", []string{"h100", "h200"}, 0.8),
		entry("hbm_derate", []string{"h200", "h100"}, 0.85))
	if !hasDuplicate(s) {
		t.Error("the same scope written in a different order was treated as distinct")
	}
}

func TestTwoUnscopedEntriesUnderOneNameAreADuplicate(t *testing.T) {
	s := set(entry("hbm_derate", nil, 0.8), entry("hbm_derate", nil, 0.85))
	if !hasDuplicate(s) {
		t.Error("two unscoped entries under one name were accepted")
	}
}

func TestScopeKeyDistinguishesDimensions(t *testing.T) {
	// A hardware scope and a model scope naming the same string are different scopes.
	a := Scope{Hardware: []string{"x"}}
	b := Scope{Model: []string{"x"}}
	if a.Key() == b.Key() {
		t.Errorf("scopes on different dimensions collide: %q", a.Key())
	}
}
