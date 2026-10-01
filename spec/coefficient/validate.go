package coefficient

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/internal/validate"
	"github.com/inference-sim/blis-schemas/vocab"
)

// Validate performs field-level validation of a coefficient set: the vocabulary
// terms are recognized, the required fields are present, and the evidence fields
// are internally consistent.
//
// What it does not do is judge whether a value is right, or whether a set is the
// right one for a given cost model. The first needs a measurement; the second is a
// scenario-level question.
func (s *Set) Validate() *validate.Problems {
	p := &validate.Problems{}
	if s.Kind != "CoefficientSet" {
		p.Field("kind", "must be %q, got %q", "CoefficientSet", s.Kind)
	}
	if s.Name == "" {
		p.Field("name", "required: a set's identity is its name")
	}
	if len(s.Coefficients) == 0 {
		p.Field("coefficients", "a set with no entries carries nothing")
	}
	// An entry's identity is its name AND its scope, not its name alone.
	//
	// A set covering several parts carries one entry per (name, scope): the GEMM
	// efficiency asymptote is a different number on H100 than on H200, and both are
	// called gemm_eps_max_bf16 because a resolver selects between them by scope.
	// Keying on the name alone would forbid that shape, and pushing the part into the
	// name instead would duplicate in the name what the scope already states —
	// leaving a resolver no way to tell that two entries are the same quantity
	// measured on different hardware.
	//
	// Two entries sharing a name AND a scope are still an error: a resolver would
	// keep whichever came last, so the file would mean something other than it says.
	seen := map[string]bool{}
	for i, e := range s.Coefficients {
		at := fmt.Sprintf("coefficients[%d]", i)
		if e.Name == "" {
			p.Field(at+".name", "required")
		} else {
			key := e.Name + "\x00" + e.Scope.Key()
			if seen[key] {
				p.Field(at+".name",
					"duplicate entry name %q at the same scope (%s); two entries "+
						"with one name and one scope cannot both be resolved",
					e.Name, e.Scope.Describe())
			}
			seen[key] = true
		}
		e.validate(p, at)
	}
	return p
}

func (e Entry) validate(p *validate.Problems, at string) {
	if !e.Units.Valid() {
		p.Field(at+".units", "%q is not one of %v", e.Units, vocab.AllUnits())
	}
	if !e.Method.Valid() {
		p.Field(at+".method", "%q is not one of %v", e.Method, vocab.AllMethods())
	}
	// A scope that states nothing is not "holds everywhere": it is "applicability
	// unknown", and a coefficient whose applicability is unknown cannot be applied.
	if e.Scope.Empty() {
		p.Field(at+".scope",
			"at least one dimension is required; an empty scope states no applicability rather than universal applicability")
	}
	if e.Method == vocab.MethodNotCharged && e.Value != 0 {
		p.Field(at+".value",
			"method not_charged declares a deliberate zero, but value is %v", e.Value)
	}
	if e.Method == vocab.MethodCopied && e.CopiedFrom == "" {
		p.Field(at+".copied_from", "required when method is copied")
	}
	// An assumed value's only support is its reasoning, so its absence leaves a
	// reader nothing to weigh. A warning rather than an error: the value is usable,
	// and forcing prose would invite filler.
	if e.Method == vocab.MethodAssumed && e.Rationale == "" {
		p.Warnf("%s: an assumed value carries no rationale, so a reader cannot weigh it", at)
	}
	// A claim of evidence should cite it. Also a warning: the registry holds
	// pre-existing entries that are measured and uncited, and failing them would
	// block adoption rather than improve the data.
	if e.Method.Evidenced() && len(e.Sources) == 0 {
		p.Warnf("%s: method %s claims evidence but cites no source", at, e.Method)
	}
	for j, src := range e.Sources {
		sat := fmt.Sprintf("%s.sources[%d]", at, j)
		if !src.Kind.Valid() {
			p.Field(sat+".kind", "%q is not one of %v", src.Kind, vocab.AllSourceKinds())
		}
		if !src.Role.Valid() {
			p.Field(sat+".role", "%q is not one of %v", src.Role, vocab.AllSourceRoles())
		}
		if src.Cite == "" {
			p.Field(sat+".cite", "required")
		}
	}
	if e.CI95 != nil && e.CI95.Low > e.CI95.High {
		p.Field(at+".ci95", "low %v exceeds high %v", e.CI95.Low, e.CI95.High)
	}
	if e.CI95 != nil && (e.Value < e.CI95.Low || e.Value > e.CI95.High) {
		p.Field(at+".value", "%v lies outside its own interval [%v, %v]",
			e.Value, e.CI95.Low, e.CI95.High)
	}
}
