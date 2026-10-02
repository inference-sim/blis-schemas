// Package blisschemas is the entry point: it composes field-level validation with
// version-scoped rule validation over a complete set of documents.
//
// The two layers stay visible in the result. A Report separates field problems from
// rule problems, because the two need different responses: a field problem is a
// malformed document and always the author's to fix, where a rule problem may mean
// the document is right and the engine version is wrong, or that a rule needs
// updating for a release it has not seen.
//
// Field validation runs first and rule validation only if it passes. A rule reading
// a malformed document produces findings that are artifacts of the malformation, and
// a reader cannot tell those from real ones.
package blisschemas

import (
	"github.com/inference-sim/blis-schemas/internal/validate"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/evaluation"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
	"github.com/inference-sim/blis-schemas/spec/workload"
)

// Bundle is a complete set of documents describing one run. Optional members are nil
// when a caller is validating a subset: a catalog contributor checks a chip alone,
// where a CI job checking a run supplies everything. The Scenario fixes the immutable
// problem and the Deployment the mutable configuration against it; validating both
// together is what checks that a deployment fits its cluster.
type Bundle struct {
	Scenario     *scenario.Scenario
	Deployment   *deployment.Deployment
	Model        *model.Graph
	Chip         *hardware.Chip
	Fabric       *hardware.Fabric
	Devices      []*hardware.StorageDevice
	Coefficients []*coefficient.Set
	Workload     *workload.Shape
	Evaluation   *evaluation.Run
}

// Report is the outcome of validating a bundle, with the two layers kept apart.
type Report struct {
	// Field holds structural problems: malformed documents, unrecognized vocabulary,
	// arithmetic that does not close. These are version-independent.
	Field *validate.Problems
	// Rule holds version-scoped problems, each attributed to a named rule.
	Rule *validate.Problems
	// RulesApplied is the engine version whose pack ran, empty if none did.
	RulesApplied string
}

// OK reports whether nothing failed in either layer.
func (r Report) OK() bool { return r.Field.OK() && r.Rule.OK() }

// Problems returns every finding from both layers, field problems first.
func (r Report) Problems() []validate.Problem {
	return append(r.Field.All(), r.Rule.All()...)
}

// Validate checks a bundle: every present document field-by-field, then the
// version-scoped rules if the field layer passed.
func Validate(b Bundle) Report {
	field := &validate.Problems{}

	if b.Scenario != nil {
		field.Merge("scenario", b.Scenario.Validate())
	}
	if b.Deployment != nil {
		field.Merge("deployment", b.Deployment.Validate())
	}
	// A deployment and the cluster it is placed on are two documents, so the checks
	// that couple them — that the pools fill the cluster and that each local
	// data-parallel width divides a node — can only run when both are present.
	if b.Scenario != nil && b.Deployment != nil {
		field.Merge("deployment", b.Deployment.ValidateAgainstCluster(
			b.Scenario.Cluster.Nodes, b.Scenario.Cluster.GPUsPerNode,
			b.Scenario.Cluster.Storage))
	}
	if b.Model != nil {
		field.Merge("model", b.Model.Validate())
	}
	if b.Chip != nil {
		field.Merge("chip", b.Chip.Validate())
	}
	if b.Fabric != nil {
		field.Merge("fabric", b.Fabric.Validate())
	}
	for _, d := range b.Devices {
		if d != nil {
			field.Merge("device:"+d.Name, d.Validate())
		}
	}
	for _, c := range b.Coefficients {
		if c != nil {
			field.Merge("coefficients:"+c.Name, c.Validate())
		}
	}
	if b.Workload != nil {
		field.Merge("workload", b.Workload.Validate())
	}
	if b.Evaluation != nil {
		field.Merge("evaluation", b.Evaluation.Validate())
	}

	rep := Report{Field: field, Rule: &validate.Problems{}}
	// Rules read a document assuming it is well formed. Running them over a
	// malformed one yields findings that are consequences of the malformation, which
	// a reader cannot separate from genuine rule violations.
	if !field.OK() {
		return rep
	}
	if b.Scenario != nil {
		rep.Rule = rules.Apply(rules.Input{
			Scenario: b.Scenario, Deployment: b.Deployment, Model: b.Model})
		if rules.Lookup(b.Scenario.EngineVersion) != nil {
			rep.RulesApplied = b.Scenario.EngineVersion
		}
	}
	return rep
}
