package simresult

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/internal/validate"
	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// Validate performs FIELD-LEVEL validation of a predicted result: required fields, the
// exactly-one operating-point rule, non-negative metrics and monotone percentiles, the
// INV-1 conservation identity, the inline-xor-external requests rule, and the field-level
// validity of the embedded Scenario and Deployment (including their coupling to the
// cluster). It does not apply engine-version-scoped rules: those are a rules pack's job,
// keyed off a Scenario, and blisschemas.Validate runs them against the Scenario a consumer
// supplies as the top-level input — not against the copy embedded here, which is kept for
// the record. Re-running version rules over the embedded input is a consumer's choice, not
// this layer's responsibility.
func (r *SimResult) Validate() *validate.Problems {
	p := &validate.Problems{}

	if r.Kind != "SimResult" {
		p.Field("kind", "must be %q, got %q", "SimResult", r.Kind)
	}
	for field, v := range map[string]string{
		"name": r.Name, "engine_version": r.EngineVersion,
	} {
		if v == "" {
			p.Field(field, "required")
		}
	}

	validateOperatingPoint(p, r.Point)
	// Summary is required and present-or-rejected: a value type would read an omitted
	// summary as a legitimate all-zero one, so it is a pointer and nil is an error.
	if r.Summary == nil {
		p.Field("summary", "required")
	} else {
		validateSummary(p, "summary", *r.Summary)
	}
	validateConservation(p, "conservation", r.Conservation)
	r.validateBreakdowns(p)
	r.validateRequests(p)
	if r.Runtime != nil && r.Runtime.WallClockMs < 0 {
		p.Field("runtime.wall_clock_ms", "must not be negative, got %v", r.Runtime.WallClockMs)
	}
	validateProvenance(p, r.Provenance)

	// The embedded input is validated as part of the result: a prediction that embeds a
	// malformed scenario or deployment is itself malformed. Each check records under an
	// explicit path prefix, as deployment and scenario validation do, so Merge keeps every
	// finding at its own path ("scenario.cluster.hardware", "deployment.pools[0].role").
	p.Merge("scenario", r.Scenario.Validate())
	p.Merge("deployment", r.Deployment.Validate())
	// The deployment is also checked against the cluster its scenario fixes — the same
	// coupling blisschemas.Validate performs for a top-level pair. The embedded result
	// carries both documents, so the checks that pools fill the cluster, that each
	// data-parallel-local width divides a node, and that offload tiers draw from the
	// declared storage can run here.
	p.Merge("deployment", r.Deployment.ValidateAgainstCluster(deployment.ClusterConstraints{
		Nodes:       r.Scenario.Cluster.Nodes,
		GPUsPerNode: r.Scenario.Cluster.GPUsPerNode,
		Storage:     r.Scenario.Cluster.Storage,
	}))

	return p
}

// validateOperatingPoint checks the exactly-one rule: a run is open-loop (a rate) or
// closed-loop (a concurrency), never both and never neither. Stating neither leaves the
// result unable to say what load produced it; stating both describes two runs.
func validateOperatingPoint(p *validate.Problems, o OperatingPoint) {
	switch {
	case o.Rate == nil && o.Concurrency == nil:
		p.Field("operating_point",
			"exactly one of rate or concurrency is required: a result must record the load it was driven at")
	case o.Rate != nil && o.Concurrency != nil:
		p.Field("operating_point",
			"rate and concurrency are mutually exclusive: a run is open-loop (rate) or closed-loop (concurrency), not both")
	case o.Rate != nil && *o.Rate <= 0:
		p.Field("operating_point.rate", "must be positive, got %v", *o.Rate)
	case o.Concurrency != nil && *o.Concurrency < 1:
		p.Field("operating_point.concurrency", "must be at least 1, got %d", *o.Concurrency)
	}
}

func validateSummary(p *validate.Problems, at string, s Summary) {
	for field, v := range map[string]float64{
		at + ".output_tokens_per_sec": s.OutputTokensPerSec,
		at + ".input_tokens_per_sec":  s.InputTokensPerSec,
		at + ".requests_per_sec":      s.RequestsPerSec,
		at + ".sched_delay_p99_ms":    s.SchedDelayP99ms,
		at + ".goodput_rps":           s.GoodputRPS,
	} {
		if v < 0 {
			p.Field(field, "must not be negative, got %v", v)
		}
	}
	if s.SLOAttainment < 0 || s.SLOAttainment > 1 {
		p.Field(at+".slo_attainment", "must lie in [0, 1], got %v", s.SLOAttainment)
	}
	validateDistribution(p, at+".ttft_ms", s.TTFT)
	validateDistribution(p, at+".itl_ms", s.ITL)
	validateDistribution(p, at+".e2e_ms", s.E2E)
}

// validateDistribution checks a latency distribution: nothing negative, and the recorded
// percentiles (the non-nil ones) non-decreasing. A nil percentile is unrecorded and
// skipped, which — because the field is a pointer — is distinct from a recorded zero: a
// p99 explicitly set to 0 below a positive p90 is a real inversion and is caught.
func validateDistribution(p *validate.Problems, at string, d Distribution) {
	if d.Mean < 0 {
		p.Field(at+".mean", "must not be negative, got %v", d.Mean)
	}
	// Walk the recorded percentiles in rank order; each must be at least the previous.
	prev, prevName := 0.0, ""
	for _, pc := range []struct {
		name string
		v    *float64
	}{{"p90", d.P90}, {"p95", d.P95}, {"p99", d.P99}} {
		if pc.v == nil {
			continue // unrecorded
		}
		if *pc.v < 0 {
			p.Field(at+"."+pc.name, "must not be negative, got %v", *pc.v)
		}
		if prevName != "" && *pc.v < prev {
			p.Field(at+"."+pc.name, "%v is below %s %v; percentiles must be non-decreasing",
				*pc.v, prevName, prev)
		}
		prev, prevName = *pc.v, pc.name
	}
}

// validateConservation checks the ledger. Negative counts are impossible; a length-capped
// request, being a completed one, cannot outnumber the completed; and the terminal buckets
// must sum to Injected — INV-1 is a hard invariant relocated into the document, so an
// imbalance in EITHER direction is a field error, not a warning. Over-accounting is
// impossible (you cannot account for more requests than entered); under-accounting means
// requests went unaccounted, which is exactly the lost-request bug INV-1 exists to catch.
func validateConservation(p *validate.Problems, at string, c Conservation) {
	for field, v := range map[string]int64{
		at + ".injected": c.Injected, at + ".completed": c.Completed,
		at + ".still_queued": c.StillQueued, at + ".still_running": c.StillRunning,
		at + ".dropped": c.Dropped, at + ".timed_out": c.TimedOut,
		at + ".length_capped": c.LengthCapped, at + ".preemptions": c.Preemptions,
		at + ".kv_alloc_failures": c.KVAllocFailures,
	} {
		if v < 0 {
			p.Field(field, "must not be negative, got %d", v)
		}
	}
	// A result predicting zero injected requests is vacuous; the floor also makes an
	// omitted (zero-valued) conservation block fail rather than pass as balanced 0 == 0.
	if c.Injected < 1 {
		p.Field(at+".injected", "must be at least 1: a result predicting zero injected requests is vacuous")
	}
	switch {
	case c.Accounted() > c.Injected:
		p.Field(at+".injected",
			"terminal buckets account for %d requests but only %d were injected",
			c.Accounted(), c.Injected)
	case c.Accounted() < c.Injected:
		p.Field(at+".injected",
			"%d injected requests are unaccounted by the terminal buckets; INV-1 requires every request to land in exactly one bucket",
			c.Injected-c.Accounted())
	}
	if c.LengthCapped > c.Completed {
		p.Field(at+".length_capped",
			"%d length-capped exceeds %d completed; a length-capped request is a completed one",
			c.LengthCapped, c.Completed)
	}
}

func (r *SimResult) validateBreakdowns(p *validate.Problems) {
	b := r.Breakdowns
	if b == nil {
		return
	}
	validateSlices(p, "breakdowns.per_class", b.PerClass)
	validateSlices(p, "breakdowns.per_model", b.PerModel)
	validateSlices(p, "breakdowns.per_tenant", b.PerTenant)
	validateSlices(p, "breakdowns.adapters", b.Adapters)
	if b.PD != nil {
		validateDistribution(p, "breakdowns.pd.prefill_ttft_ms", b.PD.PrefillTTFTms)
		validateDistribution(p, "breakdowns.pd.decode_ttft_ms", b.PD.DecodeTTFTms)
		validateDistribution(p, "breakdowns.pd.transfer_ms", b.PD.TransferMs)
		if b.PD.Transfers < 0 {
			p.Field("breakdowns.pd.transfers", "must not be negative, got %d", b.PD.Transfers)
		}
	}
	if b.Saturation != nil && b.Saturation.KneeRPS < 0 {
		p.Field("breakdowns.saturation.knee_rps", "must not be negative, got %v", b.Saturation.KneeRPS)
	}
	if b.Fitness != nil {
		for i, comp := range b.Fitness.Components {
			if comp.Name == "" {
				p.Field(fmt.Sprintf("breakdowns.fitness.components[%d].name", i), "required")
			}
		}
	}
}

// validateSlices checks a named-sub-population list: each entry needs a non-empty, unique
// name and a well-formed summary (and conservation, if present). A duplicate name would
// make a breakdown row ambiguous.
func validateSlices(p *validate.Problems, at string, slices []Slice) {
	seen := map[string]bool{}
	for i, s := range slices {
		base := fmt.Sprintf("%s[%d]", at, i)
		if s.Name == "" {
			p.Field(base+".name", "required")
		} else if seen[s.Name] {
			p.Field(base+".name", "duplicate %q", s.Name)
		}
		seen[s.Name] = true
		validateSummary(p, base+".summary", s.Summary)
		if s.Conservation != nil {
			validateConservation(p, base+".conservation", *s.Conservation)
		}
	}
}

// validateRequests enforces the inline-xor-external rule: a requests block carries inline
// rows OR a file reference, never both and never neither. Bulk rows are never forced
// inline, so the file reference is the path a large run takes.
func (r *SimResult) validateRequests(p *validate.Problems) {
	req := r.Requests
	if req == nil {
		return
	}
	hasRows, hasFile := len(req.Rows) > 0, req.File != ""
	switch {
	case hasRows && hasFile:
		p.Field("requests", "states both inline rows and an external file; they are mutually exclusive")
	case !hasRows && !hasFile:
		p.Field("requests", "states neither inline rows nor an external file; omit the block instead")
	}
	if req.Count != nil && *req.Count < 0 {
		p.Field("requests.count", "must not be negative, got %d", *req.Count)
	}
	if hasRows && req.Count != nil && *req.Count != int64(len(req.Rows)) {
		p.Field("requests.count", "%d does not match the %d inline rows", *req.Count, len(req.Rows))
	}
	for i, row := range req.Rows {
		base := fmt.Sprintf("requests.rows[%d]", i)
		if row.ID == "" {
			p.Field(base+".id", "required")
		}
		for field, v := range map[string]int{
			base + ".input_tokens": row.InputTokens, base + ".output_tokens": row.OutputTokens,
		} {
			if v < 0 {
				p.Field(field, "must not be negative, got %d", v)
			}
		}
		for field, v := range map[string]float64{base + ".ttft_ms": row.TTFTms, base + ".e2e_ms": row.E2Ems} {
			if v < 0 {
				p.Field(field, "must not be negative, got %v", v)
			}
		}
	}
}

func validateProvenance(p *validate.Problems, pr Provenance) {
	for i, c := range pr.Coefficients {
		if c.Name == "" {
			p.Field(fmt.Sprintf("provenance.coefficients[%d].name", i), "required")
		}
	}
}
