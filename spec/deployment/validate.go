package deployment

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/inference-sim/blis-schemas/internal/validate"
)

// Validate performs FIELD-LEVEL validation of a deployment IN ISOLATION: required
// fields, positive counts, internal arithmetic consistency, and the structural
// invariants that hold whatever engine version the scenario names. The checks that
// couple a deployment to the cluster it runs on live in ValidateAgainstCluster,
// because the cluster is a Scenario property this type does not carry.
//
// A structural invariant is one that follows from what parallelism is, not from an
// engine's current choices. That expert-parallel width is derived rather than stated
// is a consequence of how an expert group is built, so it belongs here. Version-scoped
// rules — whether a backend name is supported, which conditions disable async
// scheduling — do not; a rules pack keyed by engine version owns them.
func (d *Deployment) Validate() *validate.Problems {
	p := &validate.Problems{}

	if d.Kind != "Deployment" {
		p.Field("kind", "must be %q, got %q", "Deployment", d.Kind)
	}
	if d.Name == "" {
		p.Field("name", "required")
	}

	d.validatePools(p)
	d.validateOffload(p)
	d.validatePolicy(p)
	return p
}

// ClusterConstraints is the subset of a Scenario's cluster that deployment validation
// reads: the node and per-node GPU counts, and the declared storage inventory. It is a
// deployment-local type that mirrors those facts rather than importing scenario.Cluster,
// so spec/deployment stays independent of spec/scenario. Named fields also stop the two
// same-typed counts from being transposed at a call site, which a positional
// (nodes, gpusPerNode int) pair invited.
type ClusterConstraints struct {
	Nodes       int
	GPUsPerNode int
	Storage     []string
}

// ValidateAgainstCluster adds the field-level checks that couple a deployment to the
// available-hardware inventory it is placed on: that its pools' node counts sum to the
// nodes the cluster declares, that each data-parallel-local width divides a node's GPU
// count, and that every offload tier names a storage class the cluster actually lists.
//
// Pools fill the cluster exactly rather than take a subset of it: a deployment lays out
// the whole cluster it is handed, so the inventory is that cluster's full extent and not
// a pool to sub-select nodes from. This exact-fill rule is the placement contract the
// pre-split Scenario enforced, carried here unchanged — relaxing it to allow partial use
// of the declared hardware would be a behavior change, not part of this value-preserving
// split.
//
// They are separate from Validate because they need the cluster, which is a Scenario
// property; the composition layer (blisschemas.Validate) supplies it as a
// ClusterConstraints when both documents are present.
func (d *Deployment) ValidateAgainstCluster(c ClusterConstraints) *validate.Problems {
	p := &validate.Problems{}

	total := 0
	for i, pool := range d.Pools {
		total += pool.Nodes
		if pool.Parallel.DPLocal > 0 && c.GPUsPerNode > 0 &&
			c.GPUsPerNode%pool.Parallel.DPLocal != 0 {
			p.Field(fmt.Sprintf("pools[%d].parallel.dp_local", i),
				"%d does not divide gpus_per_node %d", pool.Parallel.DPLocal, c.GPUsPerNode)
		}
	}
	if len(d.Pools) > 0 && total != c.Nodes {
		p.Field("pools",
			"pool node counts sum to %d but the cluster declares %d", total, c.Nodes)
	}

	// Each offload tier draws from the cluster's declared storage inventory, as the Tier
	// doc states. The check fires only when the cluster lists storage: a cluster that
	// declares none states no inventory to constrain against, so a deployment offloading
	// against it is left as it was before the inventory existed, rather than rejected.
	if d.Offload != nil && len(c.Storage) > 0 {
		declared := make(map[string]bool, len(c.Storage))
		for _, s := range c.Storage {
			declared[s] = true
		}
		for i, t := range d.Offload.Tiers {
			if t.Device != "" && !declared[t.Device] {
				p.Field(fmt.Sprintf("offload.tiers[%d].tier", i),
					"%q is not in the cluster storage inventory %v", t.Device, c.Storage)
			}
		}
	}
	return p
}

func (d *Deployment) validatePools(p *validate.Problems) {
	if len(d.Pools) == 0 {
		p.Field("pools", "at least one pool is required")
		return
	}

	roles := map[Role]int{}
	for i, pool := range d.Pools {
		at := fmt.Sprintf("pools[%d]", i)
		if !pool.Role.Valid() {
			p.Field(at+".role", "%q is not a recognized role", pool.Role)
		}
		roles[pool.Role]++
		if pool.Nodes < 1 {
			p.Field(at+".nodes", "must be at least 1")
		}
		validateParallelism(p, at, pool.Parallel)
		validateEngine(p, at, pool.Engine)
	}

	// A disaggregated deployment needs both halves; one alone has nowhere to send or
	// receive KV, and a colocated pool alongside them describes two designs at once.
	if roles[RolePrefill] > 0 && roles[RoleDecode] == 0 {
		p.Field("pools", "a prefill pool with no decode pool has nowhere to send KV")
	}
	if roles[RoleDecode] > 0 && roles[RolePrefill] == 0 {
		p.Field("pools", "a decode pool with no prefill pool has no source of KV")
	}
	if roles[RoleColocated] > 0 && (roles[RolePrefill] > 0 || roles[RoleDecode] > 0) {
		p.Field("pools", "a colocated pool alongside disaggregated pools describes two deployments")
	}
	if roles[RolePrefill] > 0 && d.PDTransfer == nil {
		p.Field("pd_transfer", "required for a disaggregated deployment")
	}
}

func validateParallelism(p *validate.Problems, at string, pl Parallelism) {
	for field, v := range map[string]int{"tp": pl.TP, "pp": pl.PP, "dp": pl.DP} {
		if v < 1 {
			p.Field(at+".parallel."+field, "must be at least 1, got %d", v)
		}
	}
	if pl.PCP < 0 || pl.DCP < 0 {
		p.Field(at+".parallel", "context-parallel widths must not be negative")
	}
	// Prefill- and decode-context parallelism shard one sequence. Combining
	// prefill-context parallelism with data parallelism makes the expert group's
	// extent ambiguous, and engines reject it.
	if pl.PCP > 1 && pl.DP > 1 {
		p.Field(at+".parallel.pcp",
			"prefill-context parallelism and data parallelism cannot both exceed 1")
	}
	if pl.DPLocal > pl.DP && pl.DP > 0 {
		p.Field(at+".parallel.dp_local", "%d exceeds dp %d", pl.DPLocal, pl.DP)
	}
	// Expert parallelism without a mixture-of-experts model shards nothing. That
	// cross-check needs the model graph, so it is a rules-pack concern; what is
	// checkable here is that the flag and the derived width agree.
	if pl.EnableExpertParallel && pl.ExpertParallelWidth() < 2 {
		p.Warnf("%s.parallel: expert parallelism is enabled but the derived width is 1, so no expert sharding occurs", at)
	}
}

func validateEngine(p *validate.Problems, at string, e Engine) {
	for field, v := range map[string]int{
		"block_size": e.BlockSize, "max_num_batched_tokens": e.MaxNumBatchedTokens,
		"max_num_seqs": e.MaxNumSeqs, "max_model_len": e.MaxModelLen,
	} {
		if v < 0 {
			p.Field(at+".engine."+field, "must not be negative")
		}
	}
	// Stating both the boolean and the resolved name invites them to disagree, and
	// which one an implementation honours would be arbitrary.
	if e.DisableCustomAllReduce != nil && e.AllReduceBackend != "" {
		disabled := *e.DisableCustomAllReduce
		if disabled && e.AllReduceBackend == "custom" {
			p.Field(at+".engine.allreduce_backend",
				"requests the custom kernel while disable_custom_all_reduce is true")
		}
		if !disabled && e.AllReduceBackend == "nccl" {
			p.Warnf("%s.engine: allreduce_backend names nccl while disable_custom_all_reduce is false; the engine would use the custom kernel where reachable", at)
		}
	}
	// gpu_memory_utilization is a non-pointer float, so an omitted one is zero. Zero is
	// the unset sentinel — the engine picks its default — and a stated fraction lies in
	// (0, 1], so the accepted range is [0, 1]. The message states that range rather than
	// (0, 1] so it does not read as rejecting the zero the check deliberately allows.
	if e.GPUMemoryUtilization < 0 || e.GPUMemoryUtilization > 1 {
		p.Field(at+".engine.gpu_memory_utilization",
			"must lie in [0, 1] (0 means unset), got %v", e.GPUMemoryUtilization)
	}
	if e.DBO != nil && e.DBO.Enabled {
		if e.DBO.DecodeTokenThreshold < 0 || e.DBO.PrefillTokenThreshold < 0 {
			p.Field(at+".engine.dbo", "thresholds must not be negative")
		}
	}
	if e.EPLB != nil && e.EPLB.Enabled && e.EPLB.NumRedundantExperts < 0 {
		p.Field(at+".engine.eplb.num_redundant_experts", "must not be negative")
	}
	if e.Speculative != nil {
		if e.Speculative.Method == "" {
			p.Field(at+".engine.speculative.method", "required")
		}
		if e.Speculative.NumSpecTokens < 1 {
			p.Field(at+".engine.speculative.num_spec_tokens",
				"must be at least 1; omit the block instead of declaring zero drafts")
		}
	}
}

func (d *Deployment) validateOffload(p *validate.Problems) {
	o := d.Offload
	if o == nil {
		return
	}
	if len(o.Tiers) == 0 {
		p.Field("offload.tiers", "an offload block with no tiers offloads nowhere")
	}
	seen := map[string]bool{}
	for i, t := range o.Tiers {
		at := fmt.Sprintf("offload.tiers[%d]", i)
		if t.Device == "" {
			p.Field(at+".tier", "required: names a catalog device class")
		} else if seen[t.Device] {
			p.Field(at+".tier", "duplicate tier %q", t.Device)
		}
		seen[t.Device] = true
		if t.Bytes < 1 {
			p.Field(at+".bytes", "must be positive")
		}
	}
	if o.PrefetchDepth < 0 {
		p.Field("offload.prefetch_depth", "must not be negative")
	}
}

// validatePolicy field-validates the control-plane policy surface (policy.go). Each
// block is optional; an absent one is skipped. The checks confirm that enumerated
// values are drawn from the sets the simulator accepts and that numeric parameters lie
// in a usable range, mirroring the simulator's own policy-bundle, saturation-config and
// lora-config validation.
func (d *Deployment) validatePolicy(p *validate.Problems) {
	validateAdmission(p, d.Admission)
	validateRouting(p, d.Routing)
	validateScheduler(p, d.Scheduler)
	validatePreemption(p, d.Preemption)
	validateSaturation(p, d.Saturation)
	validateLoRA(p, d.LoRA)
}

func validateAdmission(p *validate.Problems, a *Admission) {
	if a == nil {
		return
	}
	// An empty policy takes the simulator default (always-admit), as the simulator's own
	// empty-string case does, so it is only checked when stated.
	if a.Policy != "" && !a.Policy.Valid() {
		p.Field("admission.policy", "%q is not a recognized admission policy; known: %s",
			a.Policy, knownValues(admissionPolicies))
	}
	if a.TokenBucketCapacity != nil && !finitePositive(*a.TokenBucketCapacity) {
		p.Field("admission.token_bucket_capacity",
			"must be a finite positive number, got %v", *a.TokenBucketCapacity)
	}
	if a.TokenBucketRefillRate != nil && !finitePositive(*a.TokenBucketRefillRate) {
		p.Field("admission.token_bucket_refill_rate",
			"must be a finite positive number, got %v", *a.TokenBucketRefillRate)
	}
	if a.TierShedThreshold != nil && *a.TierShedThreshold < 0 {
		p.Field("admission.tier_shed_threshold", "must not be negative, got %d", *a.TierShedThreshold)
	}
	// tier_shed_min_priority is a priority value, which may be negative (a sheddable
	// tier), so it carries no sign constraint.
	if a.GAIEQDThreshold != nil && !finitePositive(*a.GAIEQDThreshold) {
		p.Field("admission.gaie_qd_threshold",
			"must be a finite positive number, got %v", *a.GAIEQDThreshold)
	}
	if a.GAIEKVThreshold != nil {
		if v := *a.GAIEKVThreshold; !(v > 0 && v <= 1) {
			p.Field("admission.gaie_kv_threshold", "must lie in (0, 1], got %v", v)
		}
	}
	if a.LatencyUs < 0 {
		p.Field("admission.admission_latency_us", "must not be negative, got %d", a.LatencyUs)
	}
	for class, target := range a.SLOTargets {
		at := fmt.Sprintf("admission.slo_targets[%q]", class)
		if class == "" {
			p.Field(at, "an SLO class key must not be empty")
		}
		if target <= 0 {
			p.Field(at, "target must be positive microseconds, got %d", target)
		}
	}
	for class := range a.SLOPriorities {
		if class == "" {
			p.Field(fmt.Sprintf("admission.slo_priorities[%q]", class),
				"an SLO class key must not be empty")
		}
	}
}

func validateRouting(p *validate.Problems, r *Routing) {
	if r == nil {
		return
	}
	if r.Policy != "" && !r.Policy.Valid() {
		p.Field("routing.policy", "%q is not a recognized routing policy; known: %s",
			r.Policy, knownValues(routingPolicies))
	}
	seen := map[Scorer]bool{}
	for i, sw := range r.Scorers {
		at := fmt.Sprintf("routing.scorers[%d]", i)
		switch {
		case sw.Name == "":
			p.Field(at+".name", "required: names a routing scorer")
		case !sw.Name.Valid():
			p.Field(at+".name", "%q is not a recognized scorer; known: %s",
				sw.Name, knownValues(scorers))
		case seen[sw.Name]:
			p.Field(at+".name", "duplicate scorer %q; each scorer may appear at most once", sw.Name)
		}
		seen[sw.Name] = true
		if !finitePositive(sw.Weight) {
			p.Field(at+".weight", "must be a finite positive number, got %v", sw.Weight)
		}
	}
	if r.LatencyUs < 0 {
		p.Field("routing.routing_latency_us", "must not be negative, got %d", r.LatencyUs)
	}
	// Scorers compose only within the weighted policy; naming them under any other policy
	// has no effect, so scorers with a non-weighted policy are a warning rather than an
	// error. An empty policy takes the simulator default (round-robin), which also
	// ignores scorers, so the warning fires for the omitted case too — naming it as
	// round-robin so the author sees why the scorers are inert.
	if len(r.Scorers) > 0 && r.Policy != RoutingWeighted {
		shown := r.Policy
		if shown == "" {
			shown = RoutingRoundRobin
		}
		p.Warnf("routing: scorers are declared but policy %q ignores them; only the weighted policy consults scorers", shown)
	}
}

func validateScheduler(p *validate.Problems, s SchedulerPolicy) {
	if s != "" && !s.Valid() {
		p.Field("scheduler", "%q is not a recognized BLIS-native scheduler; known: %s",
			s, knownValues(schedulerPolicies))
	}
}

func validatePreemption(p *validate.Problems, pr *Preemption) {
	if pr == nil {
		return
	}
	if pr.Policy != "" && !pr.Policy.Valid() {
		p.Field("preemption.policy", "%q is not a recognized preemption policy; known: %s",
			pr.Policy, knownValues(preemptionPolicies))
	}
}

func validateSaturation(p *validate.Problems, s *Saturation) {
	if s == nil {
		return
	}
	seen := map[Detector]bool{}
	for i, d := range s.Detectors {
		at := fmt.Sprintf("saturation.detectors[%d]", i)
		switch {
		case d == "":
			p.Field(at, "required: names a detector")
		case !d.Valid():
			p.Field(at, "%q is not a recognized detector; known: %s", d, knownValues(detectors))
		case seen[d]:
			p.Field(at, "duplicate detector %q", d)
		}
		seen[d] = true
	}
	if c := s.Composite; c != nil && c.Sensitivity != nil && !finitePositive(*c.Sensitivity) {
		p.Field("saturation.composite.sensitivity",
			"must be a finite positive number, got %v", *c.Sensitivity)
	}
	if t := s.Threshold; t != nil && t.ThresholdMs != nil && !finitePositive(*t.ThresholdMs) {
		p.Field("saturation.threshold.threshold_ms",
			"must be a finite positive number, got %v", *t.ThresholdMs)
	}
	validateBacklogDrift(p, s.BacklogDrift)
	validatePeakRate(p, s.PeakRate)
	if s.FinalWindow != "" {
		if _, err := time.ParseDuration(s.FinalWindow); err != nil {
			p.Field("saturation.final_window", "%q is not a valid Go duration: %v", s.FinalWindow, err)
		}
	}
}

func validateBacklogDrift(p *validate.Problems, b *BacklogDriftDetector) {
	if b == nil {
		return
	}
	atLeastOne := func(field string, v *int) {
		if v != nil && *v < 1 {
			p.Field("saturation.backlog_drift."+field, "must be at least 1, got %d", *v)
		}
	}
	atLeastOne("window_size_sec", b.WindowSizeSec)
	atLeastOne("min_windows", b.MinWindows)
	atLeastOne("tail_windows", b.TailWindows)
	if b.WarmupWindows != nil && *b.WarmupWindows < 0 {
		p.Field("saturation.backlog_drift.warmup_windows", "must not be negative, got %d", *b.WarmupWindows)
	}
	posFloat := func(field string, v *float64) {
		if v != nil && !finitePositive(*v) {
			p.Field("saturation.backlog_drift."+field, "must be a finite positive number, got %v", *v)
		}
	}
	posFloat("peak_ratio", b.PeakRatio)
	// slope_k is the false-alarm calibration knob; a value of 1 or below is a legitimate
	// "maximally severe" setting, so only non-positive is rejected.
	posFloat("slope_k", b.SlopeK)
	if b.PeakRatioBand != nil {
		if v := *b.PeakRatioBand; v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			p.Field("saturation.backlog_drift.peak_ratio_band",
				"must be a non-negative finite number, got %v", v)
		}
	}
	fraction := func(field string, v *float64) {
		if v != nil && !(*v > 0 && *v <= 1) {
			p.Field("saturation.backlog_drift."+field, "must lie in (0, 1], got %v", *v)
		}
	}
	fraction("confidence_ci", b.ConfidenceCI)
	fraction("saturated_drain_ratio", b.SaturatedDrainRatio)
	fraction("transient_drain_ratio", b.TransientDrainRatio)
}

func validatePeakRate(p *validate.Problems, pr *PeakRateDetector) {
	if pr == nil {
		return
	}
	if pr.Threshold != nil && !finitePositive(*pr.Threshold) {
		p.Field("saturation.peak_rate.threshold", "must be a finite positive number, got %v", *pr.Threshold)
	}
	// overload_multiple at or below 1 collapses the detector to two levels, which the
	// simulator accepts as a deliberate setting, so only non-positive is rejected.
	if pr.OverloadMultiple != nil && !finitePositive(*pr.OverloadMultiple) {
		p.Field("saturation.peak_rate.overload_multiple",
			"must be a finite positive number, got %v", *pr.OverloadMultiple)
	}
	if pr.MinObservations != nil && *pr.MinObservations < 1 {
		p.Field("saturation.peak_rate.min_observations", "must be at least 1, got %d", *pr.MinObservations)
	}
	if pr.ConsecutiveK != nil && *pr.ConsecutiveK < 1 {
		p.Field("saturation.peak_rate.consecutive_k", "must be at least 1, got %d", *pr.ConsecutiveK)
	}
	if pr.WarmupUs != nil && *pr.WarmupUs < 0 {
		p.Field("saturation.peak_rate.warmup_us", "must not be negative, got %d", *pr.WarmupUs)
	}
}

func validateLoRA(p *validate.Problems, l *LoRA) {
	if l == nil {
		return
	}
	if l.AdapterCapacity != nil && *l.AdapterCapacity < 0 {
		p.Field("lora.adapter_capacity", "must not be negative, got %d", *l.AdapterCapacity)
	}
	if l.LoadBaseLatencyUs != nil {
		if v := *l.LoadBaseLatencyUs; v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			p.Field("lora.load_base_latency_us", "must be a non-negative finite number, got %v", v)
		}
	}
	if l.LoadBandwidthBytesUs != nil && !finitePositive(*l.LoadBandwidthBytesUs) {
		p.Field("lora.load_bandwidth_bytes_us",
			"must be a finite positive number (it divides adapter bytes), got %v", *l.LoadBandwidthBytesUs)
	}
	if l.FootprintBytesPerRank != nil && !finitePositive(*l.FootprintBytesPerRank) {
		p.Field("lora.footprint_bytes_per_rank", "must be a finite positive number, got %v", *l.FootprintBytesPerRank)
	}
	for rank, tier := range l.StepOverheadTiers {
		at := fmt.Sprintf("lora.step_overhead_tiers[%d]", rank)
		if rank < 1 {
			p.Field(at, "rank tier must be at least 1, got %d", rank)
		}
		if tier.K6 != nil {
			if v := *tier.K6; v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
				p.Field(at+".k6", "must be a non-negative finite number, got %v", v)
			}
		}
		if tier.K7 == nil {
			p.Field(at+".k7", "required: the per-tier normalization denominator")
		} else if !finitePositive(*tier.K7) {
			p.Field(at+".k7", "must be a finite positive number, got %v", *tier.K7)
		}
	}
	seen := map[string]bool{}
	for i, ad := range l.Adapters {
		at := fmt.Sprintf("lora.adapters[%d]", i)
		switch {
		case ad.ID == "":
			p.Field(at+".id", "required")
		case seen[ad.ID]:
			p.Field(at+".id", "duplicate adapter id %q", ad.ID)
		}
		seen[ad.ID] = true
		if ad.Rank < 1 {
			p.Field(at+".rank", "must be at least 1, got %d", ad.Rank)
		}
	}
	// Declaring adapters without the capacity to hold them leaves the subsystem inert
	// while implying it is active, which the simulator rejects (a pointer-to-zero
	// capacity alongside adapters is an error there).
	if len(l.Adapters) > 0 && (l.AdapterCapacity == nil || *l.AdapterCapacity < 1) {
		p.Field("lora.adapter_capacity", "at least 1 is required when adapters are declared")
	}
}

// finitePositive reports whether f is a finite number strictly greater than zero. It is
// the test the simulator applies to scorer weights and calibration knobs: a weight or a
// dial that is zero, negative, NaN or infinite is not a usable value.
func finitePositive(f float64) bool { return f > 0 && !math.IsInf(f, 0) }

// knownValues renders a sorted, comma-separated list of a string-enum validity map's
// keys, for the "known: ..." tail of an error message.
func knownValues[T ~string](m map[T]bool) string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, string(k))
	}
	sort.Strings(out)
	return strings.Join(out, ", ")
}
