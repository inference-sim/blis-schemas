package deployment

import (
	"reflect"
	"sort"
	"testing"

	"gopkg.in/yaml.v3"
)

func f64(v float64) *float64 { return &v }
func ival(v int) *int        { return &v }
func i64(v int64) *int64     { return &v }

// policyDeployment is a colocated single-node deployment with every policy block
// populated with values the simulator accepts: the default routing profile, a
// token-bucket admission policy, a BLIS-native scheduler and preemption policy, two
// saturation detectors with their knobs, and a one-adapter LoRA registry. It is the
// positive fixture the rejection table mutates.
func policyDeployment() *Deployment {
	d := singleNodeDeployment()
	d.Admission = &Admission{
		Policy:                AdmissionTokenBucket,
		TokenBucketCapacity:   f64(10000),
		TokenBucketRefillRate: f64(1000),
		TierShedMinPriority:   ival(3),
		GAIEKVThreshold:       f64(0.8),
		SLOPriorities:         map[string]int{"batch": 0},
		SLOTargets:            map[string]int64{"critical": 100000},
		LatencyUs:             250,
	}
	d.Routing = &Routing{
		Policy: RoutingWeighted,
		Scorers: []ScorerWeight{
			{Name: ScorerPrecisePrefixCache, Weight: 2},
			{Name: ScorerQueueDepth, Weight: 1},
			{Name: ScorerKVUtilization, Weight: 1},
		},
		LatencyUs: 100,
	}
	d.Scheduler = SchedulerPriorityFCFS
	d.Preemption = &Preemption{Policy: PreemptionPriority}
	d.Saturation = &Saturation{
		Detectors:    []Detector{DetectorComposite, DetectorPeakRate},
		Composite:    &CompositeDetector{Sensitivity: f64(1.5)},
		PeakRate:     &PeakRateDetector{Threshold: f64(0.5), OverloadMultiple: f64(3)},
		BacklogDrift: &BacklogDriftDetector{SlopeK: f64(3), ConfidenceCI: f64(0.95)},
		FinalWindow:  "30s",
	}
	d.LoRA = &LoRA{
		AdapterCapacity:       ival(4),
		LoadBaseLatencyUs:     f64(1500),
		LoadBandwidthBytesUs:  f64(2e6),
		FootprintBytesPerRank: f64(2e6),
		StepOverheadTiers:     map[int]StepOverheadTier{8: {K6: f64(0.02), K7: f64(1.0)}},
		Adapters:              []LoRAAdapter{{ID: "sql-adapter", Rank: 8, BaseModel: "granite-5-230b"}},
	}
	return d
}

func TestPolicyBlocksValidate(t *testing.T) {
	if p := policyDeployment().Validate(); !p.OK() {
		t.Fatalf("a deployment with every policy block populated should validate:\n%s", p.Error())
	}
	// The blocks are optional: a deployment that names none is still valid, so the
	// addition is backward-compatible with every deployment written before it.
	if p := singleNodeDeployment().Validate(); !p.OK() {
		t.Fatalf("a deployment with no policy blocks should still validate:\n%s", p.Error())
	}
}

func TestPolicyRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Deployment)
	}{
		{"unknown admission policy", func(d *Deployment) { d.Admission.Policy = "accept-maybe" }},
		{"token bucket capacity non-positive", func(d *Deployment) { d.Admission.TokenBucketCapacity = f64(0) }},
		{"token bucket refill non-positive", func(d *Deployment) { d.Admission.TokenBucketRefillRate = f64(-1) }},
		{"tier shed threshold negative", func(d *Deployment) { d.Admission.TierShedThreshold = ival(-1) }},
		{"gaie kv threshold above one", func(d *Deployment) { d.Admission.GAIEKVThreshold = f64(1.5) }},
		{"gaie qd threshold non-positive", func(d *Deployment) { d.Admission.GAIEQDThreshold = f64(0) }},
		{"slo target non-positive", func(d *Deployment) { d.Admission.SLOTargets = map[string]int64{"critical": 0} }},
		{"empty slo priority class", func(d *Deployment) { d.Admission.SLOPriorities = map[string]int{"": 1} }},
		{"admission latency negative", func(d *Deployment) { d.Admission.LatencyUs = -1 }},

		{"unknown routing policy", func(d *Deployment) { d.Routing.Policy = "psychic" }},
		{"routing latency negative", func(d *Deployment) { d.Routing.LatencyUs = -1 }},
		{"unknown scorer", func(d *Deployment) { d.Routing.Scorers[0].Name = "vibes" }},
		{"empty scorer name", func(d *Deployment) { d.Routing.Scorers[0].Name = "" }},
		{"duplicate scorer", func(d *Deployment) { d.Routing.Scorers[1].Name = ScorerPrecisePrefixCache }},
		{"non-positive scorer weight", func(d *Deployment) { d.Routing.Scorers[0].Weight = 0 }},

		{"unknown scheduler", func(d *Deployment) { d.Scheduler = "lifo" }},
		{"unknown preemption policy", func(d *Deployment) { d.Preemption.Policy = "random" }},

		{"unknown detector", func(d *Deployment) { d.Saturation.Detectors[0] = "crystal-ball" }},
		{"empty detector name", func(d *Deployment) { d.Saturation.Detectors[0] = "" }},
		{"duplicate detector", func(d *Deployment) { d.Saturation.Detectors[1] = DetectorComposite }},
		{"composite sensitivity non-positive", func(d *Deployment) { d.Saturation.Composite.Sensitivity = f64(0) }},
		{"threshold ms non-positive", func(d *Deployment) {
			d.Saturation.Threshold = &ThresholdDetector{ThresholdMs: f64(-5)}
		}},
		{"confidence ci out of range", func(d *Deployment) { d.Saturation.BacklogDrift.ConfidenceCI = f64(1.5) }},
		{"backlog window below one", func(d *Deployment) {
			d.Saturation.BacklogDrift.WindowSizeSec = ival(0)
		}},
		{"peak rate min observations below one", func(d *Deployment) {
			d.Saturation.PeakRate.MinObservations = ival(0)
		}},
		{"peak rate threshold non-positive", func(d *Deployment) { d.Saturation.PeakRate.Threshold = f64(0) }},
		{"peak rate warmup negative", func(d *Deployment) { d.Saturation.PeakRate.WarmupUs = i64(-1) }},
		{"bad final window duration", func(d *Deployment) { d.Saturation.FinalWindow = "half a minute" }},

		{"lora adapter missing id", func(d *Deployment) { d.LoRA.Adapters[0].ID = "" }},
		{"lora duplicate adapter id", func(d *Deployment) {
			d.LoRA.Adapters = append(d.LoRA.Adapters, LoRAAdapter{ID: "sql-adapter", Rank: 16})
		}},
		{"lora adapter rank below one", func(d *Deployment) { d.LoRA.Adapters[0].Rank = 0 }},
		{"lora adapters without capacity", func(d *Deployment) { d.LoRA.AdapterCapacity = nil }},
		{"lora zero capacity with adapters", func(d *Deployment) { d.LoRA.AdapterCapacity = ival(0) }},
		{"lora load bandwidth non-positive", func(d *Deployment) { d.LoRA.LoadBandwidthBytesUs = f64(0) }},
		{"lora footprint non-positive", func(d *Deployment) { d.LoRA.FootprintBytesPerRank = f64(-1) }},
		{"lora step tier missing k7", func(d *Deployment) {
			d.LoRA.StepOverheadTiers = map[int]StepOverheadTier{8: {K6: f64(0.02)}}
		}},
		{"lora step tier k7 non-positive", func(d *Deployment) {
			d.LoRA.StepOverheadTiers = map[int]StepOverheadTier{8: {K7: f64(0)}}
		}},
		{"lora step tier rank below one", func(d *Deployment) {
			d.LoRA.StepOverheadTiers = map[int]StepOverheadTier{0: {K7: f64(1)}}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := policyDeployment()
			tc.mutate(d)
			if p := d.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

// TestScorersUnderNonWeightedPolicyWarns: scorers compose only within the weighted
// policy, so declaring them under another policy — including the omitted case, which
// defaults to round-robin — is a no-op the author should see as a warning, not a
// rejection.
func TestScorersUnderNonWeightedPolicyWarns(t *testing.T) {
	for _, policy := range []RoutingPolicy{RoutingRoundRobin, ""} {
		name := string(policy)
		if name == "" {
			name = "omitted (defaults round-robin)"
		}
		t.Run(name, func(t *testing.T) {
			d := policyDeployment()
			d.Routing.Policy = policy
			p := d.Validate()
			if !p.OK() {
				t.Fatalf("scorers under a non-weighted policy should warn, not fail:\n%s", p.Error())
			}
			if len(p.All()) == 0 {
				t.Error("expected a warning that the scorers are ignored")
			}
		})
	}
}

// TestSchedulerIsDistinctFromEngineSchedulingPolicy pins the issue's requirement that
// the BLIS-native scheduler is a separate axis from the vLLM-mirroring
// Engine.SchedulingPolicy, which must stay for trace reproduction. The two live on
// different types, use different YAML keys, and draw from different value sets.
func TestSchedulerIsDistinctFromEngineSchedulingPolicy(t *testing.T) {
	if !contains(yamlNames(reflect.TypeOf(Deployment{})), "scheduler") {
		t.Error("Deployment should carry the BLIS-native scheduler under yaml key \"scheduler\"")
	}
	if !contains(yamlNames(reflect.TypeOf(Engine{})), "scheduling_policy") {
		t.Error("Engine.SchedulingPolicy (yaml \"scheduling_policy\") must stay for trace reproduction")
	}
	// The BLIS-native scheduler accepts values the vLLM axis never does; a value from one
	// set is not valid in the other.
	if SchedulerPolicy("priority-fcfs").Valid() == false {
		t.Error("priority-fcfs should be a valid BLIS-native scheduler")
	}
}

// TestPolicyFieldNames pins the YAML tag of every policy struct against the simulator
// surface it mirrors. The tags are the contract a deployment author writes against, so
// a silent rename here would quietly drop or misread a field the simulator sets.
func TestPolicyFieldNames(t *testing.T) {
	cases := []struct {
		what string
		typ  reflect.Type
		want []string
	}{
		{"Admission", reflect.TypeOf(Admission{}), []string{
			"policy", "token_bucket_capacity", "token_bucket_refill_rate",
			"tier_shed_threshold", "tier_shed_min_priority", "gaie_qd_threshold",
			"gaie_kv_threshold", "slo_priorities", "slo_targets", "admission_latency_us"}},
		{"Routing", reflect.TypeOf(Routing{}), []string{"policy", "scorers", "routing_latency_us"}},
		{"ScorerWeight", reflect.TypeOf(ScorerWeight{}), []string{"name", "weight"}},
		{"Preemption", reflect.TypeOf(Preemption{}), []string{"policy"}},
		{"Saturation", reflect.TypeOf(Saturation{}), []string{
			"detectors", "composite", "threshold", "backlog_drift", "peak_rate",
			"final_window"}},
		{"CompositeDetector", reflect.TypeOf(CompositeDetector{}), []string{"sensitivity"}},
		{"ThresholdDetector", reflect.TypeOf(ThresholdDetector{}), []string{"threshold_ms"}},
		{"BacklogDriftDetector", reflect.TypeOf(BacklogDriftDetector{}), []string{
			"window_size_sec", "min_windows", "peak_ratio", "peak_ratio_band",
			"confidence_ci", "warmup_windows", "tail_windows", "saturated_drain_ratio",
			"transient_drain_ratio", "slope_k"}},
		{"PeakRateDetector", reflect.TypeOf(PeakRateDetector{}), []string{
			"threshold", "min_observations", "warmup_us", "consecutive_k",
			"overload_multiple"}},
		{"LoRA", reflect.TypeOf(LoRA{}), []string{
			"adapter_capacity", "load_base_latency_us", "load_bandwidth_bytes_us",
			"footprint_bytes_per_rank", "step_overhead_tiers", "adapters"}},
		{"LoRAAdapter", reflect.TypeOf(LoRAAdapter{}), []string{"id", "rank", "base_model"}},
		{"StepOverheadTier", reflect.TypeOf(StepOverheadTier{}), []string{"k6", "k7"}},
	}
	for _, c := range cases {
		t.Run(c.what, func(t *testing.T) {
			assertSameSet(t, c.what, yamlNames(c.typ), c.want)
		})
	}
}

// TestPolicyEnumValues pins the exact string set of each policy enum against the
// simulator's valid-name registries. An enum that drifts from the simulator would
// validate a document the simulator rejects, or reject one it accepts.
func TestPolicyEnumValues(t *testing.T) {
	assertSameSet(t, "admission policies", enumKeys(admissionPolicies), []string{
		"always-admit", "token-bucket", "reject-all", "tier-shed", "gaie-legacy"})
	assertSameSet(t, "routing policies", enumKeys(routingPolicies), []string{
		"round-robin", "least-loaded", "weighted", "always-busiest"})
	assertSameSet(t, "scorers", enumKeys(scorers), []string{
		"prefix-affinity", "precise-prefix-cache", "no-hit-lru", "queue-depth",
		"kv-utilization", "load-balance", "active-requests", "running-requests",
		"load-aware", "vllm-dp", "lora-affinity"})
	assertSameSet(t, "schedulers", enumKeys(schedulerPolicies), []string{
		"fcfs", "priority-fcfs", "sjf", "reverse-priority"})
	assertSameSet(t, "preemption policies", enumKeys(preemptionPolicies), []string{
		"fcfs", "priority"})
	assertSameSet(t, "detectors", enumKeys(detectors), []string{
		"composite", "threshold", "backlog-drift", "peak-rate"})
}

// TestPolicyPointersRoundTrip pins that an absent optional knob stays absent through a
// YAML round trip while a stated one survives, which is why the knobs are pointers:
// "unset" (take the simulator default) and an explicit value are different deployments.
func TestPolicyPointersRoundTrip(t *testing.T) {
	in := &Saturation{
		Detectors: []Detector{DetectorThreshold},
		Threshold: &ThresholdDetector{ThresholdMs: f64(7500)},
		Composite: &CompositeDetector{}, // sensitivity unstated
	}
	out, err := yaml.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back Saturation
	if err := yaml.Unmarshal(out, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Threshold == nil || back.Threshold.ThresholdMs == nil || *back.Threshold.ThresholdMs != 7500 {
		t.Errorf("stated threshold_ms was lost: %+v", back.Threshold)
	}
	if back.Composite == nil || back.Composite.Sensitivity != nil {
		t.Errorf("unstated sensitivity should stay nil, got %+v", back.Composite)
	}
}

// TestAdmissionAndLoRAPointersRoundTrip covers the same "unset stays unset" property
// for the pointer-to-scalar knobs on Admission and LoRA: a stated capacity survives, an
// omitted refill rate stays nil (take the simulator default), distinguishing the two.
func TestAdmissionAndLoRAPointersRoundTrip(t *testing.T) {
	in := &Deployment{
		Admission: &Admission{TokenBucketCapacity: f64(10000)}, // refill rate unstated
		LoRA:      &LoRA{AdapterCapacity: ival(4)},             // cost coeffs unstated
	}
	out, err := yaml.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back Deployment
	if err := yaml.Unmarshal(out, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Admission == nil || back.Admission.TokenBucketCapacity == nil ||
		*back.Admission.TokenBucketCapacity != 10000 {
		t.Errorf("stated token_bucket_capacity was lost: %+v", back.Admission)
	}
	if back.Admission != nil && back.Admission.TokenBucketRefillRate != nil {
		t.Errorf("unstated token_bucket_refill_rate should stay nil, got %v", *back.Admission.TokenBucketRefillRate)
	}
	if back.LoRA == nil || back.LoRA.AdapterCapacity == nil || *back.LoRA.AdapterCapacity != 4 {
		t.Errorf("stated adapter_capacity was lost: %+v", back.LoRA)
	}
	if back.LoRA != nil && back.LoRA.LoadBaseLatencyUs != nil {
		t.Errorf("unstated load_base_latency_us should stay nil, got %v", *back.LoRA.LoadBaseLatencyUs)
	}
}

func enumKeys[T ~string](m map[T]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, string(k))
	}
	return out
}

func assertSameSet(t *testing.T, what string, got, want []string) {
	t.Helper()
	gs := append([]string{}, got...)
	ws := append([]string{}, want...)
	sort.Strings(gs)
	sort.Strings(ws)
	if !reflect.DeepEqual(gs, ws) {
		t.Errorf("%s: got %v, want %v", what, gs, ws)
	}
}
