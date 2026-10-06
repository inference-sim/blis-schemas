package deployment

// This file adds the control-plane POLICY surface to a Deployment: the operator
// choices that decide how requests are admitted, routed, scheduled, preempted,
// watched for saturation, and served against LoRA adapters. They sit alongside the
// physical layout (pools, offload, PD transfer) in deployment.go, and like it they
// are the tunable half of a run — a sweeper or a lens varies them against a fixed
// Scenario.
//
// They are deployment config rather than scenario or kernel for the reason the kernel
// interface states: anything whose value depends on WHEN a request arrives — queue
// waiting, admission ordering, preemption — is the simulator's concern, not the cost
// model's. That is why they are absent from a Scenario and why a cost model never
// reads them; they are operator choices, and other projects (sweeper, lens) need the
// same shapes, so they live here once.
//
// Each block mirrors an existing BLIS simulator surface so that the "enums known"
// field validation matches exactly what the simulator accepts: admission, routing
// (with its weighted scorer pipeline), the BLIS-native scheduler, and preemption come
// from the `--policy-config` bundle; saturation from `--saturation-config` plus the
// `--detectors` selection; LoRA from `--lora-config`. The enumerated values and
// defaults are owned by the simulator (inference-sim) and transcribed here.
//
// These enums are BLIS-native and version-independent: the simulator's accepted sets
// (its validAdmissionPolicies, validRoutingPolicies, validSchedulers,
// validPreemptionPolicies, validScorerNames and the detector roster) are plain
// registries, not keyed by an engine release. So the policy surface is field-validated
// here rather than in the engine-version rules pack, which gates only the per-pool vLLM
// Engine knobs (backends, cache dtypes, scheduling_policy) that move between releases.
//
// The BLIS-native scheduler here is DISTINCT from Engine.SchedulingPolicy in
// deployment.go. Engine.SchedulingPolicy mirrors vLLM's own --scheduling-policy
// (fcfs/priority) and exists to reproduce a trace faithfully; the Scheduler field
// below selects BLIS's own instance-queue ordering (fcfs/priority-fcfs/sjf/
// reverse-priority), which has no vLLM counterpart. Both are kept.
//
// The flow-control admission subsystem is NOT modeled here. It is a separate surface
// the simulator toggles with --flow-control — carrying its own saturation detector,
// request dispatch ordering, fairness policy and gateway-queue bounds in
// DeploymentConfig, outside the --policy-config bundle this file mirrors — and is a
// candidate for a later block. The SLO priority and target maps on Admission below are
// NOT part of that subsystem: they are fields of the admission bundle itself, which
// tier-shed admission and priority preemption read, so they are included here.

// --- Admission ------------------------------------------------------------------

// Admission is the first cluster gate: whether a request is accepted before it reaches
// routing. It mirrors the policy bundle's `admission` block. Every parameter beyond the
// policy name is a pointer so an absent key keeps the simulator's default rather than
// forcing a zero, which is how the simulator's own AdmissionConfig distinguishes unset
// from a deliberately-zero value.
type Admission struct {
	// Policy selects the admission behaviour. Empty takes the simulator default
	// (always-admit).
	Policy AdmissionPolicy `yaml:"policy,omitempty"`

	// Token-bucket parameters, used when Policy is token-bucket. Capacity is the
	// bucket's maximum token count (default 10000); RefillRate is tokens added per
	// second of simulated time (default 1000). Each request costs its input-token
	// count.
	TokenBucketCapacity   *float64 `yaml:"token_bucket_capacity,omitempty"`
	TokenBucketRefillRate *float64 `yaml:"token_bucket_refill_rate,omitempty"`

	// Tier-shed parameters, used when Policy is tier-shed: requests whose SLO-tier
	// priority is below the minimum are shed under overload. MinPriority defaults to 3;
	// Threshold defaults to 0.
	TierShedThreshold   *int `yaml:"tier_shed_threshold,omitempty"`
	TierShedMinPriority *int `yaml:"tier_shed_min_priority,omitempty"`

	// GAIE-legacy parameters, used when Policy is gaie-legacy: sheddable requests are
	// rejected once pool-average saturation crosses these thresholds. QD defaults to 5
	// (queue depth); KV defaults to 0.8 (KV-cache utilization fraction).
	GAIEQDThreshold *float64 `yaml:"gaie_qd_threshold,omitempty"`
	GAIEKVThreshold *float64 `yaml:"gaie_kv_threshold,omitempty"`

	// SLOPriorities overrides the default SLO-class-to-priority mapping. Absent takes
	// the GAIE defaults (critical=4, standard=3, batch=-1, sheddable=-2, background=-3);
	// a negative priority is sheddable.
	SLOPriorities map[string]int `yaml:"slo_priorities,omitempty"`
	// SLOTargets sets per-SLO-class TTFT targets in microseconds. It is a field of the
	// admission bundle, read by the simulator's SLO-aware ordering.
	SLOTargets map[string]int64 `yaml:"slo_targets,omitempty"`

	// LatencyUs is the fixed latency in microseconds the admission stage adds to each
	// request, mirroring --admission-latency. Zero (the default) adds none.
	LatencyUs int64 `yaml:"admission_latency_us,omitempty"`
}

// AdmissionPolicy is the admission behaviour. The set is the simulator's
// validAdmissionPolicies.
type AdmissionPolicy string

const (
	AdmissionAlwaysAdmit AdmissionPolicy = "always-admit"
	AdmissionTokenBucket AdmissionPolicy = "token-bucket"
	AdmissionRejectAll   AdmissionPolicy = "reject-all"
	AdmissionTierShed    AdmissionPolicy = "tier-shed"
	AdmissionGAIELegacy  AdmissionPolicy = "gaie-legacy"
)

var admissionPolicies = map[AdmissionPolicy]bool{
	AdmissionAlwaysAdmit: true, AdmissionTokenBucket: true, AdmissionRejectAll: true,
	AdmissionTierShed: true, AdmissionGAIELegacy: true,
}

// Valid reports whether a is a recognized admission policy.
func (a AdmissionPolicy) Valid() bool { return admissionPolicies[a] }

// --- Routing --------------------------------------------------------------------

// Routing selects how the cluster distributes an admitted request across instances. It
// mirrors the policy bundle's `routing` block. Scorers are read only by the weighted
// policy, where they compose into a single weighted score; the default profile is
// precise-prefix-cache:2, queue-depth:1, kv-utilization:1.
type Routing struct {
	// Policy selects the routing behaviour. Empty takes the simulator default
	// (round-robin).
	Policy RoutingPolicy `yaml:"policy,omitempty"`
	// Scorers is the weighted-pipeline scorer-weight list, mirroring the bundle's list
	// form. It is consulted only by the weighted policy.
	Scorers []ScorerWeight `yaml:"scorers,omitempty"`

	// LatencyUs is the fixed latency in microseconds the routing stage adds to each
	// request, mirroring --routing-latency. Zero (the default) adds none.
	LatencyUs int64 `yaml:"routing_latency_us,omitempty"`
}

// ScorerWeight is one entry in the weighted routing pipeline: a scorer name and its
// relative weight. Weights are relative — the simulator normalizes them to sum to one —
// so only their ratios matter.
type ScorerWeight struct {
	Name   Scorer  `yaml:"name"`
	Weight float64 `yaml:"weight"`
}

// RoutingPolicy is the routing behaviour. The set is the simulator's
// validRoutingPolicies.
type RoutingPolicy string

const (
	RoutingRoundRobin    RoutingPolicy = "round-robin"
	RoutingLeastLoaded   RoutingPolicy = "least-loaded"
	RoutingWeighted      RoutingPolicy = "weighted"
	RoutingAlwaysBusiest RoutingPolicy = "always-busiest"
)

var routingPolicies = map[RoutingPolicy]bool{
	RoutingRoundRobin: true, RoutingLeastLoaded: true, RoutingWeighted: true,
	RoutingAlwaysBusiest: true,
}

// Valid reports whether r is a recognized routing policy.
func (r RoutingPolicy) Valid() bool { return routingPolicies[r] }

// Scorer is a routing-pipeline scorer name. The set is the simulator's
// validScorerNames.
type Scorer string

const (
	ScorerPrefixAffinity     Scorer = "prefix-affinity"
	ScorerPrecisePrefixCache Scorer = "precise-prefix-cache"
	ScorerNoHitLRU           Scorer = "no-hit-lru"
	ScorerQueueDepth         Scorer = "queue-depth"
	ScorerKVUtilization      Scorer = "kv-utilization"
	ScorerLoadBalance        Scorer = "load-balance"
	ScorerActiveRequests     Scorer = "active-requests"
	ScorerRunningRequests    Scorer = "running-requests"
	ScorerLoadAware          Scorer = "load-aware"
	ScorerVLLMDP             Scorer = "vllm-dp"
	ScorerLoRAAffinity       Scorer = "lora-affinity"
)

var scorers = map[Scorer]bool{
	ScorerPrefixAffinity: true, ScorerPrecisePrefixCache: true, ScorerNoHitLRU: true,
	ScorerQueueDepth: true, ScorerKVUtilization: true, ScorerLoadBalance: true,
	ScorerActiveRequests: true, ScorerRunningRequests: true, ScorerLoadAware: true,
	ScorerVLLMDP: true, ScorerLoRAAffinity: true,
}

// Valid reports whether s is a recognized scorer.
func (s Scorer) Valid() bool { return scorers[s] }

// --- Scheduler (BLIS-native) ----------------------------------------------------

// SchedulerPolicy is the BLIS-native instance-queue ordering, selected by the bundle's
// top-level `scheduler` scalar. The set is the simulator's validSchedulers. It is
// distinct from Engine.SchedulingPolicy, which mirrors vLLM's own flag for trace
// reproduction; this one has no vLLM counterpart.
type SchedulerPolicy string

const (
	SchedulerFCFS            SchedulerPolicy = "fcfs"
	SchedulerPriorityFCFS    SchedulerPolicy = "priority-fcfs"
	SchedulerSJF             SchedulerPolicy = "sjf"
	SchedulerReversePriority SchedulerPolicy = "reverse-priority"
)

var schedulerPolicies = map[SchedulerPolicy]bool{
	SchedulerFCFS: true, SchedulerPriorityFCFS: true, SchedulerSJF: true,
	SchedulerReversePriority: true,
}

// Valid reports whether s is a recognized BLIS-native scheduler.
func (s SchedulerPolicy) Valid() bool { return schedulerPolicies[s] }

// --- Preemption -----------------------------------------------------------------

// Preemption selects how a running request is chosen for eviction when a batch must
// make room. It mirrors the policy bundle's `preemption` block. fcfs evicts the tail of
// the batch; priority evicts the least-urgent SLO tier first (the shared SLO priority
// map, overridable via Admission.SLOPriorities, decides urgency).
type Preemption struct {
	// Policy selects the victim-selection behaviour. Empty takes the simulator default
	// (fcfs).
	Policy PreemptionPolicy `yaml:"policy,omitempty"`
}

// PreemptionPolicy is the victim-selection behaviour. The set is the simulator's
// validPreemptionPolicies.
type PreemptionPolicy string

const (
	PreemptionFCFS     PreemptionPolicy = "fcfs"
	PreemptionPriority PreemptionPolicy = "priority"
)

var preemptionPolicies = map[PreemptionPolicy]bool{
	PreemptionFCFS: true, PreemptionPriority: true,
}

// Valid reports whether p is a recognized preemption policy.
func (p PreemptionPolicy) Valid() bool { return preemptionPolicies[p] }

// --- Saturation detectors -------------------------------------------------------

// Saturation configures the post-hoc saturation detectors (distinct from the real-time
// flow-control detector). Detectors names the roster to run — each must be calibrated to
// a common false-alarm rate before its score is comparable, which is why every detector
// carries a tuning knob. The per-detector blocks mirror the `--saturation-config` file;
// every knob is a pointer so an absent one keeps the detector's campaign-validated
// default.
type Saturation struct {
	// Detectors is the roster to run, mirroring the `--detectors` selection. Each name
	// must be a recognized detector.
	Detectors []Detector `yaml:"detectors,omitempty"`

	Composite    *CompositeDetector    `yaml:"composite,omitempty"`
	Threshold    *ThresholdDetector    `yaml:"threshold,omitempty"`
	BacklogDrift *BacklogDriftDetector `yaml:"backlog_drift,omitempty"`
	PeakRate     *PeakRateDetector     `yaml:"peak_rate,omitempty"`

	// FinalWindow is the trailing window the final plurality vote reduces over, as a Go
	// duration (e.g. "30s"). Empty takes the simulator default (30s).
	FinalWindow string `yaml:"final_window,omitempty"`
}

// Detector is a post-hoc saturation detector name. The set is the simulator's detector
// roster.
type Detector string

const (
	DetectorComposite    Detector = "composite"
	DetectorThreshold    Detector = "threshold"
	DetectorBacklogDrift Detector = "backlog-drift"
	DetectorPeakRate     Detector = "peak-rate"
)

var detectors = map[Detector]bool{
	DetectorComposite: true, DetectorThreshold: true, DetectorBacklogDrift: true,
	DetectorPeakRate: true,
}

// Valid reports whether d is a recognized detector.
func (d Detector) Valid() bool { return detectors[d] }

// CompositeDetector tunes the composite detector. Sensitivity scales its noise floor —
// larger fires less; the simulator default is 1.0.
type CompositeDetector struct {
	Sensitivity *float64 `yaml:"sensitivity,omitempty"`
}

// ThresholdDetector tunes the threshold detector. ThresholdMs is the mean-E2E threshold
// in milliseconds; the simulator default is 5000.
type ThresholdDetector struct {
	ThresholdMs *float64 `yaml:"threshold_ms,omitempty"`
}

// BacklogDriftDetector tunes the backlog-drift detector. SlopeK is its false-alarm
// calibration knob (default 3.0); the remaining fields mirror the simulator's
// BacklogDriftConfig, each defaulting when absent.
type BacklogDriftDetector struct {
	WindowSizeSec       *int     `yaml:"window_size_sec,omitempty"`
	MinWindows          *int     `yaml:"min_windows,omitempty"`
	PeakRatio           *float64 `yaml:"peak_ratio,omitempty"`
	PeakRatioBand       *float64 `yaml:"peak_ratio_band,omitempty"`
	ConfidenceCI        *float64 `yaml:"confidence_ci,omitempty"`
	WarmupWindows       *int     `yaml:"warmup_windows,omitempty"`
	TailWindows         *int     `yaml:"tail_windows,omitempty"`
	SaturatedDrainRatio *float64 `yaml:"saturated_drain_ratio,omitempty"`
	TransientDrainRatio *float64 `yaml:"transient_drain_ratio,omitempty"`
	SlopeK              *float64 `yaml:"slope_k,omitempty"`
}

// PeakRateDetector tunes the peak-rate detector. Threshold is its false-alarm knob in
// backlog per second (default 0.5); the band threshold < R <= OverloadMultiple*threshold
// separates BACKLOGGED from OVERLOADED.
type PeakRateDetector struct {
	Threshold        *float64 `yaml:"threshold,omitempty"`
	MinObservations  *int     `yaml:"min_observations,omitempty"`
	WarmupUs         *int64   `yaml:"warmup_us,omitempty"`
	ConsecutiveK     *int     `yaml:"consecutive_k,omitempty"`
	OverloadMultiple *float64 `yaml:"overload_multiple,omitempty"`
}

// --- LoRA -----------------------------------------------------------------------

// LoRA configures the LoRA adapter subsystem, mirroring the `--lora-config` file. The
// subsystem is inert unless AdapterCapacity is set to a positive slot count, so a
// deployment that declares adapters must also declare the capacity to hold them. The
// cost coefficients price adapter load and the per-step overhead their presence adds.
type LoRA struct {
	// AdapterCapacity is the number of adapters that may be resident on one instance at
	// once. Absent (nil) leaves the subsystem inert; declaring adapters requires a
	// positive capacity.
	AdapterCapacity *int `yaml:"adapter_capacity,omitempty"`

	// Load-cost coefficients. LoadBaseLatencyUs is the fixed per-load latency in
	// microseconds (default 1500); LoadBandwidthBytesUs divides adapter bytes to a
	// transfer time, so it must be positive (default 2e6); FootprintBytesPerRank sizes an
	// adapter's resident bytes per rank (default 2e6).
	LoadBaseLatencyUs     *float64 `yaml:"load_base_latency_us,omitempty"`
	LoadBandwidthBytesUs  *float64 `yaml:"load_bandwidth_bytes_us,omitempty"`
	FootprintBytesPerRank *float64 `yaml:"footprint_bytes_per_rank,omitempty"`

	// StepOverheadTiers maps an adapter rank to the per-step overhead coefficients that
	// its presence in a batch adds. The simulator defaults cover ranks 8, 16 and 32.
	StepOverheadTiers map[int]StepOverheadTier `yaml:"step_overhead_tiers,omitempty"`

	// Adapters is the registry of adapters a deployment serves. Empty leaves the
	// subsystem inert.
	Adapters []LoRAAdapter `yaml:"adapters,omitempty"`
}

// LoRAAdapter is one registry entry: a unique id, a positive rank, and an optional base
// model it adapts.
type LoRAAdapter struct {
	ID        string `yaml:"id"`
	Rank      int    `yaml:"rank"`
	BaseModel string `yaml:"base_model,omitempty"`
}

// StepOverheadTier is the per-step overhead coefficients for one rank tier. K6 is the
// linear coefficient (default 0); K7 is the per-tier normalization denominator and must
// be positive when the tier is declared.
type StepOverheadTier struct {
	K6 *float64 `yaml:"k6,omitempty"`
	K7 *float64 `yaml:"k7,omitempty"`
}
