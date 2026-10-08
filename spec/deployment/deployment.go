// Package deployment describes the mutable configuration a user — or an optimizer
// sweeping a configuration space — chooses against a fixed Scenario: how a model is
// laid out across the available hardware (pools, each a role at a parallelism layout
// with its own engine settings), the KV offload hierarchy below HBM, and
// prefill-to-decode KV transfer.
//
// It is the tunable half of what a single `blis run` consumes. The immutable problem
// — the model, the workload (a distributional shape or a concrete trace reference), the
// available-hardware inventory, and the coefficient/engine-version references that
// identify what a prediction is fitted against — lives in a Scenario. A Scenario and a
// Deployment together describe one run; the operating point (load level) is a run-level
// sweep axis that is neither.
//
// Three rules shape the schema, each preventing a class of error rather than
// catching it later:
//
// Derived quantities have no field. Expert-parallel width is tensor-parallel times
// the larger of prefill-context-parallel and data-parallel width; an engine builds
// the group that way and rejects the combination that would make the product
// ambiguous. A deployment that could state EP independently could describe a layout
// no engine would run, so the field does not exist.
//
// Requested settings are distinct from resolved ones. A collective backend and an
// async-scheduling preference are requests: crossing a node boundary without
// multi-node NVLink disqualifies an SM-consuming all-reduce kernel whatever the
// field says, and async scheduling has several disqualifying conditions. The
// resolver reports what it resolved, so a reader is never misled by a request.
//
// Engine settings are per pool. A prefill/decode-disaggregated deployment runs two
// kinds of engine with different backends and different graph modes; one shared
// engine block would describe neither.
package deployment

import "math"

// Deployment is the tunable configuration applied to a Scenario: the pools that lay
// the model out, the offload hierarchy, and prefill-to-decode transfer.
type Deployment struct {
	Kind string `yaml:"kind"` // "Deployment"
	Name string `yaml:"name"`

	Pools []Pool `yaml:"pools"`

	Offload    *Offload    `yaml:"offload,omitempty"`
	PDTransfer *PDTransfer `yaml:"pd_transfer,omitempty"`
}

// Role distinguishes the engines of a disaggregated deployment.
type Role string

const (
	RoleColocated Role = "colocated"
	RolePrefill   Role = "prefill"
	RoleDecode    Role = "decode"
)

var roles = map[Role]bool{RoleColocated: true, RolePrefill: true, RoleDecode: true}

// Valid reports whether r is a recognized role.
func (r Role) Valid() bool { return roles[r] }

// Pool is the nodes serving one role: a role, a node count, a parallelism layout, and
// settings. It may hold more than one engine, each running that layout with those
// settings — llm-d runs a LeaderWorkerSet of `replicas` groups, each group one engine
// over `size` nodes, so a 36-node prefill pool of dp 8 is 36 one-node engines. Nodes is
// therefore the pool's whole extent, which one engine's layout may occupy only part of.
type Pool struct {
	Role     Role        `yaml:"role"`
	Nodes    int         `yaml:"nodes"`
	Parallel Parallelism `yaml:"parallel"`
	Engine   Engine      `yaml:"engine"`
}

// Parallelism is the layout. EP is absent: see the package comment.
type Parallelism struct {
	TP int `yaml:"tp"`
	PP int `yaml:"pp"`
	DP int `yaml:"dp"`
	// DPLocal is the data-parallel rank count resident on one node. It must divide
	// the node's GPU count.
	DPLocal int `yaml:"dp_local,omitempty"`

	EnableExpertParallel bool `yaml:"enable_expert_parallel"`

	// PCP and DCP are prefill- and decode-context parallel width: sharding one
	// sequence across ranks. DCP is the axis that shards KV; TP shards it only
	// through KV-head count.
	PCP int `yaml:"pcp,omitempty"`
	DCP int `yaml:"dcp,omitempty"`
}

// ExpertParallelWidth returns the derived EP degree, or 1 when expert parallelism
// is off. It is a function rather than a field precisely so no file can disagree
// with it.
//
// The width is tp x pcp x dp: the engine lays ranks out DP x PP x PCP x TP and builds
// the expert group over the DP, PCP and TP axes together (initialize_model_parallel,
// the same at v0.29.0 and later). An unset or non-positive width counts as one, the
// same floor every other reader of these fields applies.
//
// An earlier form was tp x max(dp, pcp). That agrees with the product exactly when one
// of dp and pcp is at most 1, which was every deployment this schema admitted while
// prefill-context parallelism and data parallelism could not be combined — so the
// product changes no width that could previously validate. It differs once both exceed
// 1, which the engine supports from e6dc16cebd and llm-d deploys (DP 4 x PCP 8 is an
// expert group of 32, not 8).
//
// The product saturates at the largest int instead of wrapping, so an absurd layout
// reads as an absurdly wide group rather than a small or negative one.
func (p Parallelism) ExpertParallelWidth() int {
	if !p.EnableExpertParallel {
		return 1
	}
	width := 1
	for _, w := range []int{p.TP, p.PCP, p.DP} {
		if w < 1 {
			w = 1
		}
		if width > math.MaxInt/w {
			return math.MaxInt
		}
		width *= w
	}
	return width
}

// Engine is the per-pool settings that change step time or occupancy. Values are
// held as declared, including the tri-state cases where a nil pointer means "let
// the engine decide" and the resolver reports what it decided.
type Engine struct {
	// All2AllBackend selects the MoE dispatch/combine implementation. It sets both
	// the volume basis and the cross-node scaling: an all-gather-family backend
	// moves dense per-token bytes in two ring phases, where a routed backend moves
	// top-k-selected bytes point to point.
	All2AllBackend string `yaml:"all2all_backend,omitempty"`
	// DisableCustomAllReduce mirrors the engine's own boolean. The SM-consuming
	// all-reduce kernel is used when this is false AND the layout permits it:
	// crossing a node boundary without multi-node NVLink disqualifies it whatever
	// the field says, moving the cost from SMs to the NIC. So this is a REQUEST, and
	// the resolver reports what will actually run.
	//
	// Tri-state: nil takes the engine default, which is false — the kernel is used
	// where reachable.
	DisableCustomAllReduce *bool `yaml:"disable_custom_all_reduce,omitempty"`

	// AllReduceBackend records the resolved choice where a caller wants to state it
	// directly rather than derive it. Values are "custom" and "nccl"; it is an
	// override, and a deployment that states both this and DisableCustomAllReduce in
	// conflict is rejected.
	AllReduceBackend string `yaml:"allreduce_backend,omitempty"`
	// CUDAGraphMode selects capture strategy, which sets the host launch term.
	CUDAGraphMode string `yaml:"cudagraph_mode,omitempty"`
	// AsyncScheduling is tri-state: nil lets the engine resolve it through its
	// disqualifying conditions, which is what a real deployment does.
	AsyncScheduling *bool `yaml:"async_scheduling,omitempty"`

	// DisableCascadeAttn follows the engine default of true: cascade attention is
	// opt-in. When enabled it changes the attention primitive itself, reading a
	// shared prefix once per batch rather than once per request.
	DisableCascadeAttn *bool `yaml:"disable_cascade_attn,omitempty"`

	// EnablePrefixCaching decides whether a request may skip recomputing a prefix
	// another request already placed in the cache. It changes the WORK a prefill does,
	// not only the memory it occupies: with caching on, a matched prefix arrives as
	// already-computed tokens and only the remainder is charged, so the same prompt
	// costs a different number of scheduled tokens depending on this one field.
	//
	// A cost model cannot derive it. Whether a prefix hits depends on what else the
	// deployment served, so the engine's own setting is the only way to know whether
	// the hit was available at all.
	//
	// Tri-state: nil takes the engine default, which is TRUE in vLLM -- caching is on
	// unless disabled. A deployment that omits this therefore describes one WITH prefix
	// caching, which is why the field is a pointer: false and unstated are different
	// deployments, and a benchmark that passes --no-enable-prefix-caching cannot be
	// expressed by omission.
	//
	// The field name follows the engine's CLI flag (--enable-prefix-caching /
	// --no-enable-prefix-caching).
	EnablePrefixCaching *bool `yaml:"enable_prefix_caching,omitempty"`

	BlockSize           int `yaml:"block_size,omitempty"`
	MaxNumBatchedTokens int `yaml:"max_num_batched_tokens,omitempty"`
	MaxNumSeqs          int `yaml:"max_num_seqs,omitempty"`
	MaxModelLen         int `yaml:"max_model_len,omitempty"`

	// Quantization is the weight format the engine serves in, where it differs from
	// the format the checkpoint is stored in. A model whose config.json says bfloat16
	// can be served fp8, and the served format is what sets the weight-read bytes, the
	// compute peak and the GEMM efficiency envelope — so a deployment that omits this
	// when it set it is priced against the wrong three constants at once.
	//
	// Empty means the checkpoint's own format governs, which is the common case: a
	// checkpoint that already carries a quantization_config needs nothing here,
	// because the model graph records the format the weights are in.
	//
	// The field name follows the engine's CLI flag (--quantization).
	Quantization string `yaml:"quantization,omitempty"`

	// CacheDType is the KV cache format. With TP it is one of the two fields that
	// divide per-rank KV bytes, so a capacity answer is wrong without it.
	CacheDType string `yaml:"cache_dtype,omitempty"`
	// MambaCacheDType and MambaSSMCacheDType size a recurrent stack's state; the
	// second covers the SSM state alone where it differs from the convolutional one.
	MambaCacheDType    string `yaml:"mamba_cache_dtype,omitempty"`
	MambaSSMCacheDType string `yaml:"mamba_ssm_cache_dtype,omitempty"`
	// MambaCacheMode decides whether recurrent state is fixed per sequence or
	// proportional to the context bound — a difference of three orders of magnitude,
	// and the reason a memory interface needs both a fixed and a variable term.
	MambaCacheMode string `yaml:"mamba_cache_mode,omitempty"`

	GPUMemoryUtilization float64 `yaml:"gpu_memory_utilization,omitempty"`

	// SchedulingPolicy affects which requests join a batch and whether one is
	// preempted, not what a batch costs — so a cost model reads it only to reproduce
	// a trace faithfully, and a preempted request's recompute arrives as ordinary
	// scheduled tokens on a later step.
	//
	// The field name follows the engine's CLI flag (--scheduling-policy) rather than
	// its internal config field (policy), because a deployment is written by whoever
	// launched the engine.
	SchedulingPolicy string `yaml:"scheduling_policy,omitempty"`

	DBO         *DBO         `yaml:"dbo,omitempty"`
	EPLB        *EPLB        `yaml:"eplb,omitempty"`
	Speculative *Speculative `yaml:"speculative,omitempty"`

	// The decode-context-parallel knobs. Each changes what a DCP decode step costs, and
	// each is a REQUEST: the engine fills an unstated one with its default, and a model
	// may override the first two from its own configuration hook (set_dcp_defaults), so
	// the value that runs is the resolver's to report. They have no effect when dcp is 1.
	//
	// DCPCommBackend selects the collectives a DCP decode layer runs: "ag_rs" is a query
	// all-gather, an LSE all-gather and an output reduce-scatter; "a2a" replaces the last
	// two with one all-to-all. Empty takes the engine default, "ag_rs". Which names a
	// release accepts is that release's rules pack's to say.
	DCPCommBackend string `yaml:"dcp_comm_backend,omitempty"`
	// DCPQReplicate requests replicating the MLA query projection at load time, so each
	// rank materialises the full group-local head set and skips the query all-gather, at
	// the cost of computing the projection redundantly.
	//
	// Tri-state: nil lets the engine (or the model's hook) decide, which by default is
	// false.
	DCPQReplicate *bool `yaml:"dcp_q_replicate,omitempty"`
	// CPKVCacheInterleaveSize is how many consecutive tokens one DCP rank holds before
	// the next rank takes over, which sets each rank's share of a sequence: 1 stripes
	// token by token, block_size block by block.
	//
	// Zero means unstated, which is not the same request as 1 even though 1 is the
	// default. With a KV connector configured the engine pins the size to the block
	// size, and when it does depends on the release: v0.29.0 pins it under any KV
	// connector, stated or not; v0.31.0 only under NIXL, and only when unstated. So the
	// size that runs is the resolver's to report, and stating 1 is a different request
	// from stating nothing. The deprecated dcp_kv_cache_interleave_size it replaced is
	// deliberately not carried.
	CPKVCacheInterleaveSize int `yaml:"cp_kv_cache_interleave_size,omitempty"`
}

// DBO is dual-batch overlap: splitting a batch so one microbatch's communication
// hides under the other's compute. The thresholds are in total batch tokens, and
// which one applies turns on whether every request schedules the same token count.
//
// The YAML names match the engine's own flags (--enable-dbo,
// --dbo-decode-token-threshold) rather than being shortened by the nesting, because
// a deployment is written by whoever launched the engine and a renamed field makes
// them translate.
type DBO struct {
	Enabled               bool `yaml:"enable_dbo"`
	DecodeTokenThreshold  int  `yaml:"dbo_decode_token_threshold,omitempty"`
	PrefillTokenThreshold int  `yaml:"dbo_prefill_token_threshold,omitempty"`
}

// EPLB is expert-parallel load balancing. NumRedundantExperts is not cosmetic: with
// EPLB enabled the physical expert count must divide the EP width, so this field
// decides whether a layout is feasible at all.
type EPLB struct {
	Enabled             bool `yaml:"enable_eplb"`
	NumRedundantExperts int  `yaml:"num_redundant_experts,omitempty"`
	WindowSize          int  `yaml:"window_size,omitempty"`
	StepInterval        int  `yaml:"step_interval,omitempty"`
}

// Speculative is the draft configuration. The method name is engine-version data,
// validated by a rules pack rather than by this schema.
type Speculative struct {
	Method        string `yaml:"method"`
	NumSpecTokens int    `yaml:"num_spec_tokens"`
}

// Offload is the KV tier hierarchy below HBM.
type Offload struct {
	Tiers []Tier `yaml:"tiers"`
	// EvictionPolicy names the KV-offload cache policy. The engine registers its in-tree
	// policies with a CachePolicyFactory (lru, arc in v0.29.0) and admits out-of-tree ones
	// through a module path — the same extensible-registry shape as the connector and
	// quantization vocabularies. "lru" is the default only when the field is ABSENT; a
	// SUPPLIED name the factory does not know raises at launch rather than falling back. So
	// the v0_29 rules pack enumerates the in-tree set and the `eviction-policy-known` rule
	// WARNS on an unknown value (a probable typo, caught before launch) rather than erroring
	// (it may be a valid out-of-tree policy). It is validated by a rule, not here, because the
	// legal set is a version fact that belongs with the engine's.
	EvictionPolicy string `yaml:"eviction_policy,omitempty"`
	// PrefetchDepth is how many steps ahead a fetch is issued. It decides whether a
	// tier hit is hidden or exposed, and it is a scheduler property rather than a
	// hardware one.
	PrefetchDepth int `yaml:"prefetch_depth,omitempty"`
	// Connector names the offload implementation, whose spec determines whether a
	// fetch consumes SMs or a copy engine.
	Connector string `yaml:"connector,omitempty"`
	// Spec names the offloading spec, e.g. a CPU-only spec or a tiering spec.
	Spec string `yaml:"spec,omitempty"`
}

// Tier is one offload level: a device class from the catalog, and how much of it.
// Where the Scenario cluster declares a storage inventory, the device class must be one
// it lists (ValidateAgainstCluster checks this). A cluster that declares none states no
// inventory to draw from, so the tier is unconstrained — the shape that predates the
// inventory, kept valid; declaring storage is optional and opts a cluster into the check.
type Tier struct {
	Device string `yaml:"tier"`
	Bytes  int64  `yaml:"bytes"`
}

// PDTransfer describes prefill-to-decode KV movement. Bandwidth is absent: it
// resolves from the fabric times a registry efficiency coefficient, and is not a
// free parameter.
type PDTransfer struct {
	Connector string `yaml:"connector"`
}
