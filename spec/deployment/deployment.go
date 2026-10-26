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
// Derived quantities have no field. Expert-parallel width is the product of the
// tensor-, prefill-context- and data-parallel widths, because that is the group the
// engine builds. A deployment that could state EP independently could describe a
// layout no engine would run, so the field does not exist.
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

	// Router names the inputs of the llm-d router (the Endpoint Picker) that routes the
	// deployment's requests. Absent means the router's built-in configuration in standalone
	// mode over every pod of the deployment.
	Router *Router `yaml:"router,omitempty"`
}

// Router names the llm-d router's own inputs, exactly as a real deployment gives them to the
// router, so one set of files configures both the real router and the simulated one. Paths are
// relative to the directory of the file that holds the deployment.
//
// There is deliberately no field for a routing decision: scorers, weights, admission and flow
// control are the router's configuration, carried by the files named here and never
// restated in this schema.
type Router struct {
	// EPPFlags are the router's command-line flags, verbatim, one entry per flag (a Helm chart's
	// epp.flags map rendered as --name=value). They select the pool (--pool-name or
	// --endpoint-selector) and may name the configuration with --config-file. The schema checks
	// their form only; a consumer refuses flags it cannot honor.
	EPPFlags []string `yaml:"epp_flags,omitempty"`
	// EndpointPickerConfig is the EndpointPickerConfig file, as the router reads it with
	// --config-file. Naming the file here and in EPPFlags is an error.
	EndpointPickerConfig string `yaml:"endpoint_picker_config,omitempty"`
	// Manifests are Kubernetes manifest files. The router reads the InferencePool,
	// InferenceObjective and InferenceModelRewrite objects in them; other kinds are ignored.
	Manifests []string `yaml:"manifests,omitempty"`
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

// RoleLabel is llm-d's pod label for an engine's disaggregation role. It is derived from the
// pool's Role (RoleLabelValue); a pool's labels may repeat it only with that value.
const RoleLabel = "llm-d.ai/role"

// RoleLabelValue is the llm-d.ai/role value of a pool of role r: prefill-decode for a colocated
// pool (it serves both phases), else the role itself.
func (r Role) RoleLabelValue() string {
	if r == RoleColocated {
		return "prefill-decode"
	}
	return string(r)
}

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
	// Labels are the Kubernetes labels on the pool's model-server pods, as the pool's pod
	// template sets them. The router selects and filters endpoints by them (an InferencePool
	// selector, the llm-d.ai/role label of a disaggregated deployment). llm-d.ai/role is derived
	// from Role (RoleLabelValue); given here, it must have that value.
	Labels map[string]string `yaml:"labels,omitempty"`
}

// Parallelism is the layout. EP is absent: see the package comment.
type Parallelism struct {
	TP int `yaml:"tp"`
	PP int `yaml:"pp"`
	DP int `yaml:"dp"`
	// DPLocal is the data-parallel rank count resident on one node. It must divide
	// the node's GPU count.
	DPLocal int `yaml:"dp_local,omitempty"`

	// DataParallelLoadBalancing is how the pool's data-parallel ranks are exposed, which
	// decides what a router sees and chooses between. Empty is internal, the default.
	// See DPLoadBalancing.
	DataParallelLoadBalancing DPLoadBalancing `yaml:"data_parallel_load_balancing,omitempty"`

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
	// MaxCUDAGraphCaptureSize is compilation_config.max_cudagraph_capture_size: the
	// largest step, in tokens, a CUDA graph is captured for. A larger step runs eager,
	// so with DSpark at K=5, where a decode step verifies 6 tokens per request, a size of
	// 64 sends every step of more than 10 decoding requests eager. Nil takes the release
	// default, min(max_num_seqs x decode query length x 2, 512), or 1024 on SM100 (v0.31.0
	// vllm/config/vllm.py _set_cudagraph_sizes). The engine caps it at
	// max_num_batched_tokens; Pack.CUDAGraphCaptureSizes resolves the sizes that run.
	//
	// A pointer, because a stated 0 is not unset: the engine refuses it.
	MaxCUDAGraphCaptureSize *int `yaml:"max_cudagraph_capture_size,omitempty"`
	// CUDAGraphCaptureSizes is compilation_config.cudagraph_capture_sizes: the exact
	// step sizes, in tokens, to capture. Nil generates them from the maximum. The engine
	// drops sizes above max_num_batched_tokens, and refuses a stated maximum that is not
	// the largest size it keeps.
	CUDAGraphCaptureSizes []int `yaml:"cudagraph_capture_sizes,omitempty"`
	// DecoderReplayCUDAGraphCaptureSizes is
	// compilation_config.decoder_replay_cudagraph_capture_sizes, the decoder-replay graph sizes
	// of a YOCO model such as DeepSeek-V4.1 (vllm/config/compilation.py:565@5df63a72; new in
	// v0.32.0). Nil generates them from the sliding window.
	DecoderReplayCUDAGraphCaptureSizes []int `yaml:"decoder_replay_cudagraph_capture_sizes,omitempty"`
	// CustomOps is compilation_config.custom_ops: which ops run as custom kernels rather than
	// compiled ones, "all", "none", or +name / -name entries
	// (vllm/config/compilation.py:480@5df63a72). Nil is the engine default.
	CustomOps []string `yaml:"custom_ops,omitempty"`
	// UseTRTLLMAttention is attention_config.use_trtllm_attention: use or skip FlashInfer's
	// TRT-LLM attention backend (vllm/config/attention.py:74@5df63a72). Tri-state: nil
	// auto-detects.
	UseTRTLLMAttention *bool `yaml:"use_trtllm_attention,omitempty"`
	// MLAPrefillBackend is attention_config.mla_prefill_backend: FLASH_ATTN, FLASHINFER or
	// TRTLLM_RAGGED in v0.32.0 (vllm/config/attention.py:81@5df63a72). Empty selects
	// automatically. Which names a release accepts is its rules pack's to say.
	MLAPrefillBackend string `yaml:"mla_prefill_backend,omitempty"`
	// LinearBackend is kernel_config.linear_backend, the GEMM backend of linear layers
	// (vllm/config/kernel.py:350@5df63a72). Empty is the engine's "auto".
	LinearBackend string `yaml:"linear_backend,omitempty"`
	// Env is the engine environment variables a deployment sets that change what the engine
	// runs, such as VLLM_FLASHINFER_ALLREDUCE_BACKEND (vllm/envs.py:225@5df63a72) or
	// VLLM_MARLIN_USE_ATOMIC_ADD (vllm/envs.py:192@5df63a72), as the launch gave them. This
	// schema checks only that each name is an environment-variable name. A consumer refuses,
	// or discloses as an assumption, every key it cannot honour; it never ignores one
	// silently.
	Env map[string]string `yaml:"env,omitempty"`
	// AsyncScheduling is tri-state: nil lets the engine resolve it through its
	// disqualifying conditions, which is what a real deployment does.
	AsyncScheduling *bool `yaml:"async_scheduling,omitempty"`
	// Frontend is the API server that renders and tokenizes each request before the
	// engine sees it: python, the default, or rust (VLLM_USE_RUST_FRONTEND=1). See
	// Frontend.
	Frontend Frontend `yaml:"frontend,omitempty"`

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

	// SWABoundedReplay is CacheConfig.swa_bounded_replay, DeepSeek-V4.1's sliding-window
	// bounded replay (--swa-bounded-replay / --no-swa-bounded-replay). True keeps the
	// sliding-window cache out of prefix caching and recomputes a hit's last sliding window;
	// false lets the window cache take part in prefix caching (retained window tails).
	//
	// Tri-state: nil is the engine's resolution. vLLM v0.31.0 defaults it to true
	// (vllm/config/cache.py:242) and keeps it only under the V2 model runner, without
	// prefill context parallelism and not on ROCm (models/deepseek_v41/attention.py:519-546).
	// Only DeepSeek-V4.1 reads it.
	SWABoundedReplay *bool `yaml:"swa_bounded_replay,omitempty"`

	BlockSize           int `yaml:"block_size,omitempty"`
	MaxNumBatchedTokens int `yaml:"max_num_batched_tokens,omitempty"`
	MaxNumSeqs          int `yaml:"max_num_seqs,omitempty"`
	// LongPrefillTokenThreshold is SchedulerConfig.long_prefill_token_threshold: the most
	// tokens of one prefill scheduled in a step, so a long prompt cannot take the whole
	// token budget and stall the decodes beside it. Zero, the default, leaves prefills
	// uncapped. The engine refuses one above max_model_len (vllm/config/scheduler.py
	// verify_max_model_len), and whether a request alone in the batch is exempt is a
	// release fact (rules.Pack.LongPrefillExemptsLoneRequest).
	//
	// The field name follows the engine's CLI flag (--long-prefill-token-threshold).
	LongPrefillTokenThreshold int `yaml:"long_prefill_token_threshold,omitempty"`
	// LongPrefillTokenThresholdAdaptive floors the long-prefill cap at a fair share of the
	// token budget, max_num_batched_tokens over the queued and running requests, when the
	// threshold is nonzero (vllm/config/scheduler.py:86@5df63a72). Tri-state: nil is the
	// engine default, false.
	LongPrefillTokenThresholdAdaptive *bool `yaml:"long_prefill_token_threshold_adaptive,omitempty"`
	// MaxNumScheduledTokens is the most tokens the scheduler may issue in a step, smaller than
	// max_num_batched_tokens where the model appends tokens to a batch, as speculative decoding
	// does (vllm/config/scheduler.py:55@5df63a72). Zero means unset: the engine uses
	// max_num_batched_tokens.
	MaxNumScheduledTokens int `yaml:"max_num_scheduled_tokens,omitempty"`
	// MaxNumActiveSeqs caps the requests admitted to RUNNING without shrinking max_num_seqs,
	// which also sizes the runner and graph capture (vllm/config/scheduler.py:69@5df63a72).
	// Zero means unset: max_num_seqs is the admission limit. The engine refuses one above
	// max_num_seqs.
	MaxNumActiveSeqs int `yaml:"max_num_active_seqs,omitempty"`
	// Watermark is the fraction of KV blocks kept free when admitting waiting or preempted
	// requests, in [0, 1) (vllm/config/scheduler.py:196@5df63a72). A pointer, because 0 is a
	// value the engine accepts and nil means its default (0).
	Watermark *float64 `yaml:"watermark,omitempty"`
	// PrefillScheduleInterval admits new prefills only once every N engine steps, aligned
	// across data-parallel ranks (vllm/config/scheduler.py:203@5df63a72; applied by
	// _should_throttle_prefills, vllm/v1/engine/core.py:2318-2325@5df63a72). Zero means unset,
	// which is the engine default of 1: no throttling.
	PrefillScheduleInterval int `yaml:"prefill_schedule_interval,omitempty"`
	// StreamInterval is how many tokens the API server buffers before streaming them to the
	// client (vllm/config/scheduler.py:213@5df63a72, read by the output processor,
	// vllm/v1/engine/output_processor.py:198@5df63a72). It changes when a client sees tokens,
	// not what the scheduler does. Zero means unset, the engine default of 1.
	StreamInterval int `yaml:"stream_interval,omitempty"`
	// DisableHybridKVCacheManager allocates one KV size for every attention layer even when
	// they differ, such as full and sliding-window attention
	// (vllm/config/scheduler.py:182@5df63a72). Tri-state: nil lets the engine decide from the
	// environment and the configuration.
	DisableHybridKVCacheManager *bool `yaml:"disable_hybrid_kv_cache_manager,omitempty"`
	// MaxModelLen is the context length: omitted, auto, or a count. See MaxModelLen.
	MaxModelLen MaxModelLen `yaml:"max_model_len,omitempty"`
	// KVCacheMemoryBytes fixes the KV cache's size per GPU in bytes, instead of deriving it
	// from gpu_memory_utilization (vllm/config/cache.py:257@5df63a72). A run input, not an
	// engine outcome. Nil derives it; a stated value is positive.
	KVCacheMemoryBytes *int64 `yaml:"kv_cache_memory_bytes,omitempty"`
	// PrefixMatchUnit is the finest token boundary a prefix-cache hit can land on, which may
	// be finer than the KV block size where every group's block size divides by it
	// (vllm/config/cache.py:90@5df63a72). Zero means unset: the block size.
	PrefixMatchUnit int `yaml:"prefix_match_unit,omitempty"`
	// PrefixCacheRetentionInterval is the token interval between retained sliding-window and
	// Mamba prefix-cache checkpoints; 0 keeps only semantic checkpoints
	// (vllm/config/cache.py:159@5df63a72). A pointer, because 0 is the engine's default and a
	// stated value, and nil states nothing.
	PrefixCacheRetentionInterval *int `yaml:"prefix_cache_retention_interval,omitempty"`
	// KVEvents is the engine's KV-event publishing, which a router's precise prefix scorer
	// consumes. See KVEvents.
	KVEvents *KVEvents `yaml:"kv_events,omitempty"`
	// MaxLoRAs is the most distinct LoRA adapters one batch may hold; a request whose adapter
	// does not fit is skipped that step (vllm/config/lora.py:35@5df63a72;
	// vllm/v1/core/sched/scheduler.py:890-897, 941-952@5df63a72). Zero means unset, the engine default
	// of 1.
	MaxLoRAs int `yaml:"max_loras,omitempty"`
	// MaxCPULoRAs is how many LoRA adapters are kept loaded in CPU memory, an LRU set
	// (vllm/config/lora.py:42@5df63a72). Zero means unset: max_loras. That is the engine's own
	// reading of 0, not only this schema's: the launch path passes a non-positive
	// --max-cpu-loras as None (vllm/engine/arg_utils.py:2663-2665@5df63a72), which LoRAConfig
	// then sets to max_loras (vllm/config/lora.py:120-121@5df63a72). The engine refuses a
	// positive value below max_loras. A negative value is refused here: the CLI would read it
	// as unset, but LoRAConfig given one directly refuses it, as below max_loras, which is at
	// least 1 (vllm/config/lora.py:35,122-126@5df63a72), so no reading of it is safe to assume.
	MaxCPULoRAs int `yaml:"max_cpu_loras,omitempty"`

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
	// MoEBackend selects the expert-kernel backend (--moe-backend). Empty is the engine's
	// "auto". A mega-MoE backend changes more than the expert kernel: it is one of the
	// conditions under which DeepSeek-V4.1 turns on its own sequence parallelism, which
	// replaces each attention all-reduce with an all-gather and reduce-scatter pair.
	// Which names a release accepts is its rules pack's to say.
	MoEBackend string `yaml:"moe_backend,omitempty"`
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

	// EngramConfig is where DeepSeek-V4.1's Engram tables live and how they are shared.
	// It decides the residency of every model Gather whose residency_control is
	// engram_cpu_offload. Absent means the engine's defaults.
	EngramConfig *EngramConfig `yaml:"engram_config,omitempty"`

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
//
// The thresholds are pointers because a stated 0 and an unstated threshold are
// different launches: unstated takes the engine's default, and 0 is a value the engine
// accepts as a field and refuses at startup once dual-batch overlap is on.
type DBO struct {
	Enabled               bool `yaml:"enable_dbo"`
	DecodeTokenThreshold  *int `yaml:"dbo_decode_token_threshold,omitempty"`
	PrefillTokenThreshold *int `yaml:"dbo_prefill_token_threshold,omitempty"`
}

// EPLB is expert-parallel load balancing. NumRedundantExperts is not cosmetic: with
// EPLB enabled the EP width must divide the physical expert count, so this field
// decides whether a layout is feasible at all.
type EPLB struct {
	Enabled             bool `yaml:"enable_eplb"`
	NumRedundantExperts int  `yaml:"num_redundant_experts,omitempty"`
	WindowSize          int  `yaml:"window_size,omitempty"`
	StepInterval        int  `yaml:"step_interval,omitempty"`
}

// KVEvents is vLLM's --kv-events-config: whether the engine publishes the changes to the
// set of KV blocks its prefix cache can serve, and where (vllm/config/kv_events.py:11-47@5df63a72).
// The router's precise prefix-cache scorer consumes the stream.
type KVEvents struct {
	EnableKVCacheEvents bool `yaml:"enable_kv_cache_events"`
	// Publisher is "null" or "zmq"; empty is the engine's resolution, zmq when events are
	// enabled.
	Publisher string `yaml:"publisher,omitempty"`
	// Topic is the topic a consumer subscribes to.
	Topic string `yaml:"topic,omitempty"`
}

// kvEventPublishers are the publishers vLLM v0.32.0 accepts (vllm/config/kv_events.py:18@5df63a72).
var kvEventPublishers = map[string]bool{"null": true, "zmq": true}

// Speculative is the draft configuration. The method name is engine-version data,
// validated by a rules pack rather than by this schema.
//
// Acceptance is an input supplied by calibration (inference-sim/inference-sim#2021 B3), so
// the sampling knobs below are carried as the launch gave them and recorded, but they do not
// change which drafts a simulator accepts.
type Speculative struct {
	Method        string `yaml:"method"`
	NumSpecTokens int    `yaml:"num_spec_tokens"`
	// DraftModel names the catalog model whose graph is the drafter, where the drafter is a
	// separate checkpoint, as an EAGLE3 head is: vLLM's speculative_config.model
	// (vllm/config/speculative.py:396@5df63a72; blis-schemas#97). Empty means the target
	// graph's own speculator.
	DraftModel string `yaml:"draft_model,omitempty"`
	// KVCacheDType is the drafter's KV cache format where it differs from the target's,
	// speculative_config.kv_cache_dtype, which overrides the draft's cache_config.cache_dtype
	// (vllm/config/speculative.py:382@5df63a72). Empty means the target's.
	KVCacheDType string `yaml:"kv_cache_dtype,omitempty"`
	// DraftSampleMethod is "greedy" or "probabilistic" (vllm/config/speculative.py:85,600@5df63a72).
	DraftSampleMethod string `yaml:"draft_sample_method,omitempty"`
	// RejectionSampleMethod is "standard", "synthetic" or "block"
	// (vllm/config/speculative.py:84,529@5df63a72).
	RejectionSampleMethod string `yaml:"rejection_sample_method,omitempty"`
	// EnableAdaptiveVerification sizes the verification budget from per-request confidence;
	// DSpark only (vllm/config/speculative.py:551@5df63a72). Tri-state: nil is false.
	EnableAdaptiveVerification *bool `yaml:"enable_adaptive_verification,omitempty"`
}

// Sample-method vocabularies of vLLM v0.32.0 (vllm/config/speculative.py:84-85@5df63a72).
// They are closed literals, so an unknown value is a field error rather than a rule.
var (
	draftSampleMethods     = map[string]bool{"greedy": true, "probabilistic": true}
	rejectionSampleMethods = map[string]bool{"standard": true, "synthetic": true, "block": true}
)

// EngramConfig mirrors the cost-bearing keys of vLLM's --engram-config (v0.31.0
// vllm/config/engram.py). Each is a tri-state request: nil takes the engine default.
//
// Together they decide where each table lives and how many ranks split it. A table is
// always split by hash head across the tensor-parallel ranks. When the node's
// data-parallel replicas do not share one host copy and more than one of them is on the
// node, it is split across those replicas too, tp x local-dp ways, and each lookup is
// all-gathered over them (vllm/models/deepseek_v41/nvidia/engram.py _get_shard_info).
// When they share one, each replica keeps the TP split and the node holds one copy.
//
// embedding_across_dp is not carried: in v0.31.0 it sizes a process group that
// DeepSeek-V4.1's Engram path does not use.
type EngramConfig struct {
	// CPUOffload keeps the tables in pinned host memory, read over the host link through
	// UVA. Nil is the default, true. False allocates them in HBM, split the same way.
	CPUOffload *bool `yaml:"cpu_offload,omitempty"`
	// DPSharedMemory keeps one host copy of every tensor-parallel shard per node, in
	// /dev/shm, shared by the node's data-parallel replicas. Nil lets the engine decide,
	// which it does by enabling it when cpu_offload is on, use_thp is off, dp exceeds
	// one and elastic EP is off (resolve_dp_shared_memory). It then falls back to
	// splitting the table across the replicas when they are not co-located or /dev/shm
	// cannot hold it.
	DPSharedMemory *bool `yaml:"dp_shared_memory,omitempty"`
	// UseTHP backs a per-rank host table with transparent huge pages. It is carried
	// because it changes placement, not only page size: the engine refuses it with
	// dp_shared_memory, and its default resolution leaves sharing off when it is on.
	UseTHP *bool `yaml:"use_thp,omitempty"`
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
	// ConnectorOptions are the connector's own options, as kv_connector_extra_config gives
	// them to vLLM. See ConnectorOptions.
	ConnectorOptions *ConnectorOptions `yaml:"connector_options,omitempty"`
}

// ConnectorOptions are the KV connector options a deployment states, from vLLM's
// kv_transfer_config (vllm/config/kv_transfer.py@5df63a72) and its kv_connector_extra_config,
// each read by the connector named. Typed rather than a free map, so a misspelled option fails
// to load instead of being dropped. Every one is a tri-state request: nil is the connector's
// default.
type ConnectorOptions struct {
	// LoadAsync loads a hit asynchronously (MooncakeStoreConnector, default true:
	// vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:1397@5df63a72).
	LoadAsync *bool `yaml:"load_async,omitempty"`
	// LookupAsync looks a prefix up without blocking the scheduler (MooncakeStoreConnector,
	// default false: .../mooncake/store/scheduler.py:147@5df63a72).
	LookupAsync *bool `yaml:"lookup_async,omitempty"`
	// EnableOffload writes KV to the store as well as reading it (Mooncake's store config,
	// default false: vllm/distributed/mooncake_store.py:37,66@5df63a72).
	EnableOffload *bool `yaml:"enable_offload,omitempty"`
	// EnableCrossLayersBlocks stores one block across all layers rather than per layer
	// (MooncakeStoreConnector's layout check, .../mooncake/store/worker.py:1553@5df63a72).
	EnableCrossLayersBlocks *bool `yaml:"enable_cross_layers_blocks,omitempty"`
	// LazyOffload defers a CPU write until a block is evicted (SimpleCPUOffloadConnector,
	// default false: .../simple_cpu_offload_connector.py:104@5df63a72).
	LazyOffload *bool `yaml:"lazy_offload,omitempty"`
	// BlocksPerChunk groups GPU blocks into one offload chunk (OffloadingConnector,
	// .../offloading/config.py:99-112@5df63a72). Zero means unset; a stated one is positive.
	BlocksPerChunk int `yaml:"blocks_per_chunk,omitempty"`
	// KVLoadFailurePolicy is what a failed KV load does: "recompute" the failed blocks, or
	// "fail" the request, the default (vllm/config/kv_transfer.py:88@5df63a72).
	KVLoadFailurePolicy string `yaml:"kv_load_failure_policy,omitempty"`
}

// kvLoadFailurePolicies is vLLM v0.32.0's closed literal (vllm/config/kv_transfer.py:88@5df63a72).
var kvLoadFailurePolicies = map[string]bool{"recompute": true, "fail": true}

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

// Frontend is the API server process a pool serves through, which renders the chat
// template and tokenizes every prompt before the request reaches the engine core. vLLM
// selects it with an environment variable, not a flag: VLLM_USE_RUST_FRONTEND defaults to
// 0 (v0.31.0 vllm/envs.py:169, 1441-1443; the same at v0.29.0), and only when it is set
// does `vllm serve` launch the vllm-rs binary instead of the Python API server
// (vllm/entrypoints/cli/serve.py:64-66). The two tokenize the same prompt at very
// different rates -- the Rust frontend loads its tokenizer through fastokens -- so the
// per-prompt-token admission time a request waits before it is scheduled depends on it.
//
//   - python, the default: the Python OpenAI server, which renders and tokenizes with the
//     HF fast tokenizer before it answers (vllm/entrypoints/openai/chat_completion/
//     serving.py:285, 378).
//   - rust: the vllm-rs frontend (rust/src/chat/src/lib.rs:364-368 prepares, then
//     generates).
type Frontend string

const (
	FrontendPython Frontend = "python"
	FrontendRust   Frontend = "rust"
)

// Valid reports whether f is a recognized frontend; empty is the default, python.
func (f Frontend) Valid() bool { return f == "" || f == FrontendPython || f == FrontendRust }

// Effective returns the frontend with its default applied.
func (f Frontend) Effective() Frontend {
	if f == "" {
		return FrontendPython
	}
	return f
}

// DPLoadBalancing is how a data-parallel pool's ranks are exposed to clients, vLLM's
// three launch modes (v0.31.0 vllm/engine/arg_utils.py:536-538, 1225-1242):
//
//   - internal, the default: one API endpoint, and the engine's core client balances
//     every request across all ranks (DPLBAsyncMPClient, vllm/v1/engine/core_client.py:
//     155-166);
//   - hybrid (--data-parallel-hybrid-lb): one endpoint per node, balancing across that
//     node's ranks; a router chooses the node;
//   - external (--data-parallel-external-lb, or --data-parallel-multi-port-external-lb,
//     which launches one such server per local rank): every rank serves its own
//     endpoint (DPAsyncMPClient), and a router chooses the rank.
//
// The api server count is not part of it: --api-server-count sets how many frontend
// processes share an endpoint (vllm/entrypoints/cli/serve.py:105-127), not which ranks a
// router can reach.
type DPLoadBalancing string

const (
	DPLBInternal DPLoadBalancing = "internal"
	DPLBHybrid   DPLoadBalancing = "hybrid"
	DPLBExternal DPLoadBalancing = "external"
)

var dpLoadBalancings = map[DPLoadBalancing]bool{DPLBInternal: true, DPLBHybrid: true, DPLBExternal: true}

// Valid reports whether m is a recognized mode; empty is the default, internal.
func (m DPLoadBalancing) Valid() bool { return m == "" || dpLoadBalancings[m] }

// Mode returns the mode with its default applied.
func (m DPLoadBalancing) Mode() DPLoadBalancing {
	if m == "" {
		return DPLBInternal
	}
	return m
}
