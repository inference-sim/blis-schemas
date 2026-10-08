// Package v0_29 is the rules pack for vLLM 0.29.x.
//
// Everything here is a fact about that release, transcribed from its configuration
// surface. A later release gets its own package: copy this one, change the
// constants and the rule bodies that moved, and register it under the new version.
// Nothing in spec/ or kernel/ changes when an engine ships.
//
// Each rule states, in its Because field, what breaks without it. That is not
// decoration: a reader deciding whether to waive a rule needs to know what the rule
// is protecting, and a CI job that waives rules by name needs the list to be
// self-describing.
package v0_29

import (
	"sort"

	"github.com/inference-sim/blis-schemas/internal/validate"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// Version is the engine release this pack describes.
const Version = "0.29.0"

func set[T comparable](vs ...T) map[T]bool {
	m := make(map[T]bool, len(vs))
	for _, v := range vs {
		m[v] = true
	}
	return m
}

// Pack returns the rules pack for this version. It is a function rather than a
// package-level variable so a caller can hold two versions at once — useful when
// comparing what a scenario would mean across releases.
func Pack() *rules.Pack {
	p := &rules.Pack{
		Version: Version,

		// Transcribed from the engine's All2AllBackend literal. Twelve members:
		// the all-gather family, the point-to-point family, and the vendor
		// variants. A name outside this set is priced with the wrong volume basis.
		All2AllBackends: set("naive", "allgather_reducescatter", "pplx",
			"deepep_high_throughput", "deepep_low_latency", "deepep_v2",
			"flashinfer_all2allv", "flashinfer_nvlink_one_sided",
			"flashinfer_nvlink_two_sided", "mori_high_throughput",
			"mori_low_latency", "nixl_ep"),
		// The subset for which the engine makes the MoE input sequence-parallel,
		// which replaces the layer's all-reduce with a reduce-scatter/all-gather
		// pair. The default backend is a member, so this is not an exotic path.
		SPMoEBackends: set("allgather_reducescatter", "deepep_high_throughput",
			"deepep_low_latency", "flashinfer_nvlink_one_sided",
			"mori_high_throughput", "mori_low_latency", "nixl_ep"),

		// The engine exposes no allreduce-backend name. What it has is a boolean,
		// disable_custom_all_reduce, plus an environment-selected communicator; an
		// earlier draft of this pack invented a backend enum, which would have
		// rejected every deployment stating the real field. Kept as a set of the two
		// values the deployment field accepts so the rule shape stays uniform.
		AllReduceBackends: set("custom", "nccl"),

		// Transcribed from the CUDAGraphMode enum. Five members: the fifth,
		// FULL_DECODE_ONLY, captures decode and runs prefill eagerly, and omitting
		// it would reject a configuration the engine accepts.
		CUDAGraphModes: set("NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY",
			"FULL_AND_PIECEWISE"),

		// Transcribed from the CacheDType literal: seventeen members. The narrow
		// subset an earlier draft carried would have rejected nvfp4, which published
		// serving reports use, along with every per-token-head and turboquant format.
		CacheDTypes: set("auto", "float16", "bfloat16", "fp8", "fp8_e4m3",
			"fp8_e5m2", "fp8_inc", "fp8_ds_mla", "turboquant_k8v4",
			"turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_3bit_nc",
			"int4_per_token_head", "int8_per_token_head", "fp8_per_token_head",
			"nvfp4", "nvfp4_4over6"),

		MambaCacheModes:   set("none", "align", "all"),
		SchedulerPolicies: set("fcfs", "priority"),

		// Speculative methods accepted by this release. The multi-token-prediction
		// family is large and per-model; the members here are the ones the three
		// design documents and the published report corpus exercise.
		SpeculativeMethods: set("ngram", "ngram_gpu", "eagle", "eagle3", "medusa",
			"mlp_speculator", "draft_model", "deepseek_mtp", "glm4_moe_mtp",
			"kimi_k3_mtp", "inkling_mtp", "qwen3_next_mtp", "nemotron_h_mtp",
			"mtp", "dflash", "dspark"),
		// Methods that leave async scheduling enabled. A method outside this set
		// disables it, which moves the host term from hidden to exposed.
		AsyncCompatibleSpec: set("eagle", "eagle3", "deepseek_mtp", "glm4_moe_mtp",
			"kimi_k3_mtp", "inkling_mtp", "qwen3_next_mtp", "nemotron_h_mtp",
			"mtp", "dflash", "ngram_gpu", "draft_model", "dspark"),

		OffloadSpecs: set("CPUOffloadingSpec", "TieringOffloadingSpec"),

		// Transcribed from the QuantizationMethods literal in
		// vllm/model_executor/layers/quantization/__init__.py: thirty members, the named
		// methods and the online-quant shorthands. The engine extends QUANTIZATION_METHODS
		// at runtime for out-of-tree methods, so a name outside this set is a warning, not
		// an error — the field is still the one that decides weight width, compute peak and
		// GEMM efficiency, and a typo among the in-tree names is worth catching.
		Quantizations: set("awq", "auto_awq", "fp8", "fbgemm_fp8", "fp_quant", "modelopt",
			"modelopt_fp4", "modelopt_mxfp8", "modelopt_mixed", "auto_gptq", "gptq",
			"gptq_marlin", "awq_marlin", "humming", "compressed-tensors", "experts_int8",
			"quark", "moe_wna16", "torchao", "inc", "mxfp4", "gpt_oss_mxfp4",
			"deepseek_v4_fp8", "online", "fp8_per_tensor", "fp8_per_block",
			"fp8_per_channel", "int8_per_channel_weight_only", "nvfp4_per_token", "mxfp8"),

		// Transcribed from the KVConnectorFactory.register_connector calls in
		// vllm/distributed/kv_transfer/kv_connector/factory.py: the names an offload or
		// PD-transfer block may select. Like a quantization method the registry is
		// extensible out of tree, so an unknown name is a warning.
		Connectors: set("DecodeBenchConnector", "ExampleConnector",
			"ExampleHiddenStatesConnector", "FlexKVConnectorV1", "HF3FSKVConnector",
			"LMCacheConnectorV1", "LMCacheMPConnector", "MoRIIOConnector",
			"MooncakeConnector", "MooncakeStoreConnector", "MultiConnector",
			"NixlConnector", "NixlPullConnector", "NixlPushConnector",
			"OffloadingConnector", "SimpleCPUOffloadConnector"),

		// Transcribed from the CachePolicyFactory.register_cache_policy calls in
		// vllm/v1/kv_offload/cpu/policies/factory.py: the KV-offload eviction policies. Two
		// in-tree (lru, arc); the factory also admits out-of-tree policies, so like the
		// connector and quantization registries an unknown name is a warning, not an error.
		EvictionPolicies: set("lru", "arc"),

		CustomAllReduceWorldSizes: set(2, 4, 6, 8, 16),

		DefaultDBODecodeThreshold:  32,
		DefaultDBOPrefillThreshold: 512,

		TritonFetchPageBytes: 28 * 1024,
		TritonFetchSMs:       12,

		CascadeAttnOptIn: true,
	}
	p.Rules = ruleList(p)
	return p
}

func ruleList(p *rules.Pack) []rules.Rule {
	return []rules.Rule{
		{
			Name:    "all2all-backend-known",
			Because: "an unrecognized backend name would be priced with the default volume basis, which differs by a factor of top_k",
			Check: func(in rules.Input, out *validate.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.All2AllBackend == "" {
						return
					}
					if !p.All2AllBackends[e.All2AllBackend] {
						out.RuleErrorf("all2all-backend-known",
							"%s.engine.all2all_backend: %q is not accepted by engine %s",
							at, e.All2AllBackend, p.Version)
					}
				})
			},
		},
		{
			Name:    "allreduce-backend-known",
			Because: "the backend decides whether a reduction consumes SMs or the NIC, so an unknown name leaves the cost on no resource",
			Check: func(in rules.Input, out *validate.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.AllReduceBackend == "" {
						return
					}
					if !p.AllReduceBackends[e.AllReduceBackend] {
						out.RuleErrorf("allreduce-backend-known",
							"%s.engine.allreduce_backend: %q is not accepted by engine %s",
							at, e.AllReduceBackend, p.Version)
					}
				})
			},
		},
		{
			Name:    "enum-values-known",
			Because: "graph mode, cache dtype, mamba cache mode and scheduler policy each change a cost term, and an unknown value silently selects a default",
			Check: func(in rules.Input, out *validate.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					check := func(field, value string, allowed map[string]bool) {
						if value == "" || allowed[value] {
							return
						}
						out.RuleErrorf("enum-values-known",
							"%s.engine.%s: %q is not accepted by engine %s",
							at, field, value, p.Version)
					}
					check("cudagraph_mode", e.CUDAGraphMode, p.CUDAGraphModes)
					check("cache_dtype", e.CacheDType, p.CacheDTypes)
					check("mamba_cache_dtype", e.MambaCacheDType, p.CacheDTypes)
					check("mamba_ssm_cache_dtype", e.MambaSSMCacheDType, p.CacheDTypes)
					check("mamba_cache_mode", e.MambaCacheMode, p.MambaCacheModes)
					check("scheduling_policy", e.SchedulingPolicy, p.SchedulerPolicies)
				})
			},
		},
		{
			Name:    "speculative-method-known",
			Because: "an unknown draft method cannot be priced, and the draft pass of an MoE target is a second MoE rather than a scalar multiplier",
			Check: func(in rules.Input, out *validate.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.Speculative == nil {
						return
					}
					if !p.SpeculativeMethods[e.Speculative.Method] {
						out.RuleErrorf("speculative-method-known",
							"%s.engine.speculative.method: %q is not accepted by engine %s",
							at, e.Speculative.Method, p.Version)
					}
				})
				if in.Model != nil && in.Model.Speculator != nil {
					if !p.SpeculativeMethods[in.Model.Speculator.Method] {
						out.RuleErrorf("speculative-method-known",
							"model.speculator.method: %q is not accepted by engine %s",
							in.Model.Speculator.Method, p.Version)
					}
				}
			},
		},
		{
			Name:    "pcp-excludes-data-parallelism",
			Because: "this release refuses prefill-context parallelism combined with data parallelism at startup (\"PCP does not support data parallelism yet\", vllm/config/parallel.py), so such a layout does not run; later releases support it from e6dc16cebd, which is why the refusal is here and not a field check",
			Check: func(in rules.Input, out *validate.Problems) {
				forEachPool(in, func(at string, pool deployment.Pool) {
					if pool.Parallel.PCP > 1 && pool.Parallel.DP > 1 {
						out.RuleErrorf("pcp-excludes-data-parallelism",
							"%s: pcp %d with dp %d; engine %s refuses prefill-context parallelism combined with data parallelism",
							at, pool.Parallel.PCP, pool.Parallel.DP, Version)
					}
				})
			},
		},
		{
			Name:    "expert-divisibility-under-eplb",
			Because: "with load balancing on, the engine asserts that physical experts divide the expert-parallel width, so an indivisible layout does not start",
			Check: func(in rules.Input, out *validate.Problems) {
				if in.Model == nil {
					return
				}
				experts := totalExperts(in)
				if experts == 0 {
					return
				}
				forEachPool(in, func(at string, pool deployment.Pool) {
					e := pool.Engine
					if e.EPLB == nil || !e.EPLB.Enabled {
						return
					}
					ep := pool.Parallel.ExpertParallelWidth()
					if ep < 2 {
						return
					}
					physical := experts + e.EPLB.NumRedundantExperts
					if physical%ep != 0 {
						need := (ep - physical%ep) % ep
						out.RuleErrorf("expert-divisibility-under-eplb",
							"%s: %d experts plus %d redundant is %d, which does not divide expert-parallel width %d; %d more redundant experts would",
							at, experts, e.EPLB.NumRedundantExperts, physical, ep, need)
					}
				})
			},
		},
		{
			Name:    "expert-imbalance-without-eplb",
			Because: "without load balancing an indivisible split is legal but uneven, and the resulting static imbalance is a step-time term rather than an error",
			Check: func(in rules.Input, out *validate.Problems) {
				if in.Model == nil {
					return
				}
				experts := totalExperts(in)
				if experts == 0 {
					return
				}
				forEachPool(in, func(at string, pool deployment.Pool) {
					e := pool.Engine
					if e.EPLB != nil && e.EPLB.Enabled {
						return
					}
					ep := pool.Parallel.ExpertParallelWidth()
					if ep < 2 || experts%ep == 0 {
						return
					}
					base := experts / ep
					if base == 0 {
						out.RuleErrorf("expert-imbalance-without-eplb",
							"%s: expert-parallel width %d exceeds the model's %d experts, leaving ranks with none",
							at, ep, experts)
						return
					}
					out.RuleWarnf("expert-imbalance-without-eplb",
						"%s: %d experts over width %d leaves %d ranks with %d experts and the rest with %d, a %.2fx static imbalance the estimate must carry",
						at, experts, ep, experts%ep, base+1, base,
						float64(base+1)/float64(base))
				})
			},
		},
		{
			Name:    "custom-allreduce-reachable",
			Because: "the SM-consuming kernel supports only certain rank counts and only within one node absent multi-node NVLink, so a request outside those falls back to the NIC",
			Check: func(in rules.Input, out *validate.Problems) {
				gpusPerNode := 0
				if in.Scenario != nil {
					gpusPerNode = in.Scenario.Cluster.GPUsPerNode
				}
				forEachPool(in, func(at string, pool deployment.Pool) {
					if pool.Engine.AllReduceBackend != "custom" {
						return
					}
					tp := pool.Parallel.TP
					if !p.CustomAllReduceWorldSizes[tp] {
						out.RuleWarnf("custom-allreduce-reachable",
							"%s: tp %d is not a supported world size for the custom all-reduce, so the engine will fall back to NCCL and the cost moves from SMs to the NIC",
							at, tp)
					}
					if gpusPerNode > 0 && tp > gpusPerNode {
						out.RuleWarnf("custom-allreduce-reachable",
							"%s: tp %d spans more than the %d GPUs in a node, so the custom all-reduce needs multi-node NVLink on every rank or it falls back to NCCL",
							at, tp, gpusPerNode)
					}
				})
			},
		},
		{
			Name:    "offload-spec-known",
			Because: "the spec decides whether a fetch consumes SMs or a copy engine, which is the only way the offload path enters step time",
			Check: func(in rules.Input, out *validate.Problems) {
				if in.Deployment == nil || in.Deployment.Offload == nil {
					return
				}
				s := in.Deployment.Offload.Spec
				if s != "" && !p.OffloadSpecs[s] {
					out.RuleErrorf("offload-spec-known",
						"offload.spec: %q is not a registered spec in engine %s",
						s, p.Version)
				}
			},
		},
		{
			Name:    "quantization-known",
			Because: "the served weight format sets weight-read bytes, compute peak and the GEMM efficiency envelope at once, so a misspelled in-tree method is priced against three wrong constants; it is a warning because the engine admits out-of-tree methods this set cannot enumerate",
			Check: func(in rules.Input, out *validate.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.Quantization == "" || p.Quantizations[e.Quantization] {
						return
					}
					out.RuleWarnf("quantization-known",
						"%s.engine.quantization: %q is not an in-tree method of engine %s; if it is an out-of-tree method this is expected, otherwise check the spelling. In-tree methods are %v",
						at, e.Quantization, p.Version, sortedKeys(p.Quantizations))
				})
			},
		},
		{
			Name:    "connector-known",
			Because: "the connector names the offload/PD-transfer implementation whose spec decides whether a fetch consumes SMs or a copy engine, so an unknown name leaves the transfer on no modelled resource; a warning because the KV connector registry is extensible out of tree",
			Check: func(in rules.Input, out *validate.Problems) {
				if in.Deployment == nil {
					return
				}
				check := func(field, value string) {
					if value == "" || p.Connectors[value] {
						return
					}
					out.RuleWarnf("connector-known",
						"%s: %q is not a connector registered in engine %s; if it is an out-of-tree connector this is expected, otherwise check the spelling. Registered connectors are %v",
						field, value, p.Version, sortedKeys(p.Connectors))
				}
				if in.Deployment.Offload != nil {
					check("offload.connector", in.Deployment.Offload.Connector)
				}
				if in.Deployment.PDTransfer != nil {
					check("pd_transfer.connector", in.Deployment.PDTransfer.Connector)
				}
			},
		},
		{
			Name:    "eviction-policy-known",
			Because: "a supplied name the engine does not recognize does not silently fall back — CachePolicyFactory.get_cache_policy_cls raises ValueError at launch unless cache_policy_module_path names an out-of-tree policy; this rule catches that typo earlier, as a warning, while still permitting a real out-of-tree name the set cannot enumerate",
			Check: func(in rules.Input, out *validate.Problems) {
				if in.Deployment == nil || in.Deployment.Offload == nil {
					return
				}
				ep := in.Deployment.Offload.EvictionPolicy
				if ep == "" || p.EvictionPolicies[ep] {
					return
				}
				out.RuleWarnf("eviction-policy-known",
					"offload.eviction_policy: %q is not an in-tree policy of engine %s; if it is an out-of-tree policy this is expected, otherwise check the spelling. In-tree policies are %v",
					ep, p.Version, sortedKeys(p.EvictionPolicies))
			},
		},
		{
			Name:    "mamba-cache-mode-matches-model",
			Because: "a cache mode for recurrent layers on a model with none is a setting with no effect, and its absence on a model with them leaves the state unsized",
			Check: func(in rules.Input, out *validate.Problems) {
				if in.Model == nil {
					return
				}
				hasRecurrent := modelHasRecurrent(in)
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.MambaCacheMode != "" && !hasRecurrent {
						out.RuleWarnf("mamba-cache-mode-matches-model",
							"%s.engine.mamba_cache_mode is set but the model has no recurrent layer, so it has no effect", at)
					}
					if e.MambaCacheMode == "" && hasRecurrent {
						out.RuleWarnf("mamba-cache-mode-matches-model",
							"%s.engine: the model has recurrent layers but no mamba_cache_mode is stated, so whether their state is fixed or context-proportional is unresolved", at)
					}
				})
			},
		},
		{
			Name:    "cascade-attention-is-opt-in",
			Because: "cascade attention changes the attention primitive itself, and a deployment that leaves it unstated gets the engine default rather than the faster path",
			Check: func(in rules.Input, out *validate.Problems) {
				if !p.CascadeAttnOptIn {
					return
				}
				forEachPool(in, func(at string, pool deployment.Pool) {
					e := pool.Engine
					if e.DisableCascadeAttn != nil && !*e.DisableCascadeAttn &&
						e.Speculative != nil && e.AsyncScheduling == nil {
						out.RuleWarnf("cascade-attention-is-opt-in",
							"%s: cascade attention is requested alongside speculative decoding; with async scheduling resolved on, the engine disables cascade regardless", at)
					}
				})
			},
		},
		{
			Name:    "dbo-thresholds-stated-when-enabled",
			Because: "which threshold applies turns on batch uniformity, and an unstated threshold silently takes a default that differs by 16x between the two",
			Check: func(in rules.Input, out *validate.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.DBO == nil || !e.DBO.Enabled {
						return
					}
					if e.DBO.DecodeTokenThreshold == 0 && e.DBO.PrefillTokenThreshold == 0 {
						out.RuleWarnf("dbo-thresholds-stated-when-enabled",
							"%s.engine.dbo: enabled with no thresholds stated; engine %s defaults to %d for a uniform-decode batch and %d otherwise",
							at, p.Version, p.DefaultDBODecodeThreshold,
							p.DefaultDBOPrefillThreshold)
					}
				})
			},
		},
		{
			Name:    "sequence-parallel-moe-implied",
			Because: "on this backend with tensor and data parallelism both above one, the engine makes the MoE input sequence-parallel, which replaces the layer's all-reduce with a different pair of collectives",
			Check: func(in rules.Input, out *validate.Problems) {
				forEachPool(in, func(at string, pool deployment.Pool) {
					pl := pool.Parallel
					e := pool.Engine
					if !pl.EnableExpertParallel || pl.TP <= 1 || pl.DP <= 1 {
						return
					}
					if e.All2AllBackend == "" || !p.SPMoEBackends[e.All2AllBackend] {
						return
					}
					out.RuleWarnf("sequence-parallel-moe-implied",
						"%s: engine %s makes the MoE input sequence-parallel here (backend %q, tp %d, dp %d), so the model graph must emit a reduce-scatter and all-gather pair rather than an all-reduce",
						at, p.Version, e.All2AllBackend, pl.TP, pl.DP)
				})
			},
		},
	}
}

func forEachPool(in rules.Input, fn func(at string, pool deployment.Pool)) {
	if in.Deployment == nil {
		return
	}
	for i, pool := range in.Deployment.Pools {
		fn(poolPath(i), pool)
	}
}

func forEachEngine(in rules.Input, fn func(at string, e deployment.Engine)) {
	forEachPool(in, func(at string, pool deployment.Pool) { fn(at, pool.Engine) })
}

// sortedKeys returns a set's members in sorted order, for a membership diagnostic that
// enumerates the valid alternatives (#19 item 3): an error that names what IS accepted is
// self-correcting in one turn, where "check the spelling" leaves a generator to guess.
func sortedKeys(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func poolPath(i int) string {
	return "pools[" + itoa(i) + "]"
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	n := len(b)
	for i > 0 {
		n--
		b[n] = byte('0' + i%10)
		i /= 10
	}
	return string(b[n:])
}

// totalExperts returns the model's logical expert count, taken from the first
// grouped-GEMM node. A graph with differing expert counts per layer is possible in
// principle and absent from every model these schemas target; the maximum is used so
// the divisibility check errs toward reporting.
func totalExperts(in rules.Input) int {
	if in.Model == nil {
		return 0
	}
	best := 0
	for _, lk := range in.Model.LayerKinds {
		for _, n := range lk.Nodes {
			if n.Experts > best {
				best = n.Experts
			}
		}
	}
	return best
}

func modelHasRecurrent(in rules.Input) bool {
	if in.Model == nil {
		return false
	}
	for _, lk := range in.Model.LayerKinds {
		for _, n := range lk.Nodes {
			if n.RecurrentKind != "" {
				return true
			}
		}
	}
	return false
}
