package deployment

import (
	"fmt"

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
	return p
}

// ValidateAgainstCluster adds the field-level checks that couple a deployment to the
// available-hardware inventory it is placed on: that its pools' node counts sum to the
// nodes the cluster declares, that each data-parallel-local width divides a node's GPU
// count, and that every offload tier names a storage class the cluster actually lists.
// They are separate from Validate because they need the cluster, which is a Scenario
// property; the composition layer (blisschemas.Validate) supplies it when both documents
// are present. The cluster is passed as primitives (counts and the storage name list)
// rather than as a scenario.Cluster so this package stays independent of spec/scenario.
func (d *Deployment) ValidateAgainstCluster(nodes, gpusPerNode int, storage []string) *validate.Problems {
	p := &validate.Problems{}

	total := 0
	for i, pool := range d.Pools {
		total += pool.Nodes
		if pool.Parallel.DPLocal > 0 && gpusPerNode > 0 &&
			gpusPerNode%pool.Parallel.DPLocal != 0 {
			p.Field(fmt.Sprintf("pools[%d].parallel.dp_local", i),
				"%d does not divide gpus_per_node %d", pool.Parallel.DPLocal, gpusPerNode)
		}
	}
	if len(d.Pools) > 0 && total != nodes {
		p.Field("pools",
			"pool node counts sum to %d but the cluster declares %d", total, nodes)
	}

	// Each offload tier draws from the cluster's declared storage inventory, as the Tier
	// doc states. The check fires only when the cluster lists storage: a cluster that
	// declares none states no inventory to constrain against, so a deployment offloading
	// against it is left as it was before the inventory existed, rather than rejected.
	if d.Offload != nil && len(storage) > 0 {
		declared := make(map[string]bool, len(storage))
		for _, s := range storage {
			declared[s] = true
		}
		for i, t := range d.Offload.Tiers {
			if t.Device != "" && !declared[t.Device] {
				p.Field(fmt.Sprintf("offload.tiers[%d].tier", i),
					"%q is not in the cluster storage inventory %v", t.Device, storage)
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
