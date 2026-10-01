package scenario

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/internal/validate"
)

// Validate performs FIELD-LEVEL validation: required fields, positive counts,
// internal arithmetic consistency, and the structural invariants that hold whatever
// engine version a scenario names.
//
// A structural invariant is one that follows from what parallelism is, not from an
// engine's current choices. That a pool's rank count must divide its GPU count is
// arithmetic. That expert-parallel width is derived rather than stated is a
// consequence of how an expert group is built. Both belong here.
//
// Version-scoped rules do not. Whether a particular backend name is supported,
// which conditions disable async scheduling, what the collective thresholds default
// to — those move between releases, and a rules pack keyed by engine version owns
// them. Mixing the two would mean this file churns every time an engine ships.
func (s *Scenario) Validate() *validate.Problems {
	p := &validate.Problems{}

	if s.Kind != "Scenario" {
		p.Field("kind", "must be %q, got %q", "Scenario", s.Kind)
	}
	for field, v := range map[string]string{
		"name": s.Name, "model": s.Model, "hardware": s.Hardware,
		"engine_version": s.EngineVersion,
	} {
		if v == "" {
			p.Field(field, "required")
		}
	}
	if len(s.Coefficients) == 0 {
		p.Field("coefficients",
			"at least one set is required; an estimate with no coefficients is not an estimate")
	}

	s.validateCluster(p)
	s.validatePools(p)
	s.validateOffload(p)
	return p
}

func (s *Scenario) validateCluster(p *validate.Problems) {
	c := s.Cluster
	if c.Nodes < 1 {
		p.Field("cluster.nodes", "must be at least 1")
	}
	if c.GPUsPerNode < 1 {
		p.Field("cluster.gpus_per_node", "must be at least 1")
	}
	if c.GPUsPerRack > 0 && c.GPUsPerNode > 0 && c.GPUsPerRack%c.GPUsPerNode != 0 {
		p.Field("cluster.gpus_per_rack", "%d is not a multiple of gpus_per_node %d",
			c.GPUsPerRack, c.GPUsPerNode)
	}
	// A multi-node deployment moves bytes between nodes, and the bandwidth for that
	// is a fabric property. Without one, every cross-node cost is unpriced.
	if c.Nodes > 1 && s.Fabric == "" {
		p.Field("fabric",
			"required for a %d-node cluster: cross-node cost resolves from the fabric, not the chip",
			c.Nodes)
	}
}

func (s *Scenario) validatePools(p *validate.Problems) {
	if len(s.Pools) == 0 {
		p.Field("pools", "at least one pool is required")
		return
	}

	total := 0
	roles := map[Role]int{}
	for i, pool := range s.Pools {
		at := fmt.Sprintf("pools[%d]", i)
		if !pool.Role.Valid() {
			p.Field(at+".role", "%q is not a recognized role", pool.Role)
		}
		roles[pool.Role]++
		if pool.Nodes < 1 {
			p.Field(at+".nodes", "must be at least 1")
		}
		total += pool.Nodes
		validateParallelism(p, at, pool.Parallel, s.Cluster.GPUsPerNode)
		validateEngine(p, at, pool.Engine)
	}

	if total != s.Cluster.Nodes {
		p.Field("pools",
			"pool node counts sum to %d but the cluster declares %d", total, s.Cluster.Nodes)
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
	if roles[RolePrefill] > 0 && s.PDTransfer == nil {
		p.Field("pd_transfer", "required for a disaggregated deployment")
	}
}

func validateParallelism(p *validate.Problems, at string, pl Parallelism, gpusPerNode int) {
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
	if pl.DPLocal > 0 && gpusPerNode > 0 && gpusPerNode%pl.DPLocal != 0 {
		p.Field(at+".parallel.dp_local",
			"%d does not divide gpus_per_node %d", pl.DPLocal, gpusPerNode)
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
	if e.GPUMemoryUtilization < 0 || e.GPUMemoryUtilization > 1 {
		p.Field(at+".engine.gpu_memory_utilization",
			"must lie in (0, 1], got %v", e.GPUMemoryUtilization)
	}
	if e.MaxNumBatchedTokens > 0 && e.MaxModelLen > 0 &&
		e.MaxNumBatchedTokens < 1 {
		p.Field(at+".engine.max_num_batched_tokens", "must be positive when stated")
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

func (s *Scenario) validateOffload(p *validate.Problems) {
	o := s.Offload
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
