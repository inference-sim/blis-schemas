package deployment

import (
	"fmt"
	"math"
	"sort"

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
// count, that no engine needs more GPUs than its pool owns (or than a node holds, for
// the replicas it places there), and that every offload tier names a storage class the
// cluster actually lists.
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
		validateRankFit(p, fmt.Sprintf("pools[%d]", i), pool, c)
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

// validateRankFit refuses an engine that needs more GPUs than it has been given. Every
// rank is one device, so a layout's footprint is its rank count; nothing in the
// divisibility or node-sum checks relates the two, which is how tp 8 with pcp 4 on a
// single 8-GPU node validated while needing 32.
//
// There are two bounds, both against the engine and neither an equality:
//
//   - The engine against its pool: pp x tp x pcp x dp ranks must not exceed the GPUs in
//     the pool's nodes. DCP and expert parallelism add none — DCP reuses the
//     tensor-parallel ranks and an expert group is carved across ranks that already
//     exist — so neither enters the product.
//   - The replicas placed on a node against that node: dp_local replicas of pp x tp x
//     pcp ranks each must fit in gpus_per_node. The total cannot see this. tp 4 with
//     dp 4 and dp_local 4 over two 8-GPU nodes totals 16 and fits, yet puts 16 ranks on
//     one node of 8.
//
// Both are upper bounds because a pool is the set of nodes serving a role, not a single
// engine: 36 one-node dp-8 engines behind a router is a 36-node pool whose layout is
// dp 8, and an engine may leave devices idle. Requiring equality would reject those, so
// only the impossible case is refused — an engine larger than the room it has.
//
// The rank counts mirror the engine's own: world size is pp x tp x pcp per data-parallel
// replica, and local data-parallel rank i takes devices [i x world, (i+1) x world).
//
// A width the checks above already reported as malformed describes no layout, so this
// declines to run over it rather than invent a rank count and file a second problem
// naming the wrong field. An unset pcp is the one zero that is meaningful: it means off,
// and counts as one. A cluster that states no GPU count constrains nothing, the way an
// undeclared storage inventory does.
func validateRankFit(p *validate.Problems, at string, pool Pool, c ClusterConstraints) {
	pl := pool.Parallel
	if pl.TP < 1 || pl.PP < 1 || pl.DP < 1 || pl.PCP < 0 ||
		pool.Nodes < 1 || c.GPUsPerNode < 1 {
		return
	}
	pcp := pl.PCP
	if pcp < 1 {
		pcp = 1
	}
	// Saturating products: a width absurd enough to overflow must read as too many,
	// not wrap to a small or negative count that passes.
	replica := mulSat(mulSat(pl.PP, pl.TP), pcp)
	ranks := mulSat(replica, pl.DP)
	gpus := mulSat(pool.Nodes, c.GPUsPerNode)

	if ranks > gpus {
		p.Field(at+".parallel",
			"needs %d GPUs (pp %d x tp %d x pcp %d x dp %d, one per rank) but the pool's %d node(s) of %d GPUs provide %d",
			ranks, pl.PP, pl.TP, pcp, pl.DP, pool.Nodes, c.GPUsPerNode, gpus)
	}
	// Only when one replica fits inside a node. A replica wider than a node spans nodes,
	// and how many of its ranks sit on each depends on a node split this schema does not
	// carry, so no per-node bound is asserted there.
	if pl.DPLocal > 0 && replica <= c.GPUsPerNode {
		if local := mulSat(pl.DPLocal, replica); local > c.GPUsPerNode {
			p.Field(at+".parallel.dp_local",
				"%d replicas of %d GPUs (pp %d x tp %d x pcp %d) need %d GPUs on one node but a node has %d",
				pl.DPLocal, replica, pl.PP, pl.TP, pcp, local, c.GPUsPerNode)
		}
	}
}

// mulSat multiplies two non-negative ints, saturating at the largest int rather than
// wrapping.
func mulSat(a, b int) int {
	if a == 0 || b == 0 {
		return 0
	}
	if a > math.MaxInt/b {
		return math.MaxInt
	}
	return a * b
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
	//
	// Upstream has since narrowed this. It was a version-agnostic config check; it is
	// now raised per platform ("PCP does not support data parallelism on CUDA yet"),
	// which makes it an accelerator-specific limitation rather than a structural one,
	// and so arguably a rules-pack concern rather than a field check. It is left here
	// because moving a pre-existing check would change which deployments validate,
	// which is not this change's business — but two things depend on it, and a reader
	// re-scoping it later must handle both: the problem reported just below, and
	// ExpertParallelWidth's use of max(dp, pcp), whose agreement with the engine's
	// tp x pcp x dp product holds only where one of the two is guaranteed to be 1.
	if pl.PCP > 1 && pl.DP > 1 {
		p.Field(at+".parallel.pcp",
			"prefill-context parallelism and data parallelism cannot both exceed 1")
	}
	validateDecodeContextParallel(p, at, pl)
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

// validateDecodeContextParallel constrains decode-context-parallel width against the
// group it is carved out of. DCP has no ranks of its own: it reuses the tensor-parallel
// ranks when prefill-context parallelism is off, and the TP x PCP block when it is on.
// A width that does not partition that group describes no layout, so it is refused when
// the document loads rather than priced downstream.
//
// Two rules, mutually exclusive on whether PCP is enabled:
//
//   - PCP off: the DCP group must partition the TP group, so tp % dcp == 0.
//   - PCP on: DCP may be off, span the PCP axis, or span the whole TP x PCP block.
//     Nothing between, because any other width would straddle the two axes.
//
// Both are structural — they follow from how the group is built, need no coefficients
// and no model graph — which is why they are field checks here rather than a
// version-scoped rule. They mirror what the engine itself refuses at startup.
func validateDecodeContextParallel(p *validate.Problems, at string, pl Parallelism) {
	// Problems accumulates rather than aborting, so a width already reported above
	// arrives here unchanged. These rules are stated over a real rank group, and a
	// malformed width describes none, so running them anyway does active harm in two
	// ways: it files a second problem naming dcp when the fault is another field, and
	// — worse — it can ACCEPT a dcp that the corrected width would reject, so the
	// author sees a fresh error only on a later run. That second round is exactly
	// what an accumulating problem list exists to prevent, so every width the checks
	// above already reported is a reason to stop here rather than guess at intent.
	//
	// A negative pcp is the instructive case. Normalising it to 1 below would route
	// the deployment into the prefill-context-parallelism-OFF branch and assert, in
	// the message, that pcp is off — when the document asked for it and merely asked
	// malformedly. tp 8, pcp -2, dcp 4 is the masking half: 8 % 4 is 0 so nothing is
	// reported, while at the evident pcp of 2 the admissible set is {1, 2, 16} and 4
	// is not in it.
	if pl.TP < 1 || pl.PCP < 0 || pl.DCP < 0 {
		return
	}
	// Both widths are omitempty, so an absent field arrives as 0 meaning "not enabled".
	// The rules are stated over enabled widths, so normalise before applying them —
	// otherwise tp % dcp divides by zero for every deployment that simply omits dcp.
	// Only a zero reaches this point: a negative returned above.
	pcp, dcp := pl.PCP, pl.DCP
	if pcp < 1 {
		pcp = 1
	}
	if dcp < 1 {
		dcp = 1
	}
	if pcp == 1 {
		if pl.TP%dcp != 0 {
			p.Field(at+".parallel.dcp",
				"decode-context parallelism reuses the tensor-parallel ranks when prefill-context parallelism is off, so tp %d must be divisible by dcp %d",
				pl.TP, dcp)
		}
		return
	}
	// Name the admissible set rather than only the violation: the reader's next
	// question is which widths would work.
	if dcp != 1 && dcp != pcp && dcp != pl.TP*pcp {
		p.Field(at+".parallel.dcp",
			"with prefill-context parallelism enabled, decode-context parallelism must be disabled, span the pcp axis, or span the full tp x pcp axis; got tp %d, pcp %d, dcp %d, so the admissible widths are %v",
			pl.TP, pcp, dcp, admissibleDCP(pl.TP, pcp))
	}
}

// admissibleDCP returns the sorted, deduplicated widths the PCP-enabled rule allows.
// Deduplication matters because at tp 1 the full TP x PCP block IS the pcp axis, so a
// literal three-element list would repeat a width and read as though it were three
// distinct choices. It is only ever called with pcp above 1, which is the branch that
// has an admissible set to name.
func admissibleDCP(tp, pcp int) []int {
	seen := map[int]bool{}
	var out []int
	for _, w := range []int{1, pcp, tp * pcp} {
		if !seen[w] {
			seen[w] = true
			out = append(out, w)
		}
	}
	sort.Ints(out)
	return out
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
	// (0, 1] so it does not read as rejecting the zero the check deliberately allows. The
	// finite check runs first: a NaN would pass the range comparison (NaN < 0 and NaN > 1
	// are both false) and reach the engine as a garbage fraction.
	if p.FiniteField(at+".engine.gpu_memory_utilization", e.GPUMemoryUtilization) &&
		(e.GPUMemoryUtilization < 0 || e.GPUMemoryUtilization > 1) {
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
