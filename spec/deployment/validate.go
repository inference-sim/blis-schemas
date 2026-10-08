package deployment

import (
	"fmt"
	"math/big"
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
// count, that no engine needs more GPUs than its pool owns or places a local replica
// past the end of a node, and that every offload tier names a storage class the cluster
// actually lists.
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
// rank is one device — the engine's worker binds cuda:local_rank — so a layout's
// footprint is its rank count; nothing in the divisibility or node-sum checks relates the
// two, which is how tp 8 with pcp 4 on a single 8-GPU node validated while needing 32.
//
// Two bounds, each refusing only what no launch can run.
//
// The engine against its pool: pp x tp x pcp x dp ranks must not exceed the GPUs in the
// pool's nodes. This is the engine's own world_size_across_dp. DCP and expert parallelism
// add no ranks — DCP reuses the tensor-parallel ranks, and an expert group is built over
// ranks that already exist — so neither enters the product. It is an upper bound because
// a pool may hold several engines: llm-d runs a LeaderWorkerSet of `replicas` groups, each
// one engine over `size` nodes, so a 36-node pool of dp 8 is 36 engines of one node each.
//
// The local replicas against one node, when dp_local is stated. The engine places local
// data-parallel replica i at devices [i x world, i x world + local_world), where world is
// pp x tp x pcp and local_world is the share of one replica on one node
// (get_physical_gpu_ids_for_local_dp_rank). That share depends on the engine's node
// count n, which the schema does not carry and the launch chooses:
//
//   - n is 1 for each process llm-d starts — one vLLM per pod, joined by
//     --data-parallel-start-rank — so local_world is the whole replica.
//   - For n above 1, the engine splits each replica over n / (dp / dp_local) nodes
//     (nnodes_within_dp); that must divide world, which the multiprocessing executor
//     asserts, and local_world is world divided by it.
//
// So the bound asserted is: some n from 1 to the pool's node count gives a split under
// which every local replica's devices lie on the node. That refuses exactly what no
// launch can place — including a split that would need more nodes than the pool owns —
// and nothing the engine runs. It is the same at v0.29.0, which rounds n / (dp /
// dp_local) down, and later releases, which require it to divide: rounding n down to a
// multiple gives the same split, so the set of placeable layouts is identical.
//
// When dp_local is unset the engine infers it from the launch, so no per-node verdict is
// possible and none is given.
//
// A width the checks above already reported as malformed describes no layout, so each
// bound declines to run over it rather than invent a count and file a second problem
// naming the wrong field. That matters for more than tidiness: two negative widths
// multiply to a large positive one. An unset pcp is the one zero that is meaningful — it
// means off, and counts as one. A cluster or pool that states no room constrains nothing,
// the way an undeclared storage inventory does. The per-node bound also declines when
// the pool bound has already refused the layout: an engine that does not fit at all does
// not need a second explanation of why it does not fit on a node.
//
// Counts are exact (math/big) rather than int: a width absurd enough to overflow must still
// compare correctly against a pool that is also absurdly large, and a message must never
// print a clamped number as though it were the real one.
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
	world := product(pl.PP, pl.TP, pcp)
	ranks := new(big.Int).Mul(world, big.NewInt(int64(pl.DP)))
	gpus := product(pool.Nodes, c.GPUsPerNode)
	if ranks.Cmp(gpus) > 0 {
		p.Field(at+".parallel",
			"needs %s GPUs (pp %d x tp %d x pcp %d x dp %d, one per rank) but the pool's %d node(s) of %d GPUs provide %s",
			ranks, pl.PP, pl.TP, pcp, pl.DP, pool.Nodes, c.GPUsPerNode, gpus)
		return
	}

	// dp_local unset, above dp, or negative states no replica count to place.
	if pl.DPLocal < 1 || pl.DPLocal > pl.DP {
		return
	}
	gpn := big.NewInt(int64(c.GPUsPerNode))
	last := new(big.Int).Mul(big.NewInt(int64(pl.DPLocal-1)), world)
	if last.Cmp(gpn) >= 0 {
		// The last local replica would start past the node under every split.
		p.Field(at+".parallel.dp_local",
			"%d local replicas of %s GPUs (pp %d x tp %d x pcp %d) are placed one after another from device 0, so the last would start at device %s, but a node has %d",
			pl.DPLocal, world, pl.PP, pl.TP, pcp, last, c.GPUsPerNode)
		return
	}
	// The last replica starts on the node; its share must end there too. world fits an
	// int64 here, since the pool bound held and every factor is an int.
	groups := pl.DP / pl.DPLocal // the engine's data_parallel_node_size
	maxSplit := pool.Nodes / groups
	if maxSplit < 1 {
		maxSplit = 1 // n = 1 always gives a split of 1
	}
	room := int64(c.GPUsPerNode) - last.Int64()
	switch placeable(world.Int64(), int64(maxSplit), room) {
	case placeNo:
		p.Field(at+".parallel",
			"no node count up to the pool's %d lets the engine place %d local replicas of %s GPUs (pp %d x tp %d x pcp %d) on a %d-GPU node: replica i starts at device i x %s, so the last leaves %d GPUs for its share of the replica, and every split the pool allows (a divisor of %s over at most %d nodes) leaves a larger share",
			pool.Nodes, pl.DPLocal, world, pl.PP, pl.TP, pcp, c.GPUsPerNode, world, room, world, maxSplit)
	case placeUnknown:
		// Only for widths far beyond any real cluster: the search is bounded, and an
		// unanswered question is not a refusal.
	}
}

type placement int

const (
	placeYes placement = iota
	placeNo
	placeUnknown
)

// placeableSearchLimit bounds the divisor search below. Real layouts need a few dozen
// steps; the limit exists so an absurd width cannot make validation hang.
const placeableSearchLimit = 1 << 20

// placeable reports whether some split k — a divisor of world, at most maxSplit — leaves
// a per-node share world/k of at most room. It searches whichever of the two ranges is
// shorter: the splits themselves, or the shares.
func placeable(world, maxSplit, room int64) placement {
	lo := (world + room - 1) / room // the smallest split whose share fits
	hi := maxSplit
	if world < hi {
		hi = world
	}
	if lo > hi {
		return placeNo
	}
	shares := room
	if world < shares {
		shares = world
	}
	switch {
	case hi-lo+1 <= shares && hi-lo+1 <= placeableSearchLimit:
		for k := lo; k <= hi; k++ {
			if world%k == 0 {
				return placeYes
			}
		}
	case shares <= placeableSearchLimit:
		for share := int64(1); share <= shares; share++ {
			if world%share == 0 {
				if k := world / share; k >= lo && k <= hi {
					return placeYes
				}
			}
		}
	default:
		return placeUnknown
	}
	return placeNo
}

// product multiplies non-negative ints exactly.
func product(xs ...int) *big.Int {
	out := big.NewInt(1)
	for _, x := range xs {
		out.Mul(out, big.NewInt(int64(x)))
	}
	return out
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
	// Whether prefill-context parallelism may be combined with data parallelism is
	// not a field check: it depends on the engine release. v0.29.0 refuses it; the
	// engine supports it from e6dc16cebd, and llm-d deploys it (DP 4 x PCP 8). A
	// release that refuses it says so in its rules pack.
	validateDecodeContextParallel(p, at, pl)
	if pl.DPLocal < 0 {
		// The engine constrains it ge=0; zero is unset here, as there.
		p.Field(at+".parallel.dp_local", "must not be negative, got %d", pl.DPLocal)
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
	//
	// The full TP x PCP block is compared exactly. The engine's own product is Python's,
	// which does not overflow, and an int product would wrap: tp MaxInt, pcp 3 wraps to
	// MaxInt-2 and would admit a dcp the engine rejects.
	block := new(big.Int).Mul(big.NewInt(int64(pl.TP)), big.NewInt(int64(pcp)))
	if dcp != 1 && dcp != pcp && block.Cmp(big.NewInt(int64(dcp))) != 0 {
		p.Field(at+".parallel.dcp",
			"with prefill-context parallelism enabled, decode-context parallelism must be disabled, span the pcp axis, or span the full tp x pcp axis; got tp %d, pcp %d, dcp %d, so the admissible widths are %v",
			pl.TP, pcp, dcp, admissibleDCP(pcp, block))
	}
}

// admissibleDCP returns the sorted, deduplicated widths the PCP-enabled rule allows:
// off, the pcp axis, or the full tp x pcp block. Deduplication matters because at tp 1
// the block IS the pcp axis, so a literal three-element list would repeat a width and
// read as though it were three distinct choices. The block is exact, so the set printed
// is the set the engine would name.
func admissibleDCP(pcp int, block *big.Int) []*big.Int {
	out := []*big.Int{big.NewInt(1)}
	for _, w := range []*big.Int{big.NewInt(int64(pcp)), block} {
		dup := false
		for _, have := range out {
			if have.Cmp(w) == 0 {
				dup = true
			}
		}
		if !dup {
			out = append(out, w)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Cmp(out[j]) < 0 })
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
