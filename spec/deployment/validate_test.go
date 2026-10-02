package deployment

import "testing"

// pdDeployment is a prefill/decode-disaggregated deployment of the kind the design
// documents work through: two pools, different backends, expert parallelism. Its pool
// node counts sum to 60, which is the cluster ValidateAgainstCluster is tested against.
func pdDeployment() *Deployment {
	asyncNil := (*bool)(nil)
	return &Deployment{
		Kind: "Deployment", Name: "glm-5.3-60node-pd-h200",
		Pools: []Pool{
			{Role: RolePrefill, Nodes: 36,
				Parallel: Parallelism{TP: 1, PP: 1, DP: 8, DPLocal: 8,
					EnableExpertParallel: true},
				Engine: Engine{All2AllBackend: "deepep_high_throughput",
					AllReduceBackend: "nccl", CUDAGraphMode: "PIECEWISE",
					AsyncScheduling: asyncNil, BlockSize: 64,
					MaxNumBatchedTokens: 8192, MaxNumSeqs: 256,
					CacheDType: "fp8", GPUMemoryUtilization: 0.9}},
			{Role: RoleDecode, Nodes: 24,
				Parallel: Parallelism{TP: 1, PP: 1, DP: 16, DPLocal: 8,
					EnableExpertParallel: true},
				Engine: Engine{All2AllBackend: "deepep_low_latency",
					AllReduceBackend: "nccl", CUDAGraphMode: "FULL_AND_PIECEWISE",
					BlockSize: 64, CacheDType: "fp8", GPUMemoryUtilization: 0.9,
					EPLB: &EPLB{Enabled: true, NumRedundantExperts: 32}}},
		},
		PDTransfer: &PDTransfer{Connector: "nixl"},
	}
}

// singleNode is the colocated shape the published Granite runs use: eight GPUs,
// tensor parallel eight, one node.
func singleNodeDeployment() *Deployment {
	return &Deployment{
		Kind: "Deployment", Name: "granite-230b-h100-tp8",
		Pools: []Pool{{Role: RoleColocated, Nodes: 1,
			Parallel: Parallelism{TP: 8, PP: 1, DP: 1},
			Engine: Engine{CacheDType: "fp8", MaxNumBatchedTokens: 32768,
				GPUMemoryUtilization: 0.9, BlockSize: 16}}},
	}
}

func TestValidDeploymentsPass(t *testing.T) {
	for _, d := range []*Deployment{pdDeployment(), singleNodeDeployment()} {
		if p := d.Validate(); !p.OK() {
			t.Fatalf("%s rejected:\n%s", d.Name, p.Error())
		}
	}
}

// TestExpertParallelWidthIsDerived pins the rule that EP has no field: the width is
// a function of the other three, and no document can disagree with it.
func TestExpertParallelWidthIsDerived(t *testing.T) {
	cases := []struct {
		name string
		p    Parallelism
		want int
	}{
		{"off", Parallelism{TP: 8, DP: 1}, 1},
		{"tp1 dp8", Parallelism{TP: 1, DP: 8, EnableExpertParallel: true}, 8},
		{"tp1 dp16", Parallelism{TP: 1, DP: 16, EnableExpertParallel: true}, 16},
		{"tp2 dp8", Parallelism{TP: 2, DP: 8, EnableExpertParallel: true}, 16},
		{"pcp wider than dp", Parallelism{TP: 2, DP: 1, PCP: 4, EnableExpertParallel: true}, 8},
		{"zero guards", Parallelism{EnableExpertParallel: true}, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.p.ExpertParallelWidth(); got != c.want {
				t.Errorf("width = %d, want %d", got, c.want)
			}
		})
	}
}

func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Deployment)
	}{
		{"wrong kind", func(d *Deployment) { d.Kind = "Scenario" }},
		{"no name", func(d *Deployment) { d.Name = "" }},
		{"no pools", func(d *Deployment) { d.Pools = nil }},
		{"unknown role", func(d *Deployment) { d.Pools[0].Role = "warmup" }},
		{"zero tp", func(d *Deployment) { d.Pools[0].Parallel.TP = 0 }},
		{"pcp and dp both above one", func(d *Deployment) {
			d.Pools[0].Parallel.PCP = 2
			d.Pools[0].Parallel.DP = 8
		}},
		{"dp_local exceeds dp", func(d *Deployment) {
			d.Pools[0].Parallel.DPLocal = 99
		}},
		{"utilization above one", func(d *Deployment) {
			d.Pools[0].Engine.GPUMemoryUtilization = 1.5
		}},
		{"negative block size", func(d *Deployment) { d.Pools[0].Engine.BlockSize = -1 }},
		{"prefill without decode", func(d *Deployment) {
			d.Pools = d.Pools[:1]
		}},
		{"decode without prefill", func(d *Deployment) {
			d.Pools = d.Pools[1:]
		}},
		{"disaggregated without pd_transfer", func(d *Deployment) { d.PDTransfer = nil }},
		{"speculative with zero drafts", func(d *Deployment) {
			d.Pools[0].Engine.Speculative = &Speculative{Method: "mtp", NumSpecTokens: 0}
		}},
		{"offload with no tiers", func(d *Deployment) { d.Offload = &Offload{} }},
		{"offload tier with zero bytes", func(d *Deployment) {
			d.Offload = &Offload{Tiers: []Tier{{Device: "cpu_dram", Bytes: 0}}}
		}},
		{"duplicate offload tier", func(d *Deployment) {
			d.Offload = &Offload{Tiers: []Tier{
				{Device: "cpu_dram", Bytes: 1}, {Device: "cpu_dram", Bytes: 2}}}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := pdDeployment()
			tc.mutate(d)
			if p := d.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

// TestColocatedAlongsideDisaggregated covers the mixed-role case separately, since
// it needs three pools rather than a one-field mutation.
func TestColocatedAlongsideDisaggregated(t *testing.T) {
	d := pdDeployment()
	d.Pools = append(d.Pools, Pool{Role: RoleColocated, Nodes: 4,
		Parallel: Parallelism{TP: 1, PP: 1, DP: 1}})
	if p := d.Validate(); p.OK() {
		t.Fatal("expected rejection of a mixed-role deployment")
	}
}

// TestAllReduceRequestConflict covers the two ways a deployment can state the
// all-reduce choice inconsistently. The engine has a boolean; a resolved name is an
// override, and the pair must not contradict.
func TestAllReduceRequestConflict(t *testing.T) {
	yes, no := true, false
	d := singleNodeDeployment()
	d.Pools[0].Engine.DisableCustomAllReduce = &yes
	d.Pools[0].Engine.AllReduceBackend = "custom"
	if p := d.Validate(); p.OK() {
		t.Error("requesting the custom kernel while disabling it should fail")
	}
	// The reverse is a warning: the engine would use the kernel where reachable, so
	// naming nccl is a statement the layout may override rather than a contradiction.
	d2 := singleNodeDeployment()
	d2.Pools[0].Engine.DisableCustomAllReduce = &no
	d2.Pools[0].Engine.AllReduceBackend = "nccl"
	p2 := d2.Validate()
	if !p2.OK() {
		t.Errorf("naming nccl with the kernel enabled should warn, not fail:\n%s", p2.Error())
	}
	if len(p2.All()) == 0 {
		t.Error("expected a warning about the overridden request")
	}
	// Stating only the boolean is the ordinary case.
	d3 := singleNodeDeployment()
	d3.Pools[0].Engine.DisableCustomAllReduce = &yes
	if p := d3.Validate(); !p.OK() {
		t.Errorf("the boolean alone should be accepted:\n%s", p.Error())
	}
}

// TestValidateAgainstCluster covers the checks that couple a deployment to the
// available-hardware inventory it is placed on: these need the cluster, so they are
// separate from Validate and are exercised with the node and GPU counts a Scenario
// would supply.
func TestValidateAgainstCluster(t *testing.T) {
	// The pdDeployment pools sum to 60 nodes of 8 GPUs, which the matching cluster
	// declares, and every dp_local of 8 divides the node.
	if p := pdDeployment().ValidateAgainstCluster(ClusterConstraints{Nodes: 60, GPUsPerNode: 8}); !p.OK() {
		t.Fatalf("a deployment that fills its cluster should pass:\n%s", p.Error())
	}
	// Pool node counts that do not sum to the cluster are rejected.
	if p := pdDeployment().ValidateAgainstCluster(ClusterConstraints{Nodes: 59, GPUsPerNode: 8}); p.OK() {
		t.Error("pool nodes summing to 60 against a 59-node cluster should fail")
	}
	// A data-parallel-local width that does not divide the node is rejected.
	d := pdDeployment()
	d.Pools[0].Parallel.DPLocal = 3
	if p := d.ValidateAgainstCluster(ClusterConstraints{Nodes: 60, GPUsPerNode: 8}); p.OK() {
		t.Error("dp_local of 3 does not divide an 8-GPU node and should fail")
	}
}

// TestValidateAgainstClusterStorage covers the coupling check that an offload tier may
// only name a storage class the cluster's inventory declares. The inventory lives on the
// Scenario, so the composition layer supplies it; a cluster that lists no storage
// constrains nothing, which keeps a deployment that offloads without a declared
// inventory valid as it was before the inventory existed.
func TestValidateAgainstClusterStorage(t *testing.T) {
	withOffload := func(devices ...string) *Deployment {
		d := singleNodeDeployment()
		tiers := make([]Tier, len(devices))
		for i, dev := range devices {
			tiers[i] = Tier{Device: dev, Bytes: 1}
		}
		d.Offload = &Offload{Tiers: tiers}
		return d
	}
	inventory := []string{"cpu_dram", "nvme_gen4"}

	// A tier drawn from the declared inventory passes.
	if p := withOffload("cpu_dram").ValidateAgainstCluster(ClusterConstraints{Nodes: 1, GPUsPerNode: 8, Storage: inventory}); !p.OK() {
		t.Errorf("a tier in the inventory should pass:\n%s", p.Error())
	}
	// A tier naming a class the cluster does not list is rejected.
	if p := withOffload("optane").ValidateAgainstCluster(ClusterConstraints{Nodes: 1, GPUsPerNode: 8, Storage: inventory}); p.OK() {
		t.Error("a tier outside the cluster storage inventory should fail")
	}
	// A cluster that declares no storage inventory constrains nothing: the same offload
	// that would fail above passes, preserving pre-inventory behavior.
	if p := withOffload("optane").ValidateAgainstCluster(ClusterConstraints{Nodes: 1, GPUsPerNode: 8}); !p.OK() {
		t.Errorf("an undeclared inventory should not constrain offload:\n%s", p.Error())
	}
}
