package scenario

import "testing"

// pdScenario is a prefill/decode-disaggregated deployment of the kind the design
// documents work through: two pools, different backends, expert parallelism.
func pdScenario() *Scenario {
	asyncNil := (*bool)(nil)
	return &Scenario{
		Kind: "Scenario", Name: "glm-5.3-60node-pd-h200",
		Model: "glm-5.3", Hardware: "h200", Fabric: "ib-400g",
		Coefficients:  []string{"cost-model-primitives-h200"},
		EngineVersion: "0.29.0",
		Cluster:       Cluster{Nodes: 60, GPUsPerNode: 8},
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
// tensor parallel eight, no fabric because nothing crosses a node.
func singleNode() *Scenario {
	return &Scenario{
		Kind: "Scenario", Name: "granite-230b-h100-tp8",
		Model: "granite-5-230b", Hardware: "h100",
		Coefficients:  []string{"cost-model-primitives-h100"},
		EngineVersion: "0.29.0",
		Cluster:       Cluster{Nodes: 1, GPUsPerNode: 8},
		Pools: []Pool{{Role: RoleColocated, Nodes: 1,
			Parallel: Parallelism{TP: 8, PP: 1, DP: 1},
			Engine: Engine{CacheDType: "fp8", MaxNumBatchedTokens: 32768,
				GPUMemoryUtilization: 0.9, BlockSize: 16}}},
	}
}

func TestValidScenariosPass(t *testing.T) {
	for _, s := range []*Scenario{pdScenario(), singleNode()} {
		if p := s.Validate(); !p.OK() {
			t.Fatalf("%s rejected:\n%s", s.Name, p.Error())
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

func TestClusterGPUs(t *testing.T) {
	if got := (Cluster{Nodes: 60, GPUsPerNode: 8}).GPUs(); got != 480 {
		t.Errorf("GPUs() = %d, want 480", got)
	}
}

func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Scenario)
	}{
		{"wrong kind", func(s *Scenario) { s.Kind = "Deployment" }},
		{"no model", func(s *Scenario) { s.Model = "" }},
		{"no hardware", func(s *Scenario) { s.Hardware = "" }},
		{"no engine version", func(s *Scenario) { s.EngineVersion = "" }},
		{"no coefficients", func(s *Scenario) { s.Coefficients = nil }},
		{"zero nodes", func(s *Scenario) { s.Cluster.Nodes = 0 }},
		{"multi-node without a fabric", func(s *Scenario) { s.Fabric = "" }},
		{"rack not a multiple of node", func(s *Scenario) { s.Cluster.GPUsPerRack = 70 }},
		{"no pools", func(s *Scenario) { s.Pools = nil }},
		{"pool nodes do not sum", func(s *Scenario) { s.Pools[0].Nodes = 1 }},
		{"unknown role", func(s *Scenario) { s.Pools[0].Role = "warmup" }},
		{"zero tp", func(s *Scenario) { s.Pools[0].Parallel.TP = 0 }},
		{"pcp and dp both above one", func(s *Scenario) {
			s.Pools[0].Parallel.PCP = 2
			s.Pools[0].Parallel.DP = 8
		}},
		{"dp_local does not divide the node", func(s *Scenario) {
			s.Pools[0].Parallel.DPLocal = 3
		}},
		{"dp_local exceeds dp", func(s *Scenario) {
			s.Pools[0].Parallel.DPLocal = 99
		}},
		{"utilization above one", func(s *Scenario) {
			s.Pools[0].Engine.GPUMemoryUtilization = 1.5
		}},
		{"negative block size", func(s *Scenario) { s.Pools[0].Engine.BlockSize = -1 }},
		{"prefill without decode", func(s *Scenario) {
			s.Pools = s.Pools[:1]
			s.Cluster.Nodes = 36
		}},
		{"disaggregated without pd_transfer", func(s *Scenario) { s.PDTransfer = nil }},
		{"speculative with zero drafts", func(s *Scenario) {
			s.Pools[0].Engine.Speculative = &Speculative{Method: "mtp", NumSpecTokens: 0}
		}},
		{"offload with no tiers", func(s *Scenario) { s.Offload = &Offload{} }},
		{"offload tier with zero bytes", func(s *Scenario) {
			s.Offload = &Offload{Tiers: []Tier{{Device: "cpu_dram", Bytes: 0}}}
		}},
		{"duplicate offload tier", func(s *Scenario) {
			s.Offload = &Offload{Tiers: []Tier{
				{Device: "cpu_dram", Bytes: 1}, {Device: "cpu_dram", Bytes: 2}}}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := pdScenario()
			tc.mutate(s)
			if p := s.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

// TestColocatedAlongsideDisaggregated covers the mixed-role case separately, since
// it needs three pools rather than a one-field mutation.
func TestColocatedAlongsideDisaggregated(t *testing.T) {
	s := pdScenario()
	s.Pools = append(s.Pools, Pool{Role: RoleColocated, Nodes: 4,
		Parallel: Parallelism{TP: 1, PP: 1, DP: 1}})
	s.Cluster.Nodes = 64
	if p := s.Validate(); p.OK() {
		t.Fatal("expected rejection of a mixed-role deployment")
	}
}

// TestAllReduceRequestConflict covers the two ways a scenario can state the
// all-reduce choice inconsistently. The engine has a boolean; a resolved name is an
// override, and the pair must not contradict.
func TestAllReduceRequestConflict(t *testing.T) {
	yes, no := true, false
	s := singleNode()
	s.Pools[0].Engine.DisableCustomAllReduce = &yes
	s.Pools[0].Engine.AllReduceBackend = "custom"
	if p := s.Validate(); p.OK() {
		t.Error("requesting the custom kernel while disabling it should fail")
	}
	// The reverse is a warning: the engine would use the kernel where reachable, so
	// naming nccl is a statement the layout may override rather than a contradiction.
	s2 := singleNode()
	s2.Pools[0].Engine.DisableCustomAllReduce = &no
	s2.Pools[0].Engine.AllReduceBackend = "nccl"
	p2 := s2.Validate()
	if !p2.OK() {
		t.Errorf("naming nccl with the kernel enabled should warn, not fail:\n%s", p2.Error())
	}
	if len(p2.All()) == 0 {
		t.Error("expected a warning about the overridden request")
	}
	// Stating only the boolean is the ordinary case.
	s3 := singleNode()
	s3.Pools[0].Engine.DisableCustomAllReduce = &yes
	if p := s3.Validate(); !p.OK() {
		t.Errorf("the boolean alone should be accepted:\n%s", p.Error())
	}
}
