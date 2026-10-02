package blisschemas

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

// This file is the coverage suite: it asserts that the schemas express the
// deployments the three design documents work through and the ones a published
// report corpus measured. A schema that validates its own examples proves little;
// these cases come from outside it.

func digest() string { return strings.Repeat("c", 64) }

// graniteMoE is the 230B preview: a uniform 72-layer MoE stack, 224 experts,
// GQA over 8 KV heads. It is NOT hybrid, which is why the stack has one layer kind.
func graniteMoE() *model.Graph {
	return &model.Graph{
		Kind: "ModelGraph", Name: "granite-5-230b",
		DerivedFrom: model.Derivation{Format: "hf_config_json",
			Path: "config.json", SHA256: digest(), DeriverVersion: 1},
		Global: model.GlobalShape{HiddenSize: 3072, VocabSize: 100352,
			WeightDType: model.DTypeFP8},
		LayerKinds: []model.LayerKind{{ID: "attn_moe", Nodes: []model.Node{
			{Op: model.OpElementwise, Role: "input_norm"},
			{Op: model.OpGEMM, Role: "qkv_proj", N: 4096, K: 3072},
			{Op: model.OpAttention, AttentionKind: model.AttentionGQA,
				NumQHeads: 48, NumKVHeads: 8, HeadDim: 64},
			{Op: model.OpGEMM, Role: "o_proj", N: 3072, K: 3072},
			{Op: model.OpAllReduce, Role: "attn_out", Emit: model.EmitTensorParallel},
			{Op: model.OpGroupedGEMM, Role: "experts", N: 1536, K: 3072,
				Experts: 224, TopK: 8},
			{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
			{Op: model.OpAllReduce, Role: "mlp_out", Emit: model.EmitTensorParallelUnlessSequenceParallelMoE},
		}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}}}},
		Stack: model.Stack{Pattern: []string{"attn_moe"}, Repeat: 72},
	}
}

// hybridMLAKDA is a Kimi-shaped stack: latent attention interleaved with linear
// attention, three recurrent layers to one attention layer.
func hybridMLAKDA() *model.Graph {
	attn := model.LayerKind{ID: "mla_moe", Nodes: []model.Node{
		{Op: model.OpElementwise, Role: "input_norm"},
		{Op: model.OpAttention, AttentionKind: model.AttentionMLA,
			NumQHeads: 64, NumKVHeads: 1, HeadDim: 576,
			KVLoRARank: 512, QKRopeHeadDim: 64},
		{Op: model.OpGroupedGEMM, Role: "experts", N: 2048, K: 7168,
			Experts: 256, TopK: 8},
		{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
	}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}}}
	kda := model.LayerKind{ID: "kda_moe", Nodes: []model.Node{
		{Op: model.OpElementwise, Role: "input_norm"},
		{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentKDA,
			NumHeads: 64, StateSize: 128, StateDType: model.DTypeFP32},
		{Op: model.OpGroupedGEMM, Role: "experts", N: 2048, K: 7168,
			Experts: 256, TopK: 8},
		{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
	}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}}}
	return &model.Graph{
		Kind: "ModelGraph", Name: "kimi-k3",
		DerivedFrom: model.Derivation{Format: "hf_config_json",
			Path: "config.json", SHA256: digest(), DeriverVersion: 1},
		Global: model.GlobalShape{HiddenSize: 7168, VocabSize: 163840,
			WeightDType: model.DTypeFP8},
		LayerKinds: []model.LayerKind{kda, attn},
		Stack: model.Stack{
			Pattern: []string{"kda_moe", "kda_moe", "kda_moe", "mla_moe"},
			Repeat:  15},
		Speculator: &model.Speculator{Method: "kimi_k3_mtp", NumSpec: 1,
			Stack: model.Stack{Pattern: []string{"mla_moe"}, Repeat: 1}},
	}
}

// mambaHybrid is a Granite-hybrid-shaped stack: Mamba2 state-space layers, whose
// state is two tensors rather than one.
func mambaHybrid() *model.Graph {
	g := graniteMoE()
	g.Name = "granite-hybrid"
	g.LayerKinds = append(g.LayerKinds, model.LayerKind{
		ID: "mamba_moe", Nodes: []model.Node{
			{Op: model.OpElementwise, Role: "input_norm"},
			{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentMamba2,
				NumHeads: 128, StateSize: 128, NumGroups: 8, ConvKernel: 4,
				IntermediateSize: 8192, StateDType: model.DTypeFP32},
			{Op: model.OpGroupedGEMM, Role: "experts", N: 1536, K: 3072,
				Experts: 224, TopK: 8},
			{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
		}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}}})
	g.Stack = model.Stack{
		Pattern: []string{"mamba_moe", "mamba_moe", "mamba_moe", "attn_moe"},
		Repeat:  18}
	return g
}

// TestModelShapesAreExpressible covers the architecture families the three design
// documents and the report corpus span: uniform MoE, latent-plus-linear hybrid, and
// state-space hybrid.
func TestModelShapesAreExpressible(t *testing.T) {
	for _, g := range []*model.Graph{graniteMoE(), hybridMLAKDA(), mambaHybrid()} {
		t.Run(g.Name, func(t *testing.T) {
			if p := g.Validate(); !p.OK() {
				t.Fatalf("%s is not expressible:\n%s", g.Name, p.Error())
			}
		})
	}
	if got := graniteMoE().Stack.Layers(); got != 72 {
		t.Errorf("granite layers = %d, want 72", got)
	}
	if got := hybridMLAKDA().Stack.Layers(); got != 60 {
		t.Errorf("kimi layers = %d, want 60", got)
	}
	if got := mambaHybrid().Stack.Layers(); got != 72 {
		t.Errorf("granite-hybrid layers = %d, want 72", got)
	}
}

// corpus builds a Scenario+Deployment pair from the parts a corpus report states: the
// immutable problem (model, available-hardware inventory, coefficient/engine refs) and
// the mutable layout (pools, offload, PD transfer) chosen against it.
func corpus(name string, nodes, gpusPerNode int, fabric string,
	pools []deployment.Pool,
	mutate func(*scenario.Scenario, *deployment.Deployment)) Bundle {
	s := &scenario.Scenario{
		Kind: "Scenario", Name: name, Model: "granite-5-230b",
		Coefficients:  []string{"cost-model-primitives-h200"},
		EngineVersion: "0.29.0",
		Cluster: scenario.Cluster{Hardware: "h200", Fabric: fabric,
			Nodes: nodes, GPUsPerNode: gpusPerNode},
	}
	d := &deployment.Deployment{Kind: "Deployment", Name: name, Pools: pools}
	if mutate != nil {
		mutate(s, d)
	}
	return Bundle{Scenario: s, Deployment: d}
}

func pool(role deployment.Role, nodes int, pl deployment.Parallelism,
	e deployment.Engine) deployment.Pool {
	return deployment.Pool{Role: role, Nodes: nodes, Parallel: pl, Engine: e}
}

// TestCorpusDeploymentsAreExpressible walks the deployment shapes the published
// reports measured and the design documents work through. Each case names the knob
// it is testing, so a failure says which capability was lost.
func TestCorpusDeploymentsAreExpressible(t *testing.T) {
	tp8 := deployment.Parallelism{TP: 8, PP: 1, DP: 1}
	cases := []struct {
		what string
		b    Bundle
	}{
		{
			what: "single node, tp8, fp8 KV cache, large batched-token bound",
			b: corpus("granite-h100-tp8", 1, 8, "", []deployment.Pool{
				pool(deployment.RoleColocated, 1, tp8, deployment.Engine{
					CacheDType: "fp8", BlockSize: 16, MaxNumBatchedTokens: 32768,
					MaxNumSeqs: 1024, GPUMemoryUtilization: 0.9,
					CUDAGraphMode: "FULL_AND_PIECEWISE"}),
			}, nil),
		},
		{
			what: "CPU offload with a 1 TiB tier and a lazy connector",
			b: corpus("granite-h200-offload", 1, 8, "", []deployment.Pool{
				pool(deployment.RoleColocated, 1, tp8, deployment.Engine{
					CacheDType: "fp8", BlockSize: 16, GPUMemoryUtilization: 0.9}),
			}, func(s *scenario.Scenario, d *deployment.Deployment) {
				d.Offload = &deployment.Offload{
					Connector: "OffloadingConnector", Spec: "CPUOffloadingSpec",
					EvictionPolicy: "lru", PrefetchDepth: 2,
					Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 40}}}
			}),
		},
		{
			what: "multi-tier offload: CPU above NVMe",
			b: corpus("granite-h200-tiered", 1, 8, "", []deployment.Pool{
				pool(deployment.RoleColocated, 1, tp8, deployment.Engine{
					CacheDType: "fp8", BlockSize: 16, GPUMemoryUtilization: 0.9}),
			}, func(s *scenario.Scenario, d *deployment.Deployment) {
				d.Offload = &deployment.Offload{Spec: "TieringOffloadingSpec",
					EvictionPolicy: "arc", PrefetchDepth: 3,
					Tiers: []deployment.Tier{
						{Device: "cpu_dram", Bytes: 512 << 30},
						{Device: "nvme_gen4", Bytes: 4 << 40}}}
			}),
		},
		{
			what: "prefill/decode disaggregation across 60 nodes, wide EP",
			b: corpus("glm-5.3-60node-pd", 60, 8, "ib-400g", []deployment.Pool{
				pool(deployment.RolePrefill, 36, deployment.Parallelism{TP: 1, PP: 1,
					DP: 8, DPLocal: 8, EnableExpertParallel: true},
					deployment.Engine{All2AllBackend: "deepep_high_throughput",
						CUDAGraphMode: "PIECEWISE", CacheDType: "fp8",
						BlockSize: 64, GPUMemoryUtilization: 0.9,
						Speculative: &deployment.Speculative{Method: "glm4_moe_mtp",
							NumSpecTokens: 1}}),
				pool(deployment.RoleDecode, 24, deployment.Parallelism{TP: 1, PP: 1,
					DP: 16, DPLocal: 8, EnableExpertParallel: true},
					deployment.Engine{All2AllBackend: "deepep_low_latency",
						CUDAGraphMode: "FULL_AND_PIECEWISE", CacheDType: "fp8",
						BlockSize: 64, GPUMemoryUtilization: 0.9,
						DBO: &deployment.DBO{Enabled: true,
							DecodeTokenThreshold: 32, PrefillTokenThreshold: 512},
						EPLB: &deployment.EPLB{Enabled: true,
							NumRedundantExperts: 32, WindowSize: 1000,
							StepInterval: 3000}}),
			}, func(s *scenario.Scenario, d *deployment.Deployment) {
				d.PDTransfer = &deployment.PDTransfer{Connector: "nixl"}
			}),
		},
		{
			what: "aggregated 32-GPU serving, the alternative PD is compared against",
			b: corpus("glm-5.3-flash-32gpu", 4, 8, "ib-400g", []deployment.Pool{
				pool(deployment.RoleColocated, 4, deployment.Parallelism{TP: 1, PP: 1,
					DP: 32, DPLocal: 8, EnableExpertParallel: true},
					deployment.Engine{All2AllBackend: "deepep_low_latency",
						CacheDType: "fp8", BlockSize: 64,
						GPUMemoryUtilization: 0.9}),
			}, nil),
		},
		{
			what: "eager execution with the custom all-reduce disabled",
			b: corpus("nemotron-h100-eager", 2, 8, "roce-200g", []deployment.Pool{
				pool(deployment.RoleColocated, 2, deployment.Parallelism{TP: 8, PP: 1,
					DP: 2, DPLocal: 1, EnableExpertParallel: true},
					deployment.Engine{CUDAGraphMode: "NONE",
						AllReduceBackend: "nccl", CacheDType: "auto",
						All2AllBackend:       "deepep_high_throughput",
						GPUMemoryUtilization: 0.9}),
			}, nil),
		},
		{
			what: "hybrid stack: mamba cache mode and both state dtypes",
			b: corpus("granite-hybrid-h200", 1, 8, "", []deployment.Pool{
				pool(deployment.RoleColocated, 1, tp8, deployment.Engine{
					CacheDType: "fp8", MambaCacheDType: "auto",
					MambaSSMCacheDType: "auto", MambaCacheMode: "align",
					MaxModelLen: 131072, GPUMemoryUtilization: 0.9}),
			}, nil),
		},
		{
			what: "priority scheduling with async scheduling declined",
			b: corpus("granite-priority", 1, 8, "", []deployment.Pool{
				pool(deployment.RoleColocated, 1, tp8, deployment.Engine{
					SchedulingPolicy: "priority", AsyncScheduling: boolPtr(false),
					CacheDType: "fp8", GPUMemoryUtilization: 0.9}),
			}, nil),
		},
		{
			what: "GB200-class three-tier topology",
			b: corpus("dsr1-gb200", 18, 4, "ib-400g", []deployment.Pool{
				pool(deployment.RoleColocated, 18, deployment.Parallelism{TP: 1, PP: 1,
					DP: 72, DPLocal: 4, EnableExpertParallel: true},
					deployment.Engine{All2AllBackend: "deepep_low_latency",
						CacheDType: "fp8", GPUMemoryUtilization: 0.9,
						EPLB: &deployment.EPLB{Enabled: true,
							NumRedundantExperts: 32}}),
			}, func(s *scenario.Scenario, d *deployment.Deployment) {
				s.Cluster.GPUsPerRack = 72
			}),
		},
	}
	for _, c := range cases {
		t.Run(c.what, func(t *testing.T) {
			if rep := Validate(c.b); !rep.Field.OK() {
				t.Fatalf("not expressible:\n%s", rep.Field.Error())
			}
		})
	}
}

func boolPtr(b bool) *bool { return &b }

// TestGraniteAtEP72NeedsRedundantExperts is the cross-document rule the corpus
// motivates: 224 experts do not divide a width of 72, so a load-balanced layout is
// infeasible until enough redundant experts are added.
func TestGraniteAtEP72NeedsRedundantExperts(t *testing.T) {
	build := func(redundant int) Bundle {
		b := corpus("granite-ep72", 9, 8, "ib-400g", []deployment.Pool{
			pool(deployment.RoleColocated, 9, deployment.Parallelism{TP: 1, PP: 1,
				DP: 72, DPLocal: 8, EnableExpertParallel: true},
				deployment.Engine{All2AllBackend: "deepep_low_latency",
					CacheDType: "fp8", GPUMemoryUtilization: 0.9,
					EPLB: &deployment.EPLB{Enabled: true,
						NumRedundantExperts: redundant}}),
		}, nil)
		b.Model = graniteMoE()
		return b
	}
	if rep := Validate(build(32)); rep.OK() {
		t.Fatal("224 + 32 does not divide 72; expected a rule failure")
	}
	if rep := Validate(build(64)); !rep.OK() {
		t.Fatalf("224 + 64 = 288 = 4 x 72 should pass:\n%s", renderAll(rep))
	}
}
