package v0_29

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/internal/validate"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

func graniteGraph() *model.Graph {
	return &model.Graph{
		Kind: "ModelGraph", Name: "granite-5-230b",
		DerivedFrom: model.Derivation{Format: "hf_config_json", Path: "config.json",
			SHA256:         strings.Repeat("a", 64),
			DeriverVersion: 1},
		Global: model.GlobalShape{HiddenSize: 3072, VocabSize: 100352,
			WeightDType: model.DTypeFP8},
		LayerKinds: []model.LayerKind{{ID: "attn_moe", Nodes: []model.Node{
			{Op: model.OpAttention, AttentionKind: model.AttentionGQA,
				NumQHeads: 48, NumKVHeads: 8, HeadDim: 64},
			{Op: model.OpGroupedGEMM, N: 1536, K: 3072, Experts: 224, TopK: 8},
		}}},
		Stack: model.Stack{Pattern: []string{"attn_moe"}, Repeat: 72},
	}
}

func dep(pools ...deployment.Pool) *deployment.Deployment {
	return &deployment.Deployment{Kind: "Deployment", Name: "t", Pools: pools}
}

func colocated(pl deployment.Parallelism, e deployment.Engine) deployment.Pool {
	return deployment.Pool{Role: deployment.RoleColocated, Nodes: 1, Parallel: pl, Engine: e}
}

// run fires the pack's rules over a deployment, pairing it with the scenario its pool
// node counts imply so the cluster-aware rules (custom-allreduce-reachable) have the
// inventory they read.
func run(t *testing.T, d *deployment.Deployment, g *model.Graph) *validate.Problems {
	t.Helper()
	nodes := 0
	for _, pl := range d.Pools {
		nodes += pl.Nodes
	}
	s := &scenario.Scenario{
		Kind: "Scenario", Name: "t", Model: "granite-5-230b",
		Coefficients: []string{"c"}, EngineVersion: Version,
		Cluster: scenario.Cluster{Hardware: "h200", Nodes: nodes, GPUsPerNode: 8},
	}
	p := &validate.Problems{}
	for _, r := range Pack().Rules {
		r.Check(rules.Input{Scenario: s, Deployment: d, Model: g}, p)
	}
	return p
}

func fired(p *validate.Problems, rule string) bool {
	for _, it := range p.All() {
		if it.Rule == rule {
			return true
		}
	}
	return false
}

// TestPackRuleSet pins the rule names. A rule silently dropped during a refactor
// would otherwise leave a passing suite and no coverage.
func TestPackRuleSet(t *testing.T) {
	want := []string{
		"all2all-backend-known",
		"allreduce-backend-known",
		"cascade-attention-is-opt-in",
		"connector-known",
		"custom-allreduce-reachable",
		"dbo-thresholds-stated-when-enabled",
		"enum-values-known",
		"eviction-policy-known",
		"expert-divisibility-under-eplb",
		"expert-imbalance-without-eplb",
		"mamba-cache-mode-matches-model",
		"offload-spec-known",
		"quantization-known",
		"sequence-parallel-moe-implied",
		"speculative-method-known",
	}
	got := Pack().Names()
	if len(got) != len(want) {
		t.Fatalf("rule count = %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("rule[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// TestEveryRuleStatesItsReason: a rule a reader cannot evaluate is a rule they will
// waive blindly.
func TestEveryRuleStatesItsReason(t *testing.T) {
	for _, r := range Pack().Rules {
		if r.Because == "" {
			t.Errorf("rule %q states no reason", r.Name)
		}
		if r.Check == nil {
			t.Errorf("rule %q has no check", r.Name)
		}
	}
}

func TestUnknownBackendsRejected(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{All2AllBackend: "telepathy", AllReduceBackend: "osmosis"}))
	p := run(t, s, graniteGraph())
	for _, rule := range []string{"all2all-backend-known", "allreduce-backend-known"} {
		if !fired(p, rule) {
			t.Errorf("%s did not fire on an unknown backend", rule)
		}
	}
}

func TestKnownBackendsAccepted(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{All2AllBackend: "allgather_reducescatter",
			AllReduceBackend: "nccl", CUDAGraphMode: "FULL",
			CacheDType: "fp8", SchedulingPolicy: "priority"}))
	if p := run(t, s, graniteGraph()); !p.OK() {
		t.Fatalf("accepted values were rejected:\n%s", p.Error())
	}
}

// TestExpertDivisibility is the Granite case: 224 experts do not divide an
// expert-parallel width of 72, and the rule must say how many redundant experts
// would fix it.
func TestExpertDivisibility(t *testing.T) {
	pl := deployment.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}
	// With load balancing on, an indivisible layout is an error.
	s := dep(colocated(pl, deployment.Engine{
		EPLB: &deployment.EPLB{Enabled: true, NumRedundantExperts: 32}}))
	p := run(t, s, graniteGraph())
	if !fired(p, "expert-divisibility-under-eplb") {
		t.Fatal("divisibility rule did not fire on 224+32 over 72")
	}
	if !strings.Contains(p.Error(), "32 more redundant") {
		t.Errorf("message should name the shortfall; got:\n%s", p.Error())
	}
	// 224 + 64 = 288 = 4 x 72.
	ok := dep(colocated(pl, deployment.Engine{
		EPLB: &deployment.EPLB{Enabled: true, NumRedundantExperts: 64}}))
	if p := run(t, ok, graniteGraph()); fired(p, "expert-divisibility-under-eplb") {
		t.Errorf("divisibility rule fired on a divisible layout:\n%s", p.Error())
	}
}

// TestExpertImbalanceWithoutEPLB: the same layout is legal without load balancing,
// and carries a static imbalance the estimate must account for.
func TestExpertImbalanceWithoutEPLB(t *testing.T) {
	pl := deployment.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}
	s := dep(colocated(pl, deployment.Engine{}))
	p := run(t, s, graniteGraph())
	if !fired(p, "expert-imbalance-without-eplb") {
		t.Fatal("imbalance rule did not fire on 224 over 72")
	}
	// Legal, so a warning rather than an error.
	if !p.OK() {
		t.Errorf("an indivisible split without EPLB should warn, not fail:\n%s", p.Error())
	}
	if !strings.Contains(p.Error(), "1.33x") {
		t.Errorf("message should quantify the imbalance; got:\n%s", p.Error())
	}
}

// TestExpertParallelWidthExceedsExperts is an error rather than a warning: a rank
// with no expert cannot run the layer at all.
func TestExpertParallelWidthExceedsExperts(t *testing.T) {
	pl := deployment.Parallelism{TP: 1, PP: 1, DP: 256, EnableExpertParallel: true}
	s := dep(colocated(pl, deployment.Engine{}))
	p := run(t, s, graniteGraph())
	if p.OK() {
		t.Fatal("expected an error when width exceeds the expert count")
	}
}

func TestSequenceParallelMoEImplied(t *testing.T) {
	// The default backend with both tp and dp above one implies sequence-parallel
	// MoE, which changes which collectives the graph must emit.
	pl := deployment.Parallelism{TP: 2, PP: 1, DP: 8, EnableExpertParallel: true}
	s := dep(colocated(pl, deployment.Engine{All2AllBackend: "allgather_reducescatter"}))
	if p := run(t, s, graniteGraph()); !fired(p, "sequence-parallel-moe-implied") {
		t.Fatal("rule did not fire for tp=2, dp=8 on the default backend")
	}
	// tp=1 does not trigger it, which is why the design's own examples do not.
	pl1 := deployment.Parallelism{TP: 1, PP: 1, DP: 8, EnableExpertParallel: true}
	s1 := dep(colocated(pl1, deployment.Engine{All2AllBackend: "allgather_reducescatter"}))
	if p := run(t, s1, graniteGraph()); fired(p, "sequence-parallel-moe-implied") {
		t.Error("rule fired at tp=1, where the engine does not apply it")
	}
}

func TestCustomAllReduceReachability(t *testing.T) {
	// A world size outside the supported set falls back.
	s := dep(colocated(deployment.Parallelism{TP: 3, PP: 1, DP: 1},
		deployment.Engine{AllReduceBackend: "custom"}))
	if p := run(t, s, graniteGraph()); !fired(p, "custom-allreduce-reachable") {
		t.Error("rule did not fire on tp=3")
	}
	// Spanning nodes needs multi-node NVLink.
	s2 := dep(
		colocated(deployment.Parallelism{TP: 16, PP: 1, DP: 1},
			deployment.Engine{AllReduceBackend: "custom"}),
	)
	if p := run(t, s2, graniteGraph()); !fired(p, "custom-allreduce-reachable") {
		t.Error("rule did not fire on tp=16 across 8-GPU nodes")
	}
	// tp=8 inside one node is reachable.
	s3 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{AllReduceBackend: "custom"}))
	if p := run(t, s3, graniteGraph()); fired(p, "custom-allreduce-reachable") {
		t.Errorf("rule fired on a reachable layout:\n%s", p.Error())
	}
}

func TestMambaCacheModeMatchesModel(t *testing.T) {
	// A mode on a model with no recurrent layer has no effect.
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{MambaCacheMode: "align"}))
	if p := run(t, s, graniteGraph()); !fired(p, "mamba-cache-mode-matches-model") {
		t.Error("rule did not fire for a mode on a non-hybrid model")
	}
	// A hybrid model with no mode leaves the state unsized.
	hybrid := graniteGraph()
	hybrid.LayerKinds[0].Nodes = append(hybrid.LayerKinds[0].Nodes, model.Node{
		Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentMamba2,
		NumHeads: 128, StateSize: 128, NumGroups: 8, ConvKernel: 4,
		IntermediateSize: 8192})
	s2 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, deployment.Engine{}))
	if p := run(t, s2, hybrid); !fired(p, "mamba-cache-mode-matches-model") {
		t.Error("rule did not fire for a hybrid model with no mode")
	}
}

func TestOffloadSpecKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, deployment.Engine{}))
	s.Offload = &deployment.Offload{Spec: "MagicOffloadingSpec",
		Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 40}}}
	if p := run(t, s, graniteGraph()); !fired(p, "offload-spec-known") {
		t.Fatal("rule did not fire on an unregistered spec")
	}
	s.Offload.Spec = "CPUOffloadingSpec"
	if p := run(t, s, graniteGraph()); fired(p, "offload-spec-known") {
		t.Error("rule fired on a registered spec")
	}
}

func TestDBOThresholdsStatedWhenEnabled(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{DBO: &deployment.DBO{Enabled: true}}))
	p := run(t, s, graniteGraph())
	if !fired(p, "dbo-thresholds-stated-when-enabled") {
		t.Fatal("rule did not fire when thresholds were omitted")
	}
	if !strings.Contains(p.Error(), "32") || !strings.Contains(p.Error(), "512") {
		t.Errorf("message should name both defaults; got:\n%s", p.Error())
	}
}

func TestSpeculativeMethodKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{Speculative: &deployment.Speculative{
			Method: "crystal_ball", NumSpecTokens: 2}}))
	if p := run(t, s, graniteGraph()); !fired(p, "speculative-method-known") {
		t.Fatal("rule did not fire on an unknown method")
	}
	s2 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{Speculative: &deployment.Speculative{
			Method: "glm4_moe_mtp", NumSpecTokens: 1}}))
	if p := run(t, s2, graniteGraph()); fired(p, "speculative-method-known") {
		t.Error("rule fired on an accepted method")
	}
}

// TestQuantizationKnown: an in-tree method passes, a typo warns. It is a warning rather
// than an error because the engine admits out-of-tree methods the pack cannot enumerate,
// so the rule flags a probable typo without rejecting a possibly-valid name.
func TestQuantizationKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{Quantization: "fp8_but_misspelled"}))
	p := run(t, s, graniteGraph())
	if !fired(p, "quantization-known") {
		t.Fatal("rule did not fire on an unrecognized quantization method")
	}
	// The warning must enumerate the valid set (#19 item 3): naming an in-tree method the
	// author could have meant is what makes the diagnostic self-correcting.
	if !strings.Contains(p.Error(), "compressed-tensors") {
		t.Errorf("the warning should enumerate in-tree methods; got:\n%s", p.Error())
	}
	s2 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		deployment.Engine{Quantization: "compressed-tensors"}))
	if p := run(t, s2, graniteGraph()); fired(p, "quantization-known") {
		t.Error("rule fired on an in-tree quantization method")
	}
}

// TestConnectorKnown checks both connector sites — the offload block and the pd_transfer
// block — since a deployment can name a connector in either. An unknown name warns; a
// registered one is quiet.
func TestConnectorKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, deployment.Engine{}))
	s.Offload = &deployment.Offload{Connector: "TelepathyConnector",
		Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 40}}}
	s.PDTransfer = &deployment.PDTransfer{Connector: "NixlConnector"}
	p := run(t, s, graniteGraph())
	if !fired(p, "connector-known") {
		t.Fatal("rule did not fire on an unregistered offload connector")
	}
	if !strings.Contains(p.Error(), "offload.connector") {
		t.Errorf("the warning should name the offload connector site; got:\n%s", p.Error())
	}
	// The warning enumerates the registered connectors (#19 item 3).
	if !strings.Contains(p.Error(), "NixlConnector") {
		t.Errorf("the warning should enumerate registered connectors; got:\n%s", p.Error())
	}

	// Both connectors registered: quiet.
	s.Offload.Connector = "OffloadingConnector"
	if p := run(t, s, graniteGraph()); fired(p, "connector-known") {
		t.Errorf("rule fired on two registered connectors:\n%s", p.Error())
	}

	// A bad pd_transfer connector is caught on its own.
	s2 := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, deployment.Engine{}))
	s2.PDTransfer = &deployment.PDTransfer{Connector: "CarrierPigeonConnector"}
	if p := run(t, s2, graniteGraph()); !fired(p, "connector-known") {
		t.Fatal("rule did not fire on an unregistered pd_transfer connector")
	}
}

// TestEvictionPolicyKnown: an in-tree policy (lru/arc) passes, a typo warns and the warning
// enumerates the in-tree set. A warning, not an error, because the CachePolicyFactory admits
// out-of-tree policies — the same shape as the connector and quantization rules (#19).
func TestEvictionPolicyKnown(t *testing.T) {
	s := dep(colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, deployment.Engine{}))
	s.Offload = &deployment.Offload{EvictionPolicy: "most_recently_used_typo",
		Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 40}}}
	p := run(t, s, graniteGraph())
	if !fired(p, "eviction-policy-known") {
		t.Fatal("rule did not fire on an unrecognized eviction policy")
	}
	if !strings.Contains(p.Error(), "lru") || !strings.Contains(p.Error(), "arc") {
		t.Errorf("the warning should enumerate in-tree policies lru and arc; got:\n%s", p.Error())
	}

	// An in-tree policy is quiet.
	s.Offload.EvictionPolicy = "arc"
	if p := run(t, s, graniteGraph()); fired(p, "eviction-policy-known") {
		t.Errorf("rule fired on an in-tree policy:\n%s", p.Error())
	}
}
