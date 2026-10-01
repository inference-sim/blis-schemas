package v0_29

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/internal/validate"
	"github.com/inference-sim/blis-schemas/rules"
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

func scen(pools ...scenario.Pool) *scenario.Scenario {
	nodes := 0
	for _, p := range pools {
		nodes += p.Nodes
	}
	return &scenario.Scenario{
		Kind: "Scenario", Name: "t", Model: "granite-5-230b", Hardware: "h200",
		Coefficients: []string{"c"}, EngineVersion: Version,
		Cluster: scenario.Cluster{Nodes: nodes, GPUsPerNode: 8},
		Pools:   pools,
	}
}

func colocated(pl scenario.Parallelism, e scenario.Engine) scenario.Pool {
	return scenario.Pool{Role: scenario.RoleColocated, Nodes: 1, Parallel: pl, Engine: e}
}

func run(t *testing.T, s *scenario.Scenario, g *model.Graph) *validate.Problems {
	t.Helper()
	p := &validate.Problems{}
	for _, r := range Pack().Rules {
		r.Check(rules.Input{Scenario: s, Model: g}, p)
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
		"custom-allreduce-reachable",
		"dbo-thresholds-stated-when-enabled",
		"enum-values-known",
		"expert-divisibility-under-eplb",
		"expert-imbalance-without-eplb",
		"mamba-cache-mode-matches-model",
		"offload-spec-known",
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
	s := scen(colocated(scenario.Parallelism{TP: 8, PP: 1, DP: 1},
		scenario.Engine{All2AllBackend: "telepathy", AllReduceBackend: "osmosis"}))
	p := run(t, s, graniteGraph())
	for _, rule := range []string{"all2all-backend-known", "allreduce-backend-known"} {
		if !fired(p, rule) {
			t.Errorf("%s did not fire on an unknown backend", rule)
		}
	}
}

func TestKnownBackendsAccepted(t *testing.T) {
	s := scen(colocated(scenario.Parallelism{TP: 8, PP: 1, DP: 1},
		scenario.Engine{All2AllBackend: "allgather_reducescatter",
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
	pl := scenario.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}
	// With load balancing on, an indivisible layout is an error.
	s := scen(colocated(pl, scenario.Engine{
		EPLB: &scenario.EPLB{Enabled: true, NumRedundantExperts: 32}}))
	p := run(t, s, graniteGraph())
	if !fired(p, "expert-divisibility-under-eplb") {
		t.Fatal("divisibility rule did not fire on 224+32 over 72")
	}
	if !strings.Contains(p.Error(), "32 more redundant") {
		t.Errorf("message should name the shortfall; got:\n%s", p.Error())
	}
	// 224 + 64 = 288 = 4 x 72.
	ok := scen(colocated(pl, scenario.Engine{
		EPLB: &scenario.EPLB{Enabled: true, NumRedundantExperts: 64}}))
	if p := run(t, ok, graniteGraph()); fired(p, "expert-divisibility-under-eplb") {
		t.Errorf("divisibility rule fired on a divisible layout:\n%s", p.Error())
	}
}

// TestExpertImbalanceWithoutEPLB: the same layout is legal without load balancing,
// and carries a static imbalance the estimate must account for.
func TestExpertImbalanceWithoutEPLB(t *testing.T) {
	pl := scenario.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}
	s := scen(colocated(pl, scenario.Engine{}))
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
	pl := scenario.Parallelism{TP: 1, PP: 1, DP: 256, EnableExpertParallel: true}
	s := scen(colocated(pl, scenario.Engine{}))
	p := run(t, s, graniteGraph())
	if p.OK() {
		t.Fatal("expected an error when width exceeds the expert count")
	}
}

func TestSequenceParallelMoEImplied(t *testing.T) {
	// The default backend with both tp and dp above one implies sequence-parallel
	// MoE, which changes which collectives the graph must emit.
	pl := scenario.Parallelism{TP: 2, PP: 1, DP: 8, EnableExpertParallel: true}
	s := scen(colocated(pl, scenario.Engine{All2AllBackend: "allgather_reducescatter"}))
	if p := run(t, s, graniteGraph()); !fired(p, "sequence-parallel-moe-implied") {
		t.Fatal("rule did not fire for tp=2, dp=8 on the default backend")
	}
	// tp=1 does not trigger it, which is why the design's own examples do not.
	pl1 := scenario.Parallelism{TP: 1, PP: 1, DP: 8, EnableExpertParallel: true}
	s1 := scen(colocated(pl1, scenario.Engine{All2AllBackend: "allgather_reducescatter"}))
	if p := run(t, s1, graniteGraph()); fired(p, "sequence-parallel-moe-implied") {
		t.Error("rule fired at tp=1, where the engine does not apply it")
	}
}

func TestCustomAllReduceReachability(t *testing.T) {
	// A world size outside the supported set falls back.
	s := scen(colocated(scenario.Parallelism{TP: 3, PP: 1, DP: 1},
		scenario.Engine{AllReduceBackend: "custom"}))
	if p := run(t, s, graniteGraph()); !fired(p, "custom-allreduce-reachable") {
		t.Error("rule did not fire on tp=3")
	}
	// Spanning nodes needs multi-node NVLink.
	s2 := scen(
		colocated(scenario.Parallelism{TP: 16, PP: 1, DP: 1},
			scenario.Engine{AllReduceBackend: "custom"}),
	)
	if p := run(t, s2, graniteGraph()); !fired(p, "custom-allreduce-reachable") {
		t.Error("rule did not fire on tp=16 across 8-GPU nodes")
	}
	// tp=8 inside one node is reachable.
	s3 := scen(colocated(scenario.Parallelism{TP: 8, PP: 1, DP: 1},
		scenario.Engine{AllReduceBackend: "custom"}))
	if p := run(t, s3, graniteGraph()); fired(p, "custom-allreduce-reachable") {
		t.Errorf("rule fired on a reachable layout:\n%s", p.Error())
	}
}

func TestMambaCacheModeMatchesModel(t *testing.T) {
	// A mode on a model with no recurrent layer has no effect.
	s := scen(colocated(scenario.Parallelism{TP: 8, PP: 1, DP: 1},
		scenario.Engine{MambaCacheMode: "align"}))
	if p := run(t, s, graniteGraph()); !fired(p, "mamba-cache-mode-matches-model") {
		t.Error("rule did not fire for a mode on a non-hybrid model")
	}
	// A hybrid model with no mode leaves the state unsized.
	hybrid := graniteGraph()
	hybrid.LayerKinds[0].Nodes = append(hybrid.LayerKinds[0].Nodes, model.Node{
		Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentMamba2,
		NumHeads: 128, StateSize: 128, NumGroups: 8, ConvKernel: 4,
		IntermediateSize: 8192})
	s2 := scen(colocated(scenario.Parallelism{TP: 8, PP: 1, DP: 1}, scenario.Engine{}))
	if p := run(t, s2, hybrid); !fired(p, "mamba-cache-mode-matches-model") {
		t.Error("rule did not fire for a hybrid model with no mode")
	}
}

func TestOffloadSpecKnown(t *testing.T) {
	s := scen(colocated(scenario.Parallelism{TP: 8, PP: 1, DP: 1}, scenario.Engine{}))
	s.Offload = &scenario.Offload{Spec: "MagicOffloadingSpec",
		Tiers: []scenario.Tier{{Device: "cpu_dram", Bytes: 1 << 40}}}
	if p := run(t, s, graniteGraph()); !fired(p, "offload-spec-known") {
		t.Fatal("rule did not fire on an unregistered spec")
	}
	s.Offload.Spec = "CPUOffloadingSpec"
	if p := run(t, s, graniteGraph()); fired(p, "offload-spec-known") {
		t.Error("rule fired on a registered spec")
	}
}

func TestDBOThresholdsStatedWhenEnabled(t *testing.T) {
	s := scen(colocated(scenario.Parallelism{TP: 8, PP: 1, DP: 1},
		scenario.Engine{DBO: &scenario.DBO{Enabled: true}}))
	p := run(t, s, graniteGraph())
	if !fired(p, "dbo-thresholds-stated-when-enabled") {
		t.Fatal("rule did not fire when thresholds were omitted")
	}
	if !strings.Contains(p.Error(), "32") || !strings.Contains(p.Error(), "512") {
		t.Errorf("message should name both defaults; got:\n%s", p.Error())
	}
}

func TestSpeculativeMethodKnown(t *testing.T) {
	s := scen(colocated(scenario.Parallelism{TP: 8, PP: 1, DP: 1},
		scenario.Engine{Speculative: &scenario.Speculative{
			Method: "crystal_ball", NumSpecTokens: 2}}))
	if p := run(t, s, graniteGraph()); !fired(p, "speculative-method-known") {
		t.Fatal("rule did not fire on an unknown method")
	}
	s2 := scen(colocated(scenario.Parallelism{TP: 8, PP: 1, DP: 1},
		scenario.Engine{Speculative: &scenario.Speculative{
			Method: "glm4_moe_mtp", NumSpecTokens: 1}}))
	if p := run(t, s2, graniteGraph()); fired(p, "speculative-method-known") {
		t.Error("rule fired on an accepted method")
	}
}
