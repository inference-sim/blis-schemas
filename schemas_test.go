package blisschemas

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
	"github.com/inference-sim/blis-schemas/vocab"
)

func bundle() Bundle {
	return Bundle{
		Scenario: &scenario.Scenario{
			Kind: "Scenario", Name: "granite-230b-h200-tp8",
			Model:         "granite-5-230b",
			Coefficients:  []string{"cost-model-primitives-h200"},
			EngineVersion: "0.29.0",
			Cluster:       scenario.Cluster{Hardware: "h200", Nodes: 1, GPUsPerNode: 8},
		},
		Deployment: &deployment.Deployment{
			Kind: "Deployment", Name: "granite-230b-h200-tp8",
			Pools: []deployment.Pool{{Role: deployment.RoleColocated, Nodes: 1,
				Parallel: deployment.Parallelism{TP: 8, PP: 1, DP: 1},
				Engine: deployment.Engine{CacheDType: "fp8", BlockSize: 16,
					MaxNumBatchedTokens: 32768, GPUMemoryUtilization: 0.9}}},
		},
		Model: &model.Graph{
			Kind: "ModelGraph", Name: "granite-5-230b",
			DerivedFrom: model.Derivation{Format: "hf_config_json",
				Path: "config.json", SHA256: strings.Repeat("b", 64),
				DeriverVersion: 1},
			Global: model.GlobalShape{HiddenSize: 3072, VocabSize: 100352,
				WeightDType: model.DTypeFP8},
			LayerKinds: []model.LayerKind{{ID: "attn_moe", Nodes: []model.Node{
				{Op: model.OpAttention, AttentionKind: model.AttentionGQA,
					NumQHeads: 48, NumKVHeads: 8, HeadDim: 64},
				{Op: model.OpGroupedGEMM, N: 1536, K: 3072, Experts: 224, TopK: 8},
			}}},
			Stack: model.Stack{Pattern: []string{"attn_moe"}, Repeat: 72},
		},
		Chip: &hardware.Chip{Name: "h200",
			Provenance: vocab.ProvenanceVendorSpec, BF16Peak: 989.5,
			FP8Peak: 1979.0, MemoryBandwidthTBs: 4.8, MemoryGiB: 141,
			IntraNodeBwGBps: 450, SMCount: 132, GPUsPerNode: 8},
		Coefficients: []*coefficient.Set{{
			Kind: "CoefficientSet", Name: "cost-model-primitives-h200",
			Coefficients: []coefficient.Entry{{
				Name: "gemm_eps_max_bf16", Value: 0.72,
				Units: vocab.UnitDimensionless, Method: vocab.MethodMeasured,
				Fitted: true,
				Scope:  coefficient.Scope{Hardware: []string{"h200"}},
				Sources: []coefficient.Source{{Kind: vocab.SourceModel,
					Cite: "operator table", Role: vocab.RolePrimary}},
			}},
		}},
	}
}

func TestValidBundlePasses(t *testing.T) {
	rep := Validate(bundle())
	if !rep.OK() {
		t.Fatalf("a valid bundle was rejected:\n%s", renderAll(rep))
	}
	if rep.RulesApplied != "0.29.0" {
		t.Errorf("RulesApplied = %q, want 0.29.0", rep.RulesApplied)
	}
}

// TestUnknownEngineVersionIsReported is the property that keeps silence from
// looking like a pass: a version with no pack must say so.
func TestUnknownEngineVersionIsReported(t *testing.T) {
	b := bundle()
	b.Scenario.EngineVersion = "99.0.0"
	rep := Validate(b)
	if rep.RulesApplied != "" {
		t.Errorf("RulesApplied = %q, want empty", rep.RulesApplied)
	}
	if len(rep.Rule.All()) == 0 {
		t.Fatal("an unknown engine version produced no finding, which is indistinguishable from a checked pass")
	}
	if !strings.Contains(rep.Rule.Error(), "no rules pack") {
		t.Errorf("finding should say no pack applied; got:\n%s", rep.Rule.Error())
	}
	// It is a warning, not an error: the documents are well formed.
	if !rep.OK() {
		t.Error("an unknown version should warn rather than fail")
	}
}

// TestRulesDoNotRunOnMalformedDocuments pins the layering: a rule reading a
// malformed document yields findings that are artifacts of the malformation.
func TestRulesDoNotRunOnMalformedDocuments(t *testing.T) {
	b := bundle()
	b.Scenario.Kind = "NotAScenario"
	b.Deployment.Pools[0].Engine.All2AllBackend = "telepathy" // would fire a rule
	rep := Validate(b)
	if rep.Field.OK() {
		t.Fatal("expected a field problem")
	}
	if len(rep.Rule.All()) != 0 {
		t.Errorf("rules ran over a malformed document:\n%s", rep.Rule.Error())
	}
	if rep.RulesApplied != "" {
		t.Errorf("RulesApplied = %q, want empty", rep.RulesApplied)
	}
}

// TestLayersAreDistinguishable: a caller must be able to tell a malformed document
// from one that a particular engine would refuse.
func TestLayersAreDistinguishable(t *testing.T) {
	b := bundle()
	b.Deployment.Pools[0].Engine.All2AllBackend = "telepathy"
	rep := Validate(b)
	if !rep.Field.OK() {
		t.Fatalf("field layer should pass; got:\n%s", rep.Field.Error())
	}
	if rep.Rule.OK() {
		t.Fatal("rule layer should fail on an unknown backend")
	}
	for _, p := range rep.Rule.Errors() {
		if p.Rule == "" {
			t.Errorf("rule finding carries no rule name: %s", p)
		}
	}
}

func TestRegisteredVersions(t *testing.T) {
	vs := rules.Versions()
	if len(vs) == 0 {
		t.Fatal("no rules pack is registered, so no scenario can be rule-checked")
	}
	found := false
	for _, v := range vs {
		if v == "0.29.0" {
			found = true
		}
	}
	if !found {
		t.Errorf("0.29.0 is not registered; got %v", vs)
	}
}

// TestDeploymentWithoutScenarioIsRejected guards the composition layer against a bundle
// that would otherwise silently under-validate. A Deployment with no Scenario skips the
// cluster-fit check and every version-scoped engine rule (both need the scenario), so a
// deployment that does not fit its hardware would come back green. The layer reports a
// field problem instead. The mirror holds: a Scenario with no Deployment is a complete
// problem on its own and still validates.
func TestDeploymentWithoutScenarioIsRejected(t *testing.T) {
	b := bundle()

	rep := Validate(Bundle{Deployment: b.Deployment})
	if rep.Field.OK() {
		t.Fatal("a deployment with no scenario should be a field problem, not a pass")
	}
	if rep.RulesApplied != "" {
		t.Errorf("RulesApplied = %q, want empty when no scenario is present", rep.RulesApplied)
	}

	// The mirror: a scenario with no deployment is a complete problem and still validates.
	if rep := Validate(Bundle{Scenario: b.Scenario}); !rep.OK() {
		t.Fatalf("a scenario with no deployment should pass:\n%s", renderAll(rep))
	}
}

func TestPartialBundlesValidate(t *testing.T) {
	// A catalog contributor checks one chip, with no scenario at all.
	rep := Validate(Bundle{Chip: bundle().Chip})
	if !rep.OK() {
		t.Fatalf("a chip-only bundle was rejected:\n%s", renderAll(rep))
	}
	if rep.RulesApplied != "" {
		t.Error("no scenario means no rules should apply")
	}
}

// TestOffloadTierMustBeInClusterStorage pins the inventory coupling at the layer that
// owns it: the storage classes live on the Scenario's cluster, so only the composition
// layer — which sees both documents — can check that a deployment's offload tiers draw
// from the declared inventory.
func TestOffloadTierMustBeInClusterStorage(t *testing.T) {
	b := bundle()
	b.Scenario.Cluster.Storage = []string{"cpu_dram", "nvme_gen4"}

	// A tier drawn from the declared inventory passes.
	b.Deployment.Offload = &deployment.Offload{
		Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1}}}
	if rep := Validate(b); !rep.OK() {
		t.Fatalf("a tier drawn from the cluster inventory should pass:\n%s", renderAll(rep))
	}

	// A tier naming a class the cluster does not list is a field problem.
	b.Deployment.Offload = &deployment.Offload{
		Tiers: []deployment.Tier{{Device: "optane", Bytes: 1}}}
	if rep := Validate(b); rep.Field.OK() {
		t.Fatal("an offload tier outside the cluster storage inventory should be a field problem")
	}
}

// TestEngineMustFitItsPool is the composition-level statement of the rank-count rule:
// no layer used to relate a layout's rank count to the GPUs it was placed on, so the
// issue's example — tp 8 with pcp 4 on one 8-GPU node, needing 32 devices — passed both
// the deployment validator and the cluster-coupling checks. It goes through Validate
// because the claim was about the whole pipeline, not one function.
func TestEngineMustFitItsPool(t *testing.T) {
	// The ordinary bundle is tp 8 on one 8-GPU node: eight ranks on eight devices.
	if rep := Validate(bundle()); !rep.OK() {
		t.Fatalf("tp 8 on an 8-GPU node should fit:\n%s", renderAll(rep))
	}

	b := bundle()
	b.Deployment.Pools[0].Parallel = deployment.Parallelism{TP: 8, PP: 1, DP: 1, PCP: 4}
	rep := Validate(b)
	if rep.Field.OK() {
		t.Fatal("tp 8 with pcp 4 needs 32 GPUs and must not validate on an 8-GPU node")
	}
	found := false
	for _, p := range rep.Field.Errors() {
		if p.Path == "deployment.pools[0].parallel" && strings.Contains(p.Message, "needs 32 GPUs") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected a problem at deployment.pools[0].parallel naming 32 GPUs, got:\n%s",
			renderAll(rep))
	}

	// The same layout fits once the pool owns enough nodes. The rule is about room, not
	// about the layout being unusual: pcp 4 is fine when there is somewhere to put it.
	// A multi-node cluster must also name its fabric, which is unrelated to what is
	// being tested but part of a well-formed scenario.
	b.Scenario.Cluster.Nodes = 4
	b.Scenario.Cluster.Fabric = "ib-400g"
	b.Deployment.Pools[0].Nodes = 4
	if rep := Validate(b); !rep.Field.OK() {
		t.Fatalf("the same layout on four 8-GPU nodes should fit:\n%s", renderAll(rep))
	}
}

func renderAll(r Report) string {
	var b strings.Builder
	for _, p := range r.Problems() {
		b.WriteString(p.String())
		b.WriteString("\n")
	}
	return b.String()
}
