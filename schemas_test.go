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

func renderAll(r Report) string {
	var b strings.Builder
	for _, p := range r.Problems() {
		b.WriteString(p.String())
		b.WriteString("\n")
	}
	return b.String()
}
