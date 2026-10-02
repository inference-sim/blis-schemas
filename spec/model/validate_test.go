package model

import "testing"

// validGraph is a Granite-shaped uniform MoE stack: 224 experts, GQA attention,
// one layer kind repeated. Each negative test mutates one field of a copy, so a
// failure names exactly the rule that fired rather than a document that was wrong
// in several ways at once.
func validGraph() *Graph {
	return &Graph{
		Kind: "ModelGraph",
		Name: "granite-5-230b",
		DerivedFrom: Derivation{
			Format: "hf_config_json", Path: "config.json",
			SHA256:         "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
			DeriverVersion: 1,
		},
		Global: GlobalShape{HiddenSize: 3072, VocabSize: 100352, WeightDType: DTypeFP8},
		LayerKinds: []LayerKind{{
			ID: "attn_moe",
			Nodes: []Node{
				{Op: OpElementwise, Role: "input_norm"},
				{Op: OpGEMM, Role: "qkv_proj", N: 4096, K: 3072},
				{Op: OpAttention, AttentionKind: AttentionGQA, NumQHeads: 48, NumKVHeads: 8, HeadDim: 64},
				{Op: OpGEMM, Role: "o_proj", N: 3072, K: 3072},
				{Op: OpAllReduce, Role: "attn_out", Emit: EmitTensorParallel},
				{Op: OpGroupedGEMM, Role: "experts", N: 1536, K: 3072, Experts: 224, TopK: 8},
				{Op: OpAll2All, Role: "moe", Emit: EmitExpertParallel},
			},
			Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}},
		}},
		Stack: Stack{Pattern: []string{"attn_moe"}, Repeat: 72},
	}
}

func TestValidGraphPasses(t *testing.T) {
	if p := validGraph().Validate(); !p.OK() {
		t.Fatalf("a valid graph was rejected:\n%s", p.Error())
	}
}

func TestStackLayers(t *testing.T) {
	if got := validGraph().Stack.Layers(); got != 72 {
		t.Errorf("Layers() = %d, want 72", got)
	}
	hybrid := Stack{Pattern: []string{"a", "a", "a", "b"}, Repeat: 10}
	if got := hybrid.Layers(); got != 40 {
		t.Errorf("hybrid Layers() = %d, want 40", got)
	}
}

// TestRejects is the negative suite. A validator that cannot fail proves nothing,
// so every check above has a case here that provokes it.
func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Graph)
	}{
		{"wrong kind", func(g *Graph) { g.Kind = "Model" }},
		{"no name", func(g *Graph) { g.Name = "" }},
		{"short digest", func(g *Graph) { g.DerivedFrom.SHA256 = "abc" }},
		{"no derivation format", func(g *Graph) { g.DerivedFrom.Format = "" }},
		{"zero deriver version", func(g *Graph) { g.DerivedFrom.DeriverVersion = 0 }},
		{"zero hidden size", func(g *Graph) { g.Global.HiddenSize = 0 }},
		{"unknown weight dtype", func(g *Graph) { g.Global.WeightDType = "fp6" }},
		{"no layer kinds", func(g *Graph) { g.LayerKinds = nil }},
		{"duplicate layer-kind id", func(g *Graph) {
			g.LayerKinds = append(g.LayerKinds, g.LayerKinds[0])
		}},
		{"pattern names unknown kind", func(g *Graph) {
			g.Stack.Pattern = []string{"nope"}
		}},
		{"zero repeat", func(g *Graph) { g.Stack.Repeat = 0 }},
		{"unknown op", func(g *Graph) { g.LayerKinds[0].Nodes[0].Op = "Magic" }},
		{"edge out of range", func(g *Graph) {
			g.LayerKinds[0].Edges = append(g.LayerKinds[0].Edges, [2]int{0, 99})
		}},
		{"self edge", func(g *Graph) {
			g.LayerKinds[0].Edges = append(g.LayerKinds[0].Edges, [2]int{2, 2})
		}},
		{"cyclic graph", func(g *Graph) {
			g.LayerKinds[0].Edges = append(g.LayerKinds[0].Edges, [2]int{6, 0})
		}},
		{"collective without condition", func(g *Graph) {
			g.LayerKinds[0].Nodes[4].Emit = EmitAlways
		}},
		{"gemm without n", func(g *Graph) { g.LayerKinds[0].Nodes[1].N = 0 }},
		{"gemm carrying head count", func(g *Graph) {
			g.LayerKinds[0].Nodes[1].NumQHeads = 8
		}},
		{"grouped gemm top_k exceeds experts", func(g *Graph) {
			g.LayerKinds[0].Nodes[5].TopK = 500
		}},
		{"grouped gemm without experts", func(g *Graph) {
			g.LayerKinds[0].Nodes[5].Experts = 0
		}},
		{"attention n_kv exceeds n_q", func(g *Graph) {
			g.LayerKinds[0].Nodes[2].NumKVHeads = 99
		}},
		{"unknown attention kind", func(g *Graph) {
			g.LayerKinds[0].Nodes[2].AttentionKind = "linear"
		}},
		{"mla with more than one kv head", func(g *Graph) {
			g.LayerKinds[0].Nodes[2].AttentionKind = AttentionMLA
			g.LayerKinds[0].Nodes[2].NumKVHeads = 8
		}},
		{"swa without a window", func(g *Graph) {
			g.LayerKinds[0].Nodes[2].AttentionKind = AttentionSWA
		}},
		{"collective carrying a shape", func(g *Graph) {
			g.LayerKinds[0].Nodes[6].N = 1024
		}},
		{"speculator with zero drafts", func(g *Graph) {
			g.Speculator = &Speculator{Method: "glm4_moe_mtp", NumSpec: 0,
				Stack: Stack{Pattern: []string{"attn_moe"}, Repeat: 1}}
		}},
		{"speculator without a stack", func(g *Graph) {
			g.Speculator = &Speculator{Method: "glm4_moe_mtp", NumSpec: 1}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := validGraph()
			tc.mutate(g)
			if p := g.Validate(); p.OK() {
				t.Fatalf("expected rejection, got none")
			}
		})
	}
}

// TestRecurrentNodeRequirements covers the hybrid path separately, since its
// required parameters differ by recurrent kind.
func TestRecurrentNodeRequirements(t *testing.T) {
	mamba := Node{Op: OpRecurrentUpdate, RecurrentKind: RecurrentMamba2,
		NumHeads: 128, StateSize: 128, NumGroups: 8, ConvKernel: 4,
		IntermediateSize: 8192, StateDType: DTypeFP32}
	g := validGraph()
	g.LayerKinds[0].Nodes[2] = mamba
	if p := g.Validate(); !p.OK() {
		t.Fatalf("a well-formed mamba2 node was rejected:\n%s", p.Error())
	}
	// Mamba2 needs its convolution parameters; KDA does not.
	for _, drop := range []func(*Node){
		func(n *Node) { n.NumGroups = 0 },
		func(n *Node) { n.ConvKernel = 0 },
		func(n *Node) { n.IntermediateSize = 0 },
		func(n *Node) { n.StateSize = 0 },
	} {
		g := validGraph()
		n := mamba
		drop(&n)
		g.LayerKinds[0].Nodes[2] = n
		if p := g.Validate(); p.OK() {
			t.Errorf("expected rejection of an incomplete mamba2 node")
		}
	}
	kda := Node{Op: OpRecurrentUpdate, RecurrentKind: RecurrentKDA,
		NumHeads: 64, StateSize: 128}
	g2 := validGraph()
	g2.LayerKinds[0].Nodes[2] = kda
	if p := g2.Validate(); !p.OK() {
		t.Fatalf("a KDA node without convolution parameters was rejected:\n%s", p.Error())
	}
}

func TestLatentKVAndDTypeBytes(t *testing.T) {
	for _, k := range []AttentionKind{AttentionMLA, AttentionSparseMLA} {
		if !k.LatentKV() {
			t.Errorf("%s should report a latent KV cache", k)
		}
	}
	for _, k := range []AttentionKind{AttentionGQA, AttentionSWA} {
		if k.LatentKV() {
			t.Errorf("%s should not report a latent KV cache", k)
		}
	}
	for d, want := range map[DType]float64{
		DTypeFP32: 4, DTypeBF16: 2, DTypeFP8: 1, DTypeNVFP4: 0.5,
	} {
		if got := d.Bytes(); got != want {
			t.Errorf("%s.Bytes() = %v, want %v", d, got, want)
		}
	}
}

// A per-node weight dtype exists for mixed-precision MoE: DeepSeek-V4-Pro stores its routed
// experts at fp4 beside fp8 everywhere else. These check the field is governed rather than
// merely accepted -- an unrecognized width that fell back to the global one would reinstate
// the bug the field was added to fix.
func TestPerNodeWeightDType(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  func(*Graph)
		ok   bool
	}{
		{"absent is valid: the global dtype governs", func(g *Graph) {}, true},
		{"a recognized dtype on a GroupedGEMM is valid", func(g *Graph) {
			for i := range g.LayerKinds[0].Nodes {
				if g.LayerKinds[0].Nodes[i].Op == OpGroupedGEMM {
					g.LayerKinds[0].Nodes[i].WeightDType = DTypeNVFP4
				}
			}
		}, true},
		{"an unrecognized dtype is rejected", func(g *Graph) {
			for i := range g.LayerKinds[0].Nodes {
				if g.LayerKinds[0].Nodes[i].Op == OpGroupedGEMM {
					g.LayerKinds[0].Nodes[i].WeightDType = "fp6"
				}
			}
		}, false},
		{"a dtype on a node holding no parameters is rejected", func(g *Graph) {
			for i := range g.LayerKinds[0].Nodes {
				if g.LayerKinds[0].Nodes[i].Op == OpAttention {
					g.LayerKinds[0].Nodes[i].WeightDType = DTypeFP8
				}
			}
		}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := validGraph()
			tc.set(g)
			p := g.Validate()
			if got := p.OK(); got != tc.ok {
				t.Fatalf("OK() = %v, want %v: %v", got, tc.ok, p.Error())
			}
		})
	}
}
