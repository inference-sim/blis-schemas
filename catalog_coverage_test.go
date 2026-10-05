package blisschemas

import (
	"testing"

	"github.com/inference-sim/blis-schemas/spec/model"
)

// This suite asserts that the model schema expresses every architecture family in
// blis-catalog, not only the three the design documents work through. The shapes
// below are taken from the committed vendor configurations; each case names the
// schema capability it exercises, so a failure says which capability was lost rather
// than only which model broke.
//
// Nine families cover the catalog's thirty-two models: dense attention (Llama,
// Mistral, Qwen3, Yi, CodeLlama), sliding-window (Qwen2.5), routed MoE (Mixtral,
// Qwen3-MoE, Llama-4), latent-attention MoE (DeepSeek-V2-Lite), latent MoE with
// sparse indexing (GLM-5.x), compressed sparse-MLA MoE (DeepSeek-V4-Pro), latent
// plus linear attention (Kimi-K3), state-space plus MoE (NemotronH), and
// sliding-window MoE (Inkling).

func derivation(path string) model.Derivation {
	d := model.Derivation{Format: "hf_config_json", Path: path, DeriverVersion: 1}
	for i := 0; i < 64; i++ {
		d.SHA256 += "d"
	}
	return d
}

func graph(name string, hidden, vocab int, dt model.DType,
	kinds []model.LayerKind, stack model.Stack) *model.Graph {
	return &model.Graph{
		Kind: "ModelGraph", Name: name, DerivedFrom: derivation("config.json"),
		Global: model.GlobalShape{HiddenSize: hidden, VocabSize: vocab,
			WeightDType: dt},
		LayerKinds: kinds, Stack: stack,
	}
}

// dense is the Llama/Mistral/Qwen3/Yi shape: GQA attention and a dense MLP.
func dense() *model.Graph {
	return graph("llama-3.1-70b-instruct", 8192, 128256, model.DTypeBF16,
		[]model.LayerKind{{ID: "dense", Nodes: []model.Node{
			{Op: model.OpElementwise, Role: "input_norm"},
			{Op: model.OpGEMM, Role: "qkv_proj", N: 10240, K: 8192},
			{Op: model.OpAttention, AttentionKind: model.AttentionGQA,
				NumQHeads: 64, NumKVHeads: 8, HeadDim: 128},
			{Op: model.OpGEMM, Role: "o_proj", N: 8192, K: 8192},
			{Op: model.OpAllReduce, Role: "attn_out", Emit: model.EmitTensorParallel},
			{Op: model.OpGEMM, Role: "mlp_gate_up", N: 57344, K: 8192},
			{Op: model.OpGEMM, Role: "mlp_down", N: 8192, K: 28672},
			{Op: model.OpAllReduce, Role: "mlp_out", Emit: model.EmitTensorParallel},
		}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}}}},
		model.Stack{Pattern: []string{"dense"}, Repeat: 80})
}

// slidingWindow is the Qwen2.5 shape: the same as dense, with a bounded per-token
// read rather than a context-length one.
func slidingWindow() *model.Graph {
	g := dense()
	g.Name = "qwen2.5-7b-instruct"
	g.Global = model.GlobalShape{HiddenSize: 3584, VocabSize: 152064,
		WeightDType: model.DTypeBF16}
	g.LayerKinds[0].Nodes[2] = model.Node{Op: model.OpAttention,
		AttentionKind: model.AttentionSWA, NumQHeads: 28, NumKVHeads: 4,
		HeadDim: 128, Window: 131072}
	g.Stack = model.Stack{Pattern: []string{"dense"}, Repeat: 28}
	return g
}

// routedMoE is the Mixtral/Qwen3-MoE shape: a routed expert layer, no shared expert.
func routedMoE() *model.Graph {
	return graph("mixtral-8x7b-v0.1", 4096, 32000, model.DTypeBF16,
		[]model.LayerKind{{ID: "moe", Nodes: []model.Node{
			{Op: model.OpElementwise, Role: "input_norm"},
			{Op: model.OpAttention, AttentionKind: model.AttentionGQA,
				NumQHeads: 32, NumKVHeads: 8, HeadDim: 128},
			{Op: model.OpGroupedGEMM, Role: "experts", N: 14336, K: 4096,
				Experts: 8, TopK: 2},
			{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
		}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}}}},
		model.Stack{Pattern: []string{"moe"}, Repeat: 32})
}

// latentMoE is the DeepSeek-V2-Lite shape: latent attention with shared experts.
func latentMoE() *model.Graph {
	return graph("deepseek-v2-lite", 2048, 102400, model.DTypeBF16,
		[]model.LayerKind{{ID: "mla_moe", Nodes: []model.Node{
			{Op: model.OpElementwise, Role: "input_norm"},
			{Op: model.OpAttention, AttentionKind: model.AttentionMLA,
				NumQHeads: 16, NumKVHeads: 1, HeadDim: 576,
				KVLoRARank: 512, QKRopeHeadDim: 64},
			{Op: model.OpGroupedGEMM, Role: "experts", N: 1408, K: 2048,
				Experts: 64, TopK: 6, SharedExperts: 2,
				SharedIntermediateSize: 2816},
			{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
		}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}}}},
		model.Stack{Pattern: []string{"mla_moe"}, Repeat: 27})
}

// glm53 is the GLM-5.3 shape: latent attention with a sparse index, a shared expert,
// and dense layers before the sparse ones — so the stack has two kinds.
func glm53() *model.Graph {
	sparseAttn := model.Node{Op: model.OpAttention,
		AttentionKind: model.AttentionSparseMLA, NumQHeads: 64, NumKVHeads: 1,
		HeadDim: 192, KVLoRARank: 512, QKRopeHeadDim: 64, IndexTopK: 2048}
	denseLayer := model.LayerKind{ID: "mla_dense", Nodes: []model.Node{
		{Op: model.OpElementwise, Role: "input_norm"},
		sparseAttn,
		{Op: model.OpGEMM, Role: "mlp_gate_up", N: 32768, K: 6144},
		{Op: model.OpGEMM, Role: "mlp_down", N: 6144, K: 16384},
		{Op: model.OpAllReduce, Role: "mlp_out", Emit: model.EmitTensorParallel},
	}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}}}
	sparseLayer := model.LayerKind{ID: "mla_sparse", Nodes: []model.Node{
		{Op: model.OpElementwise, Role: "input_norm"},
		sparseAttn,
		{Op: model.OpGroupedGEMM, Role: "experts", N: 2048, K: 6144,
			Experts: 256, TopK: 8, SharedExperts: 1, SharedIntermediateSize: 2048},
		{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
	}, Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}}}
	g := graph("glm-5.3", 6144, 151552, model.DTypeBF16,
		[]model.LayerKind{denseLayer, sparseLayer},
		// Three dense layers, then sparse throughout: expressed as a leading
		// pattern would need a prologue, so the repeated unit carries both.
		// Three dense layers then seventy-five sparse: a prologue, because the
		// sequence has no repeating unit that includes the dense run.
		model.Stack{Prologue: []string{"mla_dense", "mla_dense", "mla_dense"},
			Pattern: []string{"mla_sparse"}, Repeat: 75})
	g.Speculator = &model.Speculator{Method: "glm4_moe_mtp", NumSpec: 1,
		Stack: model.Stack{Pattern: []string{"mla_sparse"}, Repeat: 1}}
	return g
}

// kimiK3 is the Kimi-K3 shape: linear attention on most layers, latent attention
// every fourth, 896 experts with two shared.
func kimiK3() *model.Graph {
	moeNodes := func(attn model.Node) []model.Node {
		return []model.Node{
			{Op: model.OpElementwise, Role: "input_norm"},
			attn,
			{Op: model.OpGroupedGEMM, Role: "experts", N: 3072, K: 7168,
				Experts: 896, TopK: 16, SharedExperts: 2,
				SharedIntermediateSize: 3072},
			{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
		}
	}
	edges := [][2]int{{0, 1}, {1, 2}, {2, 3}}
	kda := model.LayerKind{ID: "kda_moe", Nodes: moeNodes(model.Node{
		Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentKDA,
		NumHeads: 96, StateSize: 128, StateDType: model.DTypeFP32}), Edges: edges}
	mla := model.LayerKind{ID: "mla_moe", Nodes: moeNodes(model.Node{
		Op: model.OpAttention, AttentionKind: model.AttentionMLA,
		NumQHeads: 96, NumKVHeads: 1, HeadDim: 576, KVLoRARank: 512,
		QKRopeHeadDim: 64}), Edges: edges}
	g := graph("kimi-k3", 7168, 163840, model.DTypeFP8,
		[]model.LayerKind{kda, mla},
		// Full attention on every fourth layer: twenty-three complete groups of
		// four, then one trailing linear-attention layer, for ninety-three.
		model.Stack{Pattern: []string{"kda_moe", "kda_moe", "kda_moe", "mla_moe"},
			Repeat: 23, Epilogue: []string{"kda_moe"}})
	// The committed configuration nests these shapes under a text sub-object beside a
	// vision tower, so the graph prices the decoder alone and says so.
	g.Modality = model.ModalityTextDecoderOfMultimodal
	return g
}

// nemotronH is the NemotronH shape: a Mamba2 state-space layer, a separate MoE
// layer, and an attention layer, alternating on a declared pattern. It also uses
// latent MoE, which narrows the expert input below the hidden size.
func nemotronH() *model.Graph {
	mamba := model.LayerKind{ID: "mamba", Nodes: []model.Node{
		{Op: model.OpElementwise, Role: "input_norm"},
		{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentMamba2,
			NumHeads: 256, StateSize: 128, NumGroups: 8, ConvKernel: 4,
			IntermediateSize: 16384, StateDType: model.DTypeFP32},
		{Op: model.OpAllReduce, Role: "mixer_out", Emit: model.EmitTensorParallel},
	}, Edges: [][2]int{{0, 1}, {1, 2}}}
	moe := model.LayerKind{ID: "moe", Nodes: []model.Node{
		{Op: model.OpElementwise, Role: "input_norm"},
		{Op: model.OpGroupedGEMM, Role: "experts", N: 5120, K: 8192,
			Experts: 512, TopK: 22, SharedExperts: 1,
			SharedIntermediateSize: 10240, LatentSize: 2048},
		{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
	}, Edges: [][2]int{{0, 1}, {1, 2}}}
	attn := model.LayerKind{ID: "attention", Nodes: []model.Node{
		{Op: model.OpElementwise, Role: "input_norm"},
		{Op: model.OpAttention, AttentionKind: model.AttentionGQA,
			NumQHeads: 64, NumKVHeads: 2, HeadDim: 128},
		{Op: model.OpAllReduce, Role: "attn_out", Emit: model.EmitTensorParallel},
	}, Edges: [][2]int{{0, 1}, {1, 2}}}
	g := graph("nemotron-3-ultra-550b-a55b-bf16", 8192, 131072, model.DTypeBF16,
		[]model.LayerKind{mamba, moe, attn},
		// The committed configuration declares a 108-entry layer vector with no
		// repeating unit, so it is stated literally rather than compressed. This is
		// the case a pattern-and-repeat schema alone could not express.
		model.Stack{Prologue: nemotronLayerVector()})
	g.Speculator = &model.Speculator{Method: "nemotron_h_mtp", NumSpec: 1,
		Stack: model.Stack{Pattern: []string{"attention", "moe"}, Repeat: 1}}
	return g
}

// nemotronLayerVector is the committed layer sequence, transcribed verbatim from the
// vendor configuration: 48 state-space layers, 48 MoE layers and 12 attention layers
// in a declared order with no repeating unit that divides 108.
//
// It is stated literally rather than generated. An earlier version of this file built
// it from a motif and produced 54/41/13, which the composition test caught: a
// sequence that looks periodic at a glance is not, and guessing its rule silently
// describes a different model.
func nemotronLayerVector() []string {
	return []string{
		"mamba", "moe", "mamba", "moe", "mamba", "moe",
		"mamba", "attention", "moe", "mamba", "moe", "mamba",
		"moe", "mamba", "attention", "moe", "mamba", "moe",
		"mamba", "moe", "mamba", "moe", "mamba", "attention",
		"moe", "mamba", "moe", "mamba", "moe", "mamba",
		"moe", "mamba", "attention", "moe", "mamba", "moe",
		"mamba", "moe", "mamba", "attention", "moe", "mamba",
		"moe", "mamba", "moe", "mamba", "moe", "mamba",
		"attention", "moe", "mamba", "moe", "mamba", "moe",
		"mamba", "moe", "mamba", "attention", "moe", "mamba",
		"moe", "mamba", "moe", "mamba", "attention", "moe",
		"mamba", "moe", "mamba", "moe", "mamba", "moe",
		"mamba", "attention", "moe", "mamba", "moe", "mamba",
		"moe", "mamba", "moe", "mamba", "attention", "moe",
		"mamba", "moe", "mamba", "moe", "mamba", "attention",
		"moe", "mamba", "moe", "mamba", "moe", "mamba",
		"moe", "mamba", "attention", "moe", "mamba", "moe",
		"mamba", "moe", "mamba", "moe", "mamba", "moe",
	}
}

// nemotronNVFP4 is the same architecture at a narrower weight dtype, which is the
// only difference between the catalog's two Nemotron variants.
func nemotronNVFP4() *model.Graph {
	g := nemotronH()
	g.Name = "nemotron-3-ultra-550b-a55b-nvfp4"
	g.Global.WeightDType = model.DTypeNVFP4
	return g
}

// inkling is the Inkling shape: sliding-window attention interleaved with full
// attention, and a routed MoE with two shared experts.
func inkling() *model.Graph {
	moe := func(attn model.Node) []model.Node {
		return []model.Node{
			{Op: model.OpElementwise, Role: "input_norm"},
			attn,
			{Op: model.OpGroupedGEMM, Role: "experts", N: 2048, K: 6144,
				Experts: 256, TopK: 6, SharedExperts: 2,
				SharedIntermediateSize: 2048},
			{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
		}
	}
	edges := [][2]int{{0, 1}, {1, 2}, {2, 3}}
	swa := model.LayerKind{ID: "swa_moe", Nodes: moe(model.Node{
		Op: model.OpAttention, AttentionKind: model.AttentionSWA,
		NumQHeads: 64, NumKVHeads: 16, HeadDim: 128, Window: 512}), Edges: edges}
	full := model.LayerKind{ID: "full_moe", Nodes: moe(model.Node{
		Op: model.OpAttention, AttentionKind: model.AttentionGQA,
		NumQHeads: 64, NumKVHeads: 8, HeadDim: 128}), Edges: edges}
	g := graph("inkling", 6144, 151552, model.DTypeBF16,
		[]model.LayerKind{swa, full},
		model.Stack{Pattern: []string{"swa_moe", "swa_moe", "swa_moe", "swa_moe",
			"swa_moe", "full_moe"}, Repeat: 11})
	g.Speculator = &model.Speculator{Method: "inkling_mtp", NumSpec: 1,
		Stack: model.Stack{Pattern: []string{"full_moe"}, Repeat: 1}}
	// Also multimodal in the catalog: a vision and an audio tower sit beside the
	// text shapes this graph prices.
	g.Modality = model.ModalityTextDecoderOfMultimodal
	return g
}

// deepseekV4Pro is the DeepSeek-V4-Pro shape: compressed sparse-MLA attention over an
// MoE stack. One layer kind both selects a top-k of the latent cache and compresses it
// (csa4_moe: index_topk and compress_ratio 4); the other compresses heavily with no
// top-k selection (csa128_moe: compress_ratio 128, no index_topk) — the compress-only
// sparse-MLA shape this schema gained CompressRatio to express.
func deepseekV4Pro() *model.Graph {
	moe := func(attn model.Node) []model.Node {
		return []model.Node{
			{Op: model.OpElementwise, Role: "input_norm"},
			attn,
			{Op: model.OpGroupedGEMM, Role: "experts", N: 2048, K: 7168,
				Experts: 256, TopK: 8, SharedExperts: 1, SharedIntermediateSize: 2048},
			{Op: model.OpAll2All, Role: "moe", Emit: model.EmitExpertParallel},
		}
	}
	edges := [][2]int{{0, 1}, {1, 2}, {2, 3}}
	csa4 := model.LayerKind{ID: "csa4_moe", Nodes: moe(model.Node{
		Op: model.OpAttention, AttentionKind: model.AttentionSparseMLA,
		NumQHeads: 128, NumKVHeads: 1, HeadDim: 512, QKRopeHeadDim: 64,
		IndexTopK: 1024, CompressRatio: 4}), Edges: edges}
	csa128 := model.LayerKind{ID: "csa128_moe", Nodes: moe(model.Node{
		Op: model.OpAttention, AttentionKind: model.AttentionSparseMLA,
		NumQHeads: 128, NumKVHeads: 1, HeadDim: 512, QKRopeHeadDim: 64,
		CompressRatio: 128}), Edges: edges}
	return graph("deepseek-v4-pro", 7168, 129280, model.DTypeFP8,
		[]model.LayerKind{csa128, csa4},
		// A csa128 prologue, then csa128/csa4 pairs repeated: 1 + 2*30 = 61 layers.
		model.Stack{Prologue: []string{"csa128_moe"},
			Pattern: []string{"csa128_moe", "csa4_moe"}, Repeat: 30})
}

// TestCatalogArchitectureFamiliesAreExpressible walks every family in the catalog.
func TestCatalogArchitectureFamiliesAreExpressible(t *testing.T) {
	cases := []struct {
		capability string
		g          *model.Graph
		layers     int
	}{
		{"dense GQA attention with a dense MLP", dense(), 80},
		{"sliding-window attention", slidingWindow(), 28},
		{"routed MoE with no shared expert", routedMoE(), 32},
		{"latent attention with shared experts", latentMoE(), 27},
		{"a dense prologue before sparse layers, sparse indexing, MTP", glm53(), 78},
		{"linear plus latent attention, 896 experts, odd layer count", kimiK3(), 93},
		{"a non-periodic layer vector stated literally, latent MoE", nemotronH(), 108},
		{"the same architecture at a narrower weight dtype", nemotronNVFP4(), 108},
		{"sliding-window and full attention with shared experts", inkling(), 66},
		{"compress-only and top-k-plus-compress sparse-MLA (DeepSeek-V4-Pro)", deepseekV4Pro(), 61},
	}
	for _, c := range cases {
		t.Run(c.capability, func(t *testing.T) {
			if p := c.g.Validate(); !p.OK() {
				t.Fatalf("%s is not expressible:\n%s", c.g.Name, p.Error())
			}
			if got := c.g.Stack.Layers(); got != c.layers {
				t.Errorf("%s: layers = %d, want %d", c.g.Name, got, c.layers)
			}
		})
	}
}

// TestEveryPrimitiveAndKindIsExercised guards against a family passing because the
// suite never uses the schema feature it needs. A capability with no case is a
// capability with no coverage.
func TestEveryPrimitiveAndKindIsExercised(t *testing.T) {
	graphs := []*model.Graph{dense(), slidingWindow(), routedMoE(), latentMoE(),
		glm53(), kimiK3(), nemotronH(), nemotronNVFP4(), inkling()}

	ops := map[model.Op]bool{}
	attnKinds := map[model.AttentionKind]bool{}
	recurrentKinds := map[model.RecurrentKind]bool{}
	dtypes := map[model.DType]bool{}
	sawShared, sawLatentMoE, sawSpeculator, sawMultiKind := false, false, false, false

	for _, g := range graphs {
		dtypes[g.Global.WeightDType] = true
		if g.Speculator != nil {
			sawSpeculator = true
		}
		if len(g.LayerKinds) > 1 {
			sawMultiKind = true
		}
		for _, lk := range g.LayerKinds {
			for _, n := range lk.Nodes {
				ops[n.Op] = true
				if n.AttentionKind != "" {
					attnKinds[n.AttentionKind] = true
				}
				if n.RecurrentKind != "" {
					recurrentKinds[n.RecurrentKind] = true
				}
				if n.SharedExperts > 0 {
					sawShared = true
				}
				if n.LatentSize > 0 {
					sawLatentMoE = true
				}
			}
		}
	}

	for _, op := range []model.Op{model.OpGEMM, model.OpGroupedGEMM,
		model.OpAttention, model.OpRecurrentUpdate, model.OpElementwise,
		model.OpAllReduce, model.OpAll2All} {
		if !ops[op] {
			t.Errorf("no catalog family exercises %s", op)
		}
	}
	for _, k := range []model.AttentionKind{model.AttentionGQA, model.AttentionMLA,
		model.AttentionSparseMLA, model.AttentionSWA} {
		if !attnKinds[k] {
			t.Errorf("no catalog family exercises attention kind %s", k)
		}
	}
	for _, k := range []model.RecurrentKind{model.RecurrentMamba2, model.RecurrentKDA} {
		if !recurrentKinds[k] {
			t.Errorf("no catalog family exercises recurrent kind %s", k)
		}
	}
	for _, d := range []model.DType{model.DTypeBF16, model.DTypeFP8, model.DTypeNVFP4} {
		if !dtypes[d] {
			t.Errorf("no catalog family exercises weight dtype %s", d)
		}
	}
	if !sawShared {
		t.Error("no family exercises shared experts, which four catalog models use")
	}
	if !sawLatentMoE {
		t.Error("no family exercises latent MoE, which narrows the expert input")
	}
	if !sawSpeculator {
		t.Error("no family exercises a speculator stack")
	}
	if !sawMultiKind {
		t.Error("no family exercises a multi-kind stack, which every hybrid needs")
	}
	// Three catalog models nest their text shapes beside a vision or audio tower. A
	// graph for one of those must declare that it prices the decoder alone, or a
	// reader has no way to know a multimodal request is under-predicted.
	sawMultimodal := false
	for _, g := range graphs {
		if g.Modality == model.ModalityTextDecoderOfMultimodal {
			sawMultimodal = true
		}
	}
	if !sawMultimodal {
		t.Error("no family declares itself the text decoder of a multimodal model, though three catalog models are")
	}
}

// TestLayerCountsMatchTheCommittedConfigurations pins each family's layer count
// against the vendor configuration's own num_hidden_layers. Without this, a test
// graph could pass validation while describing a model with the wrong depth — the
// failure an earlier draft of this suite actually had, where three of the counts
// were assumed rather than derived.
func TestLayerCountsMatchTheCommittedConfigurations(t *testing.T) {
	// Taken from blis-catalog's committed config.json files.
	want := map[string]int{
		"llama-3.1-70b-instruct":          80,
		"qwen2.5-7b-instruct":             28,
		"mixtral-8x7b-v0.1":               32,
		"deepseek-v2-lite":                27,
		"glm-5.3":                         78,
		"kimi-k3":                         93,
		"nemotron-3-ultra-550b-a55b-bf16": 108,
		"inkling":                         66,
		"deepseek-v4-pro":                 61,
	}
	for _, g := range []*model.Graph{dense(), slidingWindow(), routedMoE(),
		latentMoE(), glm53(), kimiK3(), nemotronH(), inkling(), deepseekV4Pro()} {
		exp, ok := want[g.Name]
		if !ok {
			t.Errorf("%s has no expected layer count", g.Name)
			continue
		}
		if got := g.Stack.Layers(); got != exp {
			t.Errorf("%s: layers = %d, the configuration declares %d", g.Name, got, exp)
		}
	}
}

// TestNemotronLayerVectorComposition checks the literal vector's make-up against the
// committed configuration's counts, since a hand-built sequence is easy to get wrong.
func TestNemotronLayerVectorComposition(t *testing.T) {
	v := nemotronLayerVector()
	if len(v) != 108 {
		t.Fatalf("vector length = %d, want 108", len(v))
	}
	counts := map[string]int{}
	for _, k := range v {
		counts[k]++
	}
	// The configuration declares 48 state-space, 48 MoE and 12 attention layers.
	for kind, want := range map[string]int{"mamba": 48, "moe": 48, "attention": 12} {
		if counts[kind] != want {
			t.Errorf("%s layers = %d, the configuration declares %d",
				kind, counts[kind], want)
		}
	}
}

// TestStackExpansionShapes covers the three stack forms the schema supports, since
// each exists for a model the catalog holds.
func TestStackExpansionShapes(t *testing.T) {
	uniform := model.Stack{Pattern: []string{"a"}, Repeat: 4}
	if got := uniform.Expand(); len(got) != 4 || got[0] != "a" {
		t.Errorf("uniform expansion = %v", got)
	}
	withPrologue := model.Stack{Prologue: []string{"d", "d"},
		Pattern: []string{"s"}, Repeat: 3}
	want := []string{"d", "d", "s", "s", "s"}
	got := withPrologue.Expand()
	if len(got) != len(want) {
		t.Fatalf("prologue expansion = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] = %q, want %q", i, got[i], want[i])
		}
	}
	literal := model.Stack{Prologue: []string{"a", "b", "c"}}
	if got := literal.Layers(); got != 3 {
		t.Errorf("literal stack layers = %d, want 3", got)
	}
	if got := literal.Expand(); len(got) != 3 {
		t.Errorf("literal expansion = %v", got)
	}
}
