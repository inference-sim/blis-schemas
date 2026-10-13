package model

import (
	"strings"
	"testing"
)

// The keys added for DeepSeek-V4.1 (#47-#54) each get a passing case and a case for every
// check, built on one small graph that uses all of them, so each failure names the rule
// that fired.

// v41Graph is a three-layer stack, a source layer that owns a compressed cache and two
// readers, plus a Gather, a replicated GEMM and a parallel-block speculator whose context
// GEMM aliases its draft layer's.
func v41Graph() *Graph {
	g := validGraph()
	g.Global.WeightScale = &WeightScale{Elements: 32, Bytes: 1}
	g.Global.ContextWindow = 1048576
	attn := func(owned bool) Node {
		return Node{Op: OpAttention, AttentionKind: AttentionSparseMLA, NumQHeads: 64,
			NumKVHeads: 1, HeadDim: 512, Window: 128, IndexTopK: 512, CompressRatio: 2,
			Caches: []Cache{
				{Name: "swa", Kind: CacheWindow, Owned: true, BytesPerEntry: 584, Window: 128},
				{Name: "compressed", Kind: CacheKV, Owned: owned, BytesPerEntry: 584, TokensPerEntry: 2},
			}}
	}
	g.Embedding = []Node{{Op: OpGather, Role: "embed_tokens", Rows: 100352, RowBytes: 6144,
		RowsPerToken: 1, OutBytesPerToken: 6144, Shard: ShardRows}}
	g.LayerKinds = []LayerKind{
		{ID: "source", Nodes: []Node{
			{Op: OpGEMM, Role: "fused_wqa_wkv", N: 1792, K: 3072, TP: TPReplicated},
			attn(true),
			{Op: OpGather, Role: "engram_lookup", Rows: 384006168, RowBytes: 264, RowsPerToken: 24,
				OutBytesPerToken: 12288, Residency: ResidencyHost,
				ResidencyControl: ResidencyEngramCPUOffload, Shard: ShardHeads},
			{Op: OpAllGather, Role: "engram_rows", Emit: EmitTensorParallel, BytesPerToken: 12288},
			{Op: OpElementwise, Role: "k_store", PerTokens: 2},
			{Op: OpElementwise, Role: "mask", Context: &ContextWork{BytesPerEntry: 4,
				Per: PerQueryToken, DecodeExtent: ExtentMaxModelLen}},
		}},
		{ID: "reader", Nodes: []Node{attn(false)}},
		{ID: "draft", Nodes: []Node{
			{Op: OpGEMM, Role: "fused_wqa_wkv", N: 1792, K: 3072, TP: TPReplicated},
		}},
		{ID: "draft_context", Nodes: []Node{
			{Op: OpGEMM, Role: "context_wqa_wkv", N: 1792, K: 3072, Aliases: "draft.fused_wqa_wkv"},
		}},
	}
	g.Stack = Stack{Prologue: []string{"source"}, Pattern: []string{"reader"}, Repeat: 2}
	g.Head = []Node{{Op: OpGEMM, Role: "lm_head", N: 100352, K: 3072, TP: TPColumn}}
	g.Speculator = &Speculator{Method: "dspark", NumSpec: 5, Drafting: DraftingParallelBlock,
		Stack:        Stack{Pattern: []string{"draft"}, Repeat: 1},
		ContextStack: &Stack{Pattern: []string{"draft_context"}, Repeat: 1},
		Head:         []Node{{Op: OpGEMM, Role: "draft_lm_head", N: 100352, K: 3072, Aliases: "head.lm_head"}},
		SequentialHead: []Node{{Op: OpGather, Role: "markov_w1", Rows: 100352, RowBytes: 512,
			RowsPerToken: 1, OutBytesPerToken: 512}},
	}
	g.ArchOverrides = []ArchOverride{{CapabilityFamily: 10,
		Nodes:  []NodeOverride{{Role: "fused_wqa_wkv", WeightDType: DTypeBF16}},
		Caches: []CacheOverride{{Name: "swa", BytesPerEntry: 528}}}}
	return g
}

// A node stated twice for the two sequence-parallel layouts may carry the conditions
// without the warning a deployment condition on a non-collective node draws.
func TestSequenceParallelPairOnAGEMM(t *testing.T) {
	g := v41Graph()
	g.Global.Architecture = "DeepseekV41ForCausalLM"
	g.LayerKinds[1].Nodes = append(g.LayerKinds[1].Nodes,
		Node{Op: OpGEMM, Role: "wkv", N: 25600, K: 6144, TP: TPColumn, Emit: EmitTensorParallelUnlessSequenceParallel},
		Node{Op: OpGEMM, Role: "wkv_sp", N: 25600, K: 6144, TP: TPReplicated, Emit: EmitSequenceParallel})
	if p := g.Validate(); len(p.All()) != 0 {
		t.Fatalf("the pair drew findings:\n%s", p.Error())
	}
}

func TestV41GraphPasses(t *testing.T) {
	if p := v41Graph().Validate(); !p.OK() || len(p.All()) != 0 {
		t.Fatalf("a valid graph drew findings:\n%s", p.Error())
	}
}

func TestV41FieldRejects(t *testing.T) {
	source := func(g *Graph) []Node { return g.LayerKinds[0].Nodes }
	cases := []struct {
		name, want string
		mutate     func(*Graph)
	}{
		// Caches.
		{"cache on a GEMM", "only an attention node reads a cache", func(g *Graph) {
			source(g)[0].Caches = []Cache{{Name: "x", Kind: CacheKV, Owned: true, BytesPerEntry: 1}}
		}},
		{"unnamed cache", "caches[0].name", func(g *Graph) { source(g)[1].Caches[0].Name = "" }},
		{"unknown cache kind", "not a recognized cache kind", func(g *Graph) { source(g)[1].Caches[0].Kind = "ring" }},
		{"zero entry bytes", "bytes_per_entry", func(g *Graph) { source(g)[1].Caches[1].BytesPerEntry = 0 }},
		{"window cache without window", "requires a positive window", func(g *Graph) { source(g)[1].Caches[0].Window = 0; source(g)[1].Window = 0 }},
		{"window on a kv cache", "only a window cache is bounded", func(g *Graph) { source(g)[1].Caches[1].Window = 128 }},
		{"state cache without entries", "positive entry count", func(g *Graph) {
			source(g)[1].Caches = append(source(g)[1].Caches, Cache{Name: "ring", Kind: CacheState, Owned: true, BytesPerEntry: 4096})
		}},
		{"entries on a kv cache", "only a state cache", func(g *Graph) { source(g)[1].Caches[1].Entries = 8 }},
		{"kv granularity disagrees with the read", "reads at compress_ratio 2", func(g *Graph) { source(g)[1].Caches[1].TokensPerEntry = 1 }},
		{"window disagrees with the read", "reads a window of 128", func(g *Graph) { source(g)[1].Caches[0].Window = 64 }},
		{"duplicate name in a layer", "duplicate cache name", func(g *Graph) {
			source(g)[1].Caches = append(source(g)[1].Caches, source(g)[1].Caches[0])
		}},
		{"reader with no owner", "no earlier layer owns one", func(g *Graph) { source(g)[1].Caches[1].Owned = false }},
		{"reader before its owner", "no earlier layer owns one", func(g *Graph) {
			g.Stack = Stack{Prologue: []string{"reader", "source"}}
		}},
		{"reader disagrees with owner", "but layer 0 owns it", func(g *Graph) {
			g.LayerKinds[1].Nodes[0].Caches[1].BytesPerEntry = 528
		}},
		{"negative per-dtype record", "caches[1].bytes_by_cache_dtype.nvfp4_ds_mla: must be positive", func(g *Graph) {
			source(g)[1].Caches[1].BytesByCacheDType = map[string]int{"nvfp4_ds_mla": -5}
		}},
		{"empty per-dtype key", "caches[1].bytes_by_cache_dtype: a cache dtype name is empty", func(g *Graph) {
			source(g)[1].Caches[1].BytesByCacheDType = map[string]int{"": 288}
		}},
		{"reader restating paging", "caches[1].page_alignment: describes how the owning layer pages", func(g *Graph) {
			g.LayerKinds[1].Nodes[0].Caches[1].PageAlignment = 576
		}},
		{"reader missing the owner's per-dtype record", "but layer 0 owns it", func(g *Graph) {
			source(g)[1].Caches[1].BytesByCacheDType = map[string]int{"nvfp4_ds_mla": 288}
		}},
		{"reader with a different per-dtype record", "but layer 0 owns it", func(g *Graph) {
			source(g)[1].Caches[1].BytesByCacheDType = map[string]int{"nvfp4_ds_mla": 288}
			g.LayerKinds[1].Nodes[0].Caches[1].BytesByCacheDType = map[string]int{"nvfp4_ds_mla": 999}
		}},
		{"window cache on a node with no window", "caches[0].window: is 128 but the node reads a window of 0", func(g *Graph) {
			source(g)[1].Window = 0
		}},
		{"index cache at the wrong ratio", "tokens_per_entry: is 1 but the node reads at compress_ratio 2", func(g *Graph) {
			source(g)[1].Caches = append(source(g)[1].Caches,
				Cache{Name: "index", Kind: CacheIndex, Owned: true, BytesPerEntry: 132})
		}},
		{"pooled kv cache on a node that does not compress", "tokens_per_entry: is 2 but the node reads at compress_ratio 1", func(g *Graph) {
			for _, lk := range g.LayerKinds[:2] {
				n := &lk.Nodes[indexOfAttention(lk.Nodes)]
				n.AttentionKind, n.CompressRatio, n.IndexTopK = AttentionMLA, 0, 0
			}
		}},
		// Gather.
		{"gather without rows", "layer_kinds[0].nodes[2].rows: must be positive", func(g *Graph) { source(g)[2].Rows = 0 }},
		{"gather reading more rows than it has", "nodes[2].rows_per_token: 24 exceeds the table's 10 rows", func(g *Graph) { source(g)[2].Rows = 10 }},
		{"gather without out bytes", "out_bytes_per_token", func(g *Graph) { source(g)[2].OutBytesPerToken = 0 }},
		{"unknown residency", "not a recognized residency", func(g *Graph) { source(g)[2].Residency = "disk" }},
		{"unknown control", "not a recognized residency control", func(g *Graph) { source(g)[2].ResidencyControl = "magic" }},
		{"unknown shard", "not a recognized shard", func(g *Graph) { source(g)[2].Shard = "cols" }},
		{"heads with one row", "rows_per_token of at least 2", func(g *Graph) { g.Embedding[0].Shard = ShardHeads }},
		{"table field on a GEMM", "only a Gather has a table", func(g *Graph) { source(g)[0].Residency = ResidencyHost }},
		{"rows on a GEMM", "layer_kinds[0].nodes[0].rows: not priced by this op", func(g *Graph) { source(g)[0].Rows = 5 }},
		{"n on a gather", "layer_kinds[0].nodes[2].n: not priced by this op", func(g *Graph) { source(g)[2].N = 5 }},
		// Collective payload.
		{"payload on an All2All", "bytes_per_token", func(g *Graph) {
			g.LayerKinds = append(g.LayerKinds[:0:0], g.LayerKinds...)
			g.LayerKinds[1].Nodes = append(g.LayerKinds[1].Nodes,
				Node{Op: OpAll2All, Emit: EmitExpertParallel, BytesPerToken: 10})
		}},
		{"payload on a GEMM", "bytes_per_token", func(g *Graph) { source(g)[0].BytesPerToken = 10 }},
		{"negative collective payload", "must not be negative", func(g *Graph) { source(g)[3].BytesPerToken = -1 }},
		// Tensor-parallel layout.
		{"row on an Elementwise", "not a layout a Elementwise node accepts", func(g *Graph) { source(g)[4].TP = TPRow }},
		{"sharded on a GEMM", "not a layout a GEMM node accepts", func(g *Graph) { source(g)[0].TP = TPSharded }},
		{"layout on a GroupedGEMM", "not a layout a GroupedGEMM node accepts", func(g *Graph) {
			g.LayerKinds[1].Nodes = append(g.LayerKinds[1].Nodes,
				Node{Op: OpGroupedGEMM, N: 1, K: 1, Experts: 2, TopK: 1, TP: TPReplicated})
		}},
		// Per-entry and context work.
		{"per_tokens on a GEMM", "only an Elementwise node runs once per several", func(g *Graph) { source(g)[0].PerTokens = 2 }},
		{"negative per_tokens", "must not be negative", func(g *Graph) { source(g)[4].PerTokens = -2 }},
		{"context on a GEMM", "only an Elementwise node carries context", func(g *Graph) {
			source(g)[0].Context = &ContextWork{BytesPerEntry: 1, Per: PerRequest}
		}},
		{"context without a unit", "not a recognized unit", func(g *Graph) { source(g)[5].Context.Per = "" }},
		{"context with an unknown phase", "context.phase: \"decode\" is not a recognized phase", func(g *Graph) { source(g)[5].Context.Phase = "decode" }},
		{"context with an unknown extent", "context.decode_extent: \"window\" is not a recognized extent", func(g *Graph) { source(g)[5].Context.DecodeExtent = "window" }},
		{"context with negative tokens per entry", "context.tokens_per_entry: must not be negative", func(g *Graph) { source(g)[5].Context.TokensPerEntry = -2 }},
		{"context without bytes", "bytes_per_entry", func(g *Graph) { source(g)[5].Context.BytesPerEntry = 0 }},
		{"decode extent on prefill work", "only in prefill", func(g *Graph) { source(g)[5].Context.Phase = PhasePrefill }},
		// Weight scales.
		{"scale on an attention node", "only a GEMM or GroupedGEMM", func(g *Graph) {
			source(g)[1].WeightScale = &WeightScale{Elements: 32, Bytes: 1}
		}},
		{"zero scale group", "global.weight_scale.elements: must be positive", func(g *Graph) { g.Global.WeightScale.Elements = 0 }},
		// Aliases.
		{"alias on an attention node", "only a GEMM or Gather", func(g *Graph) { source(g)[1].Aliases = "head.lm_head" }},
		{"malformed alias", "must be <layer kind>.<role>", func(g *Graph) { g.Speculator.Head[0].Aliases = "lm_head" }},
		{"alias to nothing", "matches 0 nodes", func(g *Graph) { g.Speculator.Head[0].Aliases = "head.nope" }},
		{"alias to an unknown kind", "names no layer kind", func(g *Graph) { g.Speculator.Head[0].Aliases = "nope.lm_head" }},
		{"alias to a repeated kind", "runs 2 times", func(g *Graph) {
			g.LayerKinds[3].Nodes[0].Aliases = "reader.x"
		}},
		{"alias of another shape", "cannot be the weights", func(g *Graph) { g.Speculator.Head[0].N = 100 }},
		{"gather alias of another row width", "cannot be the weights", func(g *Graph) {
			g.Speculator.SequentialHead[0].Aliases = "embedding.embed_tokens"
		}},
		{"alias scope both a kind and head", "is ambiguous: a layer kind is also named \"head\"", func(g *Graph) {
			g.LayerKinds = append(g.LayerKinds, LayerKind{ID: "head", Nodes: []Node{{Op: OpElementwise}}})
			g.Stack.Epilogue = []string{"head"}
		}},
		{"alias to an alias", "is itself an alias", func(g *Graph) {
			g.LayerKinds[2].Nodes[0].Aliases = "head.lm_head"
			g.LayerKinds[2].Nodes[0].N, g.LayerKinds[2].Nodes[0].K = 100352, 3072
			g.LayerKinds[3].Nodes[0].N, g.LayerKinds[3].Nodes[0].K = 100352, 3072
		}},
		// Speculator.
		{"unknown drafting", "not a recognized drafting mode", func(g *Graph) { g.Speculator.Drafting = "tree" }},
		{"sequential head on an autoregressive drafter", "only a parallel_block drafter", func(g *Graph) {
			g.Speculator.Drafting = DraftingAutoregressive
		}},
		{"context stack names no kind", "names no layer kind", func(g *Graph) { g.Speculator.ContextStack.Pattern = []string{"nope"} }},
		{"empty context stack", "expands to no layers", func(g *Graph) { g.Speculator.ContextStack = &Stack{} }},
		// Sequence parallelism is the model's code, so its conditions need the architecture.
		{"sequence-parallel condition without an architecture", "needs global.architecture", func(g *Graph) {
			g.LayerKinds[1].Nodes = append(g.LayerKinds[1].Nodes, Node{Op: OpAllGather, Emit: EmitSequenceParallel})
		}},
		// Context window.
		{"default window above the window", "exceeds context_window", func(g *Graph) { g.Global.DefaultContextWindow = 1 << 21 }},
		{"default window alone", "stated without context_window", func(g *Graph) {
			g.Global.ContextWindow, g.Global.DefaultContextWindow = 0, 65536
		}},
		// Arch overrides.
		{"family zero", "positive compute-capability", func(g *Graph) { g.ArchOverrides[0].CapabilityFamily = 0 }},
		{"family twice", "overridden twice", func(g *Graph) { g.ArchOverrides = append(g.ArchOverrides, g.ArchOverrides[0]) }},
		{"override of an absent role", "arch_overrides[0].nodes[0].role: no node has role \"nope\"", func(g *Graph) { g.ArchOverrides[0].Nodes[0].Role = "nope" }},
		{"override of an absent cache", "arch_overrides[0].caches[0].name: no cache is named \"nope\"", func(g *Graph) { g.ArchOverrides[0].Caches[0].Name = "nope" }},
		{"override with no role", "arch_overrides[0].nodes[0].role: required", func(g *Graph) { g.ArchOverrides[0].Nodes[0].Role = "" }},
		{"override with an unknown dtype", "arch_overrides[0].nodes[0].weight_dtype: \"fp6\" is not a recognized dtype", func(g *Graph) {
			g.ArchOverrides[0].Nodes[0].WeightDType = "fp6"
		}},
		{"cache override with no name", "arch_overrides[0].caches[0].name: required", func(g *Graph) { g.ArchOverrides[0].Caches[0].Name = "" }},
		{"cache override with a negative size", "arch_overrides[0].caches[0]: sizes must not be negative", func(g *Graph) {
			g.ArchOverrides[0].Caches[0].BytesPerEntry = -1
		}},
		{"cache override with a zero per-dtype record", "bytes_by_cache_dtype.nvfp4_ds_mla: must be positive, got 0", func(g *Graph) {
			g.ArchOverrides[0].Caches[0].BytesByCacheDType = map[string]int{"nvfp4_ds_mla": 0}
		}},
		{"cache override restating nothing", "arch_overrides[0].caches[0]: restates nothing", func(g *Graph) {
			g.ArchOverrides[0].Caches[0].BytesPerEntry = 0
		}},
		{"override restating nothing", "restates nothing", func(g *Graph) { g.ArchOverrides[0].Nodes[0].WeightDType = "" }},
		{"override of a weightless role", "holds no parameters", func(g *Graph) {
			g.ArchOverrides[0].Nodes = append(g.ArchOverrides[0].Nodes, NodeOverride{Role: "mask", WeightDType: DTypeBF16})
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g := v41Graph()
			c.mutate(g)
			p := g.Validate()
			if p.OK() {
				t.Fatalf("accepted")
			}
			if !strings.Contains(p.Error(), c.want) {
				t.Errorf("want a finding containing %q, got:\n%s", c.want, p.Error())
			}
		})
	}
}

// A reader repeated across a stack reports its missing owner once, not once per layer.
func TestCacheSharingReportsOncePerLayerKind(t *testing.T) {
	g := v41Graph()
	g.Stack = Stack{Pattern: []string{"reader"}, Repeat: 5}
	g.LayerKinds[0].Nodes[1].Caches[1].Owned = false // the source kind is now unused
	n := 0
	for _, pr := range g.Validate().All() {
		if strings.Contains(pr.Message, "no earlier layer owns one") {
			n++
		}
	}
	if n != 1 {
		t.Errorf("missing owner reported %d times, want once", n)
	}
}

func TestStoredBytes(t *testing.T) {
	for _, c := range []struct {
		d    DType
		s    *WeightScale
		want float64
	}{
		{DTypeMXFP4, nil, 0.53125},
		{DTypeNVFP4, nil, 0.5625},
		{DTypeFP8, nil, 1},
		{DTypeFP8, &WeightScale{Elements: 32, Bytes: 1}, 1.03125},
		{DTypeFP8, &WeightScale{Elements: 128 * 128, Bytes: 4}, 1 + 4.0/16384},
		{DTypeINT4, &WeightScale{Elements: 32, Bytes: 2}, 0.5625},
		{DTypeBF16, nil, 2},
	} {
		if got := StoredBytes(c.d, c.s); got != c.want {
			t.Errorf("StoredBytes(%s, %+v) = %v, want %v", c.d, c.s, got, c.want)
		}
	}
}

func TestForCapabilityLeavesOtherFamiliesAlone(t *testing.T) {
	g := v41Graph()
	for _, major := range []int{8, 9, 12} {
		r, err := g.ForCapability(major)
		if err != nil {
			t.Fatal(err)
		}
		if got := r.LayerKinds[0].Nodes[0].WeightDType; got != "" {
			t.Errorf("family %d picked up the family-10 override: %q", major, got)
		}
	}
	r, err := g.ForCapability(10)
	if err != nil {
		t.Fatal(err)
	}
	// The override names a role, so it reaches every node with it: the target's and the
	// draft's fused_wqa_wkv alike.
	if r.LayerKinds[0].Nodes[0].WeightDType != DTypeBF16 || r.LayerKinds[2].Nodes[0].WeightDType != DTypeBF16 {
		t.Error("the role override did not reach every node with the role")
	}
	if g.LayerKinds[0].Nodes[0].WeightDType != "" || g.LayerKinds[0].Nodes[1].Caches[0].BytesPerEntry != 584 {
		t.Error("ForCapability modified its receiver")
	}
}

func TestTPLayoutByOp(t *testing.T) {
	for _, c := range []struct {
		l    TPLayout
		o    Op
		want bool
	}{
		{TPDefault, OpAllReduce, true}, {TPColumn, OpGEMM, true}, {TPRow, OpGEMM, true},
		{TPReplicated, OpGEMM, true}, {TPSharded, OpGEMM, false},
		{TPSharded, OpElementwise, true}, {TPReplicated, OpAttention, true},
		{TPColumn, OpAttention, false}, {TPReplicated, OpGroupedGEMM, false},
		{TPReplicated, OpAllGather, false},
	} {
		if got := c.l.ValidFor(c.o); got != c.want {
			t.Errorf("%q on %s: %v, want %v", c.l, c.o, got, c.want)
		}
	}
}

func indexOfAttention(ns []Node) int {
	for i, n := range ns {
		if n.Op == OpAttention {
			return i
		}
	}
	panic("no attention node")
}

// Validation reports every problem, not the first: an override naming two absent roles
// reports both, each at its own entry.
func TestArchOverrideReportsEveryMiss(t *testing.T) {
	g := v41Graph()
	g.ArchOverrides[0].Nodes = []NodeOverride{
		{Role: "nope1", WeightDType: DTypeBF16}, {Role: "nope2", WeightDType: DTypeBF16}}
	got := map[string]bool{}
	for _, pr := range g.Validate().All() {
		got[pr.Path] = true
	}
	for _, want := range []string{"arch_overrides[0].nodes[0].role", "arch_overrides[0].nodes[1].role"} {
		if !got[want] {
			t.Errorf("no problem at %s; got %v", want, got)
		}
	}
}

// A layer that reads another's state cache states the entries it reads, and they must
// equal the owner's; a reader cannot be refused for stating what its kind requires.
func TestStateCacheReader(t *testing.T) {
	g := v41Graph()
	ring := Cache{Name: "ring", Kind: CacheState, Owned: true, BytesPerEntry: 4096, Entries: 8}
	g.LayerKinds[0].Nodes[1].Caches = append(g.LayerKinds[0].Nodes[1].Caches, ring)
	ring.Owned = false
	g.LayerKinds[1].Nodes[0].Caches = append(g.LayerKinds[1].Nodes[0].Caches, ring)
	if p := g.Validate(); !p.OK() {
		t.Fatalf("a state-cache reader was refused:\n%s", p.Error())
	}
	g.LayerKinds[1].Nodes[0].Caches[2].Entries = 16
	if p := g.Validate(); p.OK() {
		t.Error("a reader of 16 entries from an 8-entry owner validated")
	}
}

// A reader prices its read with the owner's record under every cache dtype, on every
// architecture: for each family, the reader's entry equals the owner's under each dtype
// either lists.
func TestReadersReadTheOwnersRecord(t *testing.T) {
	g := v41Graph()
	g.LayerKinds[0].Nodes[1].Caches[1].BytesByCacheDType = map[string]int{"nvfp4_ds_mla": 288}
	g.LayerKinds[1].Nodes[0].Caches[1].BytesByCacheDType = map[string]int{"nvfp4_ds_mla": 288}
	g.ArchOverrides[0].Caches = append(g.ArchOverrides[0].Caches,
		CacheOverride{Name: "compressed", BytesPerEntry: 528, BytesByCacheDType: map[string]int{"nvfp4_ds_mla": 256}})
	if p := g.Validate(); !p.OK() {
		t.Fatal(p.Error())
	}
	for _, major := range []int{9, 10} {
		r, err := g.ForCapability(major)
		if err != nil {
			t.Fatal(err)
		}
		if p := r.Validate(); !p.OK() {
			t.Fatalf("family %d:\n%s", major, p.Error())
		}
		owner, reader := r.LayerKinds[0].Nodes[1].Caches[1], r.LayerKinds[1].Nodes[0].Caches[1]
		for _, dt := range []string{"fp8_ds_mla", "nvfp4_ds_mla", "auto"} {
			if owner.EntryBytes(dt) != reader.EntryBytes(dt) {
				t.Errorf("family %d %s: owner stores %d B, reader reads %d", major, dt, owner.EntryBytes(dt), reader.EntryBytes(dt))
			}
		}
	}
}
