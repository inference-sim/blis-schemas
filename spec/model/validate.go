package model

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/internal/validate"
)

// Validate performs FIELD-LEVEL validation: that the document is structurally a
// model graph, that every enumerated value is recognized, and that each node
// carries exactly the parameters its op prices.
//
// It knows nothing about engine versions. Whether a speculative method name exists
// in some release, or whether a backend is still supported, is a version-scoped
// question and lives in a rules pack. Keeping the two apart is what lets this
// function stay unchanged while engines move.
func (g *Graph) Validate() *validate.Problems {
	p := &validate.Problems{}

	if g.Kind != "ModelGraph" {
		p.Field("kind", "must be %q, got %q", "ModelGraph", g.Kind)
	}
	if g.Name == "" {
		p.Field("name", "required")
	}
	if !g.Modality.Valid() {
		p.Field("modality", "%q is not a recognized modality", g.Modality)
	}
	g.validateDerivation(p)
	g.validateGlobal(p)
	g.validateLayerKinds(p)
	g.validateStack(p)
	g.validateSpeculator(p)

	for i, n := range g.Head {
		n.validate(p, fmt.Sprintf("head[%d]", i))
	}
	return p
}

func (g *Graph) validateDerivation(p *validate.Problems) {
	d := g.DerivedFrom
	if d.Format == "" {
		p.Field("derived_from.format", "required: a graph must say which dialect it was derived from")
	}
	if d.Path == "" {
		p.Field("derived_from.path", "required")
	}
	// A digest is what makes a derived artifact auditable against its source. A
	// graph without one can drift silently, so its absence is an error rather than
	// a warning.
	if len(d.SHA256) != 64 {
		p.Field("derived_from.sha256",
			"must be a 64-character hex digest of the source file, got %d characters",
			len(d.SHA256))
	}
	if d.DeriverVersion < 1 {
		p.Field("derived_from.deriver_version", "must be at least 1")
	}
}

func (g *Graph) validateGlobal(p *validate.Problems) {
	if g.Global.HiddenSize < 1 {
		p.Field("global.hidden_size", "must be positive")
	}
	if g.Global.VocabSize < 1 {
		p.Field("global.vocab_size", "must be positive")
	}
	if !g.Global.WeightDType.Valid() {
		p.Field("global.weight_dtype", "%q is not a recognized dtype", g.Global.WeightDType)
	}
}

func (g *Graph) validateLayerKinds(p *validate.Problems) {
	if len(g.LayerKinds) == 0 {
		p.Field("layer_kinds", "at least one layer kind is required")
		return
	}
	seen := map[string]bool{}
	for i, lk := range g.LayerKinds {
		at := fmt.Sprintf("layer_kinds[%d]", i)
		if lk.ID == "" {
			p.Field(at+".id", "required")
		} else if seen[lk.ID] {
			p.Field(at+".id", "duplicate layer-kind id %q", lk.ID)
		}
		seen[lk.ID] = true

		if len(lk.Nodes) == 0 {
			p.Field(at+".nodes", "a layer kind with no nodes prices nothing")
		}
		for j, n := range lk.Nodes {
			n.validate(p, fmt.Sprintf("%s.nodes[%d]", at, j))
		}
		validateEdges(p, at, len(lk.Nodes), lk.Edges)
	}
}

// validateEdges checks that every edge indexes a real node, that no node is its own
// predecessor, and that the graph is acyclic. A cycle would make a layer's cost
// undefined, so it is rejected rather than traversed defensively at run time.
func validateEdges(p *validate.Problems, at string, n int, edges [][2]int) {
	adj := make([][]int, n)
	for k, e := range edges {
		from, to := e[0], e[1]
		if from < 0 || from >= n {
			p.Field(fmt.Sprintf("%s.edges[%d]", at, k),
				"from-index %d is out of range for %d nodes", from, n)
			continue
		}
		if to < 0 || to >= n {
			p.Field(fmt.Sprintf("%s.edges[%d]", at, k),
				"to-index %d is out of range for %d nodes", to, n)
			continue
		}
		if from == to {
			p.Field(fmt.Sprintf("%s.edges[%d]", at, k), "self-edge on node %d", from)
			continue
		}
		adj[from] = append(adj[from], to)
	}
	if cyc := findCycle(adj); cyc >= 0 {
		p.Field(at+".edges", "graph is cyclic; a cycle reaches node %d", cyc)
	}
}

// findCycle returns a node on a cycle, or -1. Iterative depth-first search with
// three colours; the graph is small enough that clarity beats cleverness.
func findCycle(adj [][]int) int {
	const (
		white = 0
		grey  = 1
		black = 2
	)
	colour := make([]int, len(adj))
	for start := range adj {
		if colour[start] != white {
			continue
		}
		stack := []int{start}
		for len(stack) > 0 {
			v := stack[len(stack)-1]
			switch colour[v] {
			case white:
				colour[v] = grey
				for _, w := range adj[v] {
					if colour[w] == grey {
						return w
					}
					if colour[w] == white {
						stack = append(stack, w)
					}
				}
			case grey:
				colour[v] = black
				stack = stack[:len(stack)-1]
			default:
				stack = stack[:len(stack)-1]
			}
		}
	}
	return -1
}

func (g *Graph) validateStack(p *validate.Problems) {
	s := g.Stack
	// A stack must produce at least one layer. Either a repeated pattern or a
	// literal run satisfies that; a model with no periodicity uses the run alone.
	if s.Layers() < 1 {
		p.Field("stack",
			"expands to no layers: state a pattern with a repeat, or a literal prologue")
	}
	if s.Repeat < 0 {
		p.Field("stack.repeat", "must not be negative")
	}
	if len(s.Pattern) > 0 && s.Repeat < 1 {
		p.Field("stack.repeat", "a pattern is stated but never repeated")
	}
	if len(s.Pattern) == 0 && s.Repeat > 0 {
		p.Field("stack.pattern", "a repeat is stated but there is no pattern to repeat")
	}
	ids := map[string]bool{}
	for _, lk := range g.LayerKinds {
		ids[lk.ID] = true
	}
	for field, seq := range map[string][]string{
		"prologue": s.Prologue, "pattern": s.Pattern, "epilogue": s.Epilogue,
	} {
		for i, id := range seq {
			if !ids[id] {
				p.Field(fmt.Sprintf("stack.%s[%d]", field, i),
					"%q names no layer kind", id)
			}
		}
	}
	// Every declared kind should appear, or it prices nothing and is more likely a
	// leftover than an intention.
	used := map[string]bool{}
	for _, id := range s.Expand() {
		used[id] = true
	}
	for _, lk := range g.LayerKinds {
		if !used[lk.ID] {
			p.Warnf("layer kind %q is declared but never used by the stack", lk.ID)
		}
	}
}

func (g *Graph) validateSpeculator(p *validate.Problems) {
	s := g.Speculator
	if s == nil {
		return
	}
	if s.Method == "" {
		p.Field("speculator.method", "required")
	}
	if s.NumSpec < 1 {
		p.Field("speculator.num_spec_tokens",
			"must be at least 1; omit the speculator block instead of declaring zero drafts")
	}
	if len(s.Stack.Pattern) == 0 {
		p.Field("speculator.stack.pattern",
			"required: a draft module is its own stack, because for an MoE target the draft pass is a second MoE")
	}
	if s.Stack.Repeat < 1 {
		p.Field("speculator.stack.repeat", "must be at least 1")
	}
}

// validate checks one node: that its op is known, that the parameters its op prices
// are present and positive, and that it carries no parameter belonging to a
// different op. The last check is what keeps the union-of-fields Node honest.
func (n Node) validate(p *validate.Problems, at string) {
	if !n.Op.Valid() {
		p.Field(at+".op", "%q is not a recognized primitive", n.Op)
		return
	}
	if !n.Emit.Valid() {
		p.Field(at+".emit", "%q is not a recognized emit condition", n.Emit)
	}
	// A collective's existence depends on the layout, so it must say under which
	// layouts it exists. An unconditional collective would be emitted even at tp=1.
	if n.Op.Collective() && n.Emit == EmitAlways {
		p.Field(at+".emit",
			"a collective node must name an emit condition; an unconditional %s would be emitted at tp=1", n.Op)
	}
	if !n.Op.Collective() && n.Emit != EmitAlways {
		// Allowed, but unusual enough to flag: a non-collective conditional node
		// usually means the graph is encoding a deployment rather than a model.
		p.Warnf("%s: a condition on a non-collective node encodes deployment in the model graph", at)
	}

	// A per-node weight dtype is only meaningful where the node holds parameters, and an
	// unrecognized one must be an error rather than silently falling back to the global
	// width: the whole point of the override is that the two differ.
	if n.WeightDType != "" {
		switch {
		case !n.WeightDType.Valid():
			p.Field(at+".weight_dtype", "%q is not a recognized dtype", n.WeightDType)
		case n.Op != OpGEMM && n.Op != OpGroupedGEMM:
			p.Field(at+".weight_dtype", "only a GEMM or GroupedGEMM holds parameters")
		}
	}

	switch n.Op {
	case OpGEMM:
		requirePositive(p, at, "n", n.N)
		requirePositive(p, at, "k", n.K)
		forbid(p, at, map[string]int{
			"experts": n.Experts, "top_k": n.TopK, "n_q": n.NumQHeads,
			"n_kv": n.NumKVHeads, "state_size": n.StateSize,
		})
	case OpGroupedGEMM:
		requirePositive(p, at, "n", n.N)
		requirePositive(p, at, "k", n.K)
		requirePositive(p, at, "experts", n.Experts)
		requirePositive(p, at, "top_k", n.TopK)
		if n.TopK > n.Experts {
			p.Field(at+".top_k", "top_k %d exceeds experts %d", n.TopK, n.Experts)
		}
		if n.SharedExperts < 0 {
			p.Field(at+".shared_experts", "must not be negative")
		}
		if n.SharedIntermediateSize < 0 {
			p.Field(at+".shared_intermediate_size", "must not be negative")
		}
		if n.SharedIntermediateSize > 0 && n.SharedExperts == 0 {
			p.Field(at+".shared_intermediate_size",
				"stated without shared_experts, so it sizes nothing")
		}
		if n.LatentSize < 0 {
			p.Field(at+".latent_size", "must not be negative")
		}
		forbid(p, at, map[string]int{"n_q": n.NumQHeads, "state_size": n.StateSize})
	case OpAttention:
		if !n.AttentionKind.Valid() {
			p.Field(at+".kind", "%q is not a recognized attention kind", n.AttentionKind)
		}
		requirePositive(p, at, "n_q", n.NumQHeads)
		requirePositive(p, at, "n_kv", n.NumKVHeads)
		requirePositive(p, at, "d_h", n.HeadDim)
		if n.NumKVHeads > n.NumQHeads {
			p.Field(at+".n_kv", "n_kv %d exceeds n_q %d", n.NumKVHeads, n.NumQHeads)
		}
		if n.AttentionKind.LatentKV() && n.NumKVHeads != 1 {
			// A latent cache is one vector per token: an engine pins the head count
			// to one, so any other value describes no real deployment.
			p.Field(at+".n_kv",
				"%s stores one latent vector per token, so n_kv must be 1, got %d",
				n.AttentionKind, n.NumKVHeads)
		}
		if n.AttentionKind == AttentionSWA && n.Window < 1 {
			p.Field(at+".window", "sliding-window attention requires a positive window")
		}
		if n.AttentionKind == AttentionSparseMLA && n.IndexTopK < 1 &&
			n.CompressRatio < 1 {
			// A sparse latent read is bounded either by a top-k selection or by a
			// compressed stream. DeepSeek-V4's ratio-128 layers use the second with no
			// top-k at all, so requiring index_topk alone would reject a real layer.
			p.Field(at+".index_topk",
				"sparse MLA requires a positive index_topk or compress_ratio")
		}
		if n.CompressRatio < 0 {
			p.Field(at+".compress_ratio", "must not be negative")
		}
		if n.CompressRatio > 0 && !n.AttentionKind.LatentKV() {
			p.Field(at+".compress_ratio",
				"stated on %s, which reads no compressed latent stream",
				n.AttentionKind)
		}
		forbid(p, at, map[string]int{"experts": n.Experts, "state_size": n.StateSize})
	case OpRecurrentUpdate:
		if !n.RecurrentKind.Valid() {
			p.Field(at+".recurrent_kind", "%q is not a recognized recurrent kind", n.RecurrentKind)
		}
		requirePositive(p, at, "n_heads", n.NumHeads)
		requirePositive(p, at, "state_size", n.StateSize)
		if n.RecurrentKind == RecurrentMamba2 {
			requirePositive(p, at, "n_groups", n.NumGroups)
			requirePositive(p, at, "conv_kernel", n.ConvKernel)
			requirePositive(p, at, "intermediate_size", n.IntermediateSize)
		}
		if n.StateDType != "" && !n.StateDType.Valid() {
			p.Field(at+".state_dtype", "%q is not a recognized dtype", n.StateDType)
		}
		forbid(p, at, map[string]int{"experts": n.Experts, "n_q": n.NumQHeads})
	case OpElementwise:
		// Bytes per token may be derived from hidden size, so it is optional; a
		// negative value is still a mistake.
		if n.BytesPerToken < 0 {
			p.Field(at+".bytes_per_token", "must not be negative")
		}
	case OpAllReduce, OpAllGather, OpReduceScatter, OpAll2All:
		// A collective's byte volume follows from the activation shape and the
		// layout, so it carries no shape of its own. Anything else is a mistake.
		forbid(p, at, map[string]int{
			"n": n.N, "k": n.K, "experts": n.Experts, "n_q": n.NumQHeads,
			"n_kv": n.NumKVHeads, "state_size": n.StateSize,
		})
	}
}

func requirePositive(p *validate.Problems, at, field string, v int) {
	if v < 1 {
		p.Field(at+"."+field, "must be positive for this op, got %d", v)
	}
}

// forbid reports any parameter set on a node whose op does not price it. Silence
// would let a typo — a head count on a GEMM — survive review looking deliberate.
func forbid(p *validate.Problems, at string, fields map[string]int) {
	for name, v := range fields {
		if v != 0 {
			p.Field(at+"."+name, "not priced by this op; remove it")
		}
	}
}
