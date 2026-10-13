package model

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-schemas/internal/validate"
)

// validateAliases resolves every aliases reference to exactly one weight-holding node of
// the same shape. An alias that resolved to nothing, to several nodes, or to a node that
// is itself an alias would leave the residency it claims is counted elsewhere counted
// nowhere, or twice.
func (g *Graph) validateAliases(p *validate.Problems) {
	kinds := map[string]int{}
	for i, lk := range g.LayerKinds {
		kinds[lk.ID] = i
	}
	// A layer kind instantiated more than once holds one copy of its weights per
	// instance, so "the node in that kind" names no single weight.
	instances := map[string]int{}
	stacks := []Stack{g.Stack}
	if s := g.Speculator; s != nil {
		stacks = append(stacks, s.Stack)
		if s.ContextStack != nil {
			stacks = append(stacks, *s.ContextStack)
		}
	}
	for _, st := range stacks {
		for _, id := range st.Expand() {
			instances[id]++
		}
	}
	g.EachNode(func(at string, n Node) {
		if n.Aliases == "" {
			return
		}
		// Split at the last dot: a layer-kind id may contain one, a role does not need to.
		i := strings.LastIndex(n.Aliases, ".")
		scope, role, ok := "", "", i > 0
		if ok {
			scope, role = n.Aliases[:i], n.Aliases[i+1:]
		}
		if !ok || scope == "" || role == "" {
			p.Field(at+".aliases", "%q must be <layer kind>.<role>, head.<role> or embedding.<role>", n.Aliases)
			return
		}
		var pool []Node
		var poolAt string
		li, isKind := kinds[scope]
		switch {
		case isKind && (scope == "head" || scope == "embedding"):
			p.Field(at+".aliases", "%q is ambiguous: a layer kind is also named %q", n.Aliases, scope)
			return
		case scope == "head":
			pool, poolAt = g.Head, "head"
		case scope == "embedding":
			pool, poolAt = g.Embedding, "embedding"
		case isKind:
			if c := instances[scope]; c != 1 {
				p.Field(at+".aliases", "layer kind %q runs %d times, so it holds no single weight to alias", scope, c)
				return
			}
			pool, poolAt = g.LayerKinds[li].Nodes, fmt.Sprintf("layer_kinds[%d].nodes", li)
		default:
			p.Field(at+".aliases", "%q names no layer kind, and is neither head nor embedding", scope)
			return
		}
		var hits []int
		for i, m := range pool {
			if m.Role == role {
				hits = append(hits, i)
			}
		}
		if len(hits) != 1 {
			p.Field(at+".aliases", "%q matches %d nodes; it must match exactly one", n.Aliases, len(hits))
			return
		}
		target, targetAt := pool[hits[0]], fmt.Sprintf("%s[%d]", poolAt, hits[0])
		switch {
		case targetAt == at:
			p.Field(at+".aliases", "a node cannot alias itself")
		case target.Aliases != "":
			p.Field(at+".aliases", "%s is itself an alias; name the node that holds the weights", targetAt)
		case !sameWeights(n, target):
			p.Field(at+".aliases", "%s holds a %s; it cannot be the weights of this %s", targetAt, weightShape(target), weightShape(n))
		}
	})
}

// sameWeights reports whether two nodes can read one weight tensor: two GEMMs of equal n
// and k, two tables of equal rows and row width, or a GEMM over a table's rows, which is
// a tied LM head reading its embedding. The last compares the vocabulary only: a row's
// bytes are k times an element width the graph does not state for a table.
func sameWeights(a, b Node) bool {
	switch {
	case a.Op == OpGEMM && b.Op == OpGEMM:
		return a.N == b.N && a.K == b.K
	case a.Op == OpGather && b.Op == OpGather:
		return a.Rows == b.Rows && a.RowBytes == b.RowBytes
	case a.Op == OpGEMM && b.Op == OpGather:
		return int64(a.N) == b.Rows
	case a.Op == OpGather && b.Op == OpGEMM:
		return a.Rows == int64(b.N)
	}
	return false
}

func weightShape(n Node) string {
	switch n.Op {
	case OpGEMM:
		return fmt.Sprintf("GEMM of %d x %d", n.N, n.K)
	case OpGather:
		return fmt.Sprintf("table of %d rows of %d B", n.Rows, n.RowBytes)
	}
	return string(n.Op) + " node"
}

// validateArchOverrides checks each override is well formed and that every role and
// cache it names exists, by applying it. An override naming nothing would read as a
// statement about the served model that changes nothing.
func (g *Graph) validateArchOverrides(p *validate.Problems) {
	seen := map[int]bool{}
	for i, o := range g.ArchOverrides {
		at := fmt.Sprintf("arch_overrides[%d]", i)
		if o.CapabilityFamily < 1 {
			p.Field(at+".capability_family", "must be a positive compute-capability major version, got %d", o.CapabilityFamily)
		} else if seen[o.CapabilityFamily] {
			p.Field(at+".capability_family", "family %d is overridden twice", o.CapabilityFamily)
		}
		seen[o.CapabilityFamily] = true
		if len(o.Nodes) == 0 && len(o.Caches) == 0 {
			p.Field(at, "overrides nothing; remove it")
		}
		for j, no := range o.Nodes {
			nat := fmt.Sprintf("%s.nodes[%d]", at, j)
			if no.Role == "" {
				p.Field(nat+".role", "required")
			}
			if no.WeightDType == "" && no.WeightScale == nil {
				p.Field(nat, "restates nothing; give a weight_dtype or a weight_scale")
			}
			if no.WeightDType != "" && !no.WeightDType.Valid() {
				p.Field(nat+".weight_dtype", "%q is not a recognized dtype", no.WeightDType)
			}
			validateWeightScale(p, nat+".weight_scale", no.WeightScale)
			g.EachNode(func(nodeAt string, n Node) {
				if n.Role == no.Role && n.Op != OpGEMM && n.Op != OpGroupedGEMM {
					p.Field(nat+".role", "%q matches %s, a %s, which holds no parameters", no.Role, nodeAt, n.Op)
				}
			})
		}
		for j, co := range o.Caches {
			cat := fmt.Sprintf("%s.caches[%d]", at, j)
			if co.Name == "" {
				p.Field(cat+".name", "required")
			}
			if co.BytesPerEntry < 0 || co.PageAlignment < 0 {
				p.Field(cat, "sizes must not be negative")
			}
			if co.BytesPerEntry == 0 && co.PageAlignment == 0 && len(co.BytesByCacheDType) == 0 {
				p.Field(cat, "restates nothing; give a size")
			}
			for k, v := range co.BytesByCacheDType {
				if v < 1 {
					p.Field(cat+".bytes_by_cache_dtype."+k, "must be positive, got %d", v)
				}
			}
		}
		if o.CapabilityFamily >= 1 {
			if _, err := g.ForCapability(o.CapabilityFamily); err != nil {
				p.Field(at, "%v", err)
			}
		}
	}
}
