package rules

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-schemas/spec/model"
)

// KVCachePacking is how a release groups a model's stated caches into KV cache groups
// that share one block pool, as data. vLLM draws every group's blocks from one pool, and
// every block costs the widest group's page bytes (vllm/v1/core/kv_cache_utils.py
// _get_kv_cache_bytes_per_block; "Groups alias from byte 0"), so a model whose caches
// have different page sizes pays for padding in every block. Pack.KVCacheGroups applies
// it.
type KVCachePacking struct {
	// Modeled is whether the pack states the grouping at all. A release whose grouping
	// is not transcribed is refused rather than guessed.
	Modeled bool
	// StateCacheBuckets is whether a kind: state cache is its own bucket, a
	// CircularBufferSpec the packer may split down to one state per group to fit the
	// block the other buckets need (v0.31.0). v0.29.0 stores DeepSeek-V4's compressor
	// state as a SlidingWindowMLASpec whose window the graph does not state, so a state
	// cache cannot be placed there.
	StateCacheBuckets bool
	// LoneWindowIsState is whether a sliding-window bucket splits like a state bucket
	// when no balanced mixed-page bucket fixes a repeat pattern (v0.31.0 is_state_bucket).
	LoneWindowIsState bool
	// BlockStrideAlignment is whether the block's bytes round up to the least common
	// multiple of every layer's block_stride_alignment (v0.31.0).
	BlockStrideAlignment bool
}

// String renders the packing for the generated documentation.
func (k KVCachePacking) String() string {
	if !k.Modeled {
		return "not modeled"
	}
	parts := []string{"packed groups share one block pool"}
	if k.StateCacheBuckets {
		parts = append(parts, "state caches split to fit the block")
	}
	if k.LoneWindowIsState {
		parts = append(parts, "a lone window bucket splits like state")
	}
	if k.BlockStrideAlignment {
		parts = append(parts, "the block rounds to the layers' stride alignment")
	}
	return strings.Join(parts, "; ")
}

// KVCacheLayer is one engine KV cache layer: one owned cache of one layer of the graph.
// vLLM registers each such cache as its own layer with its own spec.
type KVCacheLayer struct {
	Name  string
	Cache model.Cache
}

// OwnedKVCacheLayers lists every owned cache of a graph, in the order the engine meets
// them: the target's stack, layer by layer, then the speculator's, each layer's caches in
// the order its nodes state them. They are named layers.<i>.<cache> and
// speculator.layers.<i>.<cache>. A cache a layer reads but does not own is not a layer.
func OwnedKVCacheLayers(g *model.Graph) []KVCacheLayer {
	kinds := map[string]*model.LayerKind{}
	for i := range g.LayerKinds {
		kinds[g.LayerKinds[i].ID] = &g.LayerKinds[i]
	}
	var out []KVCacheLayer
	walk := func(prefix string, s model.Stack) {
		for i, id := range s.Expand() {
			lk := kinds[id]
			if lk == nil {
				continue
			}
			for _, n := range lk.Nodes {
				for _, c := range n.Caches {
					if c.Owned {
						out = append(out, KVCacheLayer{Name: fmt.Sprintf("%slayers.%d.%s", prefix, i, c.Name), Cache: c})
					}
				}
			}
		}
	}
	walk("", g.Stack)
	if g.Speculator != nil {
		walk("speculator.", g.Speculator.Stack)
	}
	return out
}

// KVCacheGroupsInput is what the grouping reads besides the caches.
type KVCacheGroupsInput struct {
	// Layers are the owned caches in engine order (OwnedKVCacheLayers). The order decides
	// which layers share a group, not the bytes per block.
	Layers []KVCacheLayer
	// EngineBlockSize is the deployment's block_size, for a cache that states none.
	EngineBlockSize int
	// CacheDType is the cache dtype that runs, which picks each entry's record.
	CacheDType string
	// BlockStrideAlignment is the least common multiple of the block_stride_alignment the
	// engine's backends state for these layers, which the graph does not carry: on
	// DeepSeek-V4.1 the indexer's lcm(512, page alignment) when indexer_sparse_logits is
	// on (vllm/models/deepseek_v41/attention.py). Zero means none.
	BlockStrideAlignment int
}

// KVCacheGroup is one KV cache group: layers sharing a block table.
type KVCacheGroup struct {
	Layers []string
	// BlockSize is the group's tokens per block.
	BlockSize int
	// PageBytes is the bytes its layers occupy in one block, the sum of their pages.
	PageBytes int
}

// KVCacheGrouping is a model's KV cache groups and the bytes each pool block costs.
type KVCacheGrouping struct {
	// Packed is whether the mixed-page-size packer formed the groups; otherwise every
	// layer shares one group.
	Packed bool
	Groups []KVCacheGroup
	// BytesPerBlock is what one block of the shared pool costs: the widest group's page
	// bytes, rounded to the block-stride alignment. Every group's block costs this much,
	// whatever its own layers occupy.
	BytesPerBlock int
	// PageBytes is each layer's own page size, by layer name.
	PageBytes map[string]int
}

// GroupOf returns the index of the group holding the named layer, or -1.
func (g KVCacheGrouping) GroupOf(layer string) int {
	for i, gr := range g.Groups {
		for _, n := range gr.Layers {
			if n == layer {
				return i
			}
		}
	}
	return -1
}

// KVCacheGroups groups stated caches the way this release's engine does, and returns the
// bytes one block of the shared pool costs. A consumer sizes a sequence in whole blocks of
// BytesPerBlock in each group, as the engine does, rather than each layer at its own page.
//
// It follows vLLM's get_kv_cache_groups for the paths stated caches take. Every layer of
// one uniform type is one group (_get_kv_cache_groups_uniform_type). Layers with more than
// one page size are packed by _get_packed_kv_cache_groups (v0.31.0
// vllm/v1/core/kv_cache_utils.py:2001-2120): bucketed by uniform type, balanced buckets
// split into groups of the same number of pattern repeats, chosen by _approximate_gcd to
// minimise padding, and state buckets split to fit the block the others need. The block
// is the widest group (_get_kv_cache_bytes_per_block, :1606-1632, called at :1768).
//
// A cache's uniform type follows the spec the engine gives it: kv and index caches are
// full-attention specs at their block size, window caches sliding-window specs keyed by
// their window, and state caches circular buffers of Entries rows. The packer assumes a
// block-outermost layout, which is the only one that expresses mixed pages, and the hybrid
// KV cache manager on.
//
// It refuses what it does not model rather than guess: a release whose grouping is not
// transcribed, several uniform types sharing one page size (the engine's
// _get_kv_cache_groups_uniform_page_size), a state cache where the release stores it
// as a spec the graph cannot state, and a page of zero bytes.
func (p *Pack) KVCacheGroups(in KVCacheGroupsInput) (KVCacheGrouping, error) {
	k := p.KVCachePacking
	if !k.Modeled {
		return KVCacheGrouping{}, fmt.Errorf("engine %s: KV cache grouping is not transcribed in its rules pack", p.Version)
	}
	if len(in.Layers) == 0 {
		return KVCacheGrouping{}, fmt.Errorf("no owned caches to group")
	}
	layers := make([]packLayer, 0, len(in.Layers))
	pages := map[string]int{}
	for _, l := range in.Layers {
		pl, err := p.packLayerOf(l, in)
		if err != nil {
			return KVCacheGrouping{}, err
		}
		if _, dup := pages[l.Name]; dup {
			return KVCacheGrouping{}, fmt.Errorf("layer %q is stated twice", l.Name)
		}
		pages[l.Name] = pl.page
		layers = append(layers, pl)
	}

	var groups []KVCacheGroup
	packed := false
	switch {
	case sameType(layers):
		g := KVCacheGroup{BlockSize: layers[0].block}
		for _, l := range layers {
			g.Layers = append(g.Layers, l.name)
			g.PageBytes += l.page
		}
		groups = []KVCacheGroup{g}
	case distinctPages(layers) > 1:
		var err error
		if groups, err = k.pack(layers); err != nil {
			return KVCacheGrouping{}, err
		}
		packed = true
	default:
		return KVCacheGrouping{}, fmt.Errorf("the caches are of several uniform types with one page size; " +
			"the engine groups them by _get_kv_cache_groups_uniform_page_size, which is not modeled")
	}

	bytes := 0
	for _, g := range groups {
		if g.PageBytes > bytes {
			bytes = g.PageBytes
		}
	}
	if k.BlockStrideAlignment && in.BlockStrideAlignment > 1 {
		bytes = roundUp(bytes, in.BlockStrideAlignment)
	}
	return KVCacheGrouping{Packed: packed, Groups: groups, BytesPerBlock: bytes, PageBytes: pages}, nil
}

// packLayer is one layer as the packer sees it: its uniform-type key and page.
type packLayer struct {
	name  string
	key   string // equal keys are one uniform type
	state bool   // a circular-buffer state cache
	win   bool   // a sliding-window cache
	block int
	page  int
}

func (p *Pack) packLayerOf(l KVCacheLayer, in KVCacheGroupsInput) (packLayer, error) {
	c := l.Cache
	pl := packLayer{name: l.Name}
	entry := c.EntryBytes(in.CacheDType)
	switch c.Kind {
	case model.CacheKV, model.CacheIndex, model.CacheWindow:
		pl.block = c.BlockSize
		if pl.block < 1 {
			pl.block = in.EngineBlockSize
		}
		if pl.block < 1 {
			return pl, fmt.Errorf("layer %q states no block size and the engine block size is unset", l.Name)
		}
		// AttentionSpec: num_states = block_size // tokens_per_state, padded to the
		// alignment (_apply_alignment_padding).
		pl.page = pl.block / c.EntryTokens() * entry
		if c.Kind == model.CacheWindow {
			pl.win = true
			pl.key = fmt.Sprintf("window:%d:%d", c.Window, pl.block)
		} else {
			pl.key = fmt.Sprintf("full:%d", pl.block)
		}
	case model.CacheState:
		if !p.KVCachePacking.StateCacheBuckets {
			return pl, fmt.Errorf("layer %q: engine %s stores a state cache as a spec the graph does not state", l.Name, p.Version)
		}
		// CircularBufferSpec: block_size is the ring's row count, one row per state.
		pl.block, pl.page, pl.state = c.Entries, c.Entries*entry, true
		pl.key = fmt.Sprintf("state:%d", pl.block)
	default:
		return pl, fmt.Errorf("layer %q: cache kind %q has no engine spec", l.Name, c.Kind)
	}
	if c.PageAlignment > 1 {
		pl.page = roundUp(pl.page, c.PageAlignment)
	}
	if pl.page < 1 {
		return pl, fmt.Errorf("layer %q has a page of %d bytes", l.Name, pl.page)
	}
	return pl, nil
}

// bucket is one uniform-type bucket and its layers by page size, in first-seen order.
type bucket struct {
	layers   []packLayer
	pages    []int            // distinct page sizes, first-seen order
	byPage   map[int][]string // layer names per page size
	balanced bool
}

func (b *bucket) maxLayersPerPage() int {
	m := 0
	for _, names := range b.byPage {
		if len(names) > m {
			m = len(names)
		}
	}
	return m
}

func (b *bucket) widest(n int) int {
	total := 0
	for _, pg := range b.pages {
		total += cdiv(len(b.byPage[pg]), n) * pg
	}
	return total
}

// pack is _get_packed_kv_cache_groups.
func (k KVCachePacking) pack(layers []packLayer) ([]KVCacheGroup, error) {
	var buckets []*bucket
	for _, l := range layers {
		var into *bucket
		for _, b := range buckets {
			if b.layers[0].key == l.key {
				into = b
				break
			}
		}
		if into == nil {
			into = &bucket{byPage: map[int][]string{}}
			buckets = append(buckets, into)
		}
		into.layers = append(into.layers, l)
		if _, seen := into.byPage[l.page]; !seen {
			into.pages = append(into.pages, l.page)
		}
		into.byPage[l.page] = append(into.byPage[l.page], l.name)
	}
	for _, b := range buckets {
		counts := map[int]bool{}
		for _, names := range b.byPage {
			counts[len(names)] = true
		}
		b.balanced = len(counts) == 1
	}

	minRepeats := 0
	var balancedRepeats []int
	for _, b := range buckets {
		if !b.balanced {
			continue
		}
		balancedRepeats = append(balancedRepeats, b.maxLayersPerPage())
		if len(b.pages) > 1 && b.maxLayersPerPage() > minRepeats {
			minRepeats = b.maxLayersPerPage()
		}
	}
	repeats := 0 // zero: none
	if minRepeats > 0 {
		repeats = approximateGCD(balancedRepeats, minRepeats)
	}
	numGroupsFor := func(b *bucket) int {
		if b.balanced && repeats > 0 {
			return cdiv(b.maxLayersPerPage(), repeats)
		}
		return 1
	}
	isState := func(b *bucket) bool {
		first := b.layers[0]
		return first.state || (k.LoneWindowIsState && repeats == 0 && first.win)
	}
	anchor := 0
	for _, b := range buckets {
		n := numGroupsFor(b)
		if isState(b) {
			n = len(b.layers)
		}
		if w := b.widest(n); w > anchor {
			anchor = w
		}
	}

	var groups []KVCacheGroup
	for _, b := range buckets {
		n := numGroupsFor(b)
		if anchor > 0 && isState(b) {
			perBlock := anchor / b.layers[0].page
			if perBlock < 1 {
				perBlock = 1
			}
			if c := cdiv(len(b.layers), perBlock); c > n {
				n = c
			}
		}
		block := b.layers[0].block
		pageOf := map[string]int{}
		for _, l := range b.layers {
			pageOf[l.name] = l.page
		}
		if n == 1 {
			g := KVCacheGroup{BlockSize: block}
			for _, l := range b.layers {
				g.Layers = append(g.Layers, l.name)
				g.PageBytes += l.page
			}
			groups = append(groups, g)
			continue
		}
		// pattern_repeats = zip(*page_size_layers.values()), which stops at the shortest
		// list. The engine would leave an unbalanced bucket's surplus layers in no group;
		// that is refused rather than reproduced.
		if !b.balanced {
			return nil, fmt.Errorf("an unbalanced bucket of %d layers would be split %d ways, "+
				"which leaves layers in no group", len(b.layers), n)
		}
		patternRepeats := len(b.byPage[b.pages[0]])
		for i := 0; i < n; i++ {
			g := KVCacheGroup{BlockSize: block}
			for r := i; r < patternRepeats; r += n {
				for _, pg := range b.pages {
					name := b.byPage[pg][r]
					g.Layers = append(g.Layers, name)
					g.PageBytes += pageOf[name]
				}
			}
			groups = append(groups, g)
		}
	}
	return groups, nil
}

// approximateGCD is _approximate_gcd: the chunk d in [lower, max(values)] minimising the
// total upward padding sum((d - x%d) % d), ties to the larger d.
func approximateGCD(values []int, lower int) int {
	minD := lower
	if minD < 1 {
		minD = 1
	}
	maxD := 0
	for _, v := range values {
		if v > maxD {
			maxD = v
		}
	}
	if minD > maxD {
		return minD
	}
	best, bestPad := minD, -1
	for d := minD; d <= maxD; d++ {
		pad := 0
		for _, x := range values {
			pad += (d - x%d) % d
		}
		if bestPad < 0 || pad < bestPad || (pad == bestPad && d > best) {
			best, bestPad = d, pad
		}
	}
	return best
}

func sameType(layers []packLayer) bool {
	for _, l := range layers[1:] {
		if l.key != layers[0].key {
			return false
		}
	}
	return true
}

func distinctPages(layers []packLayer) int {
	seen := map[int]bool{}
	for _, l := range layers {
		seen[l.page] = true
	}
	return len(seen)
}

func cdiv(a, b int) int { return (a + b - 1) / b }

func roundUp(x, m int) int { return cdiv(x, m) * m }
