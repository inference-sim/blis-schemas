package rules_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	blisschemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/rules"
	v0_29 "github.com/inference-sim/blis-schemas/rules/v0_29"
	v0_31 "github.com/inference-sim/blis-schemas/rules/v0_31"
	"github.com/inference-sim/blis-schemas/spec/model"
)

const kvDir = "testdata/kvgroups"

// loadLayers reads a *.layers.json file, the form golden.py reads too.
func loadLayers(t *testing.T, name string) []rules.KVCacheLayer {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(kvDir, name+".layers.json"))
	if err != nil {
		t.Fatal(err)
	}
	var raw []struct {
		Name                                      string
		Kind                                      string
		Bytes, TPE, Window, Entries, Block, Align int
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatal(err)
	}
	out := make([]rules.KVCacheLayer, len(raw))
	for i, r := range raw {
		out[i] = rules.KVCacheLayer{Name: r.Name, Cache: model.Cache{
			Name: r.Name, Kind: model.CacheKind(r.Kind), Owned: true, BytesPerEntry: r.Bytes,
			TokensPerEntry: r.TPE, Window: r.Window, Entries: r.Entries, BlockSize: r.Block,
			PageAlignment: r.Align}}
	}
	return out
}

type golden struct {
	Bytes  int     `json:"bytes"`
	Groups [][]any `json:"groups"`
}

// TestKVCacheGroupsMatchEngineGoldens compares the grouping with what vLLM's own
// _get_packed_kv_cache_groups and _get_kv_cache_bytes_per_block return for the same layers
// (testdata/kvgroups/golden.py ran them), group by group and layer by layer.
func TestKVCacheGroupsMatchEngineGoldens(t *testing.T) {
	for _, c := range []struct {
		layers string
		pack   *rules.Pack
		block  int
		align  int
	}{
		{"synth-nostate", v0_29.Pack(), 64, 0},
		{"synth-nostate", v0_31.Pack(), 64, 0},
		{"synth-state", v0_31.Pack(), 64, 0},
		{"synth-state", v0_31.Pack(), 64, 1024},
		{"deepseek-v4.1-example", v0_31.Pack(), 64, 0},
		{"deepseek-v4.1-example", v0_31.Pack(), 256, 0},
	} {
		name := c.layers + "." + "v" + strings.ReplaceAll(c.pack.Version, ".", "_")
		name = strings.TrimSuffix(name, "_0") + ".block" + strconv.Itoa(c.block) + ".align" + strconv.Itoa(c.align)
		t.Run(name, func(t *testing.T) {
			b, err := os.ReadFile(filepath.Join(kvDir, name+".json"))
			if err != nil {
				t.Fatal(err)
			}
			var want golden
			if err := json.Unmarshal(b, &want); err != nil {
				t.Fatal(err)
			}
			got, err := c.pack.KVCacheGroups(rules.KVCacheGroupsInput{
				Layers: loadLayers(t, c.layers), EngineBlockSize: c.block,
				CacheDType: "fp8_ds_mla", BlockStrideAlignment: c.align})
			if err != nil {
				t.Fatal(err)
			}
			if !got.Packed || got.BytesPerBlock != want.Bytes || len(got.Groups) != len(want.Groups) {
				t.Fatalf("packed %v, %d B per block in %d groups; engine: %d B in %d groups",
					got.Packed, got.BytesPerBlock, len(got.Groups), want.Bytes, len(want.Groups))
			}
			for i, wg := range want.Groups {
				var names []string
				for _, n := range wg[2].([]any) {
					names = append(names, n.(string))
				}
				g := got.Groups[i]
				if g.BlockSize != int(wg[0].(float64)) || g.PageBytes != int(wg[1].(float64)) ||
					!reflect.DeepEqual(g.Layers, names) {
					t.Errorf("group %d = %+v, engine %v", i, g, wg)
				}
			}
		})
	}
}

// OwnedKVCacheLayers walks the target stack, then the speculator's, keeping owned caches.
func TestOwnedKVCacheLayersWalkTheStacks(t *testing.T) {
	g := &model.Graph{
		LayerKinds: []model.LayerKind{
			{ID: "src", Nodes: []model.Node{{Caches: []model.Cache{
				{Name: "kv", Kind: model.CacheKV, Owned: true, BytesPerEntry: 10},
				{Name: "ring", Kind: model.CacheState, Owned: true, BytesPerEntry: 4, Entries: 8}}}}},
			{ID: "reader", Nodes: []model.Node{{Caches: []model.Cache{
				{Name: "kv", Kind: model.CacheKV, Owned: false, BytesPerEntry: 10}}}}},
		},
		Stack:      model.Stack{Prologue: []string{"src"}, Pattern: []string{"reader"}, Repeat: 2},
		Speculator: &model.Speculator{Stack: model.Stack{Pattern: []string{"src"}, Repeat: 1}},
	}
	var names []string
	for _, l := range rules.OwnedKVCacheLayers(g) {
		names = append(names, l.Name)
	}
	want := []string{"layers.0.kv", "layers.0.ring", "speculator.layers.0.kv", "speculator.layers.0.ring"}
	if !reflect.DeepEqual(names, want) {
		t.Errorf("OwnedKVCacheLayers = %v, want %v", names, want)
	}
}

func kv(name string, block, bytes int) rules.KVCacheLayer {
	return rules.KVCacheLayer{Name: name, Cache: model.Cache{Name: name, Kind: model.CacheKV,
		Owned: true, BytesPerEntry: bytes, BlockSize: block}}
}

// One uniform type is one group, and the block is every layer's page.
func TestKVCacheGroupsUniformType(t *testing.T) {
	p := v0_31.Pack()
	got, err := p.KVCacheGroups(rules.KVCacheGroupsInput{
		Layers:          []rules.KVCacheLayer{kv("a", 0, 100), kv("b", 0, 300)},
		EngineBlockSize: 16, BlockStrideAlignment: 1000})
	if err != nil {
		t.Fatal(err)
	}
	if got.Packed || len(got.Groups) != 1 || got.Groups[0].PageBytes != 6400 || got.BytesPerBlock != 7000 {
		t.Errorf("got %+v", got)
	}
	if got.GroupOf("b") != 0 || got.GroupOf("missing") != -1 || got.PageBytes["a"] != 1600 {
		t.Errorf("lookups wrong: %+v", got)
	}
	// v0.29.0 does not round the block to a stride alignment.
	got, _ = v0_29.Pack().KVCacheGroups(rules.KVCacheGroupsInput{
		Layers:          []rules.KVCacheLayer{kv("a", 0, 100), kv("b", 0, 300)},
		EngineBlockSize: 16, BlockStrideAlignment: 1000})
	if got.BytesPerBlock != 6400 {
		t.Errorf("v0.29.0 BytesPerBlock = %d, want 6400", got.BytesPerBlock)
	}
}

// The grouping refuses what it does not model rather than return a plausible number.
func TestKVCacheGroupsRefuses(t *testing.T) {
	window := func(name string, w int) rules.KVCacheLayer {
		return rules.KVCacheLayer{Name: name, Cache: model.Cache{Name: name, Kind: model.CacheWindow,
			Owned: true, BytesPerEntry: 100, Window: w}}
	}
	ring := rules.KVCacheLayer{Name: "ring", Cache: model.Cache{Name: "ring", Kind: model.CacheState,
		Owned: true, BytesPerEntry: 4096, Entries: 8}}
	for _, c := range []struct {
		name     string
		pack     *rules.Pack
		in       rules.KVCacheGroupsInput
		errorHas string
	}{
		{"no layers", v0_31.Pack(), rules.KVCacheGroupsInput{EngineBlockSize: 16}, "no owned caches"},
		{"unmodeled release", &rules.Pack{Version: "9.9.9"},
			rules.KVCacheGroupsInput{Layers: []rules.KVCacheLayer{kv("a", 0, 1)}, EngineBlockSize: 16}, "not transcribed"},
		{"no block size", v0_31.Pack(),
			rules.KVCacheGroupsInput{Layers: []rules.KVCacheLayer{kv("a", 0, 1)}}, "no block size"},
		{"duplicate layer", v0_31.Pack(),
			rules.KVCacheGroupsInput{Layers: []rules.KVCacheLayer{kv("a", 0, 1), kv("a", 0, 1)}, EngineBlockSize: 16}, "stated twice"},
		{"several types, one page", v0_31.Pack(),
			rules.KVCacheGroupsInput{Layers: []rules.KVCacheLayer{kv("a", 0, 100), window("w", 128)}, EngineBlockSize: 16},
			"_get_kv_cache_groups_uniform_page_size"},
		{"state cache on v0.29.0", v0_29.Pack(),
			rules.KVCacheGroupsInput{Layers: []rules.KVCacheLayer{kv("a", 0, 100), ring}, EngineBlockSize: 16},
			"does not state"},
		{"page of zero bytes", v0_31.Pack(),
			rules.KVCacheGroupsInput{Layers: []rules.KVCacheLayer{{Name: "z", Cache: model.Cache{Kind: model.CacheKV,
				Owned: true, BytesPerEntry: 100, TokensPerEntry: 32}}}, EngineBlockSize: 16}, "page of 0 bytes"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := c.pack.KVCacheGroups(c.in)
			if err == nil || !strings.Contains(err.Error(), c.errorHas) {
				t.Fatalf("want an error containing %q, got %v", c.errorHas, err)
			}
		})
	}
}

// Packing mixed pages costs bytes that per-layer pages miss: the shared block is the
// widest group, so a group narrower than it pays padding in every block.
func TestKVCacheGroupsPaddingIsCharged(t *testing.T) {
	got, err := v0_31.Pack().KVCacheGroups(rules.KVCacheGroupsInput{
		Layers: loadLayers(t, "deepseek-v4.1-example"), EngineBlockSize: 64, CacheDType: "fp8_ds_mla"})
	if err != nil {
		t.Fatal(err)
	}
	narrow := 0
	for _, g := range got.Groups {
		if g.PageBytes < got.BytesPerBlock {
			narrow++
		}
	}
	if narrow == 0 {
		t.Errorf("every group fills the %d B block; the example mixes page sizes", got.BytesPerBlock)
	}
}

// The committed layers file is the example graph's own walk, so its goldens describe that
// graph.
func TestExampleLayersAreTheGraphs(t *testing.T) {
	g, err := blisschemas.LoadModelGraph(filepath.Join("..", "docs", "examples", "model-graph-deepseek-v4.1.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	got := rules.OwnedKVCacheLayers(g)
	want := loadLayers(t, "deepseek-v4.1-example")
	if len(got) != len(want) {
		t.Fatalf("the graph has %d owned caches, the layers file %d", len(got), len(want))
	}
	for i := range got {
		gc, wc := got[i].Cache, want[i].Cache
		if got[i].Name != want[i].Name || gc.Kind != wc.Kind || gc.BytesPerEntry != wc.BytesPerEntry ||
			gc.EntryTokens() != wc.EntryTokens() || gc.Window != wc.Window || gc.Entries != wc.Entries ||
			gc.BlockSize != wc.BlockSize || gc.PageAlignment != wc.PageAlignment {
			t.Errorf("layer %d: graph %+v, file %+v", i, got[i], want[i])
		}
	}
}
