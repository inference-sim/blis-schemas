// Package rules holds validation that depends on an engine version.
//
// The separation from field-level validation is the point of this package. A field
// check asks whether a document is structurally a scenario: are the counts
// positive, does the arithmetic close, is the enumerated value one this schema
// knows. Those questions have the same answer in every engine release, so the
// answers live with the types and change only when the schema does.
//
// A rule asks whether a document describes a deployment a PARTICULAR engine would
// run. Backend names appear and disappear. Defaults flip. A feature gains a fourth
// disqualifying condition. Every one of those is a fact about a release, and
// compiling it into the schema would mean the schema churns whenever an engine ships
// — and worse, would silently misjudge a scenario pinned to an older version.
//
// So a rules pack is data about one version, registered under it. A scenario names
// its engine version; validation selects the matching pack. A version with no pack
// is reported as unvalidated rather than assumed fine, because silence about an
// unknown version is how a wrong answer gets mistaken for a checked one.
package rules

import (
	"fmt"
	"sort"
	"sync"

	"github.com/inference-sim/blis-schemas/internal/validate"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

// Input is everything a rule may read. A rule sees the whole resolved set of
// documents, because the interesting rules are cross-document: expert divisibility
// needs the deployment's layout and the model's expert count, and neither alone. The
// Scenario carries the immutable problem (engine version, cluster inventory) and the
// Deployment the mutable layout (pools, offload) a rule checks against it.
type Input struct {
	Scenario   *scenario.Scenario
	Deployment *deployment.Deployment
	Model      *model.Graph
}

// Rule is one named, version-scoped check.
//
// Name is stable and mentioned in every finding, so a CI job can waive one rule
// without waiving a whole layer, and a reader can tell which release's expectation
// fired. Because is a short statement of why the rule exists, surfaced in
// documentation rather than in every message.
type Rule struct {
	Name    string
	Because string
	Check   func(Input, *validate.Problems)
}

// Pack is the rule set for one engine version, plus the version-specific data those
// rules read. Holding the data here rather than in each rule's body is what lets a
// new release be a new Pack with different constants and the same rule bodies.
type Pack struct {
	// Version is the engine release this pack describes.
	Version string

	// All2AllBackends are the accepted MoE dispatch backend names, and
	// SPMoEBackends the subset for which the engine makes the MoE input
	// sequence-parallel.
	All2AllBackends map[string]bool
	SPMoEBackends   map[string]bool

	AllReduceBackends map[string]bool
	CUDAGraphModes    map[string]bool
	CacheDTypes       map[string]bool
	MambaCacheModes   map[string]bool
	SchedulerPolicies map[string]bool
	// SpeculativeMethods are accepted draft methods, and AsyncCompatibleSpec the
	// subset that does not disable async scheduling.
	SpeculativeMethods  map[string]bool
	AsyncCompatibleSpec map[string]bool
	// OffloadSpecs are the registered offloading spec names.
	OffloadSpecs map[string]bool

	// Quantizations are the weight-format names the engine serves, from its
	// QuantizationMethods literal. A name outside the set is priced against the wrong
	// weight width, compute peak and GEMM efficiency at once. Connectors are the KV
	// connector names registered with the engine's KVConnectorFactory, naming the
	// offload/PD-transfer implementation; an unknown one leaves the transfer on no
	// resource. EvictionPolicies are the KV-offload cache-policy names registered with the
	// engine's CachePolicyFactory (lru, arc), naming how an offload tier evicts. All three
	// are version facts, verified against the engine source like the sets above (#19), and
	// all three are registries the engine extends out of tree — so an unknown value warns
	// (probable typo) rather than errors (it might be a valid out-of-tree name).
	Quantizations    map[string]bool
	Connectors       map[string]bool
	EvictionPolicies map[string]bool

	// CustomAllReduceWorldSizes are the rank counts the SM-consuming all-reduce
	// kernel supports. A width outside the set falls back whatever was requested.
	CustomAllReduceWorldSizes map[int]bool

	// DefaultDBODecodeThreshold and DefaultDBOPrefillThreshold are the token counts
	// above which dual-batch overlap engages, by batch uniformity.
	DefaultDBODecodeThreshold  int
	DefaultDBOPrefillThreshold int

	// TritonFetchPageBytes is the offload page size below which a CPU-to-GPU fetch
	// takes an SM-consuming kernel rather than a copy engine, and TritonFetchSMs is
	// how many SMs that path withholds.
	TritonFetchPageBytes int
	TritonFetchSMs       int

	// CascadeAttnOptIn records that cascade attention is disabled by default.
	CascadeAttnOptIn bool

	// Rules are the checks themselves.
	Rules []Rule
}

var (
	mu    sync.RWMutex
	packs = map[string]*Pack{}
)

// Register makes a pack available under its version. It panics on a duplicate:
// two packs for one version means one silently wins, and which one would depend on
// import order.
func Register(p *Pack) {
	mu.Lock()
	defer mu.Unlock()
	if p == nil || p.Version == "" {
		panic("rules: a pack must declare a version")
	}
	if _, dup := packs[p.Version]; dup {
		panic(fmt.Sprintf("rules: duplicate pack for version %q", p.Version))
	}
	packs[p.Version] = p
}

// Lookup returns the pack for a version, or nil.
func Lookup(version string) *Pack {
	mu.RLock()
	defer mu.RUnlock()
	return packs[version]
}

// Versions returns every registered version, sorted, for error messages and for a
// CI job that reports which versions it can validate.
func Versions() []string {
	mu.RLock()
	defer mu.RUnlock()
	out := make([]string, 0, len(packs))
	for v := range packs {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

// Apply runs the pack matching the scenario's engine version.
//
// An unregistered version yields one finding rather than silence. That is
// deliberate: a scenario pinned to a version this build does not know has not been
// rule-checked, and reporting nothing would be indistinguishable from passing.
func Apply(in Input) *validate.Problems {
	p := &validate.Problems{}
	if in.Scenario == nil {
		p.Errorf("no scenario to validate")
		return p
	}
	pack := Lookup(in.Scenario.EngineVersion)
	if pack == nil {
		p.RuleWarnf("engine-version-known",
			"no rules pack for engine version %q; field validation passed but no version-specific rule was applied. Registered versions: %v",
			in.Scenario.EngineVersion, Versions())
		return p
	}
	for _, r := range pack.Rules {
		r.Check(in, p)
	}
	return p
}

// Names returns the pack's rule names, sorted. Tests use it to assert that a rule
// has not been dropped, which is the failure a passing suite would otherwise hide.
func (p *Pack) Names() []string {
	out := make([]string, 0, len(p.Rules))
	for _, r := range p.Rules {
		out = append(out, r.Name)
	}
	sort.Strings(out)
	return out
}

// --- Version-scoped behaviour a cost model resolves against --------------------
//
// The methods below answer questions about what the engine will DO, where the fields
// above state what it will ACCEPT. Both are version-scoped facts about one engine
// release, so they belong together: a cost model that compiled these in would churn
// whenever an engine shipped, and would silently misjudge a scenario pinned to an older
// release.
//
// They read only the pack's own fields, so a pack for a new version gets correct
// behaviour by declaring its constants and nothing else.

// SequenceParallelMoE reports whether the engine makes the MoE input sequence-parallel.
//
// When it does, a layer's tensor-parallel all-reduce is replaced by a reduce-scatter and
// all-gather pair rather than joined by one, so the two are alternatives and a model that
// charges both over-prices the layer. All four conditions must hold: the backend must be
// one of the set that supports it, expert parallelism must be on, and both widths must
// exceed one.
func (p *Pack) SequenceParallelMoE(backend string, expertParallel bool, tp, dp int) bool {
	if !expertParallel || tp <= 1 || dp <= 1 {
		return false
	}
	return p.SPMoEBackends[backend]
}

// CustomAllReduceSupportsWidth reports whether the SM-consuming reduction kernel handles
// a given rank count.
//
// The width matters because the answer moves the cost between resources rather than
// scaling it: the custom kernel spends SMs, and NCCL spends link bandwidth. A width the
// kernel does not implement falls back to NCCL, so charging SMs for it would put the cost
// on a resource that is not being consumed.
func (p *Pack) CustomAllReduceSupportsWidth(tp int) bool {
	return p.CustomAllReduceWorldSizes[tp]
}

// TritonFetchWithholdsSMs reports how many SMs a concurrent offload fetch takes from the
// forward pass, given the page size it moves.
//
// Zero when the page size is one the engine moves with a copy engine instead, which costs
// no SMs at all. That distinction is the whole reason this takes the page size: the same
// offload configuration is free on one page size and costs 9% of an H200's SMs on another.
func (p *Pack) TritonFetchWithholdsSMs(pageBytes int) int {
	if pageBytes != p.TritonFetchPageBytes {
		return 0
	}
	return p.TritonFetchSMs
}

// DBOEngages reports whether dual-batch overlap splits a batch.
//
// The thresholds differ by region: a uniform-decode batch is measured against the decode
// threshold and anything else against the prefill one. A model that used one threshold for
// both would engage overlap on batches the engine leaves whole.
func (p *Pack) DBOEngages(enabled bool, totalTokens int, uniformDecode bool) bool {
	if !enabled {
		return false
	}
	threshold := p.DefaultDBOPrefillThreshold
	if uniformDecode {
		threshold = p.DefaultDBODecodeThreshold
	}
	return totalTokens >= threshold
}
