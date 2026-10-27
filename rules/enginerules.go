package rules

import (
	"sort"

	"github.com/inference-sim/blis-schemas/internal/validate"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
)

// Problems is the accumulating finding list a rule reports into, and Problem, Severity and
// its two values the findings in it. They are exported here so a rules pack can live in
// another module: a Rule's Check takes a Problems, and a type in an internal package cannot
// be named from outside this module.
type (
	Problems = validate.Problems
	Problem  = validate.Problem
	Severity = validate.Severity
)

const (
	SeverityError   = validate.SeverityError
	SeverityWarning = validate.SeverityWarning
)

// EngineRules is everything a consumer asks of one engine release's rules: which values
// it accepts, how it resolves what a deployment leaves unstated, and its version-scoped
// checks. It is the shape a rules pack is consumed through, so the pack itself can live
// with the engine model that owns it (blis-vllm), and a cost model or a validator takes
// the release's rules from its caller instead of looking them up here.
//
// *Pack implements it. A consumer that needs a fact the interface does not name asks for
// a method to be added here, rather than type-asserting to *Pack: an assertion silently
// skips the fact for any other implementation.
type EngineRules interface {
	// EngineVersion is the release the rules describe, such as "0.31.0".
	EngineVersion() string

	// Check runs every rule of the release over a document set.
	Check(in Input) *Problems
	// RuleNames returns the release's rule names, sorted.
	RuleNames() []string

	// SpeculativeMethodAccepted reports whether the release accepts a draft method, and
	// SpeculativeMethodNames lists the accepted ones, sorted, for an error message.
	SpeculativeMethodAccepted(method string) bool
	SpeculativeMethodNames() []string
	// AsyncCompatibleSpeculative reports whether a draft method leaves async scheduling
	// on, and known whether the release states a list at all.
	AsyncCompatibleSpeculative(method string) (compatible, known bool)

	// EngramOnHostByDefault is EngramConfig.cpu_offload's default in the release: whether an
	// Engram table lives in host memory when the deployment does not say.
	EngramOnHostByDefault() bool
	// DCPCommBackendNames are the decode-context-parallel collective sets the release
	// accepts, nil when it states none.
	DCPCommBackendNames() map[string]bool
	// MegaMoEBackend reports whether a --moe-backend value runs a model's mega-MoE path.
	MegaMoEBackend(name string) bool

	SequenceParallelMoE(backend string, expertParallel bool, tp, dp int) bool
	ModelSequenceParallel(architecture string, pl deployment.Parallelism, moeBackend string) bool
	CustomAllReduceSupportsWidth(tp int) bool
	TritonFetchWithholdsSMs(pageBytes int) int
	DBOEngages(enabled bool, totalTokens int, uniformDecode bool) bool
	EffectiveAll2AllBackend(backend string) string
	ResolveCacheDType(architecture string, family, tp int, requested string) string
	WindowPagesPerRequest(c model.Cache, engineBlock, inFlight, extraRetained, maxModelLen int) int
	KVCacheGroups(in KVCacheGroupsInput) (KVCacheGrouping, error)
	CUDAGraphCaptureSizes(in CUDAGraphCaptureInput) (CUDAGraphCapture, bool, error)
	ResolveDPLoadBalancing(pl deployment.Parallelism) deployment.DPLoadBalancing
}

var _ EngineRules = (*Pack)(nil)

// EngineVersion implements EngineRules.
func (p *Pack) EngineVersion() string { return p.Version }

// Check implements EngineRules: every rule of the pack, in order.
func (p *Pack) Check(in Input) *Problems {
	out := &validate.Problems{}
	for _, r := range p.Rules {
		r.Check(in, out)
	}
	return out
}

// RuleNames implements EngineRules.
func (p *Pack) RuleNames() []string { return p.Names() }

// SpeculativeMethodAccepted implements EngineRules.
func (p *Pack) SpeculativeMethodAccepted(method string) bool { return p.SpeculativeMethods[method] }

// SpeculativeMethodNames implements EngineRules.
func (p *Pack) SpeculativeMethodNames() []string { return sortedKeys(p.SpeculativeMethods) }

// AsyncCompatibleSpeculative implements EngineRules.
func (p *Pack) AsyncCompatibleSpeculative(method string) (compatible, known bool) {
	if p.AsyncCompatibleSpec == nil {
		return false, false
	}
	return p.AsyncCompatibleSpec[method], true
}

// EngramOnHostByDefault implements EngineRules. It reads the EngramCPUOffloadDefault field,
// which a method of the same name could not coexist with.
func (p *Pack) EngramOnHostByDefault() bool { return p.EngramCPUOffloadDefault }

// DCPCommBackendNames implements EngineRules. It returns a copy, so a caller cannot edit
// the pack.
func (p *Pack) DCPCommBackendNames() map[string]bool {
	if p.DCPCommBackends == nil {
		return nil
	}
	out := make(map[string]bool, len(p.DCPCommBackends))
	for k, v := range p.DCPCommBackends {
		out[k] = v
	}
	return out
}

// MegaMoEBackend implements EngineRules, normalizing the name as the engine does.
func (p *Pack) MegaMoEBackend(name string) bool { return p.MegaMoEBackends[NormalizeMoEBackend(name)] }

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k, ok := range m {
		if ok {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}
