package simresult

import (
	"bytes"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

func floatPtr(f float64) *float64 { return &f }
func intPtr(i int) *int           { return &i }
func int64Ptr(i int64) *int64     { return &i }

// valid builds a well-formed prediction: an open-loop run of a single-node tp8 Granite
// deployment, with a balanced conservation ledger and a fully-valid embedded input. Each
// test mutates one facet and asserts the one law it is about.
func valid() *SimResult {
	return &SimResult{
		Kind: "SimResult", Name: "granite-230b-h200-tp8@10rps",
		EngineVersion: "0.29.0",
		Point:         OperatingPoint{Rate: floatPtr(10)},
		Summary: &Summary{
			OutputTokensPerSec: 287.5, InputTokensPerSec: 4761.5, RequestsPerSec: 9.8,
			TTFT:            Distribution{Mean: 52.5, P90: floatPtr(90), P95: floatPtr(110), P99: floatPtr(168)},
			ITL:             Distribution{Mean: 6.7, P90: floatPtr(7), P95: floatPtr(7.5), P99: floatPtr(9)},
			E2E:             Distribution{Mean: 3200, P90: floatPtr(4100), P95: floatPtr(4500), P99: floatPtr(5200)},
			SchedDelayP99ms: 12, GoodputRPS: 9.4, SLOAttainment: 0.96,
		},
		Conservation: Conservation{
			Injected: 100, Completed: 96, StillRunning: 2, StillQueued: 1, Dropped: 1,
			LengthCapped: 4, Preemptions: 7, KVAllocFailures: 0,
		},
		Scenario: scenario.Scenario{
			Kind: "Scenario", Name: "granite-230b-h200-tp8",
			Model: "granite-5-230b", Coefficients: []string{"cost-model-primitives-h200"},
			EngineVersion: "0.29.0",
			Cluster:       scenario.Cluster{Hardware: "h200", Nodes: 1, GPUsPerNode: 8},
		},
		Deployment: deployment.Deployment{
			Kind: "Deployment", Name: "granite-230b-h200-tp8",
			Pools: []deployment.Pool{{Role: deployment.RoleColocated, Nodes: 1,
				Parallel: deployment.Parallelism{TP: 8, PP: 1, DP: 1},
				Engine:   deployment.Engine{CacheDType: "fp8", GPUMemoryUtilization: 0.9}}},
		},
		Provenance: Provenance{
			Catalog: "blis-catalog@0.1.1", Registry: "blis-registry@abc123",
			Coefficients: []CoefficientOrigin{
				{Name: "gemm_eps_max_bf16", Set: "cost-model-primitives-h200",
					Method: "measured", Scope: "h200"}},
		},
	}
}

func TestValidResultPasses(t *testing.T) {
	if p := valid().Validate(); !p.OK() {
		t.Fatalf("a valid result was rejected:\n%s", p.Error())
	}
}

// TestRejects pins every field-level law in one table; each case names the law it breaks,
// so a failure says which guard was lost rather than only that something broke.
func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*SimResult)
	}{
		{"wrong kind", func(r *SimResult) { r.Kind = "EvaluationRun" }},
		{"no name", func(r *SimResult) { r.Name = "" }},
		{"no engine version", func(r *SimResult) { r.EngineVersion = "" }},

		// Operating point: exactly one of rate / concurrency.
		{"operating point neither", func(r *SimResult) { r.Point = OperatingPoint{} }},
		{"operating point both", func(r *SimResult) {
			r.Point = OperatingPoint{Rate: floatPtr(10), Concurrency: intPtr(8)}
		}},
		{"non-positive rate", func(r *SimResult) { r.Point = OperatingPoint{Rate: floatPtr(0)} }},
		{"concurrency below one", func(r *SimResult) {
			r.Point = OperatingPoint{Concurrency: intPtr(0)}
		}},

		// Mandatory sections must be present, not read as an all-zero default.
		{"summary omitted", func(r *SimResult) { r.Summary = nil }},
		{"zero injected (vacuous / omitted conservation)", func(r *SimResult) {
			r.Conservation = Conservation{}
		}},

		// Summary.
		{"negative throughput", func(r *SimResult) { r.Summary.OutputTokensPerSec = -1 }},
		{"slo attainment above one", func(r *SimResult) { r.Summary.SLOAttainment = 1.5 }},
		{"negative latency mean", func(r *SimResult) { r.Summary.TTFT.Mean = -1 }},
		{"negative recorded percentile", func(r *SimResult) {
			r.Summary.TTFT.P95 = floatPtr(-1)
		}},
		{"percentiles out of order", func(r *SimResult) {
			r.Summary.TTFT = Distribution{Mean: 50, P90: floatPtr(200), P95: floatPtr(100), P99: floatPtr(300)}
		}},
		// A percentile RECORDED as zero below a positive one is a real inversion the
		// pointer fields let us catch — a plain float64 would read the 0 as unrecorded.
		{"recorded zero percentile below a positive one", func(r *SimResult) {
			r.Summary.TTFT = Distribution{Mean: 1, P90: floatPtr(5), P95: floatPtr(0), P99: floatPtr(6)}
		}},

		// Conservation.
		{"negative count", func(r *SimResult) { r.Conservation.Completed = -1 }},
		{"over-accounted", func(r *SimResult) { r.Conservation.Completed = 200 }},
		{"length-capped exceeds completed", func(r *SimResult) {
			r.Conservation.LengthCapped = r.Conservation.Completed + 1
		}},

		// Requests.
		{"requests both inline and file", func(r *SimResult) {
			r.Requests = &Requests{File: "rows.csv", Rows: []RequestRow{{ID: "a"}}}
		}},
		{"requests neither", func(r *SimResult) { r.Requests = &Requests{} }},
		{"requests count mismatch", func(r *SimResult) {
			r.Requests = &Requests{Rows: []RequestRow{{ID: "a"}}, Count: int64Ptr(9)}
		}},
		// A count RECORDED as 0 beside inline rows is a mismatch the pointer field lets us
		// catch — a plain int64 would read the 0 as "unstated" and skip the check.
		{"requests count recorded zero with rows", func(r *SimResult) {
			r.Requests = &Requests{Rows: []RequestRow{{ID: "a"}}, Count: int64Ptr(0)}
		}},
		{"requests row missing id", func(r *SimResult) {
			r.Requests = &Requests{Rows: []RequestRow{{ID: ""}}}
		}},
		{"requests row negative tokens", func(r *SimResult) {
			r.Requests = &Requests{Rows: []RequestRow{{ID: "a", InputTokens: -1}}}
		}},

		// Runtime.
		{"negative wall clock", func(r *SimResult) { r.Runtime = &Runtime{WallClockMs: -1} }},

		// Breakdowns.
		{"duplicate slice name", func(r *SimResult) {
			r.Breakdowns = &Breakdowns{PerClass: []Slice{{Name: "x"}, {Name: "x"}}}
		}},
		{"slice missing name", func(r *SimResult) {
			r.Breakdowns = &Breakdowns{PerTenant: []Slice{{Name: ""}}}
		}},
		{"fitness component missing name", func(r *SimResult) {
			r.Breakdowns = &Breakdowns{Fitness: &Fitness{Score: 1,
				Components: []FitnessComponent{{Name: "", Value: 1}}}}
		}},
		{"negative pd transfers", func(r *SimResult) {
			r.Breakdowns = &Breakdowns{PD: &PD{Transfers: -1}}
		}},

		// Provenance.
		{"coefficient origin missing name", func(r *SimResult) {
			r.Provenance.Coefficients = []CoefficientOrigin{{Name: ""}}
		}},

		// Embedded input — a malformed scenario or deployment makes the result malformed.
		{"embedded scenario invalid", func(r *SimResult) { r.Scenario.Model = "" }},
		{"embedded deployment invalid", func(r *SimResult) { r.Deployment.Pools = nil }},
		{"deployment does not fit cluster", func(r *SimResult) { r.Scenario.Cluster.Nodes = 2 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := valid()
			tc.mutate(r)
			if p := r.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

// TestOperatingPointIsExactlyOne is the headline law of the type the epic flags for
// scrutiny: a run is open-loop (a rate) or closed-loop (a concurrency), and recording
// which keeps the result self-describing. Both forms are valid; neither-and-both are not.
func TestOperatingPointIsExactlyOne(t *testing.T) {
	open := valid()
	open.Point = OperatingPoint{Rate: floatPtr(10)}
	if p := open.Validate(); !p.OK() {
		t.Fatalf("an open-loop point should pass:\n%s", p.Error())
	}
	if !open.Point.OpenLoop() {
		t.Error("OpenLoop() should be true when a rate is set")
	}

	closed := valid()
	closed.Point = OperatingPoint{Concurrency: intPtr(8)}
	if p := closed.Validate(); !p.OK() {
		t.Fatalf("a closed-loop point should pass:\n%s", p.Error())
	}
	if closed.Point.OpenLoop() {
		t.Error("OpenLoop() should be false when a concurrency is set")
	}
}

// TestConservationIdentity is INV-1 as a law. The terminal buckets sum to Injected; the
// diagnostic counters (length-capped, preemptions, kv-alloc-failures) are NOT part of
// that sum, so raising them while the buckets stay balanced must not break the identity.
// This is the distinction the issue's conservation list conflates and this type keeps.
func TestConservationIdentity(t *testing.T) {
	c := Conservation{Injected: 100, Completed: 96, StillRunning: 2, StillQueued: 1, Dropped: 1}
	if c.Accounted() != 100 {
		t.Fatalf("Accounted() = %d, want 100", c.Accounted())
	}
	if !c.Balanced() {
		t.Fatal("a ledger whose buckets sum to injected should be Balanced()")
	}
	// Diagnostic counters ride alongside without entering the identity.
	c.LengthCapped, c.Preemptions, c.KVAllocFailures = 50, 999, 7
	if !c.Balanced() {
		t.Fatal("diagnostic counters must not affect the conservation identity")
	}
	if c.Accounted() != 100 {
		t.Fatalf("diagnostic counters changed Accounted() to %d", c.Accounted())
	}
}

// TestImbalanceFailsBothDirections: INV-1 is a hard invariant relocated into the document,
// so a ledger that does not balance is a field error, not a mere warning — in EITHER
// direction. Under-accounting is the lost-request bug INV-1 exists to catch;
// over-accounting is impossible. A balanced ledger passes.
func TestImbalanceFailsBothDirections(t *testing.T) {
	under := valid()
	under.Conservation = Conservation{Injected: 100, Completed: 90} // 10 unaccounted
	if under.Validate().OK() {
		t.Fatal("under-accounting must fail: INV-1 requires the buckets to sum to injected")
	}
	over := valid()
	over.Conservation = Conservation{Injected: 100, Completed: 100, Dropped: 5} // 105 > 100
	if over.Validate().OK() {
		t.Fatal("over-accounting must fail: you cannot account for more than were injected")
	}
	balanced := valid()
	balanced.Conservation = Conservation{Injected: 100, Completed: 100}
	if p := balanced.Validate(); !p.OK() {
		t.Fatalf("a balanced ledger should pass:\n%s", p.Error())
	}
}

// TestRequestsAreOptInAndEitherInlineOrExternal covers the bulk-data law: a result need
// not carry per-request rows at all; when it does, it inlines them OR references an
// external file, and the two forms are exclusive so bulk rows are never forced inline.
func TestRequestsAreOptInAndEitherInlineOrExternal(t *testing.T) {
	// Opt-out: no requests block at all is valid.
	if p := valid().Validate(); !p.OK() {
		t.Fatalf("a result with no requests block should pass:\n%s", p.Error())
	}
	// Inline rows.
	inline := valid()
	inline.Requests = &Requests{Count: int64Ptr(1), Rows: []RequestRow{
		{ID: "r1", InputTokens: 8015, OutputTokens: 496, TTFTms: 52, E2Ems: 3200}}}
	if p := inline.Validate(); !p.OK() {
		t.Fatalf("inline rows should pass:\n%s", p.Error())
	}
	// External file reference, with the count stated so a reader need not open it.
	external := valid()
	external.Requests = &Requests{File: "run1-requests.csv", Count: int64Ptr(100000)}
	if p := external.Validate(); !p.OK() {
		t.Fatalf("an external file reference should pass:\n%s", p.Error())
	}
}

// TestRuntimeIsSeparable demonstrates the quarantine: the runtime block serializes as an
// additive tail, and stripping it leaves the rest of the document byte-identical to one
// that never carried it — so a determinism check (INV-6) excludes exactly runtime and
// nothing leaks outside it.
func TestRuntimeIsSeparable(t *testing.T) {
	withRT := valid()
	withRT.Runtime = &Runtime{WallClockMs: 1234, Host: "node-7", Build: "deadbeef",
		StartedAt: "2026-10-02T00:00:00Z", FinishedAt: "2026-10-02T00:00:01Z"}
	noRT := valid()

	a, err := yaml.Marshal(withRT)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	withRT.Runtime = nil
	b, _ := yaml.Marshal(withRT)
	c, _ := yaml.Marshal(noRT)

	if bytes.Equal(a, b) {
		t.Fatal("the runtime block did not serialize, so the test proves nothing")
	}
	if !bytes.Equal(b, c) {
		t.Fatalf("stripping runtime did not restore byte-identity:\n--- stripped ---\n%s\n--- never had ---\n%s", b, c)
	}
}

// TestMarshalIsDeterministic guards against a non-deterministic region creeping in (a map
// field would serialize in random order). The whole document must marshal identically
// across repeated calls — the precondition INV-6 rests on, once runtime is excluded.
func TestMarshalIsDeterministic(t *testing.T) {
	r := valid()
	r.Breakdowns = &Breakdowns{
		PerClass: []Slice{{Name: "interactive", Summary: *r.Summary}},
		Fitness: &Fitness{Score: 0.82, Components: []FitnessComponent{
			{Name: "goodput", Value: 9.4, Weight: 0.7},
			{Name: "p99_e2e", Value: 5200, Weight: 0.3}}},
	}
	first, _ := yaml.Marshal(r)
	for i := 0; i < 8; i++ {
		again, _ := yaml.Marshal(r)
		if !bytes.Equal(first, again) {
			t.Fatalf("marshal %d differed; a non-deterministic region crept in", i)
		}
	}
}
