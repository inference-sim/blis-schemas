// Package simresult describes what one BLIS run PREDICTS: the output document a single
// `blis run` emits, and that a sweeper or llm-d-lens consumes to rank configurations.
//
// It is deliberately a different type from evaluation.Run, and the distinction is the
// whole reason the package exists. An evaluation.Run is a MEASURED ground-truth sweep —
// a harness driving a real server across a range of concurrency Points, the record a
// prediction is scored against. A SimResult is a PREDICTION — one run of the simulator
// at one operating point, the thing being scored. Conflating them in one type would
// blur the boundary a comparison rests on and force a concurrency-sweep shape onto a
// single-run result; keeping them apart lets a calibration report line the two up
// (prediction vs observed) without either pretending to be the other.
//
// Four shapes recur and each exists to prevent a class of error:
//
// The operating point is recorded here, not in the embedded input. The load level a run
// was driven at (an arrival rate OR a closed-loop concurrency, never both) is a run-level
// sweep axis supplied OUTSIDE the Scenario/Deployment input — a Scenario fixes the
// immutable problem and a Deployment the mutable layout, and neither carries the load.
// A result that omitted it would not be self-describing: two predictions of the same
// scenario+deployment at different loads are two facts, not an error bar. This mirrors
// the role Concurrency plays in evaluation.Run.Point.
//
// Non-determinism is quarantined. The document is byte-deterministic given the resolved
// scenario+deployment, the catalog/registry revision, and the seed (INV-6) — EXCEPT the
// Runtime block, which carries wall-clock, host and build facts that change run to run.
// It is a single, clearly-named region so a comparison, a golden test, or a run/replay
// parity check (INV-13) can exclude exactly it and assert equality over everything else.
//
// Bulk data stays external. Per-request rows are opt-in, and when a run is large they
// are referenced as an external sidecar file by path rather than inlined — a document
// carries configuration and references, not a megabyte of CSV. Requests states one or
// the other, never both.
//
// The input is embedded, fully resolved. A SimResult carries the Scenario and Deployment
// that produced it, after resolution, so the prediction is reproducible from the document
// alone, and so feeding a result's embedded input back to `replay` is well defined. This
// is the config-vs-bulk-data split the epic draws: the resolved Scenario+Deployment are
// configuration — small, and the exact thing a reader needs to reproduce the run — so they
// are embedded rather than referenced by hash or path; only BULK data (per-request rows,
// raw traces) is externalized. Provenance records what the prediction rests on — the
// catalog and registry revisions, and every coefficient the kernel used — because a
// prediction that cannot state its basis cannot be audited.
package simresult

import (
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

// SimResult is one simulator run's predicted output.
type SimResult struct {
	Kind string `yaml:"kind"` // "SimResult"
	Name string `yaml:"name"`

	// EngineVersion is the engine release this prediction models. Like evaluation.Run's,
	// it is recorded on the result itself: it may differ from the embedded scenario's
	// engine_version (the release the coefficients were fitted against), and when it does
	// a downstream comparison is across versions — recording both makes that visible.
	EngineVersion string `yaml:"engine_version"`

	// Point is the load level the run was driven at: a rate XOR a concurrency. See the
	// package comment for why it lives here and not in the embedded input.
	Point OperatingPoint `yaml:"operating_point"`

	// Summary is the headline aggregate: throughput, the latency distributions, and the
	// SLO figures. It is what a reader looks at first and a sweeper ranks on. A pointer and
	// required: an omitted summary must be rejected, not read as an all-zero one — a value
	// type could not tell the two apart, and an all-zero summary is itself legitimate (an
	// all-dropped run completes nothing).
	Summary *Summary `yaml:"summary"`

	// Conservation is INV-1 relocated into the document: every injected request is
	// accounted for in exactly one terminal bucket, with a few diagnostic counters beside.
	// Required (its Injected floor of 1 makes an omitted, zero-valued block fail).
	Conservation Conservation `yaml:"conservation"`

	// Breakdowns disaggregate the summary along whichever axes a run exercised. Absent
	// when a run has nothing to break down (a single class, single tenant, no PD).
	Breakdowns *Breakdowns `yaml:"breakdowns,omitempty"`

	// Requests is opt-in per-request detail: inline rows for a small run, or a reference
	// to an external sidecar file for a large one. Absent unless the run requested it.
	Requests *Requests `yaml:"requests,omitempty"`

	// Runtime is the ONLY non-deterministic region of the document (see the package
	// comment). Absent for a byte-deterministic emission that records nothing wall-clock.
	Runtime *Runtime `yaml:"runtime,omitempty"`

	// Scenario and Deployment are the fully-resolved input that produced this result,
	// embedded for reproducibility. They are validated as part of the result.
	Scenario   scenario.Scenario     `yaml:"scenario"`
	Deployment deployment.Deployment `yaml:"deployment"`

	// Provenance records the catalog/registry revisions and the coefficients the kernel
	// used, so the prediction can be audited and reproduced.
	Provenance Provenance `yaml:"provenance"`
}

// OperatingPoint is the load level a run was driven at. Exactly one of its two fields
// is set: Rate for an open-loop run (arrivals per second), Concurrency for a closed-loop
// one (simultaneous in-flight requests). They are pointers so "not this kind of run" is
// distinct from "zero", and so the exactly-one rule is checkable rather than guessed from
// a zero that could be either an unset field or a real value.
type OperatingPoint struct {
	// Rate is the open-loop arrival rate in requests per second. Set iff the run was
	// open-loop.
	Rate *float64 `yaml:"rate,omitempty"`
	// Concurrency is the closed-loop count of simultaneous requests. Set iff the run was
	// closed-loop.
	Concurrency *int `yaml:"concurrency,omitempty"`
}

// OpenLoop reports whether the run was driven by an arrival rate rather than a fixed
// concurrency. Meaningful only once the point has validated (exactly one field set).
func (o OperatingPoint) OpenLoop() bool { return o.Rate != nil }

// Summary is the headline aggregate of a run.
type Summary struct {
	// Throughput. Output and input token rates are separate because an input-heavy
	// workload judged on output throughput looks idle while doing more total work;
	// RequestsPerSec is the completed-request rate, the denominator goodput compares to.
	OutputTokensPerSec float64 `yaml:"output_tokens_per_sec"`
	InputTokensPerSec  float64 `yaml:"input_tokens_per_sec,omitempty"`
	RequestsPerSec     float64 `yaml:"requests_per_sec,omitempty"`

	// Latency distributions, milliseconds: mean plus the tail. The three are not
	// interchangeable — TTFT includes admission and prefill, ITL is per emitted token
	// during decode, and E2E carries completion overhead that lands on neither.
	TTFT Distribution `yaml:"ttft_ms"`
	ITL  Distribution `yaml:"itl_ms"`
	E2E  Distribution `yaml:"e2e_ms"`

	// SchedDelayP99ms is the p99 of time a request waited before its first scheduled
	// step. Only the tail is reported: the mean scheduling delay is uninformative, the
	// p99 is where queue build-up shows.
	SchedDelayP99ms float64 `yaml:"sched_delay_p99_ms,omitempty"`

	// GoodputRPS is the rate of requests that completed within their SLO; SLOAttainment
	// is the fraction that did, in [0, 1]. Goodput is the quantity a capacity decision
	// turns on — total throughput that misses the SLO is not useful work. The SLO TARGETS
	// these are scored against are a property of the deployment's admission/SLO policy, not
	// of the result: once that policy surface lands (epic S2) it rides in the embedded
	// Deployment, from where these derived figures become recomputable. This result records
	// the figures; it does not redefine the policy that produced them.
	GoodputRPS    float64 `yaml:"goodput_rps,omitempty"`
	SLOAttainment float64 `yaml:"slo_attainment,omitempty"`
}

// Distribution is a mean plus tail percentiles, all in one unit (milliseconds for the
// latency metrics). Each percentile is a pointer so an unrecorded tail (nil) is distinct
// from a recorded zero — a run that computes only the mean and p99 leaves p90 and p95
// nil rather than claiming they are zero. The recorded percentiles must be non-negative
// and, being order statistics, non-decreasing.
type Distribution struct {
	Mean float64  `yaml:"mean"`
	P90  *float64 `yaml:"p90,omitempty"`
	P95  *float64 `yaml:"p95,omitempty"`
	P99  *float64 `yaml:"p99,omitempty"`
}

// Conservation is INV-1 made part of the document. Every injected request lands in
// exactly one TERMINAL bucket, and the sum of the buckets equals Injected — the
// five-term single-instance form of the invariant, with a cluster run folding its
// router/gateway/encode rejections into Dropped.
//
// Three fields are NOT terminal buckets and are deliberately excluded from that sum:
// LengthCapped is a FINISH REASON (a request that hit its length cap still Completed, so
// counting it again would double-count); Preemptions and KVAllocFailures count EVENTS,
// not requests — one request may be preempted many times. They ride along as diagnostics
// because they explain a result, not because they account for a request.
type Conservation struct {
	// Injected is the left-hand side: every request that entered the simulation.
	Injected int64 `yaml:"injected"`

	// Terminal buckets — each injected request lands in exactly one.
	Completed    int64 `yaml:"completed"`
	StillQueued  int64 `yaml:"still_queued,omitempty"`
	StillRunning int64 `yaml:"still_running,omitempty"`
	// Dropped is every non-completion terminal disposition: unservable drops, and, for a
	// cluster run, the router/gateway/encode rejections and in-flight evictions the
	// twelve-term INV-1 names separately.
	Dropped  int64 `yaml:"dropped,omitempty"`
	TimedOut int64 `yaml:"timed_out,omitempty"`

	// Diagnostic counters — see the type comment; excluded from the identity.
	LengthCapped    int64 `yaml:"length_capped,omitempty"`
	Preemptions     int64 `yaml:"preemptions,omitempty"`
	KVAllocFailures int64 `yaml:"kv_alloc_failures,omitempty"`
}

// Accounted sums the terminal buckets: the right-hand side of the INV-1 identity. The
// diagnostic counters are excluded by construction.
func (c Conservation) Accounted() int64 {
	return c.Completed + c.StillQueued + c.StillRunning + c.Dropped + c.TimedOut
}

// Balanced reports whether every injected request is accounted for in exactly one
// terminal bucket — the field-level statement of INV-1.
func (c Conservation) Balanced() bool { return c.Accounted() == c.Injected }

// Breakdowns disaggregate the run along the axes it exercised. Each member is present
// only when a run has that axis: a single-class, single-tenant, non-disaggregated run
// leaves them all empty, which is why the whole block is optional on the result.
type Breakdowns struct {
	// PerClass, PerModel, PerTenant and Adapters share the Slice shape: a named
	// sub-population reporting the same Summary (and optional Conservation) as the whole
	// run. The axis is which field the slice sits under; the metrics are the same shape.
	PerClass  []Slice `yaml:"per_class,omitempty"`
	PerModel  []Slice `yaml:"per_model,omitempty"`
	PerTenant []Slice `yaml:"per_tenant,omitempty"`
	Adapters  []Slice `yaml:"adapters,omitempty"`

	// PD splits latency across a prefill/decode-disaggregated deployment. Present only
	// for a disaggregated run.
	PD *PD `yaml:"pd,omitempty"`

	// Saturation records whether the run was saturated and by which detector; Fitness is
	// the derived score a sweeper ranks on. Both are optional: a run may ask for neither.
	Saturation *Saturation `yaml:"saturation,omitempty"`
	Fitness    *Fitness    `yaml:"fitness,omitempty"`
}

// Slice is one named sub-population's metrics. Conservation is optional: a per-class
// summary is usually enough, and the request ledger is reported for the whole run.
type Slice struct {
	Name         string        `yaml:"name"`
	Summary      Summary       `yaml:"summary"`
	Conservation *Conservation `yaml:"conservation,omitempty"`
}

// PD is the prefill/decode split for a disaggregated run: the share of first-token
// latency attributable to each role, the KV transfer between them, and how many
// transfers occurred.
type PD struct {
	PrefillTTFTms Distribution `yaml:"prefill_ttft_ms,omitempty"`
	DecodeTTFTms  Distribution `yaml:"decode_ttft_ms,omitempty"`
	TransferMs    Distribution `yaml:"transfer_ms,omitempty"`
	Transfers     int64        `yaml:"transfers,omitempty"`
}

// Saturation records the post-hoc saturation verdict for the run. KneeRPS, when a
// detector found one, is the load at which the run saturated.
type Saturation struct {
	Saturated bool    `yaml:"saturated"`
	Detector  string  `yaml:"detector,omitempty"`
	KneeRPS   float64 `yaml:"knee_rps,omitempty"`
}

// Fitness is the single scalar a sweeper ranks configurations by, with the weighted
// components it was composed from. It is a DERIVED score, not a measurement, which is
// why it is a breakdown rather than part of the Summary.
type Fitness struct {
	Score      float64            `yaml:"score"`
	Components []FitnessComponent `yaml:"components,omitempty"`
}

// FitnessComponent is one named term of a fitness score, with the weight it entered at.
// A slice rather than a map so the order and byte-serialization are deterministic.
type FitnessComponent struct {
	Name   string  `yaml:"name"`
	Value  float64 `yaml:"value"`
	Weight float64 `yaml:"weight,omitempty"`
}

// Requests is opt-in per-request detail. A result either inlines the rows (a small run,
// a test) or references an external sidecar file by path (a large run) — never both, and
// bulk rows are never forced inline. The schema does not open the file; a reader resolves
// the path relative to the document.
type Requests struct {
	// File references an external sidecar holding the per-request rows (e.g. a CSV). It
	// is a path, resolved by the reader relative to the result document.
	File string `yaml:"file,omitempty"`
	// Rows is the inline per-request detail, for a run small enough to embed.
	Rows []RequestRow `yaml:"rows,omitempty"`
	// Count is the number of rows. It is a pointer so an unstated count (nil) is distinct
	// from a recorded zero — a reader can tell "size not given" from "no rows". May be
	// stated alongside File so the size is known without opening it; when Rows is inline
	// and Count is given, they must agree.
	Count *int64 `yaml:"count,omitempty"`
}

// RequestRow is one request's recorded outcome, for the inline convenience case. The
// external file is the canonical bulk form; this carries the fields a small run most
// often inlines.
type RequestRow struct {
	ID           string  `yaml:"id"`
	InputTokens  int     `yaml:"input_tokens,omitempty"`
	OutputTokens int     `yaml:"output_tokens,omitempty"`
	TTFTms       float64 `yaml:"ttft_ms,omitempty"`
	E2Ems        float64 `yaml:"e2e_ms,omitempty"`
	FinishReason string  `yaml:"finish_reason,omitempty"`
}

// Runtime is the quarantined non-deterministic block: wall-clock, host and build facts
// that change run to run. Nothing a downstream comparison or a determinism check depends
// on belongs here — that is the point of keeping it in one place.
type Runtime struct {
	WallClockMs float64 `yaml:"wall_clock_ms,omitempty"`
	Host        string  `yaml:"host,omitempty"`
	Build       string  `yaml:"build,omitempty"`
	// StartedAt and FinishedAt are ISO 8601 timestamps.
	StartedAt  string `yaml:"started_at,omitempty"`
	FinishedAt string `yaml:"finished_at,omitempty"`
}

// Provenance records what the prediction rests on: the catalog and registry revisions
// resolved against (a git SHA or tag), and every coefficient the kernel used. It is a
// value rather than an optional block because a prediction should always state its basis
// — though a pure-analytical run may legitimately use no coefficients, so the list may
// be empty.
type Provenance struct {
	Catalog      string              `yaml:"catalog,omitempty"`
	Registry     string              `yaml:"registry,omitempty"`
	Coefficients []CoefficientOrigin `yaml:"coefficients,omitempty"`
}

// CoefficientOrigin is one row of the provenance trail: a registry entry the kernel used,
// with its evidence. It is the document form of the kernel's own CoefficientOrigin —
// the same four facts, carrying yaml tags because this one is serialized.
type CoefficientOrigin struct {
	Name   string `yaml:"name"`
	Set    string `yaml:"set,omitempty"`
	Method string `yaml:"method,omitempty"`
	Scope  string `yaml:"scope,omitempty"`
}
