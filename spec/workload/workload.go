// Package workload describes the traffic a deployment is measured against. Traffic
// comes at two fidelities. A Shape is a distribution — how long prompts are, how long
// completions are, and how much prefix requests share — because a cost model is a
// function of an actual batch and a mean hides the shape that matters: two corpora with
// the same mean input length, one tightly clustered and one spanning 52,000 to 93,000
// tokens, saturate the same deployment in different ways and at different concurrencies.
// A trace is a concrete corpus — real per-request lengths, arrival timestamps, and a
// prefix TREE — captured in an external file this package references by path.
//
// A Scenario's "what traffic" slot is a Binding: a sum type over those two fidelities.
// The slot is single on purpose, so a scenario names traffic once, distributional or
// concrete, rather than carrying two fields that could disagree.
package workload

import "sort"

// Shape is one traffic class. Token counts are per request.
type Shape struct {
	Name string `yaml:"name"`

	// PrefixTokens is the shared prefix length, which becomes the cached fraction a
	// cost model consumes. Zero means no sharing.
	PrefixTokens int `yaml:"prefix_tokens"`

	Prompt Distribution `yaml:"prompt"`
	Output Distribution `yaml:"output"`
}

// Distribution is a token-count distribution. Mean and StdDev describe the bulk;
// Min and Max bound it, and they matter because a scheduler's behaviour at the
// tail is what sets a knee.
type Distribution struct {
	Mean   int `yaml:"tokens"`
	StdDev int `yaml:"tokens_stdev,omitempty"`
	Min    int `yaml:"tokens_min,omitempty"`
	Max    int `yaml:"tokens_max,omitempty"`
}

// Binding is a Scenario's "what traffic" slot: a sum type with two arms, exactly one of
// which is set. Shape references a catalog workload Shape — the distributional arm a
// Scenario has always carried; Trace references an external captured trace — the
// concrete arm. A Scenario that poses a capacity question with no traffic omits the
// binding entirely (a nil *Binding on the Scenario); a Binding that is present must
// choose one arm.
//
// The two arms answer one question — what traffic a deployment is measured against — at
// different fidelities, which is why they share a single slot rather than sitting as two
// independent fields that could both be set or both omitted. A Shape is distributional:
// a cost model draws prompt and output lengths from it and shares one scalar prefix
// (Shape.PrefixTokens) across requests. A trace is concrete: real per-request lengths,
// arrival timestamps, and a prefix TREE — per-request prefix groups and lengths in the
// referenced data rows — that a single scalar cannot express. The prefix structure of a
// trace therefore lives in the referenced trace, never collapsed onto Shape.PrefixTokens.
type Binding struct {
	// Shape names a catalog workload Shape — the distributional arm. It is a name, not an
	// inline Shape, because the Shape lives in the catalog and is validated there; a
	// Scenario only identifies which one.
	Shape string `yaml:"shape,omitempty"`

	// Trace references an external captured trace — the concrete arm. The bulk
	// per-request rows stay in the file it names; only the reference and the small header
	// metadata are carried in the document.
	Trace *TraceRef `yaml:"trace,omitempty"`
}

// TraceRef references a captured TraceV2 trace by PATH, never by value. A real trace is a
// header plus a per-request data CSV that can run to millions of rows; inlining those
// rows into a Scenario would make the document unbounded and defeat the reason bulk data
// lives in files at all — the same reason the catalog and registry own their data rather
// than a scenario inlining it. So this type carries the path to the data file, optional
// integrity and provenance, and the SMALL header metadata as structured fields, and has
// no field in which per-request rows could be written.
type TraceRef struct {
	// Data is the path to the bulk per-request data CSV. It is the only reference to the
	// rows; the rows themselves are never represented in the document.
	Data string `yaml:"data"`

	// SHA256 is the optional content digest of the data file, so a consumer can detect a
	// trace that changed under a reference that did not. A 64-character hex digest when
	// present; absent when the producer recorded none.
	SHA256 string `yaml:"sha256,omitempty"`

	// Rows is the optional expected row COUNT of the data file — provenance and a cheap
	// cross-check against the file a consumer opens, NOT the rows themselves. Zero means
	// unstated.
	Rows int `yaml:"rows,omitempty"`

	// Header is the small trace header metadata: the few structured fields that describe
	// the trace as a whole, as distinct from its per-request rows.
	Header TraceHeader `yaml:"header"`
}

// TraceHeader is the metadata a TraceV2 carries alongside its data CSV: what produced the
// trace, in what units, under what seed, against what server, and to what SLO targets. It
// is bounded and small — a handful of scalars and two optional sub-blocks — which is why
// it is inlined where the per-request rows are not.
type TraceHeader struct {
	// Version is the TraceV2 schema version the data file was written against — a
	// compatibility signal, so a consumer that cannot read a version refuses rather than
	// misreads.
	Version int `yaml:"trace_version"`

	// TimeUnit names the unit of the data file's timestamp columns (microseconds in
	// practice). A trace whose timestamps carry no stated unit is ambiguous, so it is
	// required, and the set is CLOSED — an unrecognized unit is an error, following this
	// repo's rule that a unit cannot be invented at the call site. The set admits the
	// several spellings the producing tools use for one unit (observe writes "us", the
	// converters write "microseconds"): accepting a known synonym is a producer
	// convenience, where accepting an arbitrary string would let a typo through as a unit
	// nothing downstream can map.
	TimeUnit TimeUnit `yaml:"time_unit"`

	// Mode records which pipeline produced the trace: a real server observed, a workload
	// generated, or a trace replayed. It is the field that keeps an observed corpus
	// distinguishable from a synthetic one.
	Mode Mode `yaml:"mode"`

	// WorkloadSeed is the workload RNG seed, when one was recorded. It is a pointer because
	// a seed of 0 is a real seed distinct from "no seed recorded": a generated trace
	// carries one, an observed trace of a real server does not. The yaml key is
	// workload_seed, matching the TraceV2 header the serving pipeline writes, so an
	// operator does not translate it.
	WorkloadSeed *int64 `yaml:"workload_seed,omitempty"`

	// Server records the configuration of the server that produced the trace, as
	// provenance. It is distinct from the Deployment being simulated — an observed trace
	// records the real server it came from, which is what a calibration compares a
	// prediction against. Absent when the producing pipeline recorded none.
	Server *TraceServer `yaml:"server,omitempty"`

	// GoodputSLOTargets are the per-class TTFT/ITL/E2E thresholds the trace was measured
	// against, keyed by SLO class name, so that observe -> replay -> calibrate carry one
	// SLO definition. Absent when the trace defined none. The yaml key is
	// goodput_slo_targets, matching the TraceV2 header the serving pipeline writes, so an
	// operator does not translate it.
	//
	// The key space is OPEN on purpose: an SLO class is a user-defined admission class
	// (the source system carries whatever names an operator configured — "critical",
	// "sheddable", and so on), not a vocabulary this schema owns, so the keys are not
	// checked against a closed set. Which classes are legal, and how admission treats
	// them, is a deployment-policy concern (the S2 surface), not a property of the trace a
	// workload references.
	GoodputSLOTargets map[string]SLODimTargets `yaml:"goodput_slo_targets,omitempty"`
}

// Mode is which pipeline produced a trace. The set is closed: an unrecognized mode is an
// error, because the distinction between an observed corpus and a synthetic one is a
// property these documents exist to preserve, and it survives only if the words for it
// cannot be invented at the call site.
type Mode string

const (
	// ModeReal: a real server's latencies observed (blis observe).
	ModeReal Mode = "real"
	// ModeGenerated: a workload synthesized from a shape or converted from an external
	// trace (blis run / blis convert).
	ModeGenerated Mode = "generated"
	// ModeReplayed: a trace replayed through the simulator (blis replay).
	ModeReplayed Mode = "replayed"
)

var modes = map[Mode]bool{ModeReal: true, ModeGenerated: true, ModeReplayed: true}

// Valid reports whether m is a recognized mode.
func (m Mode) Valid() bool { return modes[m] }

// AllModes returns every mode, sorted, for an error message that names what was allowed
// rather than only what was rejected.
func AllModes() []string { return enumerate(modes) }

// TimeUnit is the unit of a trace's timestamp columns. The set is closed, but it carries
// more than one member per physical unit on purpose: the tools that write traces spell
// microseconds as both "us" and "microseconds", and a schema that rejected one of its own
// producers' spellings would be wrong, where one that accepts any string would not be a
// vocabulary at all.
type TimeUnit string

const (
	TimeUnitNanoseconds      TimeUnit = "ns"
	TimeUnitNanosecondsLong  TimeUnit = "nanoseconds"
	TimeUnitMicroseconds     TimeUnit = "us"
	TimeUnitMicrosecondsLong TimeUnit = "microseconds"
	TimeUnitMilliseconds     TimeUnit = "ms"
	TimeUnitMillisecondsLong TimeUnit = "milliseconds"
	TimeUnitSeconds          TimeUnit = "s"
	TimeUnitSecondsLong      TimeUnit = "seconds"
)

var timeUnits = map[TimeUnit]bool{
	TimeUnitNanoseconds: true, TimeUnitNanosecondsLong: true,
	TimeUnitMicroseconds: true, TimeUnitMicrosecondsLong: true,
	TimeUnitMilliseconds: true, TimeUnitMillisecondsLong: true,
	TimeUnitSeconds: true, TimeUnitSecondsLong: true,
}

// Valid reports whether u is a recognized time unit.
func (u TimeUnit) Valid() bool { return timeUnits[u] }

// AllTimeUnits returns every recognized time unit, sorted, for an error that names what
// was allowed.
func AllTimeUnits() []string { return enumerate(timeUnits) }

// enumerate returns the members of a closed string vocabulary, sorted, for error text.
func enumerate[T ~string](set map[T]bool) []string {
	out := make([]string, 0, len(set))
	for v := range set {
		out = append(out, string(v))
	}
	sort.Strings(out)
	return out
}

// TraceServer is the configuration of the server that produced a trace. The field names
// follow a serving engine's own configuration surface, so an operator who launched the
// server does not translate them. Every field is optional: a producer that did not record
// a setting omits it rather than inventing a default.
//
// The integer fields are plain ints, not pointers, because zero is not a meaningful value
// for any of them — a tensor-parallel width, block size, sequence cap or context length
// of 0 describes no server — so an omitted field decoding to 0 reads unambiguously as
// "unset", and validation only rejects a negative. This is the same convention, and the
// same zero-is-unset sentinel, that deployment.Engine uses for the same quantities; the
// pointer treatment (see WorkloadSeed) is reserved for fields where a recorded 0 differs from an
// absent one.
type TraceServer struct {
	Type                 string  `yaml:"type,omitempty"`
	Model                string  `yaml:"model,omitempty"`
	TensorParallel       int     `yaml:"tensor_parallel,omitempty"`
	MaxNumSeqs           int     `yaml:"max_num_seqs,omitempty"`
	BlockSize            int     `yaml:"block_size,omitempty"`
	GPUMemoryUtilization float64 `yaml:"gpu_memory_utilization,omitempty"`
	// MaxModelLen is int, not int64, to match deployment.Engine.MaxModelLen: the two
	// describe the same quantity (a context length, in the millions at most), and a
	// schema that typed one field wider than its sibling would invite a reader to think
	// the two meant different things.
	MaxModelLen int `yaml:"max_model_len,omitempty"`
}

// SLODimTargets is one SLO class's latency thresholds, in milliseconds. A zero on any
// dimension means that dimension is unconstrained for the class — no completion has zero
// latency, so zero carries no other meaning — which is why these are bare floats rather
// than pointers.
type SLODimTargets struct {
	TTFTMs float64 `yaml:"ttft_ms,omitempty"`
	ITLMs  float64 `yaml:"itl_ms,omitempty"`
	E2EMs  float64 `yaml:"e2e_ms,omitempty"`
}
