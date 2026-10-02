// Package kernel is the interface a cost model implements and a discrete-event
// simulator calls.
//
// Every method is a pure function of its arguments and the resolved configuration:
// same inputs, same outputs, no internal mutation, safe for concurrent use. That is
// not a style preference. It is what lets the kernel be called at any simulated
// instant, tested without a scheduler attached, and memoized by batch shape.
//
// The boundary follows from purity. Anything whose value depends on WHEN a request
// arrives belongs to the simulator: queue waiting, admission ordering, preemption.
// Anything that follows from a batch and the deployment belongs here. A KV transfer
// is the instructive case — it is concurrent with execution and gates when a request
// becomes schedulable, so it sets an arrival time rather than a step time, and a
// kernel that added it to step time would charge for latency the engine already hid.
package kernel

import (
	"time"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// Kernel prices one instantiated deployment. Obtain one from a constructor that
// resolves a scenario; there is deliberately no Init method on the interface,
// because an interface-level Init admits an unconstructed value and makes every
// other method's behaviour depend on whether it was called — the mutable state
// purity rules out.
type Kernel interface {
	// --- Memory occupancy ---------------------------------------------------

	// FixedBytes is occupancy independent of the request set, per rank.
	FixedBytes() MemoryBreakdown

	// SequenceFixedBytes is per-sequence occupancy that does not vary with prompt
	// length: recurrent state, per-sequence bookkeeping. Zero for a pure-attention
	// model. It is computed from the layout rather than stored, because it divides
	// with tensor-parallel width and its group count pads up to divide that width.
	SequenceFixedBytes() int64

	// SequenceVariableBytes is per-sequence occupancy as a function of token count,
	// quantized to the engine's page size. Two settings divide it: tensor-parallel
	// width, down to a floor of one KV head, and the cache dtype.
	SequenceVariableBytes(tokens int) int64

	// --- Step time -----------------------------------------------------------

	// StepTime prices one forward pass over a batch.
	StepTime(Batch) StepEstimate

	// --- Transfers -----------------------------------------------------------

	// TierTime prices one transfer between GPU and an offload tier. Two links are
	// crossed on a CPU tier — the host interconnect and host memory — and the slower
	// one binds, so an implementation takes the minimum rather than trusting either.
	TierTime(tier string, dir Direction, bytes int64, queueDepth int) time.Duration

	// PDTransferTime prices moving one request's KV between pools. The byte count is
	// not tokens times a constant: it depends on the attention kind and on which
	// parallelism axis shards KV.
	PDTransferTime(tokens int, from, to Placement) time.Duration

	// --- Host overheads ------------------------------------------------------
	// These price engine work around a request rather than GPU work inside a step.
	// They belong here because the cost model already carries a host term for kernel
	// launch, and splitting host cost across two owners would leave a caller
	// sourcing some coefficients from the kernel and others from elsewhere.

	// AdmissionOverhead is CPU time before a request can be scheduled:
	// tokenization, template application, multimodal preprocessing. It scales with
	// prompt length. It excludes queue waiting, which depends on engine busyness and
	// therefore cannot live in a pure function.
	AdmissionOverhead(promptTokens int) time.Duration

	// OutputTokenOverhead is per-emitted-token host work: incremental
	// detokenization and streaming. It runs in the output path rather than the
	// forward pass, which is why it does not enter StepTime.
	OutputTokenOverhead() time.Duration

	// CompletionOverhead is fixed per-request work at finish: finish-reason
	// determination, serialization, teardown. It lands on end-to-end latency and not
	// on time-to-first-token.
	CompletionOverhead() time.Duration

	// --- Provenance ----------------------------------------------------------

	// Provenance reports every coefficient this kernel used, with its evidence. A
	// prediction that cannot state what it rests on cannot be audited.
	Provenance() []CoefficientOrigin

	// Resolved reports the configuration after resolution, including requested
	// settings the layout overrode.
	Resolved() Resolution
}

// MemoryBreakdown decomposes fixed occupancy rather than returning a scalar,
// because a caller deciding between "shard further" and "raise utilization" needs
// to know which term dominates.
type MemoryBreakdown struct {
	Weights        int64 // parameters resident on this rank
	ActivationPeak int64 // transient scratch at the batched-token bound
	CUDAGraph      int64 // capture cost; zero when capture is off
	EPLBRedundant  int64 // redundant-expert replicas; zero when EPLB is off
	CommBuffers    int64 // collective and all-to-all buffers
}

// Total returns the sum. Callers that only need a feasibility answer use it; the
// fields exist for callers that need to act on the result.
func (m MemoryBreakdown) Total() int64 {
	return m.Weights + m.ActivationPeak + m.CUDAGraph + m.EPLBRedundant + m.CommBuffers
}

// Resource is the axis a cost lands on. A communication op appears as ResourceSM
// when the resolved backend is an SM-consuming kernel, which is why the resolved
// backend and not the requested one determines it.
type Resource string

const (
	ResourceSM         Resource = "sm"
	ResourceHBM        Resource = "hbm"
	ResourceNVLink     Resource = "nvlink"
	ResourceNIC        Resource = "nic"
	ResourcePCIe       Resource = "pcie"
	ResourceCopyEngine Resource = "copy_engine"
	ResourceHost       Resource = "host"
)

// StepEstimate brackets a step and names its bottleneck.
type StepEstimate struct {
	// Overlap assumes perfect overlap WITHIN a stage; NoOverlap assumes none anywhere.
	//
	// The qualifier is the whole content of the estimate. A stage is a set of operations
	// that really do run concurrently — one layer, since the next layer needs this one's
	// output — so Overlap takes the max over resources inside a stage and sums across
	// stages. It is not a max over the step: that would assume every layer overlaps every
	// other, which understates a step badly wherever no single resource dominates, and
	// single-request decode is exactly that case.
	//
	// Reporting both matters because a configuration whose verdict flips between them
	// needs measurement rather than prediction, and a single number hides that.
	Overlap   time.Duration
	NoOverlap time.Duration

	// Bottleneck is the resource that did the most work over the step. It distinguishes
	// "add GPUs" from "raise the batch size" as the next action, and the answer changes
	// with batch size within one deployment.
	//
	// Not "the resource that set Overlap", which it cannot be. A step composes per stage
	// — max over resources within a stage, sum across stages, since one layer's output is
	// the next layer's input — so Overlap is a sum of per-stage maxima and no single
	// resource's total equals it. Which resource to relieve is the actionable question,
	// and answering it does not require that one resource dominate every stage.
	Bottleneck Resource

	// PerResource is the per-resource sum, for a caller that needs to see why.
	PerResource map[Resource]time.Duration
}

// Batch is what the simulator must supply: per-request shape plus the engine and
// resource state a cost depends on. Nothing here is derivable from the deployment
// alone, which is why each field exists.
type Batch struct {
	Reqs []ReqShape

	// DecodeThreshold selects the attention kernel per request. It is engine
	// configuration rather than request state, and the classifier needs both.
	DecodeThreshold int

	// SMBudget is the SM count available this step: the chip's count, less any
	// withheld by a concurrent offload fetch. It is absorbed entirely when a step is
	// bandwidth-bound, which is the per-stage max rule working as intended.
	SMBudget int
}

// ReqShape is one request's contribution. The three token counts are what an
// engine's own region classifier reads; passing a region enum instead would move an
// engine implementation detail into the caller.
type ReqShape struct {
	// Scheduled is tokens scheduled this step. Under speculation it is the draft
	// length plus one, so a value of one does not imply a decode.
	Scheduled int
	// Computed is tokens already computed. An engine advances it optimistically at
	// dispatch and rolls it back on speculative rejection, so a simulator passes the
	// value as of batch formation.
	Computed int
	// PromptLen is the prompt length. Computed against it is what separates a
	// chunked-prefill tail from a decode, and the two use different kernels.
	PromptLen int
	// CachedTokens is the prefix-cache hit, already subtracted from Scheduled. A
	// count rather than a rate: the simulator has computed it, and a cost model that
	// took a rate would have to guess.
	CachedTokens int
}

// Tokens returns the batch's total scheduled token count, the quantity collective
// thresholds compare against.
func (b Batch) Tokens() int {
	n := 0
	for _, r := range b.Reqs {
		n += r.Scheduled
	}
	return n
}

// UniformDecode reports whether every request schedules the same token count, equal
// to the uniform decode width. It decides which of two collective thresholds
// applies, and a single partially-rejected speculative request breaks it.
func (b Batch) UniformDecode(uniformWidth int) bool {
	if len(b.Reqs) == 0 {
		return false
	}
	for _, r := range b.Reqs {
		if r.Scheduled != uniformWidth {
			return false
		}
	}
	return true
}

// Direction distinguishes an offload write from a read back. They are separate
// because device read and write bandwidths differ materially on flash, and pricing
// an eviction at read bandwidth understates it.
type Direction string

const (
	DirectionToTier   Direction = "to_tier"
	DirectionFromTier Direction = "from_tier"
)

// Placement locates one rank. Comparing two placements is how a transfer's link is
// chosen: same node, same rack, or across the fabric. A two-tier placement cannot
// express a rack boundary, which is why Rack is present.
type Placement struct {
	Node int
	Rack int
	Pool deployment.Role
}

// CoefficientOrigin is one row of the provenance trail.
type CoefficientOrigin struct {
	Name   string // registry entry name
	Set    string // the set it came from
	Method string // the evidence vocabulary's term
	Scope  string // the scope as recorded
}

// Resolution reports what the constructor decided where a deployment expressed a
// request. It exists so a prediction can be read without re-deriving the resolver's
// logic, and so a reader is never misled by a field the layout overrode.
type Resolution struct {
	ExpertParallelWidth int
	// AllReduceBackend is what will run, which may differ from what was requested.
	AllReduceBackend string
	// AsyncScheduling is the resolved value of a tri-state request.
	AsyncScheduling bool
	// CascadeAttention is whether it is available after every gate is applied.
	CascadeAttention bool
	// SequenceParallelMoE is whether the MoE input is made sequence-parallel, which
	// replaces an all-reduce with a reduce-scatter and all-gather pair.
	SequenceParallelMoE bool
	// Overrides records each request the layout could not honour, and why.
	Overrides []Override
}

// Override is one request the resolver declined, with the reason.
type Override struct {
	Field     string
	Requested string
	Resolved  string
	Reason    string
}
