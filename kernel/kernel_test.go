package kernel

import (
	"reflect"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// stub implements Kernel to the letter and nothing more. Its purpose is to fail
// COMPILATION if a method is added to the interface without a deliberate decision,
// and to prove the facts a consumer needs are reachable through the interface rather
// than only through a concrete implementation. That was the defect behind this test:
// a consumer that needed the resolved widths and the admission settings had to
// depend on one implementation's concrete type, which defeats the reason the
// interface exists — a second cost model could not satisfy it.
type stub struct {
	res  Resolution
	pool deployment.Pool
}

func (stub) FixedBytes() MemoryBreakdown                            { return MemoryBreakdown{} }
func (stub) SequenceFixedBytes() int64                              { return 0 }
func (stub) SequenceVariableBytes(int) int64                        { return 0 }
func (stub) StepTime(Batch) StepEstimate                            { return StepEstimate{} }
func (stub) TierTime(string, Direction, int64, int) time.Duration   { return 0 }
func (stub) PDTransferTime(int, Placement, Placement) time.Duration { return 0 }
func (stub) AdmissionOverhead(int) time.Duration                    { return 0 }
func (stub) OutputTokenOverhead() time.Duration                     { return 0 }
func (stub) CompletionOverhead() time.Duration                      { return 0 }
func (stub) Provenance() []CoefficientOrigin                        { return nil }
func (s stub) Resolved() Resolution                                 { return s.res }
func (s stub) Deployment() deployment.Pool                          { return s.pool }

var _ Kernel = stub{}

// TestConsumerNeedsOnlyTheInterface is the regression this change exists for. It
// reads, through the interface alone, every fact the consumer that hit the gap had
// to reach into a concrete type for: both resolved widths and the three engine
// settings that decide admission. If any of these ever stops being reachable from a
// kernel.Kernel, this fails to compile.
func TestConsumerNeedsOnlyTheInterface(t *testing.T) {
	var k Kernel = stub{
		res: Resolution{TensorParallelWidth: 8, DataParallelWidth: 4, ExpertParallelWidth: 32},
		pool: deployment.Pool{
			Role:     deployment.RoleDecode,
			Parallel: deployment.Parallelism{TP: 8, PP: 1, DP: 4, EnableExpertParallel: true},
			Engine:   deployment.Engine{BlockSize: 64, MaxNumSeqs: 256, MaxNumBatchedTokens: 8192},
		},
	}

	r := k.Resolved()
	if r.TensorParallel() != 8 || r.DataParallel() != 4 || r.ExpertParallel() != 32 {
		t.Errorf("resolved widths = (%d, %d, %d), want (8, 4, 32)",
			r.TensorParallel(), r.DataParallel(), r.ExpertParallel())
	}

	pool := k.Deployment()
	if pool.Role != deployment.RoleDecode {
		t.Errorf("role = %q, want %q", pool.Role, deployment.RoleDecode)
	}
	if pool.Engine.BlockSize != 64 || pool.Engine.MaxNumSeqs != 256 ||
		pool.Engine.MaxNumBatchedTokens != 8192 {
		t.Errorf("admission settings not reachable through the interface: %+v", pool.Engine)
	}
}

// TestWidthFloors pins the floor every width accessor applies. The fields exist so an
// implementation can state a width; the accessors exist because a reader of a
// zero-valued Resolution would otherwise multiply by zero. A scheduler scales its
// sequence cap, token budget and KV budget by the data-parallel width, so a zero
// there makes all three zero and the simulation runs on a deployment that admits
// nothing — a wrong answer rather than an error.
func TestWidthFloors(t *testing.T) {
	// The zero value is the case that matters: adding fields to Resolution breaks no
	// existing construction precisely because they default to zero, so every reader
	// must survive one.
	var zero Resolution
	if zero.TensorParallel() != 1 || zero.DataParallel() != 1 || zero.ExpertParallel() != 1 {
		t.Errorf("zero Resolution widths = (%d, %d, %d), want all 1",
			zero.TensorParallel(), zero.DataParallel(), zero.ExpertParallel())
	}
	// A negative is nonsense rather than unset, but it reaches the same multipliers,
	// so it gets the same floor rather than propagating a sign.
	neg := Resolution{TensorParallelWidth: -2, DataParallelWidth: -2, ExpertParallelWidth: -2}
	if neg.TensorParallel() != 1 || neg.DataParallel() != 1 || neg.ExpertParallel() != 1 {
		t.Errorf("negative widths = (%d, %d, %d), want all 1",
			neg.TensorParallel(), neg.DataParallel(), neg.ExpertParallel())
	}
	// A stated width is returned unchanged; the floor must not clamp anything real.
	set := Resolution{TensorParallelWidth: 8, DataParallelWidth: 16, ExpertParallelWidth: 128}
	if set.TensorParallel() != 8 || set.DataParallel() != 16 || set.ExpertParallel() != 128 {
		t.Errorf("stated widths = (%d, %d, %d), want (8, 16, 128)",
			set.TensorParallel(), set.DataParallel(), set.ExpertParallel())
	}
}

// TestExpertWidthIsNotRecomputableFromTheOtherTwo pins why ExpertParallelWidth is
// reported rather than derived on read. It would be tempting to drop the field and
// compute tp x dp from the other two, and for many layouts that gives the right
// answer — which is exactly what makes it a trap.
//
// Two facts defeat it, and each is a layout a deployment can state:
//
//   - Expert parallelism OFF. The width is 1 however wide tp and dp are, and
//     Resolution does not carry the enable flag, so a reader cannot tell.
//   - A prefill-context width above dp. The derivation takes tp x max(dp, pcp), and
//     Resolution does not carry pcp either.
//
// So the assertion here is a DISAGREEMENT: for each layout below, the naive tp x dp
// differs from the real width. If a future change made the two agree everywhere, the
// field would be redundant and this test should be revisited rather than deleted.
func TestExpertWidthIsNotRecomputableFromTheOtherTwo(t *testing.T) {
	cases := []struct {
		name string
		pl   deployment.Parallelism
	}{
		{"expert parallelism off", deployment.Parallelism{TP: 8, PP: 1, DP: 4}},
		{"pcp wider than dp", deployment.Parallelism{TP: 2, PP: 1, DP: 1, PCP: 4,
			EnableExpertParallel: true}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			derived := c.pl.ExpertParallelWidth()
			if naive := c.pl.TP * c.pl.DP; naive == derived {
				t.Fatalf("this case no longer distinguishes the derivation: tp*dp = %d = %d",
					naive, derived)
			}
			// What a faithful resolver reports: the derived width, not a recomputation.
			r := Resolution{
				TensorParallelWidth: c.pl.TP,
				DataParallelWidth:   c.pl.DP,
				ExpertParallelWidth: derived,
			}
			if got := r.ExpertParallel(); got != derived {
				t.Errorf("reported expert width %d, derived %d", got, derived)
			}
		})
	}
}

// TestResolutionCarriesTheWidthsSeparately pins the asymmetry this change removed.
// Before it, a consumer could read the expert width from a Resolution but had to go
// elsewhere for tensor- and data-parallel width, and "elsewhere" meant either
// re-deriving the resolver's logic or reading the Deployment and getting the request.
// Reflection rather than a field read, because the point is that the type DECLARES
// all three: a field read would still compile if two of them were removed and
// replaced by accessors over the request.
func TestResolutionCarriesTheWidthsSeparately(t *testing.T) {
	ty := reflect.TypeOf(Resolution{})
	for _, name := range []string{"TensorParallelWidth", "DataParallelWidth", "ExpertParallelWidth"} {
		f, ok := ty.FieldByName(name)
		if !ok {
			t.Errorf("Resolution does not declare %s; a consumer must not have to re-derive it", name)
			continue
		}
		if f.Type.Kind() != reflect.Int {
			t.Errorf("%s is %s, want int", name, f.Type)
		}
	}
}
