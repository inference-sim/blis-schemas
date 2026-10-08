package deployment

import (
	"math"
	"math/big"
	"math/rand"
	"strings"
	"testing"
)

// These tests state the validation rules as properties over many inputs, through the
// public validators only. An oracle here is an independent model of what the ENGINE
// does — never a copy of the code under test — and a metamorphic test relates two
// documents whose verdicts must agree or move one way, without needing an oracle at all.
// Every verdict is read from the problems a user would see: the path a problem is filed
// at, and for the rank-fit rule the message, since its per-node bound shares a path with
// older dp_local checks.
//
// Random inputs come from fixed seeds, so a failure reproduces exactly.

// --- what a user sees -------------------------------------------------------------

func dcpRejected(pl Parallelism) bool {
	d := singleNodeDeployment()
	d.Pools[0].Parallel = pl
	for _, p := range d.Validate().Errors() {
		if p.Path == "pools[0].parallel.dcp" {
			return true
		}
	}
	return false
}

// fitVerdict reports which of the two rank-fit bounds refused a layout.
func fitVerdict(pl Parallelism, nodes, gpusPerNode int) (pool, node bool) {
	for _, p := range rankFit(pl, nodes, gpusPerNode).Errors() {
		switch {
		case p.Path == "pools[0].parallel" && strings.Contains(p.Message, "GPUs (pp"):
			pool = true
		case p.Path == "pools[0].parallel.dp_local" && strings.Contains(p.Message, "would start at device"):
			node = true
		}
	}
	return pool, node
}

// --- engine models ------------------------------------------------------------------

// engineAcceptsDCP is the tp/pcp/dcp block of vllm/config/parallel.py, transcribed from
// the engine's source (lines 563-577 at 119937c6f1; 544-561 at 700a10be0b, identical).
// The engine's fields are constrained ge=1, so it is defined only for widths of one or
// more.
func engineAcceptsDCP(tp, pcp, dcp int) bool {
	if pcp == 1 {
		return tp%dcp == 0 // "DCP reuses the TP ranks when PCP is disabled."
	}
	return dcp == 1 || dcp == pcp || dcp == tp*pcp
}

// enginePlacesOnOneNode simulates get_physical_gpu_ids_for_local_dp_rank: local replica
// i is given devices [i*world, i*world + world/k), where k is the number of nodes one
// replica's ranks are spread over (k divides world). It is written as a search over every
// split, device range by device range, so it shares no arithmetic with the rule.
func enginePlacesOnOneNode(world, dpLocal, gpusPerNode int) bool {
	for k := 1; k <= world; k++ {
		if world%k != 0 {
			continue
		}
		perNode, fits := world/k, true
		for i := 0; i < dpLocal && fits; i++ {
			fits = i*world+perNode <= gpusPerNode
		}
		if fits {
			return true
		}
	}
	return false
}

// --- differential: the rules agree with the engine ---------------------------------

// TestDCPMatchesTheEngine compares every verdict over a grid wider than any deployment
// uses, not a sample: tp 1-16, pcp 1-8, dcp 1-64.
func TestDCPMatchesTheEngine(t *testing.T) {
	cases, mismatches := 0, 0
	for tp := 1; tp <= 16; tp++ {
		for pcp := 1; pcp <= 8; pcp++ {
			for dcp := 1; dcp <= 64; dcp++ {
				cases++
				got := !dcpRejected(Parallelism{TP: tp, PP: 1, DP: 1, PCP: pcp, DCP: dcp})
				if want := engineAcceptsDCP(tp, pcp, dcp); got != want {
					mismatches++
					if mismatches <= 5 {
						t.Errorf("tp %d pcp %d dcp %d: validator accepts=%v, engine accepts=%v",
							tp, pcp, dcp, got, want)
					}
				}
			}
		}
	}
	if cases != 8192 {
		t.Fatalf("grid covered %d cases, want 8192", cases)
	}
}

// TestPerNodeBoundMatchesEnginePlacement: the per-node bound refuses a layout exactly
// when no node split lets the engine place every local replica on one node — so it
// never refuses a layout the engine can run, and never admits one it cannot.
func TestPerNodeBoundMatchesEnginePlacement(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	refused := 0
	const n = 20000
	for i := 0; i < n; i++ {
		gpn := []int{1, 2, 4, 8, 16}[r.Intn(5)]
		dp := 1 + r.Intn(16)
		pl := Parallelism{TP: 1 + r.Intn(16), PP: 1 + r.Intn(3), DP: dp, PCP: r.Intn(3),
			DPLocal: 1 + r.Intn(dp)}
		world := pl.TP * pl.PP * max(pl.PCP, 1)
		want := !enginePlacesOnOneNode(world, pl.DPLocal, gpn)
		_, got := fitVerdict(pl, 1+r.Intn(64), gpn)
		if got {
			refused++
		}
		if got != want {
			t.Fatalf("%+v on %d-GPU nodes (world %d): validator refuses=%v, engine cannot place=%v",
				pl, gpn, world, got, want)
		}
	}
	// Guard against a generator that only ever produces one verdict.
	if refused == 0 || refused == n {
		t.Fatalf("degenerate sample: %d of %d refused", refused, n)
	}
}

// TestPoolBoundMatchesExactRankCount: the pool bound refuses exactly the layouts whose
// rank count exceeds the pool's GPUs, computed in arbitrary precision — including widths
// and pools large enough to overflow a 64-bit product on either side.
func TestPoolBoundMatchesExactRankCount(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	width := func() int {
		switch r.Intn(5) {
		case 0:
			return math.MaxInt - r.Intn(3)
		case 1:
			return 1 + r.Intn(300)
		default:
			return 1 + r.Intn(8)
		}
	}
	refused := 0
	const n = 20000
	for i := 0; i < n; i++ {
		pl := Parallelism{TP: width(), PP: width(), DP: width(), PCP: r.Intn(5)}
		nodes, gpn := width(), []int{1, 2, 4, 8}[r.Intn(4)]
		need := big.NewInt(1)
		for _, f := range []int{pl.TP, pl.PP, max(pl.PCP, 1), pl.DP} {
			need.Mul(need, big.NewInt(int64(f)))
		}
		have := new(big.Int).Mul(big.NewInt(int64(nodes)), big.NewInt(int64(gpn)))
		want := need.Cmp(have) > 0
		got, _ := fitVerdict(pl, nodes, gpn)
		if got {
			refused++
		}
		if got != want {
			t.Fatalf("%+v on %d x %d: validator refuses=%v, needs %s of %s", pl, nodes, gpn, got, need, have)
		}
	}
	if refused == 0 || refused == n {
		t.Fatalf("degenerate sample: %d of %d refused", refused, n)
	}
}

// --- metamorphic: relations between documents --------------------------------------

// TestOmittedWidthIsTheSameAsOne: pcp and dcp are omitempty, so an absent one arrives as
// zero meaning "not enabled". A document that omits either must be judged exactly as one
// that states 1 — the same problems, word for word — under both validators.
func TestOmittedWidthIsTheSameAsOne(t *testing.T) {
	report := func(pl Parallelism, nodes int) string {
		d := singleNodeDeployment()
		d.Pools[0].Nodes = nodes
		d.Pools[0].Parallel = pl
		return d.Validate().Error() + "\n--\n" +
			d.ValidateAgainstCluster(ClusterConstraints{Nodes: nodes, GPUsPerNode: 8}).Error()
	}
	for tp := 1; tp <= 16; tp++ {
		for dp := 1; dp <= 4; dp++ {
			for nodes := 1; nodes <= 4; nodes++ {
				for _, dcp := range []int{0, 1, 2, 3, 4, 8} {
					omitted := report(Parallelism{TP: tp, PP: 1, DP: dp, PCP: 0, DCP: dcp}, nodes)
					stated := report(Parallelism{TP: tp, PP: 1, DP: dp, PCP: 1, DCP: dcp}, nodes)
					if omitted != stated {
						t.Fatalf("tp %d dp %d dcp %d on %d nodes: pcp omitted and pcp 1 differ:\n%s\nvs\n%s",
							tp, dp, dcp, nodes, omitted, stated)
					}
				}
				for _, pcp := range []int{0, 1, 2, 4} {
					omitted := report(Parallelism{TP: tp, PP: 1, DP: dp, PCP: pcp, DCP: 0}, nodes)
					stated := report(Parallelism{TP: tp, PP: 1, DP: dp, PCP: pcp, DCP: 1}, nodes)
					if omitted != stated {
						t.Fatalf("tp %d dp %d pcp %d on %d nodes: dcp omitted and dcp 1 differ:\n%s\nvs\n%s",
							tp, dp, pcp, nodes, omitted, stated)
					}
				}
			}
		}
	}
}

// randomLayout draws a well-formed layout and room for it, biased so that both verdicts
// are common.
func randomLayout(r *rand.Rand) (Parallelism, int, int) {
	dp := 1 + r.Intn(16)
	pl := Parallelism{TP: 1 + r.Intn(16), PP: 1 + r.Intn(4), DP: dp, PCP: r.Intn(4)}
	if r.Intn(2) == 0 {
		pl.DPLocal = 1 + r.Intn(dp)
	}
	return pl, 1 + r.Intn(16), []int{1, 2, 4, 8}[r.Intn(4)]
}

// TestMoreRoomNeverAddsARankFitProblem: whatever fits a pool fits one with more nodes, and
// one with nodes twice as large. (Doubling keeps every dp_local that divided a node
// dividing it, so the older divisibility check cannot interfere.)
func TestMoreRoomNeverAddsARankFitProblem(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	for i := 0; i < 20000; i++ {
		pl, nodes, gpn := randomLayout(r)
		pool, node := fitVerdict(pl, nodes, gpn)
		for _, room := range [][2]int{{nodes + 1 + r.Intn(8), gpn}, {nodes, gpn * 2}} {
			p2, n2 := fitVerdict(pl, room[0], room[1])
			if (p2 && !pool) || (n2 && !node) {
				t.Fatalf("%+v: fits %d x %d but not %d x %d", pl, nodes, gpn, room[0], room[1])
			}
		}
	}
}

// TestWiderLayoutsNeverLoseARankFitProblem: whatever does not fit still does not fit when
// any width grows. Growing dp alone cannot move the per-node bound (it reads dp_local and
// the replica), so that bound is only checked for the replica's own widths.
func TestWiderLayoutsNeverLoseARankFitProblem(t *testing.T) {
	r := rand.New(rand.NewSource(13))
	grow := map[string]func(*Parallelism){
		"tp":  func(p *Parallelism) { p.TP++ },
		"pp":  func(p *Parallelism) { p.PP++ },
		"pcp": func(p *Parallelism) { p.PCP = max(p.PCP, 1) + 1 },
		"dp":  func(p *Parallelism) { p.DP++ },
	}
	for i := 0; i < 20000; i++ {
		pl, nodes, gpn := randomLayout(r)
		pool, node := fitVerdict(pl, nodes, gpn)
		for axis, g := range grow {
			wider := pl
			g(&wider)
			p2, n2 := fitVerdict(wider, nodes, gpn)
			if pool && !p2 {
				t.Fatalf("%+v exceeds %d x %d, but growing %s made it fit", pl, nodes, gpn, axis)
			}
			if node && !n2 && axis != "dp" {
				t.Fatalf("%+v cannot be placed on a %d-GPU node, but growing %s placed it", pl, gpn, axis)
			}
		}
	}
}

// TestDCPAndExpertParallelismAddNoRanks: neither changes how many devices a layout
// needs — DCP reuses the TP ranks, an expert group spans ranks that already exist — so
// neither may change a rank-fit verdict.
func TestDCPAndExpertParallelismAddNoRanks(t *testing.T) {
	r := rand.New(rand.NewSource(17))
	for i := 0; i < 20000; i++ {
		pl, nodes, gpn := randomLayout(r)
		pool, node := fitVerdict(pl, nodes, gpn)
		for _, v := range []Parallelism{
			func() Parallelism { q := pl; q.DCP = q.TP; return q }(),
			func() Parallelism { q := pl; q.DCP = 1 + r.Intn(64); return q }(),
			func() Parallelism { q := pl; q.EnableExpertParallel = !q.EnableExpertParallel; return q }(),
		} {
			if p2, n2 := fitVerdict(v, nodes, gpn); p2 != pool || n2 != node {
				t.Fatalf("%+v vs %+v on %d x %d: verdict changed from (%v, %v) to (%v, %v)",
					pl, v, nodes, gpn, pool, node, p2, n2)
			}
		}
	}
}

// TestReplicaAxesAreInterchangeable: a replica's footprint is pp x tp x pcp, so permuting
// the three widths among themselves cannot change either verdict.
func TestReplicaAxesAreInterchangeable(t *testing.T) {
	r := rand.New(rand.NewSource(19))
	for i := 0; i < 20000; i++ {
		pl, nodes, gpn := randomLayout(r)
		pl.PCP = 1 + r.Intn(4) // a stated width, so it can trade places with the others
		pool, node := fitVerdict(pl, nodes, gpn)
		for _, perm := range [][3]int{{pl.PP, pl.TP, pl.PCP}, {pl.PCP, pl.PP, pl.TP}, {pl.TP, pl.PCP, pl.PP}} {
			q := pl
			q.TP, q.PP, q.PCP = perm[0], perm[1], perm[2]
			if p2, n2 := fitVerdict(q, nodes, gpn); p2 != pool || n2 != node {
				t.Fatalf("%+v vs %+v on %d x %d: verdict changed under permutation", pl, q, nodes, gpn)
			}
		}
	}
}

// TestScalingReplicasWithTheirNodesPreservesFit: k times the replicas on k times the
// nodes needs exactly k times the GPUs it is given, so the pool verdict cannot change.
func TestScalingReplicasWithTheirNodesPreservesFit(t *testing.T) {
	r := rand.New(rand.NewSource(23))
	for i := 0; i < 20000; i++ {
		pl, nodes, gpn := randomLayout(r)
		pl.DPLocal = 0 // the per-node bound is about one node; scaling is about the pool
		pool, _ := fitVerdict(pl, nodes, gpn)
		k := 2 + r.Intn(7)
		scaled := pl
		scaled.DP *= k
		if p2, _ := fitVerdict(scaled, nodes*k, gpn); p2 != pool {
			t.Fatalf("%+v on %d x %d fits=%v, but x%d replicas on x%d nodes fits=%v",
				pl, nodes, gpn, !pool, k, k, !p2)
		}
	}
}
