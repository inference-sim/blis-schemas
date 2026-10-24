package enginetest

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/engine"
)

func fifoCfg() FIFOConfig {
	return FIFOConfig{MaxRunning: 2, TokenBudget: 8, StepBaseUs: 10, StepPerTokenUs: 1}
}

func newFIFO(cfg FIFOConfig) func() (engine.Engine, error) {
	return func() (engine.Engine, error) { return NewFIFO(cfg) }
}

func req(id string, prompt, output int) engine.Request {
	r := engine.Request{ID: id, Prompt: make([]int32, prompt), Output: make([]int32, output)}
	for i := range r.Prompt {
		r.Prompt[i] = int32(i + 1)
	}
	return r
}

func at(t engine.Time, r engine.Request) Arrival { return Arrival{At: t, Req: r} }

// The example from #2021 B8: token budget 8; A has 10 prompt tokens and B 6, and each makes
// 2 tokens. Step 1 schedules A 8; step 2 A 2 and B 6 (both prefills complete, first tokens);
// step 3 one decode each, and both finish.
func TestFIFOReproducesTheIssueExample(t *testing.T) {
	tr := TestEngine(t, newFIFO(fifoCfg()), []Arrival{at(0, req("A", 10, 2)), at(0, req("B", 6, 2))}, Options{})
	want := []string{"A:8", "A:2 B:6", "A:1 B:1"}
	if len(tr.Steps) != len(want) {
		t.Fatalf("%d steps, want %d: %+v", len(tr.Steps), len(want), tr.Steps)
	}
	for i, s := range tr.Steps {
		var parts []string
		for _, r := range s.Batch.Reqs {
			parts = append(parts, fmt.Sprintf("%s:%d", r.ID, r.Tokens))
		}
		if got := strings.Join(parts, " "); got != want[i] {
			t.Errorf("step %d schedules %q, want %q", i+1, got, want[i])
		}
	}
	for _, id := range []string{"A", "B"} {
		if f := tr.Finished[id]; f.Reason != engine.FinishStop || f.OutputTokens != 2 {
			t.Errorf("%s finished %+v, want stop after 2 tokens", id, f)
		}
	}
}

// A request held until an instant is admitted at that instant, and only through the wake:
// nothing else happens in between.
func TestTheWakeIsTheOnlyWayAHeldRequestRuns(t *testing.T) {
	r := req("held", 4, 1)
	r.Tags = map[string]string{"fifo.hold_until": "500"}
	tr := TestEngine(t, newFIFO(fifoCfg()), []Arrival{at(0, r)}, Options{})
	if len(tr.Steps) == 0 || tr.Steps[0].Start != 500 {
		t.Fatalf("first step starts at %v, want 500 (the hold)", tr.Steps)
	}
}

// A request that loads KV first runs only after its transfer completes, at the instant the
// driver reports it.
func TestATransferGatesAdmission(t *testing.T) {
	r := req("loads", 4, 1)
	r.Tags = map[string]string{"fifo.load_tokens": "4"}
	tr := TestEngine(t, newFIFO(fifoCfg()), []Arrival{at(0, r)}, Options{
		TransferTime: func(engine.Transfer) engine.Duration { return 250 },
	})
	if len(tr.Steps) == 0 || tr.Steps[0].Start != 250 {
		t.Fatalf("first step starts at %v, want 250 (the transfer's end)", tr.Steps)
	}
}

// With two batches allowed in flight, the second is scheduled before the first completes,
// and they complete in order.
func TestTwoBatchesInFlight(t *testing.T) {
	cfg := fifoCfg()
	cfg.InFlight, cfg.MaxRunning, cfg.TokenBudget = 2, 1, 4
	tr := TestEngine(t, newFIFO(cfg), []Arrival{at(0, req("A", 4, 1)), at(0, req("B", 4, 1))}, Options{})
	if len(tr.Steps) < 2 || tr.Steps[1].Start != tr.Steps[0].End {
		t.Fatalf("steps %+v: the second batch should start as the first ends", tr.Steps)
	}
}

// Drafts accept as the outcome says, and a request never emits past its assigned output.
func TestDraftsAcceptAsSupplied(t *testing.T) {
	cfg := fifoCfg()
	cfg.Drafts = 3
	tr := TestEngine(t, newFIFO(cfg), []Arrival{at(0, req("A", 2, 9))}, Options{
		Accepted: func(engine.Batch, engine.Scheduled) int { return 2 },
	})
	// 1 output token from prefill, then 3 per decode (2 accepted + bonus): 1+3+3+2 = 9.
	if f := tr.Finished["A"]; f.OutputTokens != 9 || f.Reason != engine.FinishStop {
		t.Fatalf("A finished %+v, want stop after 9", f)
	}
	if n := len(tr.Steps); n != 4 {
		t.Errorf("%d steps, want 4 (prefill, then three decodes)", n)
	}
}

// A timeout finishes a request through Finish, and its later completions are ignored.
func TestATimeoutFinishesThroughFinish(t *testing.T) {
	tr := TestEngine(t, newFIFO(fifoCfg()), []Arrival{at(0, req("slow", 4, 50))}, Options{
		Deadlines: map[string]engine.Time{"slow": 100},
	})
	if f := tr.Finished["slow"]; f.Reason != engine.FinishTimeout {
		t.Fatalf("slow finished %+v, want a timeout", f)
	}
}

func TestProtocolErrors(t *testing.T) {
	// Two batches in flight at once: a budget of 4 admits A alone, and B in the second batch.
	cfg := fifoCfg()
	cfg.InFlight, cfg.TokenBudget = 2, 4
	if err := CheckProtocolErrors(newFIFO(cfg), req("A", 4, 2), req("B", 4, 2)); err != nil {
		t.Fatal(err)
	}
}

// The FIFO engine is built from an opaque spec through the registry, as a simulator builds
// any engine; its config decodes strictly.
func TestFIFOBuildsThroughTheRegistry(t *testing.T) {
	engine.Register(FIFOFactory{})
	var node yaml.Node
	if err := yaml.Unmarshal([]byte("max_running: 2\ntoken_budget: 8\nstep_base_us: 10\nstep_per_token_us: 1\n"), &node); err != nil {
		t.Fatal(err)
	}
	spec := engine.Spec{Kind: FIFOKind, Config: *node.Content[0]}
	build := func() (engine.Engine, error) { return engine.New(spec) }
	TestEngine(t, build, []Arrival{at(0, req("A", 10, 2)), at(0, req("B", 6, 2))}, Options{})

	var bad yaml.Node
	if err := yaml.Unmarshal([]byte("max_running: 2\ntoken_budget: 8\nmax_runnign: 3\n"), &bad); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.New(engine.Spec{Kind: FIFOKind, Config: *bad.Content[0]}); err == nil {
		t.Error("a misspelled key built an engine; the block must decode strictly")
	}
	if _, err := engine.New(engine.Spec{Kind: "no-such-engine"}); err == nil ||
		!strings.Contains(err.Error(), FIFOKind) {
		t.Errorf("an unknown kind returned %v; want an error naming the registered kinds", err)
	}
}

// --- The driver catches broken engines. Each is FIFO with one law broken. -------------

type mute struct{ *FIFO }

// NextWake never reports the hold, so a held request strands.
func (mute) NextWake(engine.Time) (engine.Time, bool) { return 0, false }

type reusesIDs struct{ *FIFO }

func (e reusesIDs) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.FIFO.Schedule(now)
	if !b.Empty() {
		e.FIFO.inFlight[len(e.FIFO.inFlight)-1].ID = 1
		b.ID = 1
	}
	return b, err
}

type ignoresLimit struct{ *FIFO }

func (e ignoresLimit) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.FIFO.Schedule(now)
	if errors.Is(err, engine.ErrNoCapacity) {
		return engine.Batch{}, nil
	}
	return b, err
}

type finishesTwice struct{ *FIFO }

func (e finishesTwice) Complete(id engine.BatchID, o engine.Outcome, end engine.Time) (engine.Result, error) {
	r, err := e.FIFO.Complete(id, o, end)
	r.Finished = append(r.Finished, r.Finished...)
	return r, err
}

// wakesNow reports a wake at the instant it is asked, which the contract forbids.
type wakesNow struct{ *FIFO }

func (e wakesNow) NextWake(now engine.Time) (engine.Time, bool) {
	if _, ok := e.FIFO.NextWake(now); ok {
		return now, true
	}
	return 0, false
}

// freeSteps prices a step at zero, so time would not advance.
type freeSteps struct{ *FIFO }

func (freeSteps) StepTime(engine.Batch) engine.Duration { return 0 }

// finishesAStranger reports a request it was never given as finished.
type finishesAStranger struct{ *FIFO }

func (e finishesAStranger) Complete(id engine.BatchID, o engine.Outcome, end engine.Time) (engine.Result, error) {
	r, err := e.FIFO.Complete(id, o, end)
	r.Finished = append(r.Finished, engine.Finished{ID: "stranger", Reason: engine.FinishStop})
	return r, err
}

// overEmits reports more output tokens for a request than one step can sample.
type overEmits struct{ *FIFO }

func (e overEmits) Complete(id engine.BatchID, o engine.Outcome, end engine.Time) (engine.Result, error) {
	r, err := e.FIFO.Complete(id, o, end)
	for rid := range r.Emitted {
		r.Emitted[rid] += 2
	}
	return r, err
}

func TestTheDriverCatchesBrokenEngines(t *testing.T) {
	held := req("held", 4, 1)
	held.Tags = map[string]string{"fifo.hold_until": "500"}
	two := []Arrival{at(0, req("A", 4, 2)), at(0, req("B", 4, 2))}
	for _, c := range []struct {
		name     string
		wrap     func(*FIFO) engine.Engine
		cfg      func(*FIFOConfig)
		arrivals []Arrival
		want     string
	}{
		{"no wake strands a held request", func(f *FIFO) engine.Engine { return mute{f} }, nil, []Arrival{at(0, held)}, "never finish"},
		{"a reused batch ID", func(f *FIFO) engine.Engine { return reusesIDs{f} }, nil, two, "does not follow"},
		{"the in-flight limit ignored", func(f *FIFO) engine.Engine { return ignoresLimit{f} }, func(c *FIFOConfig) { c.MaxRunning = 1 }, two, "want ErrNoCapacity"},
		{"a request finished twice", func(f *FIFO) engine.Engine { return finishesTwice{f} }, nil, two, "finished twice"},
		{"a wake at the instant asked", func(f *FIFO) engine.Engine { return wakesNow{f} }, nil, []Arrival{at(0, held)}, "want an instant after now"},
		{"a step that takes no time", func(f *FIFO) engine.Engine { return freeSteps{f} }, nil, two, "want at least 1"},
		{"a never-added request finished", func(f *FIFO) engine.Engine { return finishesAStranger{f} }, nil, two, "never added"},
		{"more tokens emitted than a step samples", func(f *FIFO) engine.Engine { return overEmits{f} }, nil, two, "one sampled plus"},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg := fifoCfg()
			if c.cfg != nil {
				c.cfg(&cfg)
			}
			build := func() (engine.Engine, error) {
				f, err := NewFIFO(cfg)
				if err != nil {
					return nil, err
				}
				return c.wrap(f), nil
			}
			_, err := Drive(build, c.arrivals, Options{MaxSteps: 1000})
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("Drive returned %v, want an error containing %q", err, c.want)
			}
		})
	}
}
