package embedtest_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/embed"
	"github.com/inference-sim/blis-schemas/embed/embedtest"
)

func config(c stubConfig) embed.Config {
	b, _ := json.Marshal(c)
	return embed.Config{Config: b}
}

// suite is the stub router's Suite, as a router's own test would write one.
func suite(m mutation) embedtest.Suite {
	evict := config(stubConfig{FlowControl: true, Capacity: 4, Evict: true})
	return embedtest.Suite{
		New:        newStub(m),
		Seed:       7,
		Route:      config(stubConfig{}),
		Hold:       config(stubConfig{FlowControl: true, TTL: "5s", Capacity: 1}),
		KV:         config(stubConfig{KV: true}),
		Evict:      &evict,
		Refuse:     config(stubConfig{Plugins: []string{"scorer", "unhostable"}}),
		RefuseType: "unhostable",
	}
}

// TestStubPasses: a router that keeps the contract passes.
func TestStubPasses(t *testing.T) {
	embedtest.Run(t, suite(correct))
}

// TestMutationsFail: every mutation of the stub, each breaking one rule, is caught, by the
// part of the scenario that checks that rule. This is the check's own mutation test: a check
// that let any of these pass would not be checking its rule.
func TestMutationsFail(t *testing.T) {
	for _, c := range []struct {
		m    mutation
		want string // a substring of some problem
	}{
		{saturatedAsAdmit, "hold: request-2: want a saturated rejection, got admission"},
		{ttlAsCancelled, "want a ttl-expired rejection, got context-cancelled"},
		{noEndpointsAsUnavl, "no-endpoints: request-1: want a no-endpoints rejection, got unavailable"},
		{admitNotShed, "admission: request-1: want a admission rejection, got dispatched"},
		{badBodyDispatched, "unclassified: request-1: want a unclassified rejection"},
		{randAtConstruction, "construction drew randomness"},
		{wallClockSeed, "determinism: two runs on seed 7 made different decisions"},
		{leakOnStop, "hold: request-5: decided after Stop"},
		{loopAfterStop, "an Every loop ran"},
		{headersDropped, "want every header passed in"},
		{ignoresAllow, "refuse: construction succeeded although Allow refused unhostable"},
		{ignoresClear, "the other endpoint's blocks were cleared"},
		{ignoresRemoved, "the other endpoint removed the blocks it had published"},
		{neverEvicts, "evict: request-1: not evicted"},
		{routesNotReady, "after it was made not ready"},
		{removeKeeps, "after the other was removed"},
		{doneTwice, "done called 2 times"},
		{noFlowControlFacts, "hold: FlowControl() = {Enabled:false"},
		{evictSignalNoEvict, "route: request-1: an EvictionSignal from a configuration that does not evict"},
		{scoresForStrangers, `a score for "192.0.2.1:8000", which is no endpoint`},
		{failedConstructLeak, "a failed construction left an Every loop running"},
		{parkedForever, "route: 1 goroutines started with Runtime.Go are still live after Stop"},
	} {
		t.Run(string(c.m), func(t *testing.T) {
			problems := embedtest.Problems(suite(c.m))
			for _, p := range problems {
				if strings.Contains(p.Error(), c.want) {
					return
				}
			}
			t.Errorf("no problem contains %q; got %d:", c.want, len(problems))
			for _, p := range problems {
				t.Log(p)
			}
		})
	}
}

// TestRuntimeIsDeterministic: two Runtimes on one seed give the same streams, and distinct
// stream names give distinct streams.
func TestRuntimeIsDeterministic(t *testing.T) {
	a, b := embedtest.NewRuntime(3), embedtest.NewRuntime(3)
	for range 8 {
		if x, y := a.Rand("s").Float64(), b.Rand("s").Float64(); x != y {
			t.Fatalf("stream s differs between runtimes on one seed: %v, %v", x, y)
		}
	}
	if a.Rand("t").Float64() == a.Rand("u").Float64() {
		t.Error("streams t and u gave the same first draw")
	}
}
