package deployment

import (
	"math/rand"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestLongPrefillTokenThreshold(t *testing.T) {
	for _, c := range []struct {
		n  int
		ok bool
	}{{0, true}, {8192, true}, {-1, false}} {
		p := (&Deployment{Kind: "Deployment", Name: "d", Pools: []Pool{{
			Role: RoleColocated, Nodes: 1, Parallel: Parallelism{TP: 1, PP: 1, DP: 1},
			Engine: Engine{LongPrefillTokenThreshold: c.n}}}}).Validate()
		if p.OK() != c.ok {
			t.Errorf("long_prefill_token_threshold %d: ok=%v, want %v\n%s", c.n, p.OK(), c.ok, p.Error())
		}
	}
	// Load, marshal, load: a stated threshold comes back as itself, and the default
	// (zero, off) is omitted rather than written.
	r := rand.New(rand.NewSource(5))
	for i := 0; i < 500; i++ {
		e := Engine{LongPrefillTokenThreshold: r.Intn(1 << 20)}
		out, err := yaml.Marshal(e)
		if err != nil {
			t.Fatal(err)
		}
		var back Engine
		if err := yaml.Unmarshal(out, &back); err != nil || back.LongPrefillTokenThreshold != e.LongPrefillTokenThreshold {
			t.Fatalf("%d round-tripped through %q to %d (%v)", e.LongPrefillTokenThreshold, out, back.LongPrefillTokenThreshold, err)
		}
		if e.LongPrefillTokenThreshold == 0 && containsKey(string(out), "long_prefill_token_threshold") {
			t.Fatalf("the default was written: %s", out)
		}
	}
}
