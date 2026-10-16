package deployment

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// swa_bounded_replay is tri-state: unstated (the engine resolves it), true and false each
// load and write back as themselves, and the unstated value is not written.
func TestSWABoundedReplayRoundTrips(t *testing.T) {
	for _, src := range []string{"{}", "{swa_bounded_replay: true}", "{swa_bounded_replay: false}"} {
		var in Engine
		if err := yaml.Unmarshal([]byte(src), &in); err != nil {
			t.Fatal(err)
		}
		if (in.SWABoundedReplay == nil) != (src == "{}") {
			t.Fatalf("%s loaded as %v", src, in.SWABoundedReplay)
		}
		out, err := yaml.Marshal(in)
		if err != nil {
			t.Fatal(err)
		}
		var back Engine
		if err := yaml.Unmarshal(out, &back); err != nil {
			t.Fatal(err)
		}
		switch {
		case in.SWABoundedReplay == nil:
			if back.SWABoundedReplay != nil || strings.Contains(string(out), "swa_bounded_replay") {
				t.Fatalf("an unstated swa_bounded_replay was written: %s", out)
			}
		case back.SWABoundedReplay == nil || *back.SWABoundedReplay != *in.SWABoundedReplay:
			t.Fatalf("%s round-tripped through %q to %v", src, out, back.SWABoundedReplay)
		}
	}
}
