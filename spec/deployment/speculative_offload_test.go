package deployment

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The speculative knobs a launch states beside method and K: an external drafter by catalog
// name (blis-schemas#97), the drafter's KV dtype, and the sampling knobs. Acceptance is
// supplied (inference-sim#2021 B3), so the sampling knobs are recorded, not acted on; their
// closed vocabularies are still checked, since a misspelled one is a launch vLLM refuses.
func TestSpeculativeKnobs(t *testing.T) {
	spec := func(s Speculative) *Deployment {
		s.Method, s.NumSpecTokens = "eagle3", 3
		return engineOnly(Engine{Speculative: &s})
	}
	for _, c := range []struct {
		name string
		s    Speculative
		bad  string
	}{
		{"none stated", Speculative{}, ""},
		{"an external drafter", Speculative{DraftModel: "minimax-m3-eagle3-gqa"}, ""},
		{"a blank drafter name", Speculative{DraftModel: "  "}, "speculative.draft_model"},
		{"drafter kv dtype", Speculative{KVCacheDType: "fp8"}, ""},
		{"probabilistic drafting", Speculative{DraftSampleMethod: "probabilistic"}, ""},
		{"an unknown draft sample method", Speculative{DraftSampleMethod: "beam"}, "speculative.draft_sample_method"},
		{"block rejection sampling", Speculative{RejectionSampleMethod: "block"}, ""},
		{"an unknown rejection sample method", Speculative{RejectionSampleMethod: "strict"}, "speculative.rejection_sample_method"},
		{"adaptive verification", Speculative{EnableAdaptiveVerification: ptrBool(true)}, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := spec(c.s).Validate()
			if c.bad == "" {
				if !p.OK() {
					t.Fatalf("refused:\n%s", p.Error())
				}
				return
			}
			if p.OK() || !strings.Contains(p.Error(), ".engine."+c.bad) {
				t.Fatalf("want a finding on engine.%s, got:\n%s", c.bad, p.Error())
			}
		})
	}
	// An absent draft_model is today's document: the target graph's own speculator.
	var s Speculative
	if err := yaml.Unmarshal([]byte("method: dspark\nnum_spec_tokens: 5\n"), &s); err != nil || s.DraftModel != "" {
		t.Fatalf("a document without draft_model loaded as %+v (%v)", s, err)
	}
}

// The offload connector's options, as kv_transfer_config.kv_connector_extra_config gives
// them, typed rather than a free map so a misspelled option fails to load.
func TestOffloadConnectorOptions(t *testing.T) {
	off := func(o *ConnectorOptions) *Deployment {
		d := engineOnly(Engine{})
		d.Offload = &Offload{Tiers: []Tier{{Device: "cpu-dram", Bytes: 1 << 30}}, ConnectorOptions: o}
		return d
	}
	for _, c := range []struct {
		name string
		o    *ConnectorOptions
		bad  string
	}{
		{"none", nil, ""},
		{"mooncake async", &ConnectorOptions{LoadAsync: ptrBool(true), LookupAsync: ptrBool(true)}, ""},
		{"cross-layer blocks", &ConnectorOptions{EnableCrossLayersBlocks: ptrBool(true)}, ""},
		{"lazy offload", &ConnectorOptions{LazyOffload: ptrBool(true)}, ""},
		{"blocks per chunk", &ConnectorOptions{BlocksPerChunk: 4}, ""},
		{"blocks per chunk negative", &ConnectorOptions{BlocksPerChunk: -1}, "offload.connector_options.blocks_per_chunk"},
		{"recompute on failure", &ConnectorOptions{KVLoadFailurePolicy: "recompute"}, ""},
		{"an unknown failure policy", &ConnectorOptions{KVLoadFailurePolicy: "retry"}, "offload.connector_options.kv_load_failure_policy"},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := off(c.o).Validate()
			if c.bad == "" {
				if !p.OK() {
					t.Fatalf("refused:\n%s", p.Error())
				}
				return
			}
			if p.OK() || !strings.Contains(p.Error(), c.bad) {
				t.Fatalf("want a finding on %s, got:\n%s", c.bad, p.Error())
			}
		})
	}
	// Strict decoding: a misspelled option is a load error, not a silently dropped setting.
	var o Offload
	dec := yaml.NewDecoder(strings.NewReader("tiers: [{tier: cpu-dram, bytes: 1}]\nconnector_options: {load_asynch: true}\n"))
	dec.KnownFields(true)
	if err := dec.Decode(&o); err == nil {
		t.Error("a misspelled connector option loaded")
	}
}
