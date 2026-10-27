package deployment

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The offload connector's options, as kv_transfer_config.kv_connector_extra_config gives
// them, typed rather than a free map so a misspelled option fails to load.
func TestOffloadConnectorOptions(t *testing.T) {
	off := func(o *ConnectorOptions) *Deployment {
		d := &Deployment{Kind: "Deployment", Name: "d", Pools: []Pool{{
			Role: RoleColocated, Nodes: 1, Parallel: Parallelism{TP: 1, PP: 1, DP: 1}}}}
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

func ptrBool(v bool) *bool { return &v }
