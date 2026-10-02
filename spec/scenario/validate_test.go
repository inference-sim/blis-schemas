package scenario

import "testing"

// multiNode is the immutable problem behind a prefill/decode cluster: a model on 60
// H200 nodes behind a 400G fabric. The mutable layout that runs on it is a Deployment,
// which this package no longer carries.
func multiNode() *Scenario {
	return &Scenario{
		Kind: "Scenario", Name: "glm-5.3-60node-h200",
		Model: "glm-5.3", Coefficients: []string{"cost-model-primitives-h200"},
		EngineVersion: "0.29.0",
		Cluster: Cluster{Hardware: "h200", Fabric: "ib-400g", Nodes: 60,
			GPUsPerNode: 8, Storage: []string{"cpu_dram", "nvme_gen4"}},
	}
}

// singleNode is the colocated problem the published Granite runs pose: eight GPUs on
// one node, no fabric because nothing crosses a node.
func singleNode() *Scenario {
	return &Scenario{
		Kind: "Scenario", Name: "granite-230b-h100",
		Model: "granite-5-230b", Coefficients: []string{"cost-model-primitives-h100"},
		EngineVersion: "0.29.0",
		Cluster:       Cluster{Hardware: "h100", Nodes: 1, GPUsPerNode: 8},
	}
}

func TestValidScenariosPass(t *testing.T) {
	for _, s := range []*Scenario{multiNode(), singleNode()} {
		if p := s.Validate(); !p.OK() {
			t.Fatalf("%s rejected:\n%s", s.Name, p.Error())
		}
	}
}

func TestClusterGPUs(t *testing.T) {
	if got := (Cluster{Nodes: 60, GPUsPerNode: 8}).GPUs(); got != 480 {
		t.Errorf("GPUs() = %d, want 480", got)
	}
}

func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Scenario)
	}{
		{"wrong kind", func(s *Scenario) { s.Kind = "Deployment" }},
		{"no model", func(s *Scenario) { s.Model = "" }},
		{"no hardware", func(s *Scenario) { s.Cluster.Hardware = "" }},
		{"no engine version", func(s *Scenario) { s.EngineVersion = "" }},
		{"no coefficients", func(s *Scenario) { s.Coefficients = nil }},
		{"zero nodes", func(s *Scenario) { s.Cluster.Nodes = 0 }},
		{"multi-node without a fabric", func(s *Scenario) { s.Cluster.Fabric = "" }},
		{"rack not a multiple of node", func(s *Scenario) { s.Cluster.GPUsPerRack = 70 }},
		{"empty storage class", func(s *Scenario) {
			s.Cluster.Storage = []string{""}
		}},
		{"duplicate storage class", func(s *Scenario) {
			s.Cluster.Storage = []string{"cpu_dram", "cpu_dram"}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := multiNode()
			tc.mutate(s)
			if p := s.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}
