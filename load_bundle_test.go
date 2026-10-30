package blisschemas

import (
	"strings"
	"testing"
)

const bundleScenario = `kind: Scenario
name: granite-230b-h200-tp8
model: granite-5-230b
coefficients: [cost-model-primitives-h200]
engine_version: "0.32.0"
workload:
  shape: chatbot
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
`

const bundleDeployment = `kind: Deployment
name: granite-230b-h200-tp8
pools:
  - role: colocated
    nodes: 1
    parallel:
      tp: 8
      pp: 1
      dp: 1
    engine:
      block_size: 16
`

func TestLoadBundle(t *testing.T) {
	path := write(t, "b.yaml", bundleScenario+"---\n"+bundleDeployment)
	sc, dep, err := LoadBundle(path)
	if err != nil {
		t.Fatalf("LoadBundle: %v", err)
	}
	if sc.Name != "granite-230b-h200-tp8" || sc.EngineVersion != "0.32.0" || sc.Cluster.Hardware != "h200" {
		t.Errorf("scenario did not parse: %+v", sc)
	}
	if len(dep.Pools) != 1 || dep.Pools[0].Parallel.TP != 8 {
		t.Fatalf("deployment did not parse: %+v", dep.Pools)
	}
	var e map[string]any
	if err := dep.Pools[0].Engine.Decode(&e); err != nil || e["block_size"] != 16 {
		t.Errorf("engine block = %v (err %v), want block_size 16", e, err)
	}
}

// The bundle loader is the scenario and deployment loaders over one stream: it must decode
// each half to exactly what LoadScenario and LoadDeployment decode from the half alone.
func TestLoadBundleMatchesTheSingleDocumentLoaders(t *testing.T) {
	sc, dep, err := LoadBundle(write(t, "b.yaml", bundleScenario+"---\n"+bundleDeployment))
	if err != nil {
		t.Fatalf("LoadBundle: %v", err)
	}
	wantSc, err := LoadScenario(write(t, "s.yaml", bundleScenario))
	if err != nil {
		t.Fatalf("LoadScenario: %v", err)
	}
	wantDep, err := LoadDeployment(write(t, "d.yaml", bundleDeployment))
	if err != nil {
		t.Fatalf("LoadDeployment: %v", err)
	}
	if sc.Name != wantSc.Name || sc.Model != wantSc.Model || sc.EngineVersion != wantSc.EngineVersion ||
		sc.Cluster.Hardware != wantSc.Cluster.Hardware || sc.Cluster.GPUsPerNode != wantSc.Cluster.GPUsPerNode {
		t.Errorf("scenario half differs from LoadScenario:\n got %+v\nwant %+v", sc, wantSc)
	}
	if dep.Name != wantDep.Name || len(dep.Pools) != len(wantDep.Pools) ||
		dep.Pools[0].Parallel != wantDep.Pools[0].Parallel || dep.Pools[0].Role != wantDep.Pools[0].Role {
		t.Errorf("deployment half differs from LoadDeployment:\n got %+v\nwant %+v", dep, wantDep)
	}
}

func TestLoadBundleRefusals(t *testing.T) {
	for _, c := range []struct {
		name, body, want string
	}{
		{"empty file", "", "is empty"},
		{"no deployment", bundleScenario, "holds a scenario but no deployment"},
		{"no pools", bundleScenario + "---\nkind: Deployment\nname: x\npools: []\n", "declares no pools"},
		{"unknown scenario field", "kind: Scenario\nnmae: typo\n---\n" + bundleDeployment, "decoding the scenario"},
		{"unknown deployment field", bundleScenario + "---\nkind: Deployment\npoolz: []\n", "decoding the deployment"},
	} {
		t.Run(c.name, func(t *testing.T) {
			path := write(t, "b.yaml", c.body)
			_, _, err := LoadBundle(path)
			if err == nil || !strings.Contains(err.Error(), c.want) || !strings.Contains(err.Error(), path) {
				t.Fatalf("err = %v, want one containing %q and the path", err, c.want)
			}
		})
	}
	if _, _, err := LoadBundle("/nonexistent/bundle.yaml"); err == nil || !strings.Contains(err.Error(), "opening") {
		t.Fatalf("missing file: err = %v, want an opening error", err)
	}
}
