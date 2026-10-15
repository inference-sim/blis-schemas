package workload

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// These tests re-derive the replay descriptor's aiperf constants from an aiperf 0.12.0
// source tree at AIPERF_SOURCE (its src/aiperf directory), as the rules packs' tests do
// from VLLM_SOURCE. Without one they skip and say so; AIPERF_SOURCE_REQUIRED turns the
// skip into a failure.

func aiperfSource(t *testing.T) string {
	t.Helper()
	root := os.Getenv("AIPERF_SOURCE")
	skip := t.Skipf
	if os.Getenv("AIPERF_SOURCE_REQUIRED") != "" {
		skip = t.Fatalf
	}
	if root == "" {
		skip("AIPERF_SOURCE is unset: the replay descriptor's aiperf constants were NOT checked against aiperf 0.12.0")
	}
	if _, err := os.Stat(filepath.Join(root, "common", "scenario")); err != nil {
		skip("AIPERF_SOURCE=%s has no common/scenario: the aiperf constants were NOT checked", root)
	}
	return root
}

func readAiperf(t *testing.T, root string, parts ...string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(append([]string{root}, parts...)...))
	if err != nil {
		t.Fatalf("reading %v: %v (the aiperf layout may have moved)", parts, err)
	}
	return string(b)
}

// Every alias the table names reads the repository plugins.yaml fixes for it, and the
// rolling set is exactly the aliases plugins.yaml calls rolling.
func TestAgentXLoaderReposMatchSource(t *testing.T) {
	root := aiperfSource(t)
	var plugins struct {
		Loaders map[string]struct {
			Description string `yaml:"description"`
			Metadata    struct {
				HFDatasetName string `yaml:"hf_dataset_name"`
			} `yaml:"metadata"`
		} `yaml:"public_dataset_loader"`
	}
	if err := yaml.Unmarshal([]byte(readAiperf(t, root, "plugin", "plugins.yaml")), &plugins); err != nil {
		t.Fatal(err)
	}
	for alias, repo := range AgentXLoaders {
		if alias == loaderWekaHF || alias == loaderWekaTrace {
			continue
		}
		l, ok := plugins.Loaders[alias]
		if !ok {
			t.Errorf("%s is not a public_dataset_loader in plugins.yaml", alias)
			continue
		}
		if l.Metadata.HFDatasetName != repo {
			t.Errorf("%s reads %s in plugins.yaml; the table says %s", alias, l.Metadata.HFDatasetName, repo)
		}
		rolling := strings.Contains(strings.Join(strings.Fields(l.Description), " "), "rolling alias that tracks the current default")
		if rolling != AgentXRollingLoaders[alias] {
			t.Errorf("%s: rolling in plugins.yaml %v, in the table %v", alias, rolling, AgentXRollingLoaders[alias])
		}
	}
}
