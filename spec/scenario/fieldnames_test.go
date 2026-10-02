package scenario

import (
	"reflect"
	"strings"
	"testing"
)

// The Engine struct's YAML names are the contract a deployment author writes
// against, so each must be the engine's own name for the setting rather than an
// invented one. An earlier draft carried three that no engine accepts — an
// allreduce_backend enum where the engine has a boolean, a cache-dtype set with four
// of seventeen members, and scheduler_policy where the flag is scheduling-policy.
// All three would have rejected valid deployments.
//
// This test cannot reach into an engine checkout, so it pins the names instead.
// Verification against a live tree belongs in whatever CI has that checkout; what
// this guards is silent renaming during a refactor here.
func TestEngineFieldNames(t *testing.T) {
	// Names taken from the engine's configuration surface. Where the CLI flag and
	// the internal field differ, the CLI name is used, because a scenario is written
	// by whoever launched the deployment.
	want := map[string]string{
		"all2all_backend":           "VLLM_ALL2ALL_BACKEND / --all2all-backend",
		"disable_custom_all_reduce": "--disable-custom-all-reduce",
		"allreduce_backend":         "resolved override, not an engine field",
		"cudagraph_mode":            "-O/--compilation-config cudagraph_mode",
		"async_scheduling":          "--async-scheduling",
		"disable_cascade_attn":      "--disable-cascade-attn",
		"enable_prefix_caching":     "--enable-prefix-caching / --no-enable-prefix-caching",
		"block_size":                "--block-size",
		"max_num_batched_tokens":    "--max-num-batched-tokens",
		"max_num_seqs":              "--max-num-seqs",
		"max_model_len":             "--max-model-len",
		"quantization":              "--quantization",
		"cache_dtype":               "--kv-cache-dtype",
		"mamba_cache_dtype":         "--mamba-cache-dtype",
		"mamba_ssm_cache_dtype":     "--mamba-ssm-cache-dtype",
		"mamba_cache_mode":          "--mamba-cache-mode",
		"gpu_memory_utilization":    "--gpu-memory-utilization",
		"scheduling_policy":         "--scheduling-policy",
		"dbo":                       "--enable-dbo and its thresholds",
		"eplb":                      "--enable-eplb and --eplb-config",
		"speculative":               "--speculative-config",
	}
	got := yamlNames(reflect.TypeOf(Engine{}))
	if len(got) != len(want) {
		t.Errorf("Engine has %d yaml fields, the mapping covers %d: %v",
			len(got), len(want), got)
	}
	for _, name := range got {
		if _, ok := want[name]; !ok {
			t.Errorf("Engine field %q has no engine counterpart recorded; either it is invented or the mapping needs updating", name)
		}
	}
	for name := range want {
		if !contains(got, name) {
			t.Errorf("the mapping records %q but Engine no longer declares it", name)
		}
	}
}

// TestParallelismFieldNames does the same for the layout, and asserts that no field
// named ep exists: the width is derived, and a field for it could disagree.
func TestParallelismFieldNames(t *testing.T) {
	got := yamlNames(reflect.TypeOf(Parallelism{}))
	want := []string{"tp", "pp", "dp", "dp_local", "enable_expert_parallel",
		"pcp", "dcp"}
	if len(got) != len(want) {
		t.Errorf("Parallelism has %d yaml fields, want %d: %v", len(got), len(want), got)
	}
	for _, w := range want {
		if !contains(got, w) {
			t.Errorf("Parallelism is missing %q", w)
		}
	}
	for _, forbidden := range []string{"ep", "expert_parallel_size"} {
		if contains(got, forbidden) {
			t.Errorf("Parallelism declares %q; the expert-parallel width is derived so a field could disagree with it", forbidden)
		}
	}
}

func yamlNames(t reflect.Type) []string {
	var out []string
	for i := 0; i < t.NumField(); i++ {
		tag := t.Field(i).Tag.Get("yaml")
		if tag == "" || tag == "-" {
			continue
		}
		out = append(out, strings.Split(tag, ",")[0])
	}
	return out
}

func contains(list []string, want string) bool {
	for _, s := range list {
		if s == want {
			return true
		}
	}
	return false
}
