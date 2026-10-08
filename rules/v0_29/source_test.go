package v0_29

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// This file re-derives the pack's constants from an engine checkout, when one is
// available at VLLM_SOURCE. It exists because every constant here was transcribed by
// hand, and hand transcription is where the errors were: an earlier draft carried
// four of seventeen cache dtypes, four of five graph modes, a backend enum for a
// setting that is a boolean, and a field name the engine does not use. Three of those
// would have rejected valid deployments.
//
// constants_test.go pins the values so a refactor cannot quietly narrow them. This
// file checks the values are right in the first place, which no amount of internal
// pinning can establish.
//
// Without a checkout the tests SKIP and say so. A skipped test reads like a passing
// one in a summary line, so the message names what was not checked.

func vllmSource(t *testing.T) string {
	t.Helper()
	root := os.Getenv("VLLM_SOURCE")
	if root == "" {
		t.Skip("VLLM_SOURCE is unset: the pack's constants were NOT checked against " +
			"an engine checkout. Set it to a vllm tree at version " + Version +
			" to verify them.")
	}
	if _, err := os.Stat(filepath.Join(root, "vllm", "config")); err != nil {
		t.Skipf("VLLM_SOURCE=%s has no vllm/config: the pack's constants were NOT checked", root)
	}
	return root
}

// readSource returns one engine file's contents, failing rather than skipping: the
// tree exists, so a missing file means the layout moved and the transcription's basis
// is gone.
func readSource(t *testing.T, root string, parts ...string) string {
	t.Helper()
	path := filepath.Join(append([]string{root}, parts...)...)
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v (the engine layout may have moved; the pack's "+
			"constants can no longer be verified against it)", path, err)
	}
	return string(b)
}

// literalMembers extracts the string members of a Python Literal type alias.
func literalMembers(t *testing.T, src, name string) []string {
	t.Helper()
	re := regexp.MustCompile(`(?s)\b` + name + `\s*=\s*Literal\[(.*?)\]`)
	m := re.FindStringSubmatch(src)
	if m == nil {
		t.Fatalf("no Literal named %s found; the engine may have changed how it "+
			"declares this vocabulary", name)
	}
	out := regexp.MustCompile(`"([^"]+)"`).FindAllStringSubmatch(m[1], -1)
	vals := make([]string, 0, len(out))
	for _, g := range out {
		vals = append(vals, g[1])
	}
	sort.Strings(vals)
	return vals
}

func assertSameSet(t *testing.T, what string, pack map[string]bool, source []string) {
	t.Helper()
	inSource := map[string]bool{}
	for _, s := range source {
		inSource[s] = true
	}
	var extra, missing []string
	for k := range pack {
		if !inSource[k] {
			extra = append(extra, k)
		}
	}
	for _, s := range source {
		if !pack[s] {
			missing = append(missing, s)
		}
	}
	sort.Strings(extra)
	sort.Strings(missing)
	if len(extra) > 0 {
		t.Errorf("%s: the pack accepts %v, which the engine does not; a deployment "+
			"using one would validate here and fail to launch", what, extra)
	}
	if len(missing) > 0 {
		t.Errorf("%s: the engine accepts %v, which the pack does not; a valid "+
			"deployment would be rejected", what, missing)
	}
}

func TestAll2AllBackendsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "parallel.py")
	assertSameSet(t, "All2AllBackends", Pack().All2AllBackends,
		literalMembers(t, src, "All2AllBackend"))
}

func TestDCPCommBackendsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "parallel.py")
	assertSameSet(t, "DCPCommBackends", Pack().DCPCommBackends,
		literalMembers(t, src, "DCPCommBackend"))
}

// The cache-dtype set is the one an earlier draft got most wrong, so it is checked
// member by member rather than only by size.
func TestCacheDTypesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "cache.py")
	assertSameSet(t, "CacheDTypes", Pack().CacheDTypes,
		literalMembers(t, src, "CacheDType"))
}

func TestMambaCacheModesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "cache.py")
	assertSameSet(t, "MambaCacheModes", Pack().MambaCacheModes,
		literalMembers(t, src, "MambaCacheMode"))
}

func TestSchedulerPoliciesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "scheduler.py")
	assertSameSet(t, "SchedulerPolicies", Pack().SchedulerPolicies,
		literalMembers(t, src, "SchedulerPolicy"))
}

// CUDAGraphMode is an enum class rather than a Literal, so it is parsed differently.
// The fifth member an earlier draft missed, FULL_DECODE_ONLY, is declared as a tuple
// of two other members; the pattern accepts both forms.
func TestCUDAGraphModesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "compilation.py")
	i := strings.Index(src, "class CUDAGraphMode(")
	if i < 0 {
		t.Fatal("no CUDAGraphMode enum found; the engine may have moved it")
	}
	body := src[i:]
	if j := strings.Index(body, "\n\nclass "); j > 0 {
		body = body[:j]
	}
	re := regexp.MustCompile(`(?m)^\s{4}([A-Z][A-Z_]*)\s*=`)
	var members []string
	for _, m := range re.FindAllStringSubmatch(body, -1) {
		members = append(members, m[1])
	}
	sort.Strings(members)
	assertSameSet(t, "CUDAGraphModes", Pack().CUDAGraphModes, members)
}

func TestOffloadSpecsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "v1", "kv_offload", "factory.py")
	re := regexp.MustCompile(`register_spec\(\s*"(\w+)"`)
	var specs []string
	for _, m := range re.FindAllStringSubmatch(src, -1) {
		specs = append(specs, m[1])
	}
	sort.Strings(specs)
	assertSameSet(t, "OffloadSpecs", Pack().OffloadSpecs, specs)
}

// Quantizations is the QuantizationMethods literal — a plain Literal[...] like the cache
// and backend sets, so it uses the same extractor. The engine extends QUANTIZATION_METHODS
// at runtime for out-of-tree methods, but the literal is the in-tree set the pack mirrors;
// a divergence here means a method was added or renamed in the release.
func TestQuantizationsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "model_executor", "layers", "quantization", "__init__.py")
	assertSameSet(t, "Quantizations", Pack().Quantizations,
		literalMembers(t, src, "QuantizationMethods"))
}

// Connectors is a registry like OffloadSpecs: names passed to register_connector rather
// than a Literal, so it is extracted the same way. The first string argument of each call
// is the connector name a deployment selects.
func TestConnectorsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "distributed", "kv_transfer", "kv_connector", "factory.py")
	re := regexp.MustCompile(`register_connector\(\s*"([^"]+)"`)
	var names []string
	for _, m := range re.FindAllStringSubmatch(src, -1) {
		names = append(names, m[1])
	}
	sort.Strings(names)
	assertSameSet(t, "Connectors", Pack().Connectors, names)
}

// EvictionPolicies is the CachePolicyFactory registry — register_cache_policy calls in
// vllm/v1/kv_offload/cpu/policies/factory.py, the same shape as the connector registry (the
// name is the call's first string argument, on the line after the paren). A divergence means
// a built-in policy was added or renamed in the release.
func TestEvictionPoliciesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "v1", "kv_offload", "cpu", "policies", "factory.py")
	re := regexp.MustCompile(`register_cache_policy\(\s*"([^"]+)"`)
	var names []string
	for _, m := range re.FindAllStringSubmatch(src, -1) {
		names = append(names, m[1])
	}
	sort.Strings(names)
	assertSameSet(t, "EvictionPolicies", Pack().EvictionPolicies, names)
}

// The numeric constants are single values rather than sets, so each is matched
// against its declaration site.
func TestNumericConstantsMatchSource(t *testing.T) {
	root := vllmSource(t)
	p := Pack()

	parallel := readSource(t, root, "vllm", "config", "parallel.py")
	for field, want := range map[string]int{
		"dbo_decode_token_threshold":  p.DefaultDBODecodeThreshold,
		"dbo_prefill_token_threshold": p.DefaultDBOPrefillThreshold,
	} {
		re := regexp.MustCompile(field + `: int = Field\(default=(\d+)`)
		m := re.FindStringSubmatch(parallel)
		if m == nil {
			t.Errorf("%s: no default found in the engine's parallel config", field)
			continue
		}
		if m[1] != itoa(want) {
			t.Errorf("%s: pack has %d, the engine declares %s", field, want, m[1])
		}
	}

	triton := readSource(t, root, "vllm", "v1", "kv_offload", "cpu",
		"swap_blocks_triton.py")
	if m := regexp.MustCompile(`NUM_SMS = (\d+)`).FindStringSubmatch(triton); m != nil {
		if m[1] != itoa(p.TritonFetchSMs) {
			t.Errorf("TritonFetchSMs: pack has %d, the engine declares %s",
				p.TritonFetchSMs, m[1])
		}
	} else {
		t.Error("NUM_SMS not found in the engine's Triton swap-blocks module")
	}
	if m := regexp.MustCompile(`THRESHOLD_BYTES = (\d+) \* 1024`).FindStringSubmatch(triton); m != nil {
		if m[1] != itoa(p.TritonFetchPageBytes/1024) {
			t.Errorf("TritonFetchPageBytes: pack has %d KiB, the engine declares %s KiB",
				p.TritonFetchPageBytes/1024, m[1])
		}
	} else {
		t.Error("THRESHOLD_BYTES not found in the engine's Triton swap-blocks module")
	}

	custom := readSource(t, root, "vllm", "distributed", "device_communicators",
		"custom_all_reduce.py")
	m := regexp.MustCompile(`_SUPPORTED_WORLD_SIZES = \[([\d, ]+)\]`).FindStringSubmatch(custom)
	if m == nil {
		t.Fatal("_SUPPORTED_WORLD_SIZES not found; the custom all-reduce may have moved")
	}
	declared := map[int]bool{}
	for _, part := range strings.Split(m[1], ",") {
		declared[atoi(strings.TrimSpace(part))] = true
	}
	for k := range p.CustomAllReduceWorldSizes {
		if !declared[k] {
			t.Errorf("world size %d is in the pack but not supported by the engine", k)
		}
	}
	for k := range declared {
		if !p.CustomAllReduceWorldSizes[k] {
			t.Errorf("world size %d is supported by the engine but absent from the pack", k)
		}
	}

	modelCfg := readSource(t, root, "vllm", "config", "model.py")
	optIn := strings.Contains(modelCfg, "disable_cascade_attn: bool = True")
	if optIn != p.CascadeAttnOptIn {
		t.Errorf("CascadeAttnOptIn: pack has %v, the engine's default says %v",
			p.CascadeAttnOptIn, optIn)
	}
}

// The sequence-parallel MoE subset is derived from a property body rather than a
// declaration, so it is extracted from that function's text.
func TestSPMoEBackendsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "parallel.py")
	i := strings.Index(src, "def use_sequence_parallel_moe(self) -> bool:")
	if i < 0 {
		t.Fatal("use_sequence_parallel_moe not found; the engine may have changed " +
			"how it decides sequence-parallel MoE")
	}
	body := src[i:]
	if j := strings.Index(body, "\n    @property"); j > 0 {
		body = body[:j]
	}
	var names []string
	for _, m := range regexp.MustCompile(`"([a-z0-9_]+)"`).FindAllStringSubmatch(body, -1) {
		names = append(names, m[1])
	}
	sort.Strings(names)
	assertSameSet(t, "SPMoEBackends", Pack().SPMoEBackends, names)
}

// The engine has no allreduce-backend enum. This test asserts the absence, because
// the pack's shape depends on it: an earlier draft invented such an enum.
func TestNoAllReduceBackendEnumInSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "parallel.py")
	if regexp.MustCompile(`AllReduceBackend\s*=\s*Literal\[`).MatchString(src) {
		t.Error("the engine now declares an AllReduceBackend literal; the pack " +
			"models this as a boolean and should be revisited")
	}
	if !strings.Contains(src, "disable_custom_all_reduce: bool") {
		t.Error("disable_custom_all_reduce is no longer a boolean; the deployment " +
			"field mirrors it and should be revisited")
	}
}

// itoa is declared in pack.go and reused here.

func atoi(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return -1
		}
		n = n*10 + int(c-'0')
	}
	return n
}
