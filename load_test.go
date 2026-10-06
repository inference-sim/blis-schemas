package blisschemas

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/workload"
)

// The yaml tags on every spec type are a contract with the files in blis-catalog and
// blis-registry. Until something parses a file, those tags are untested: a misspelled
// tag compiles, validates, and silently drops the field it was meant to read.

func write(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("writing %s: %v", name, err)
	}
	return path
}

func TestLoadScenario(t *testing.T) {
	path := write(t, "s.yaml", `
kind: Scenario
name: granite-230b-h200-tp8
model: granite-5-230b
coefficients: [cost-model-primitives-h200]
engine_version: "0.29.0"
workload:
  shape: chatbot
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
  storage: [cpu_dram, nvme_gen4]
`)
	s, err := LoadScenario(path)
	if err != nil {
		t.Fatalf("LoadScenario: %v", err)
	}
	// Spot-check a field from each nesting level: a tag error at any depth would
	// leave a zero value that the type system cannot catch.
	if s.Name != "granite-230b-h200-tp8" {
		t.Errorf("name = %q", s.Name)
	}
	// The workload is a sum-type binding now, not a bare name: the distributional arm
	// names a catalog shape under `shape`.
	if s.Workload == nil || s.Workload.Shape != "chatbot" || s.Workload.Trace != nil {
		t.Errorf("workload binding did not parse as a shape ref: %+v", s.Workload)
	}
	// The hardware and fabric references and the storage inventory are folded into
	// the cluster rather than sitting loose at the top level.
	if s.Cluster.Hardware != "h200" {
		t.Errorf("cluster.hardware = %q, want h200", s.Cluster.Hardware)
	}
	if s.Cluster.GPUsPerNode != 8 {
		t.Errorf("gpus_per_node = %d, want 8", s.Cluster.GPUsPerNode)
	}
	if len(s.Cluster.Storage) != 2 || s.Cluster.Storage[0] != "cpu_dram" {
		t.Errorf("cluster.storage did not parse: %v", s.Cluster.Storage)
	}
	if rep := Validate(Bundle{Scenario: s}); !rep.Field.OK() {
		t.Errorf("a loaded scenario failed field validation:\n%s", rep.Field.Error())
	}
}

// TestLoadScenarioWithTraceBinding exercises the trace arm of the workload binding end to
// end: the data-file path, the integrity/provenance fields, and the small header metadata
// (including the server sub-block and the per-class SLO targets) must all parse through
// the strict loader, and the loaded scenario must validate. A tag error anywhere in the
// nested trace reference would leave a zero value the type system cannot catch, which is
// the one failure this package exists to rule out.
func TestLoadScenarioWithTraceBinding(t *testing.T) {
	path := write(t, "s.yaml", `
kind: Scenario
name: granite-230b-h200-replay
model: granite-5-230b
coefficients: [cost-model-primitives-h200]
engine_version: "0.29.0"
workload:
  trace:
    data: traces/agentic-run.csv
    sha256: `+strings.Repeat("a", 64)+`
    rows: 1048576
    header:
      trace_version: 3
      time_unit: microseconds
      mode: real
      workload_seed: 0
      server:
        type: vllm
        model: granite-5-230b
        tensor_parallel: 8
        max_num_seqs: 1024
        block_size: 16
        gpu_memory_utilization: 0.9
        max_model_len: 131072
      goodput_slo_targets:
        critical:
          ttft_ms: 500
          itl_ms: 50
          e2e_ms: 30000
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
`)
	s, err := LoadScenario(path)
	if err != nil {
		t.Fatalf("LoadScenario: %v", err)
	}
	b := s.Workload
	if b == nil || b.Shape != "" || b.Trace == nil {
		t.Fatalf("workload binding did not parse as a trace ref: %+v", b)
	}
	tr := b.Trace
	if tr.Data != "traces/agentic-run.csv" || tr.Rows != 1048576 || len(tr.SHA256) != 64 {
		t.Errorf("trace reference fields did not parse: %+v", tr)
	}
	h := tr.Header
	if h.Version != 3 || h.TimeUnit != "microseconds" || h.Mode != workload.ModeReal {
		t.Errorf("trace header scalars did not parse: %+v", h)
	}
	// workload_seed: 0 is a recorded zero, which the pointer must preserve as distinct
	// from absent.
	if h.WorkloadSeed == nil || *h.WorkloadSeed != 0 {
		t.Errorf("workload_seed did not parse as a recorded zero: %v", h.WorkloadSeed)
	}
	if h.Server == nil || h.Server.TensorParallel != 8 || h.Server.GPUMemoryUtilization != 0.9 {
		t.Errorf("trace server block did not parse: %+v", h.Server)
	}
	crit, ok := h.GoodputSLOTargets["critical"]
	if !ok || crit.TTFTMs != 500 || crit.ITLMs != 50 || crit.E2EMs != 30000 {
		t.Errorf("goodput_slo_targets did not parse: %+v", h.GoodputSLOTargets)
	}
	if rep := Validate(Bundle{Scenario: s}); !rep.Field.OK() {
		t.Errorf("a loaded trace-bound scenario failed field validation:\n%s", rep.Field.Error())
	}
}

// TestLoadScenarioRejectsUnknownTraceField pins strict decoding through EVERY nesting
// level the trace reference introduces: a misspelled sub-key must fail loudly rather than
// leave a zero value, the same guarantee the top-level loaders give. The trace types are
// plain structs — none implements a custom UnmarshalYAML, which is what would silently
// drop the decoder's KnownFields setting (see internal/validate/strict.go) — so strict
// decoding recurses through all of them for free. That is exactly the kind of property
// that is assumed and then quietly lost, so each new level is pinned: at the
// `workload` binding itself, directly under `trace`, under its `header`, under the
// `header.server` sub-block, and inside a `header.goodput_slo_targets.<class>` map VALUE (the one
// struct the feature reaches only through a map, where it is least obvious the strict
// check still applies).
func TestLoadScenarioRejectsUnknownTraceField(t *testing.T) {
	// head is a valid scenario up to the workload block; each case supplies the workload
	// subtree with an injection that must be rejected, naming the stray key.
	cases := []struct {
		name      string
		traceYAML string
		badKey    string
	}{
		{
			name: "unknown key under the workload binding",
			traceYAML: `  shape: chatbot
  invalid_field: yes`,
			badKey: "invalid_field",
		},
		{
			name: "unknown key directly under trace",
			traceYAML: `  trace:
    data: traces/run.csv
    rowz: 10
    header:
      trace_version: 3
      time_unit: microseconds
      mode: real`,
			badKey: "rowz",
		},
		{
			name: "unknown key under trace.header",
			traceYAML: `  trace:
    data: traces/run.csv
    header:
      trace_version: 3
      time_unit: microseconds
      mode: real
      time_uint: microseconds`,
			badKey: "time_uint",
		},
		{
			name: "unknown key under trace.header.server",
			traceYAML: `  trace:
    data: traces/run.csv
    header:
      trace_version: 3
      time_unit: microseconds
      mode: real
      server:
        tensor_parallel: 8
        tensor_paralel: 8`,
			badKey: "tensor_paralel",
		},
		{
			name: "unknown key inside a goodput_slo_targets map value",
			traceYAML: `  trace:
    data: traces/run.csv
    header:
      trace_version: 3
      time_unit: microseconds
      mode: real
      goodput_slo_targets:
        critical:
          ttft_ms: 500
          ttft_mss: 500`,
			badKey: "ttft_mss",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := write(t, "s.yaml", `
kind: Scenario
name: typo
model: granite-5-230b
coefficients: [c]
engine_version: "0.29.0"
workload:
`+tc.traceYAML+`
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
`)
			_, err := LoadScenario(path)
			if err == nil {
				t.Fatalf("a misspelled %q was accepted; the trace would validate while silently dropping it", tc.badKey)
			}
			if !strings.Contains(err.Error(), tc.badKey) {
				t.Errorf("the error should name the offending field %q, got: %v", tc.badKey, err)
			}
		})
	}
}

// TestLoadWorkload is the by-filename identity contract for a workload Shape. The file
// carries no name — exactly like a chip or a fabric — so a nameless document must load,
// LoadWorkload must stamp Shape.Name from the path stem, and the loaded shape must then
// validate clean (Validate requires a Name, now always loader-supplied). This is the
// catalog's target form from issue #21: a chatbot.yaml that is prefix/prompt/output and
// nothing else.
func TestLoadWorkload(t *testing.T) {
	path := write(t, "chatbot.yaml", `
prefix_tokens: 0
prompt:
  tokens: 256
  tokens_stdev: 100
  tokens_min: 2
  tokens_max: 800
output:
  tokens: 256
  tokens_stdev: 100
  tokens_min: 1
  tokens_max: 1024
`)
	w, err := LoadWorkload(path)
	if err != nil {
		t.Fatalf("LoadWorkload: %v", err)
	}
	// Identity is the filename stem, as with LoadChip/LoadFabric.
	if w.Name != "chatbot" {
		t.Errorf("name = %q; it should come from the filename", w.Name)
	}
	// Spot-check a field from each nesting level: a tag error at any depth would leave a
	// zero value the type system cannot catch.
	if w.PrefixTokens != 0 {
		t.Errorf("prefix_tokens = %d, want 0", w.PrefixTokens)
	}
	if w.Prompt.Mean != 256 || w.Prompt.StdDev != 100 || w.Prompt.Min != 2 || w.Prompt.Max != 800 {
		t.Errorf("prompt distribution did not parse: %+v", w.Prompt)
	}
	if w.Output.Mean != 256 || w.Output.Max != 1024 {
		t.Errorf("output distribution did not parse: %+v", w.Output)
	}
	// A loader-named shape validates: the Name requirement is satisfied by the stamp, not
	// by a line in the file.
	if p := w.Validate(); !p.OK() {
		t.Errorf("a loaded workload failed validation:\n%s", p.Error())
	}
}

// TestLoadWorkloadRejectsInFileName pins that identity is a filename fact, not a file
// field: an in-file `name:` is no longer part of the contract, so under strict decoding
// it is an unknown field and must fail loudly rather than quietly become a second source
// of truth for the identity the filename already fixes. The Name field is tagged
// `yaml:"-"`, so yaml.v3 maps no `name` key to it and KnownFields rejects the stray key.
func TestLoadWorkloadRejectsInFileName(t *testing.T) {
	path := write(t, "chatbot.yaml", `
name: not-chatbot
prefix_tokens: 0
prompt:
  tokens: 256
output:
  tokens: 256
`)
	if _, err := LoadWorkload(path); err == nil {
		t.Fatal("an in-file name was accepted; the filename and the field could disagree")
	} else if !strings.Contains(err.Error(), "name") {
		t.Errorf("the error should name the offending field %q, got: %v", "name", err)
	}
}

func TestLoadDeployment(t *testing.T) {
	path := write(t, "d.yaml", `
kind: Deployment
name: granite-230b-h200-tp8
pools:
  - role: colocated
    nodes: 1
    parallel:
      tp: 8
      pp: 1
      dp: 1
    engine:
      cache_dtype: fp8
      block_size: 16
      max_num_batched_tokens: 32768
      gpu_memory_utilization: 0.9
      scheduling_policy: priority
      eplb:
        enable_eplb: true
        num_redundant_experts: 32
      dbo:
        enable_dbo: true
        dbo_decode_token_threshold: 32
`)
	d, err := LoadDeployment(path)
	if err != nil {
		t.Fatalf("LoadDeployment: %v", err)
	}
	if d.Name != "granite-230b-h200-tp8" {
		t.Errorf("name = %q", d.Name)
	}
	if len(d.Pools) != 1 || d.Pools[0].Parallel.TP != 8 {
		t.Fatalf("pools did not parse: %+v", d.Pools)
	}
	e := d.Pools[0].Engine
	if e.CacheDType != "fp8" {
		t.Errorf("cache_dtype = %q, want fp8", e.CacheDType)
	}
	if e.SchedulingPolicy != "priority" {
		t.Errorf("scheduling_policy = %q; the tag must match the engine's flag name",
			e.SchedulingPolicy)
	}
	if e.GPUMemoryUtilization != 0.9 {
		t.Errorf("gpu_memory_utilization = %v, want 0.9", e.GPUMemoryUtilization)
	}
	// The nested blocks use the engine's flag names rather than shortened ones.
	if e.EPLB == nil || !e.EPLB.Enabled || e.EPLB.NumRedundantExperts != 32 {
		t.Errorf("eplb did not parse: %+v", e.EPLB)
	}
	if e.DBO == nil || !e.DBO.Enabled || e.DBO.DecodeTokenThreshold != 32 {
		t.Errorf("dbo did not parse: %+v", e.DBO)
	}
	// Validate the loaded deployment on its own terms; the composition layer requires a
	// scenario alongside a deployment (TestDeploymentWithoutScenarioIsRejected covers that).
	if p := d.Validate(); !p.OK() {
		t.Errorf("a loaded deployment failed field validation:\n%s", p.Error())
	}
}

// TestLoadDeploymentWithPolicies loads the policy surface added in S2 — admission,
// routing with scorers, the BLIS-native scheduler, preemption, saturation detectors,
// and LoRA — through the strict loader, so the new YAML tags are exercised against a
// real decode rather than only in-memory construction.
func TestLoadDeploymentWithPolicies(t *testing.T) {
	path := write(t, "d.yaml", `
kind: Deployment
name: granite-230b-h200-tp8-policies
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine:
      cache_dtype: fp8
      scheduling_policy: priority
admission:
  policy: token-bucket
  token_bucket_capacity: 10000.0
  token_bucket_refill_rate: 1000.0
  admission_latency_us: 250
  slo_priorities:
    batch: 0
routing:
  policy: weighted
  routing_latency_us: 100
  scorers:
    - name: precise-prefix-cache
      weight: 2.0
    - name: queue-depth
      weight: 1.0
    - name: kv-utilization
      weight: 1.0
scheduler: priority-fcfs
preemption:
  policy: priority
saturation:
  detectors: [composite, peak-rate]
  composite:
    sensitivity: 1.5
  peak_rate:
    threshold: 0.5
  final_window: 30s
lora:
  adapter_capacity: 4
  load_base_latency_us: 1500.0
  step_overhead_tiers:
    8:
      k6: 0.02
      k7: 1.0
  adapters:
    - id: sql-adapter
      rank: 8
`)
	d, err := LoadDeployment(path)
	if err != nil {
		t.Fatalf("LoadDeployment: %v", err)
	}
	// The engine's own vLLM-mirroring scheduling_policy and the BLIS-native scheduler are
	// both set, and are distinct fields.
	if got := d.Pools[0].Engine.SchedulingPolicy; got != "priority" {
		t.Errorf("engine.scheduling_policy = %q, want priority", got)
	}
	if d.Scheduler != "priority-fcfs" {
		t.Errorf("scheduler = %q, want priority-fcfs", d.Scheduler)
	}
	if d.Admission == nil || d.Admission.Policy != deployment.AdmissionTokenBucket ||
		d.Admission.TokenBucketCapacity == nil || *d.Admission.TokenBucketCapacity != 10000 {
		t.Errorf("admission did not parse: %+v", d.Admission)
	}
	if d.Routing == nil || len(d.Routing.Scorers) != 3 ||
		d.Routing.Scorers[0].Name != deployment.ScorerPrecisePrefixCache {
		t.Errorf("routing scorers did not parse: %+v", d.Routing)
	}
	if d.Saturation == nil || len(d.Saturation.Detectors) != 2 ||
		d.Saturation.Composite == nil || d.Saturation.Composite.Sensitivity == nil {
		t.Errorf("saturation did not parse: %+v", d.Saturation)
	}
	if d.LoRA == nil || len(d.LoRA.Adapters) != 1 || d.LoRA.Adapters[0].Rank != 8 ||
		d.LoRA.StepOverheadTiers[8].K7 == nil {
		t.Errorf("lora did not parse: %+v", d.LoRA)
	}
	if p := d.Validate(); !p.OK() {
		t.Errorf("a loaded policy deployment failed field validation:\n%s", p.Error())
	}
}

// TestLoadRejectsUnknownPolicyField is the strict-loader property reaching into a nested
// policy block: a key the simulator never set, misspelled inside an otherwise valid
// block, must fail loudly rather than leave the setting its author intended silently
// dropped.
func TestLoadRejectsUnknownPolicyField(t *testing.T) {
	path := write(t, "d.yaml", `
kind: Deployment
name: typo
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine: {cache_dtype: fp8}
routing:
  policy: weighted
  scorers:
    - name: queue-depth
      wieght: 1.0
`)
	_, err := LoadDeployment(path)
	if err == nil {
		t.Fatal("a misspelled nested policy field was accepted; strict decoding must reach into policy blocks")
	}
	if !strings.Contains(err.Error(), "wieght") {
		t.Errorf("the error should name the offending field, got: %v", err)
	}
}

// TestLoadRejectsUnknownFields is the property that makes strict decoding worth
// having. A misspelled key must fail loudly rather than leave a zero value. The engine
// block it misspells lives in a deployment now, so that is what is loaded.
func TestLoadRejectsUnknownFields(t *testing.T) {
	path := write(t, "d.yaml", `
kind: Deployment
name: typo
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine:
      cache_dytpe: fp8
`)
	_, err := LoadDeployment(path)
	if err == nil {
		t.Fatal("a misspelled field was accepted; the deployment would validate while " +
			"silently omitting the setting its author intended")
	}
	if !strings.Contains(err.Error(), "cache_dytpe") {
		t.Errorf("the error should name the offending field, got: %v", err)
	}
}

func TestLoadModelGraph(t *testing.T) {
	path := write(t, "g.yaml", `
kind: ModelGraph
name: granite-5-230b
modality: text_only
derived_from:
  format: hf_config_json
  path: config.json
  sha256: `+strings.Repeat("a", 64)+`
  deriver_version: 1
global:
  hidden_size: 3072
  vocab_size: 100352
  weight_dtype: fp8
layer_kinds:
  - id: attn_moe
    nodes:
      - {op: Elementwise, role: input_norm}
      - {op: Attention, kind: gqa, n_q: 48, n_kv: 8, d_h: 64}
      - {op: GroupedGEMM, role: experts, n: 1536, k: 3072, experts: 224, top_k: 8, shared_experts: 1}
      - {op: All2All, role: moe, emit: expert_parallel}
    edges: [[0, 1], [1, 2], [2, 3]]
stack:
  pattern: [attn_moe]
  repeat: 72
`)
	g, err := LoadModelGraph(path)
	if err != nil {
		t.Fatalf("LoadModelGraph: %v", err)
	}
	if g.Stack.Layers() != 72 {
		t.Errorf("layers = %d, want 72", g.Stack.Layers())
	}
	if g.Global.WeightDType != "fp8" {
		t.Errorf("weight_dtype = %q", g.Global.WeightDType)
	}
	n := g.LayerKinds[0].Nodes
	if len(n) != 4 {
		t.Fatalf("parsed %d nodes, want 4", len(n))
	}
	// The attention node's kind uses the tag "kind", not "attention_kind".
	if n[1].AttentionKind != "gqa" || n[1].NumKVHeads != 8 || n[1].HeadDim != 64 {
		t.Errorf("attention node did not parse: %+v", n[1])
	}
	if n[2].Experts != 224 || n[2].TopK != 8 || n[2].SharedExperts != 1 {
		t.Errorf("grouped-GEMM node did not parse: %+v", n[2])
	}
	if n[3].Emit != model.EmitExpertParallel {
		t.Errorf("collective condition = %q, want %q", n[3].Emit, model.EmitExpertParallel)
	}
	if len(g.LayerKinds[0].Edges) != 3 || g.LayerKinds[0].Edges[2] != [2]int{2, 3} {
		t.Errorf("edges did not parse: %v", g.LayerKinds[0].Edges)
	}
	if p := g.Validate(); !p.OK() {
		t.Errorf("a loaded graph failed validation:\n%s", p.Error())
	}
}

// TestLoadModelIdentity parses a model.yaml identity manifest end to end: the name and
// every source provenance field must arrive through the strict loader, and the loaded
// card must validate. A tag error at any depth would leave a zero value the type system
// cannot catch — the one failure this package exists to rule out.
func TestLoadModelIdentity(t *testing.T) {
	path := write(t, "model.yaml", `
name: deepseek-v4-pro
source:
  provider: huggingface
  repo: deepseek-ai/DeepSeek-V4-Pro
  revision: b5968e9190ef611bbf34a7229255be88a0e937c1
  retrieved: 2026-10-02
`)
	id, err := LoadModelIdentity(path)
	if err != nil {
		t.Fatalf("LoadModelIdentity: %v", err)
	}
	// Unlike a chip or a workload, the name is a FILE field here, not the filename stem.
	if id.Name != "deepseek-v4-pro" {
		t.Errorf("name = %q; it is a file field for a model card", id.Name)
	}
	if id.Source == nil {
		t.Fatal("source did not parse")
	}
	if id.Source.Provider != "huggingface" || id.Source.Repo != "deepseek-ai/DeepSeek-V4-Pro" {
		t.Errorf("source identity did not parse: %+v", id.Source)
	}
	if id.Source.Revision != "b5968e9190ef611bbf34a7229255be88a0e937c1" || id.Source.Retrieved != "2026-10-02" {
		t.Errorf("source provenance did not parse: %+v", id.Source)
	}
	if p := id.Validate(); !p.OK() {
		t.Errorf("a loaded identity failed validation:\n%s", p.Error())
	}
}

// TestLoadModelIdentityRejectsUnknownField pins strict decoding through the nested source
// block: a misspelled sub-key must fail loudly rather than leave a zero value, the same
// guarantee every other loader gives. Identity and Source are plain structs — neither
// implements a custom UnmarshalYAML, which is what would silently drop the decoder's
// KnownFields setting — so strict decoding recurses through the source block for free, and
// this pins it at both levels.
func TestLoadModelIdentityRejectsUnknownField(t *testing.T) {
	cases := []struct {
		name   string
		body   string
		badKey string
	}{
		{
			name: "unknown key at the top level",
			body: `
name: m
provdier: huggingface
source:
  provider: huggingface
  repo: r
  revision: v`,
			badKey: "provdier",
		},
		{
			name: "unknown key inside source",
			body: `
name: m
source:
  provider: huggingface
  repo: r
  revision: v
  revison: v2`,
			badKey: "revison",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadModelIdentity(write(t, "model.yaml", tc.body))
			if err == nil {
				t.Fatalf("a misspelled %q was accepted; the card would validate while silently dropping it", tc.badKey)
			}
			if !strings.Contains(err.Error(), tc.badKey) {
				t.Errorf("the error should name the offending field %q, got: %v", tc.badKey, err)
			}
		})
	}
}

// TestLoadCatalogModelIdentitiesIfPresent parses the committed model.yaml files
// themselves — the strongest check of the tags, since a hand-written fixture agrees with
// whatever the tags happen to say where the real files do not. Gated on BLIS_CATALOG, and
// named to run under CI's parse-committed-data job (the LoadCatalog filter).
func TestLoadCatalogModelIdentitiesIfPresent(t *testing.T) {
	root := os.Getenv("BLIS_CATALOG")
	if root == "" {
		t.Skip("BLIS_CATALOG is unset: the model.yaml tags were NOT checked against the " +
			"committed catalog files, only against fixtures in this package")
	}
	cards, _ := filepath.Glob(filepath.Join(root, "models", "*", "model.yaml"))
	if len(cards) == 0 {
		t.Skipf("BLIS_CATALOG=%s has no models/*/model.yaml", root)
	}
	for _, path := range cards {
		id, err := LoadModelIdentity(path)
		if err != nil {
			t.Errorf("LoadModelIdentity(%s): %v", path, err)
			continue
		}
		// The stated name must match the directory, and the source provenance must be
		// whole: those are what make a card an identity rather than a note.
		dir := filepath.Base(filepath.Dir(path))
		if id.Name != dir {
			t.Errorf("%s: name %q does not match directory %q", path, id.Name, dir)
		}
		if p := id.Validate(); !p.OK() {
			t.Errorf("%s failed validation:\n%s", path, p.Error())
		}
	}
	t.Logf("parsed %d model.yaml identity files from the catalog", len(cards))
}

func TestLoadCoefficientSet(t *testing.T) {
	path := write(t, "c.yaml", `
kind: CoefficientSet
name: cost-model-primitives-h200
coefficients:
  - gemm_eps_max_bf16:
      value: 0.72
      units: dimensionless
      method: measured
      fitted: true
      scope:
        hardware: [h200]
      sources:
        - {kind: model, cite: "operator table", role: primary}
      ci95: [0.70, 0.74]
      validated: tpot
`)
	s, err := LoadCoefficientSet(path)
	if err != nil {
		t.Fatalf("LoadCoefficientSet: %v", err)
	}
	if len(s.Coefficients) != 1 {
		t.Fatalf("parsed %d entries, want 1", len(s.Coefficients))
	}
	e := s.Coefficients[0]
	if e.Value != 0.72 || e.Units != "dimensionless" || e.Method != "measured" {
		t.Errorf("entry did not parse: %+v", e)
	}
	if !e.Fitted {
		t.Error("fitted did not parse")
	}
	if len(e.Scope.Hardware) != 1 || e.Scope.Hardware[0] != "h200" {
		t.Errorf("scope did not parse: %+v", e.Scope)
	}
	if len(e.Sources) != 1 || e.Sources[0].Role != "primary" {
		t.Errorf("sources did not parse: %+v", e.Sources)
	}
	if e.CI95 == nil || e.CI95.Low != 0.70 {
		t.Errorf("ci95 did not parse: %+v", e.CI95)
	}
	// The name comes from the key the entry was nested under, not from a field.
	if e.Name != "gemm_eps_max_bf16" {
		t.Errorf("name = %q; it should come from the nesting key", e.Name)
	}
	// A scalar metric decodes to a one-element list, which is how the registry
	// writes the common case.
	if len(e.Validated) != 1 || e.Validated[0] != "tpot" {
		t.Errorf("validated = %v, want [tpot] from a scalar", e.Validated)
	}
	if p := s.Validate(); !p.OK() {
		t.Errorf("a loaded set failed validation:\n%s", p.Error())
	}
}

// TestLoadCatalogFilesIfPresent parses the committed catalog files themselves. It is
// the strongest check of the tags: a hand-written fixture agrees with whatever the
// tags happen to say, where the real files do not.
func TestLoadCatalogFilesIfPresent(t *testing.T) {
	root := os.Getenv("BLIS_CATALOG")
	if root == "" {
		t.Skip("BLIS_CATALOG is unset: the yaml tags were NOT checked against the " +
			"committed catalog files, only against fixtures in this package")
	}
	chips, _ := filepath.Glob(filepath.Join(root, "hardware", "*.yaml"))
	if len(chips) == 0 {
		t.Skipf("BLIS_CATALOG=%s has no hardware/*.yaml", root)
	}
	for _, path := range chips {
		c, err := LoadChip(path)
		if err != nil {
			t.Errorf("LoadChip(%s): %v", filepath.Base(path), err)
			continue
		}
		// Every field the schema declares required must have arrived.
		if c.BF16Peak <= 0 || c.MemoryGiB <= 0 || c.IntraNodeBwGBps <= 0 {
			t.Errorf("%s: a required field did not parse: %+v", filepath.Base(path), c)
		}
	}
	fabrics, _ := filepath.Glob(filepath.Join(root, "networks", "*.yaml"))
	for _, path := range fabrics {
		f, err := LoadFabric(path)
		if err != nil {
			t.Errorf("LoadFabric(%s): %v", filepath.Base(path), err)
			continue
		}
		if f.InterNodeBwGBps <= 0 {
			t.Errorf("%s: InterNodeBwGBps did not parse", filepath.Base(path))
		}
	}
	t.Logf("parsed %d chip and %d fabric files from the catalog", len(chips), len(fabrics))
}

// TestLoadRegistryFilesIfPresent does the same for coefficient sets.
func TestLoadRegistryFilesIfPresent(t *testing.T) {
	root := os.Getenv("BLIS_REGISTRY")
	if root == "" {
		t.Skip("BLIS_REGISTRY is unset: the coefficient tags were NOT checked " +
			"against the committed registry files")
	}
	sets, _ := filepath.Glob(filepath.Join(root, "coefficients", "*.yaml"))
	if len(sets) == 0 {
		t.Skipf("BLIS_REGISTRY=%s has no coefficients/*.yaml", root)
	}
	total := 0
	for _, path := range sets {
		s, err := LoadCoefficientSet(path)
		if err != nil {
			t.Errorf("LoadCoefficientSet(%s): %v", filepath.Base(path), err)
			continue
		}
		if len(s.Coefficients) == 0 {
			t.Errorf("%s: parsed no entries", filepath.Base(path))
			continue
		}
		// Every entry must have taken its name from the key it was nested under, and
		// must carry a unit and a method: those are what make a number usable.
		for i, e := range s.Coefficients {
			if e.Name == "" {
				t.Errorf("%s: coefficients[%d] parsed without a name", filepath.Base(path), i)
			}
			if e.Units == "" || e.Method == "" {
				t.Errorf("%s: %s parsed without units or method", filepath.Base(path), e.Name)
			}
		}
		total += len(s.Coefficients)
	}
	t.Logf("parsed %d sets holding %d coefficients from the registry", len(sets), total)
}

// TestCoefficientEntryFormIsEnforced covers the nested wire form's failure modes. The
// registry writes each entry as a single-key map; a flat entry or a multi-key one
// would leave the name ambiguous or empty, and an entry without a name cannot be
// looked up by the kernel that needs it.
func TestCoefficientEntryFormIsEnforced(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "flat entry with no nesting key",
			body: `
kind: CoefficientSet
name: s
coefficients:
  - value: 0.72
    units: dimensionless
`,
		},
		{
			name: "two keys, so the name is ambiguous",
			body: `
kind: CoefficientSet
name: s
coefficients:
  - first: {value: 1, units: dimensionless, method: assumed, fitted: false, scope: {tp: [8]}}
    second: {value: 2, units: dimensionless, method: assumed, fitted: false, scope: {tp: [8]}}
`,
		},
		{
			name: "entry is a scalar rather than a map",
			body: `
kind: CoefficientSet
name: s
coefficients:
  - just_a_string
`,
		},
		{
			name: "unknown field inside an entry",
			body: `
kind: CoefficientSet
name: s
coefficients:
  - a_coefficient:
      value: 1
      units: dimensionless
      method: assumed
      fitted: false
      scope: {tp: [8]}
      confidence: high
`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := LoadCoefficientSet(write(t, "c.yaml", tc.body)); err == nil {
				t.Fatal("expected a decode error, got none")
			}
		})
	}
}

// TestCatalogCommentKeysAreAcceptedButTyposAreNot: the catalog's provenance narrative
// lives in underscore-prefixed keys, which must pass, while a misspelled real field
// must still fail. Accepting everything would make strict decoding pointless.
func TestCatalogCommentKeysAreAcceptedButTyposAreNot(t *testing.T) {
	ok := write(t, "chip.yaml", `
Provenance: vendor_spec
TFlopsPeak: 989.5
TFlopsFP8: 1979.0
BwPeakTBs: 4.8
MemoryGiB: 141.0
IntraNodeBwGBps: 450
_comment: "narrative the catalog carries alongside every number"
_comment_interconnect: "a second one"
`)
	c, err := LoadChip(ok)
	if err != nil {
		t.Fatalf("comment keys were rejected: %v", err)
	}
	if c.Name != "chip" {
		t.Errorf("name = %q; it should come from the filename", c.Name)
	}
	if c.MemoryGiB != 141.0 {
		t.Errorf("MemoryGiB = %v, want 141", c.MemoryGiB)
	}

	bad := write(t, "chip.yaml", `
Provenance: vendor_spec
TFlopsPeak: 989.5
BwPeakTBs: 4.8
MemoryGiB: 141.0
IntraNodeBwGBps: 450
MemorGiB: 999
`)
	if _, err := LoadChip(bad); err == nil {
		t.Fatal("a misspelled field was accepted; the chip would validate with the " +
			"wrong capacity and nothing would report it")
	}
}
