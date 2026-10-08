package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeSet writes one coefficient set under <dir>/coefficients/<name>, creating any
// intermediate directories so name may be nested (e.g. "sub/deep.yaml").
func writeSet(t *testing.T, dir, name, body string) {
	t.Helper()
	path := filepath.Join(dir, "coefficients", name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// A well-formed set validates: exit 0, and the summary names it on stdout.
const validSet = `
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
`

func TestRunValidSetPasses(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "primitives.yaml", validSet)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr:\n%s", code, errBuf.String())
	}
	if !strings.Contains(out.String(), "all 1 set(s) validate") {
		t.Errorf("stdout missing the summary line:\n%s", out.String())
	}
}

// A set with two entries sharing one name AND one scope is rejected: a resolver would
// keep whichever came last, so the file would mean something other than it says.
const duplicateSet = `
kind: CoefficientSet
name: dup
coefficients:
  - gemm_eps_max_bf16:
      value: 0.72
      units: dimensionless
      method: measured
      scope:
        hardware: [h200]
  - gemm_eps_max_bf16:
      value: 0.80
      units: dimensionless
      method: measured
      scope:
        hardware: [h200]
`

// The command must surface a duplicate as a failure (exit 1) with the finding on stderr.
func TestRunDuplicateScopeFails(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "dup.yaml", duplicateSet)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1; stderr:\n%s", code, errBuf.String())
	}
	if !strings.Contains(errBuf.String(), "duplicate entry name") {
		t.Errorf("stderr did not report the duplicate:\n%s", errBuf.String())
	}
}

// A directory holding both passing and failing sets must process all of them, not stop
// at the first failure: exit 1, the valid set still summarized on stdout, the bad one
// marked FAILED on stdout, and the aggregate tally on stderr. This is the multi-file
// aggregation the per-PR CI gate depends on.
func TestRunMixedPassAndFailAggregates(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "good.yaml", validSet)
	writeSet(t, dir, "dup.yaml", duplicateSet)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1; stderr:\n%s", code, errBuf.String())
	}
	so := out.String()
	if !strings.Contains(so, "good.yaml") || !strings.Contains(so, "coefficient(s)") {
		t.Errorf("stdout missing the passing set's summary row:\n%s", so)
	}
	if !strings.Contains(so, "dup.yaml") || !strings.Contains(so, "FAILED") {
		t.Errorf("stdout missing the failing set's row:\n%s", so)
	}
	if !strings.Contains(errBuf.String(), "1 of 2 set(s) failed") {
		t.Errorf("stderr missing the aggregate tally:\n%s", errBuf.String())
	}
}

// A set that fails to parse — here an unknown key inside an entry, which the strict
// loader rejects — is a load failure, also exit 1.
func TestRunUnparseableSetFails(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "bad.yaml", `
kind: CoefficientSet
name: bad
coefficients:
  - x:
      value: 1
      units: dimensionless
      method: measured
      bogus_key: 1
      scope:
        hardware: [h200]
`)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1; stderr:\n%s", code, errBuf.String())
	}
}

// prefixLines attributes every line of a multi-line diagnostic and normalizes CRLF, so a
// decoder error that spans lines does not leave later lines unprefixed or trailing a
// carriage return.
func TestPrefixLinesNormalizesCRLF(t *testing.T) {
	got := prefixLines("c.yaml", "first\r\nsecond\r\n")
	want := "c.yaml: first\nc.yaml: second\n"
	if got != want {
		t.Errorf("prefixLines = %q, want %q", got, want)
	}
}

// A non-finite coefficient value is rejected: the schema's FiniteField check runs inside
// (*coefficient.Set).Validate, so the command enforces it for free — the point of
// validating through the same schema a consumer loads these sets with.
func TestRunRejectsNonFiniteValue(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "nan.yaml", `
kind: CoefficientSet
name: nan
coefficients:
  - x:
      value: .nan
      units: dimensionless
      method: measured
      scope:
        hardware: [h200]
`)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 1 {
		t.Fatalf("exit = %d, want 1; stderr:\n%s", code, errBuf.String())
	}
	if !strings.Contains(errBuf.String(), "finite number") {
		t.Errorf("stderr did not report the non-finite value:\n%s", errBuf.String())
	}
}

func TestRunUsageOnWrongArgs(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := run(nil, &out, &errBuf); code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(errBuf.String(), "usage:") {
		t.Errorf("stderr missing usage:\n%s", errBuf.String())
	}
}

// A root with no coefficient sets is misuse, not success: exit 2 naming the path, so a
// mistyped root cannot pass as "nothing to check".
func TestRunNoSetsIsMisuse(t *testing.T) {
	dir := t.TempDir()
	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 2 {
		t.Fatalf("exit = %d, want 2; stderr:\n%s", code, errBuf.String())
	}
	if !strings.Contains(errBuf.String(), "no coefficient sets") {
		t.Errorf("stderr did not name the missing path:\n%s", errBuf.String())
	}
}

// Discovery matches blis-registry's validator/validate.py: a recursive walk under
// coefficients/ taking both .yaml and .yml, case-insensitively. A .yml set, a set in a
// nested subdirectory, and an upper-case extension must all be found and validated —
// otherwise this gate could pass over a set the registry's own gate rejects, the exact
// two-gates-diverge failure the registry CI job (inference-sim/blis-registry#20) exists
// to prevent.
func TestRunDiscoversYmlNestedAndUppercase(t *testing.T) {
	dir := t.TempDir()
	writeSet(t, dir, "flat.yaml", validSet)
	writeSet(t, dir, "alt.yml", validSet)
	writeSet(t, dir, filepath.Join("nested", "deep.yaml"), validSet)
	writeSet(t, dir, "UPPER.YAML", validSet)

	var out, errBuf bytes.Buffer
	if code := run([]string{dir}, &out, &errBuf); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr:\n%s", code, errBuf.String())
	}
	so := out.String()
	for _, want := range []string{"flat.yaml", "alt.yml", filepath.Join("nested", "deep.yaml"), "UPPER.YAML"} {
		if !strings.Contains(so, want) {
			t.Errorf("discovery missed %q:\n%s", want, so)
		}
	}
	if !strings.Contains(so, "all 4 set(s) validate") {
		t.Errorf("stdout summary wrong, want all 4 valid:\n%s", so)
	}
}
