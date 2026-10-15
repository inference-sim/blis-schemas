package measurement

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"
)

func decodeStrict(t *testing.T, b []byte, dst any) {
	t.Helper()
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	if err := dec.Decode(dst); err != nil {
		t.Fatal(err)
	}
}

func canonical() *MeasuredRun {
	r := exactRun()
	a := Assess(r, true)
	r.Verdict, r.Resolution = a.Verdict, a.Resolution
	return r
}

// TestGoldenRoundTrip pins the wire form: the fixture marshals to the committed bytes, and
// those bytes decode, strictly, back to the fixture and re-marshal to themselves.
func TestGoldenRoundTrip(t *testing.T) {
	cases := map[string]any{"run.yaml": canonical(), "sources.yaml": permitting()}
	for file, want := range cases {
		t.Run(file, func(t *testing.T) {
			path := filepath.Join("testdata", file)
			golden, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			b, err := Marshal(want)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(b, golden) {
				t.Fatalf("%s is not what the fixture marshals to:\n%s", path, b)
			}
			got := reflect.New(reflect.TypeOf(want).Elem()).Interface()
			decodeStrict(t, golden, got)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%s does not decode to the fixture", path)
			}
			again, err := Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(again, golden) {
				t.Fatal("a decoded run does not re-marshal to the same bytes")
			}
		})
	}
	var r MeasuredRun
	golden, _ := os.ReadFile(filepath.Join("testdata", "run.yaml"))
	decodeStrict(t, golden, &r)
	if p := r.ValidateWithSources(permitting()); !p.OK() {
		t.Fatalf("the golden run does not validate:\n%s", p.Error())
	}
}

func TestUnknownFieldIsAnError(t *testing.T) {
	golden, err := os.ReadFile(filepath.Join("testdata", "run.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	bad := bytes.Replace(golden, []byte("schema_version:"), []byte("verdict_override: EXACT\nschema_version:"), 1)
	var r MeasuredRun
	dec := yaml.NewDecoder(bytes.NewReader(bad))
	dec.KnownFields(true)
	if err := dec.Decode(&r); err == nil {
		t.Fatal("an unknown field decoded silently")
	}
}
