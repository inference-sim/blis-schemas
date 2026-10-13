package deployment

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// max_model_len has three states, and each spelling the engine's flag accepts must land
// on the right one: auto and -1 both mean fit-to-memory, which is not the same request
// as leaving the field out.
func TestMaxModelLenDecodes(t *testing.T) {
	for _, c := range []struct {
		in        string
		want      MaxModelLen
		auto      bool
		stated    int
		statedSet bool
	}{
		{"{}", 0, false, 0, false},
		{"{max_model_len: auto}", MaxModelLenAuto, true, 0, false},
		{"{max_model_len: AUTO}", MaxModelLenAuto, true, 0, false},
		{"{max_model_len: -1}", MaxModelLenAuto, true, 0, false},
		{"{max_model_len: 131072}", 131072, false, 131072, true},
		{"{max_model_len: 128k}", 128000, false, 128000, true},
		{"{max_model_len: 128K}", 131072, false, 131072, true},
		{"{max_model_len: 25.6k}", 25600, false, 25600, true},
		{"{max_model_len: 1M}", 1 << 20, false, 1 << 20, true},
		{"{max_model_len: '-1'}", MaxModelLenAuto, true, 0, false},
	} {
		var e Engine
		if err := yaml.Unmarshal([]byte(c.in), &e); err != nil {
			t.Fatalf("%s: %v", c.in, err)
		}
		n, ok := e.MaxModelLen.Stated()
		if e.MaxModelLen != c.want || e.MaxModelLen.Auto() != c.auto || n != c.stated || ok != c.statedSet {
			t.Errorf("%s: got %d (auto %v, stated %d %v)", c.in, e.MaxModelLen, e.MaxModelLen.Auto(), n, ok)
		}
	}
	for _, bad := range []string{"{max_model_len: atuo}", "{max_model_len: 1.5}",
		"{max_model_len: [1]}", "{max_model_len: 1.5K}", "{max_model_len: 12kb}"} {
		var e Engine
		if err := yaml.Unmarshal([]byte(bad), &e); err == nil || !strings.Contains(err.Error(), "max_model_len") {
			t.Errorf("%s: want a max_model_len error, got %v (value %d)", bad, err, e.MaxModelLen)
		}
	}
}

func TestMaxModelLenRoundTrips(t *testing.T) {
	out, err := yaml.Marshal(Engine{MaxModelLen: MaxModelLenAuto})
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(out)) != "max_model_len: auto" {
		t.Errorf("auto marshals as %q", out)
	}
	out, _ = yaml.Marshal(Engine{MaxModelLen: 4096})
	if strings.TrimSpace(string(out)) != "max_model_len: 4096" {
		t.Errorf("4096 marshals as %q", out)
	}
}

func TestMaxModelLenValidation(t *testing.T) {
	for _, c := range []struct {
		m    MaxModelLen
		want bool
	}{{0, true}, {MaxModelLenAuto, true}, {8192, true}, {-2, false}} {
		p := (&Deployment{Kind: "Deployment", Name: "d", Pools: []Pool{{
			Role: RoleColocated, Nodes: 1, Parallel: Parallelism{TP: 1, PP: 1, DP: 1},
			Engine: Engine{MaxModelLen: c.m}}}}).Validate()
		if p.OK() != c.want {
			t.Errorf("max_model_len %d: ok=%v, want %v\n%s", c.m, p.OK(), c.want, p.Error())
		}
	}
}
