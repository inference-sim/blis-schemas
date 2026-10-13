package kernel

import (
	"math"
	"strings"
	"testing"
	"time"
)

func TestAloneTimeIsSlowestLinkOrFloor(t *testing.T) {
	nic := LinkUse{Link: "node0/nic/egress", BytesPerSecond: 50e9}
	nvl := LinkUse{Link: "node0/nvlink", BytesPerSecond: 450e9}
	cases := []struct {
		name string
		d    TransferDemand
		want time.Duration
	}{
		{"no transfer", TransferDemand{}, 0},
		{"no link costs the floor", TransferDemand{Bytes: 1 << 30, Floor: 7 * time.Microsecond}, 7 * time.Microsecond},
		{"bandwidth bound", TransferDemand{Bytes: 50e9, Floor: time.Millisecond, Links: []LinkUse{nic}}, time.Second},
		{"floor bound", TransferDemand{Bytes: 50, Floor: time.Millisecond, Links: []LinkUse{nic}}, time.Millisecond},
		// The payload crosses every link in series, so the slowest binds whatever the order.
		{"slowest link binds", TransferDemand{Bytes: 50e9, Links: []LinkUse{nvl, nic}}, time.Second},
		{"zero bytes on a link costs the floor", TransferDemand{Floor: 3, Links: []LinkUse{nic}}, 3},
		{"saturates", TransferDemand{Bytes: math.MaxInt64, Links: []LinkUse{{Link: "slow", BytesPerSecond: 1e-3}}}, time.Duration(math.MaxInt64)},
	}
	for _, c := range cases {
		if got := c.d.AloneTime(); got != c.want {
			t.Errorf("%s: AloneTime() = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestCheckRejectsMalformedDemands(t *testing.T) {
	ok := LinkUse{Link: "a", BytesPerSecond: 1}
	cases := map[string]TransferDemand{
		"negative":       {Bytes: -1},
		"negative floor": {Floor: -1},
		"unnamed link":   {Links: []LinkUse{{BytesPerSecond: 1}}},
		"named twice":    {Links: []LinkUse{ok, ok}},
		"zero rate":      {Links: []LinkUse{{Link: "a"}}},
		"NaN rate":       {Links: []LinkUse{{Link: "a", BytesPerSecond: math.NaN()}}},
		"infinite rate":  {Links: []LinkUse{{Link: "a", BytesPerSecond: math.Inf(1)}}},
	}
	for name, d := range cases {
		if d.Check() == nil {
			t.Errorf("%s: Check() accepted %+v", name, d)
		}
	}
	if err := (TransferDemand{Bytes: 1, Links: []LinkUse{ok}}).Check(); err != nil {
		t.Errorf("a well-formed demand was rejected: %v", err)
	}
	if err := (TransferDemand{Links: []LinkUse{ok, ok}}).Check(); !strings.Contains(err.Error(), "twice") {
		t.Errorf("duplicate link: got %v", err)
	}
}
