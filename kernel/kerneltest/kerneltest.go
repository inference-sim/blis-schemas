// Package kerneltest holds checks of the kernel.Kernel contract that any implementation
// can run against itself, in the way testing/fstest checks an fs.FS.
//
// A law stated only in a doc comment is kept by whoever reads it. Shipping it as code
// means every implementation, and every fake a simulator's tests build, is held to the
// same reading of it.
package kerneltest

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// PDCase is one prefill-to-decode handoff to check.
type PDCase struct {
	Tokens   int
	From, To kernel.Placement
}

func (c PDCase) String() string {
	return fmt.Sprintf("%d tokens from %+v to %+v", c.Tokens, c.From, c.To)
}

// PDCases returns handoffs covering the three link classes a placement pair can select
// (same node, same rack on another node, another rack) at token counts from none to a
// long context. An implementation adds cases for any boundary its fabric model has
// beyond these.
func PDCases() []PDCase {
	from := kernel.Placement{Node: 0, Rack: 0, Pool: deployment.RolePrefill}
	tos := []kernel.Placement{
		{Node: 0, Rack: 0, Pool: deployment.RoleDecode},
		{Node: 1, Rack: 0, Pool: deployment.RoleDecode},
		{Node: 2, Rack: 1, Pool: deployment.RoleDecode},
	}
	var out []PDCase
	for _, to := range tos {
		for _, t := range []int{0, 1, 17, 4096, 131072} {
			out = append(out, PDCase{Tokens: t, From: from, To: to})
		}
	}
	return out
}

// CheckPDTransfer checks one case against the PDTransferDemand contract: the demand is
// well formed, asking for it twice gives the same answer, and alone it costs exactly
// what PDTransferTime charges.
func CheckPDTransfer(k kernel.Kernel, c PDCase) error {
	d := k.PDTransferDemand(c.Tokens, c.From, c.To)
	if err := d.Check(); err != nil {
		return fmt.Errorf("%v: malformed demand: %w", c, err)
	}
	if again := k.PDTransferDemand(c.Tokens, c.From, c.To); !reflect.DeepEqual(d, again) {
		return fmt.Errorf("%v: two calls gave different demands, %+v and %+v", c, d, again)
	}
	got, want := d.AloneTime(), k.PDTransferTime(c.Tokens, c.From, c.To)
	if got != want {
		return fmt.Errorf("%v: demand %+v costs %v alone but PDTransferTime is %v",
			c, d, got, want)
	}
	return nil
}

// CheckLinksAgree checks the comparability TransferDemand promises: two kernels built from
// one deployment, such as its prefill and decode kernels, name the same links at the same
// rates for the same handoff. Without it a simulator could not tell that a transfer one
// kernel priced shares a link with one the other priced.
func CheckLinksAgree(a, b kernel.Kernel, c PDCase) error {
	la := a.PDTransferDemand(c.Tokens, c.From, c.To).Links
	lb := b.PDTransferDemand(c.Tokens, c.From, c.To).Links
	key := func(ls []kernel.LinkUse) map[kernel.LinkID]float64 {
		m := make(map[kernel.LinkID]float64, len(ls))
		for _, l := range ls {
			m[l.Link] = l.BytesPerSecond
		}
		return m
	}
	if !reflect.DeepEqual(key(la), key(lb)) {
		return fmt.Errorf("%v: the two kernels name different links, %v and %v", c, la, lb)
	}
	return nil
}

// TestPDTransfer runs CheckPDTransfer over cases, reporting every failure.
func TestPDTransfer(t testing.TB, k kernel.Kernel, cases []PDCase) {
	t.Helper()
	for _, c := range cases {
		if err := CheckPDTransfer(k, c); err != nil {
			t.Error(err)
		}
	}
}
