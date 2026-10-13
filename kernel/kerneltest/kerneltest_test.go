package kerneltest

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// fake prices a handoff the way a real kernel does: a floor plus a payload over the NIC
// across nodes, or over NVLink on one node. skew, when set, breaks the law by charging a
// different time from the demand it reports.
type fake struct {
	kernel.Kernel // unused methods panic, which these tests never reach
	skew          time.Duration
}

func (f fake) PDTransferDemand(tokens int, from, to kernel.Placement) kernel.TransferDemand {
	if tokens <= 0 {
		return kernel.TransferDemand{}
	}
	d := kernel.TransferDemand{Bytes: int64(tokens) * 70272, Floor: 20 * time.Microsecond}
	if from.Node == to.Node {
		d.Links = []kernel.LinkUse{{Link: kernel.LinkID(fmt.Sprintf("node%d/nvlink", from.Node)), BytesPerSecond: 450e9}}
		return d
	}
	d.Links = []kernel.LinkUse{
		{Link: kernel.LinkID(fmt.Sprintf("node%d/nic/egress", from.Node)), BytesPerSecond: 50e9},
		{Link: kernel.LinkID(fmt.Sprintf("node%d/nic/ingress", to.Node)), BytesPerSecond: 50e9},
	}
	return d
}

func (f fake) PDTransferTime(tokens int, from, to kernel.Placement) time.Duration {
	return f.PDTransferDemand(tokens, from, to).AloneTime() + f.skew
}

func TestLawHoldsForAConformingKernel(t *testing.T) {
	TestPDTransfer(t, fake{}, PDCases())
}

func TestLawFailsForADivergingKernel(t *testing.T) {
	err := CheckPDTransfer(fake{skew: time.Nanosecond},
		PDCase{Tokens: 4096, From: kernel.Placement{Pool: deployment.RolePrefill},
			To: kernel.Placement{Node: 1, Pool: deployment.RoleDecode}})
	if err == nil || !strings.Contains(err.Error(), "PDTransferTime") {
		t.Fatalf("a kernel charging 1ns more than its demand passed: %v", err)
	}
}

// relinked names its NIC differently, as a kernel would that derived IDs from something
// other than the placements.
type relinked struct{ fake }

func (r relinked) PDTransferDemand(tokens int, from, to kernel.Placement) kernel.TransferDemand {
	d := r.fake.PDTransferDemand(tokens, from, to)
	for i := range d.Links {
		d.Links[i].Link = "other/" + d.Links[i].Link
	}
	return d
}

func TestLinksAgree(t *testing.T) {
	for _, c := range PDCases() {
		if err := CheckLinksAgree(fake{}, fake{skew: 1}, c); err != nil {
			t.Errorf("two kernels over one fabric: %v", err)
		}
	}
	c := PDCase{Tokens: 4096, To: kernel.Placement{Node: 1}}
	if CheckLinksAgree(fake{}, relinked{}, c) == nil {
		t.Error("kernels naming different links passed")
	}
}

func TestCasesCoverEveryLinkClass(t *testing.T) {
	var sameNode, sameRack, crossRack bool
	for _, c := range PDCases() {
		switch {
		case c.From.Node == c.To.Node:
			sameNode = true
		case c.From.Rack == c.To.Rack:
			sameRack = true
		default:
			crossRack = true
		}
	}
	if !sameNode || !sameRack || !crossRack {
		t.Errorf("cases miss a link class: same node %v, same rack %v, cross rack %v",
			sameNode, sameRack, crossRack)
	}
}
