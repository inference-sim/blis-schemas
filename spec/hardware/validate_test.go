package hardware

import (
	"testing"

	"github.com/inference-sim/blis-schemas/vocab"
)

// h200 mirrors the committed catalog entry, so a drift between this repository's
// expectations and the catalog's contents shows up as a test failure rather than a
// silently wrong estimate.
func h200() *Chip {
	return &Chip{
		Name: "h200", Provenance: vocab.ProvenanceVendorSpec,
		BF16Peak: 989.5, FP8Peak: 1979.0, MemoryBandwidthTBs: 4.8,
		MemoryGiB: 141.0, IntraNodeBwGBps: 450, SMCount: 132, GPUsPerNode: 8,
	}
}

func ib400g() *Fabric {
	return &Fabric{Name: "ib-400g", Provenance: vocab.ProvenanceVendorSpec,
		InterNodeBwGBps: 50, RDMA: true}
}

func TestValidChipAndFabricPass(t *testing.T) {
	if p := h200().Validate(); !p.OK() {
		t.Fatalf("h200 rejected:\n%s", p.Error())
	}
	if p := ib400g().Validate(); !p.OK() {
		t.Fatalf("ib-400g rejected:\n%s", p.Error())
	}
}

// TestIntraToInterRatio checks the quantity a cross-node collective is priced
// against. It is a function of the pairing, which is why neither schema owns it.
func TestIntraToInterRatio(t *testing.T) {
	cases := []struct {
		intra, inter, want float64
	}{
		{450, 50, 9},  // NVLink 4 against InfiniBand NDR
		{450, 25, 18}, // the same chip behind RoCE
		{900, 900, 1}, // multi-node NVLink: no step at the boundary
		{450, 0, 1},   // an unstated fabric must not divide by zero
	}
	for _, c := range cases {
		chip := Chip{IntraNodeBwGBps: c.intra}
		fab := Fabric{InterNodeBwGBps: c.inter}
		if got := IntraToInterRatio(chip, fab); got != c.want {
			t.Errorf("ratio(%v, %v) = %v, want %v", c.intra, c.inter, got, c.want)
		}
	}
}

func TestChipRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Chip)
	}{
		{"no name", func(c *Chip) { c.Name = "" }},
		{"unknown provenance", func(c *Chip) { c.Provenance = "guessed" }},
		{"zero bf16 peak", func(c *Chip) { c.BF16Peak = 0 }},
		{"zero bandwidth", func(c *Chip) { c.MemoryBandwidthTBs = 0 }},
		{"zero memory", func(c *Chip) { c.MemoryGiB = 0 }},
		{"zero intra-node bandwidth", func(c *Chip) { c.IntraNodeBwGBps = 0 }},
		{"negative fp8 peak", func(c *Chip) { c.FP8Peak = -1 }},
		{"fp8 not above bf16", func(c *Chip) { c.FP8Peak = 500 }},
		{"rack not a multiple of node", func(c *Chip) { c.GPUsPerRack = 70 }},
		{"intra-rack bandwidth with no rack", func(c *Chip) { c.IntraRackBwGBps = 900 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := h200()
			tc.mutate(c)
			if p := c.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

func TestFabricAndDeviceRejects(t *testing.T) {
	bad := []*Fabric{
		{Provenance: vocab.ProvenanceVendorSpec, InterNodeBwGBps: 50}, // no name
		{Name: "x", Provenance: "invented", InterNodeBwGBps: 50},
		{Name: "x", Provenance: vocab.ProvenanceVendorSpec}, // zero bandwidth
	}
	for i, f := range bad {
		if p := f.Validate(); p.OK() {
			t.Errorf("fabric case %d: expected rejection", i)
		}
	}
	good := &StorageDevice{Name: "nvme_gen4", ReadBandwidthMBs: 7000,
		WriteBandwidthMBs: 5000, BaseLatencyUs: 80}
	if p := good.Validate(); !p.OK() {
		t.Fatalf("nvme_gen4 rejected:\n%s", p.Error())
	}
	badDevices := []*StorageDevice{
		{ReadBandwidthMBs: 7000, WriteBandwidthMBs: 5000, BaseLatencyUs: 80},
		{Name: "d", WriteBandwidthMBs: 5000, BaseLatencyUs: 80},
		{Name: "d", ReadBandwidthMBs: 7000, BaseLatencyUs: 80},
		{Name: "d", ReadBandwidthMBs: 7000, WriteBandwidthMBs: 5000}, // zero latency
	}
	for i, d := range badDevices {
		if p := d.Validate(); p.OK() {
			t.Errorf("device case %d: expected rejection", i)
		}
	}
}
