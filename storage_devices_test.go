package blisschemas_test

import (
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/hardware"
)

// The committed storage tiers must load with their names intact: TierTime looks a tier
// up by name, so a device loaded anonymously prices every offload transfer as unknown.
func TestStorageDevicesLoadWithTheirNames(t *testing.T) {
	devices, err := schemas.LoadStorageDevices(
		"/Users/sri/Documents/Projects/blis-catalog/devices/storage.yaml")
	if err != nil {
		t.Skipf("catalog unavailable: %v", err)
	}
	want := map[string]bool{
		"nvme_gen4": false, "nvme_gen3": false, "sata_ssd": false,
		"cpu_dram": false, "s3": false,
	}
	for _, d := range devices {
		if d.Name == "" {
			t.Error("a device loaded with no name")
		}
		if _, ok := want[d.Name]; !ok {
			t.Errorf("unexpected tier %q", d.Name)
			continue
		}
		want[d.Name] = true
		if d.ReadBandwidthMBs <= 0 || d.WriteBandwidthMBs <= 0 {
			t.Errorf("%s: bandwidths did not load (%v read, %v write)",
				d.Name, d.ReadBandwidthMBs, d.WriteBandwidthMBs)
		}
		for _, p := range schemas.Validate(schemas.Bundle{
			Devices: []*hardware.StorageDevice{d},
		}).Problems() {
			t.Errorf("%s: %s", d.Name, p.String())
		}
	}
	for name, seen := range want {
		if !seen {
			t.Errorf("tier %q did not load", name)
		}
	}
}

// A CPU tier is orders of magnitude faster than object storage, and the cost model's
// tier arithmetic depends on that ordering rather than on the absolute figures.
func TestStorageTiersAreOrderedBySpeed(t *testing.T) {
	devices, err := schemas.LoadStorageDevices(
		"/Users/sri/Documents/Projects/blis-catalog/devices/storage.yaml")
	if err != nil {
		t.Skipf("catalog unavailable: %v", err)
	}
	byName := map[string]float64{}
	latency := map[string]float64{}
	for _, d := range devices {
		byName[d.Name] = d.ReadBandwidthMBs
		latency[d.Name] = d.BaseLatencyUs
	}
	if byName["cpu_dram"] <= byName["nvme_gen4"] {
		t.Errorf("cpu_dram (%v MB/s) should read faster than nvme_gen4 (%v)",
			byName["cpu_dram"], byName["nvme_gen4"])
	}
	if byName["nvme_gen4"] <= byName["nvme_gen3"] {
		t.Errorf("gen4 (%v) should read faster than gen3 (%v)",
			byName["nvme_gen4"], byName["nvme_gen3"])
	}
	if latency["s3"] <= latency["cpu_dram"]*100 {
		t.Errorf("s3 latency (%v us) should dwarf cpu_dram's (%v us)",
			latency["s3"], latency["cpu_dram"])
	}
}
