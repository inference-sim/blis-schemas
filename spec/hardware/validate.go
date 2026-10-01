package hardware

import "github.com/inference-sim/blis-schemas/internal/validate"

// Validate performs field-level validation of a chip: required positives,
// recognized provenance, and internal consistency between the packaging fields.
func (c *Chip) Validate() *validate.Problems {
	p := &validate.Problems{}
	if c.Name == "" {
		p.Field("name", "required")
	}
	if !c.Provenance.Valid() {
		p.Field("Provenance", "%q is not a recognized provenance", c.Provenance)
	}
	if c.BF16Peak <= 0 {
		p.Field("TFlopsPeak", "must be positive")
	}
	if c.MemoryBandwidthTBs <= 0 {
		p.Field("BwPeakTBs", "must be positive")
	}
	if c.MemoryGiB <= 0 {
		p.Field("MemoryGiB", "must be positive")
	}
	if c.IntraNodeBwGBps <= 0 {
		p.Field("IntraNodeBwGBps", "must be positive")
	}
	if c.FP8Peak < 0 {
		p.Field("TFlopsFP8", "must not be negative; omit it on parts without native FP8")
	}
	if c.NVFP4Peak < 0 {
		p.Field("TFlopsNVFP4", "must not be negative; omit it where the format is emulated")
	}
	// FP8 at or below BF16 would mean the narrower format buys nothing, which is not
	// a property any shipped part has. It is far more likely a transcription slip.
	if c.FP8Peak > 0 && c.FP8Peak <= c.BF16Peak {
		p.Field("TFlopsFP8",
			"FP8 peak %.1f does not exceed BF16 peak %.1f; check the transcription",
			c.FP8Peak, c.BF16Peak)
	}
	if c.SMCount < 0 {
		p.Field("SMCount", "must not be negative")
	}
	if c.GPUsPerRack > 0 && c.GPUsPerNode > 0 && c.GPUsPerRack%c.GPUsPerNode != 0 {
		p.Field("GPUsPerRack", "%d is not a multiple of GPUsPerNode %d",
			c.GPUsPerRack, c.GPUsPerNode)
	}
	if c.IntraRackBwGBps > 0 && c.GPUsPerRack == 0 {
		p.Field("IntraRackBwGBps",
			"an intra-rack bandwidth without GPUsPerRack describes a tier with no extent")
	}
	return p
}

// Validate performs field-level validation of a fabric.
func (f *Fabric) Validate() *validate.Problems {
	p := &validate.Problems{}
	if f.Name == "" {
		p.Field("name", "required")
	}
	if !f.Provenance.Valid() {
		p.Field("Provenance", "%q is not a recognized provenance", f.Provenance)
	}
	if f.InterNodeBwGBps <= 0 {
		p.Field("InterNodeBwGBps", "must be positive")
	}
	return p
}

// Validate performs field-level validation of a storage device class.
func (d *StorageDevice) Validate() *validate.Problems {
	p := &validate.Problems{}
	if d.Name == "" {
		p.Field("name", "required")
	}
	if d.ReadBandwidthMBs <= 0 {
		p.Field("read_bandwidth", "must be positive")
	}
	if d.WriteBandwidthMBs <= 0 {
		p.Field("write_bandwidth", "must be positive")
	}
	// Zero base latency would make an arbitrarily small transfer free, which no
	// device is. It is the term that dominates small transfers.
	if d.BaseLatencyUs <= 0 {
		p.Field("base_latency", "must be positive; it dominates small transfers")
	}
	return p
}

// IntraToInterRatio returns the bandwidth ratio a cross-node collective pays,
// given a chip and the fabric it is paired with. It is a method on neither type
// alone because the ratio is a property of the pairing, which is the reason the two
// schemas are separate.
func IntraToInterRatio(c Chip, f Fabric) float64 {
	if f.InterNodeBwGBps <= 0 {
		return 1
	}
	return c.IntraNodeBwGBps / f.InterNodeBwGBps
}
