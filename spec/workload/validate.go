package workload

import "github.com/inference-sim/blis-schemas/internal/validate"

// Validate performs field-level validation of a workload binding: that exactly one arm
// is chosen, and that the chosen arm is itself well formed. A binding is validated only
// when present; a Scenario with no traffic carries no binding, which this type never
// sees (the Scenario skips a nil one).
func (b *Binding) Validate() *validate.Problems {
	p := &validate.Problems{}
	hasShape := b.Shape != ""
	hasTrace := b.Trace != nil
	switch {
	case hasShape && hasTrace:
		p.Errorf("a workload is a shape or a trace, not both; set shape or trace, not the two together")
	case !hasShape && !hasTrace:
		p.Errorf("a workload binding chooses a shape or a trace; omit the workload entirely for a scenario with no traffic")
	case hasTrace:
		p.Merge("trace", b.Trace.Validate())
	}
	return p
}

// Validate performs field-level validation of a trace reference: that it names a data
// file, that any integrity fields are well formed, and that the header is valid.
func (t *TraceRef) Validate() *validate.Problems {
	p := &validate.Problems{}
	if t.Data == "" {
		p.Field("data", "required: the path to the bulk per-request data CSV")
	}
	// Optional, but a digest that is present and the wrong length is a copy-paste error
	// that would never match the file it is meant to guard.
	if t.SHA256 != "" && len(t.SHA256) != 64 {
		p.Field("sha256",
			"must be a 64-character hex digest of the data file, got %d characters",
			len(t.SHA256))
	}
	if t.Rows < 0 {
		p.Field("rows", "must not be negative")
	}
	p.Merge("header", t.Header.Validate())
	return p
}

// Validate performs field-level validation of the trace header metadata.
func (h *TraceHeader) Validate() *validate.Problems {
	p := &validate.Problems{}
	if h.Version < 1 {
		p.Field("trace_version",
			"must be at least 1: a trace reference states the TraceV2 version its data file was written against")
	}
	if h.TimeUnit == "" {
		p.Field("time_unit", "required: names the unit of the data file's timestamp columns")
	}
	switch {
	case h.Mode == "":
		p.Field("mode", "required: names the pipeline that produced the trace")
	case !h.Mode.Valid():
		p.Field("mode", "%q is not one of %v", h.Mode, AllModes())
	}
	if h.Server != nil {
		p.Merge("server", h.Server.Validate())
	}
	for class, targets := range h.SLOTargets {
		if class == "" {
			p.Field("slo_targets", "an SLO class key must not be empty")
			continue
		}
		p.Merge("slo_targets."+class, targets.Validate())
	}
	return p
}

// Validate performs field-level validation of a trace's server-config provenance.
func (s *TraceServer) Validate() *validate.Problems {
	p := &validate.Problems{}
	for field, v := range map[string]int{
		"tensor_parallel": s.TensorParallel, "max_num_seqs": s.MaxNumSeqs,
		"block_size": s.BlockSize,
	} {
		if v < 0 {
			p.Field(field, "must not be negative")
		}
	}
	if s.MaxModelLen < 0 {
		p.Field("max_model_len", "must not be negative")
	}
	// gpu_memory_utilization is a non-pointer float, so an omitted one is zero. Zero is
	// the unset sentinel and a stated fraction lies in (0, 1], so the accepted range is
	// [0, 1]; the message states [0, 1] rather than (0, 1] so it does not read as
	// rejecting the zero the check deliberately allows. Mirrors deployment.Engine.
	if s.GPUMemoryUtilization < 0 || s.GPUMemoryUtilization > 1 {
		p.Field("gpu_memory_utilization",
			"must lie in [0, 1] (0 means unset), got %v", s.GPUMemoryUtilization)
	}
	return p
}

// Validate performs field-level validation of one SLO class's latency targets.
func (t SLODimTargets) Validate() *validate.Problems {
	p := &validate.Problems{}
	for field, v := range map[string]float64{
		"ttft_ms": t.TTFTMs, "itl_ms": t.ITLMs, "e2e_ms": t.E2EMs,
	} {
		if v < 0 {
			p.Field(field, "must not be negative")
		}
	}
	// A class that constrains no dimension is a declaration that says nothing; it would
	// pass goodput for every completion and read as a target that was simply never met.
	if t.TTFTMs == 0 && t.ITLMs == 0 && t.E2EMs == 0 {
		p.Errorf("sets no target on any of ttft_ms, itl_ms, e2e_ms; omit the class instead of declaring it empty")
	}
	return p
}

// Validate performs field-level validation of a traffic shape.
func (s *Shape) Validate() *validate.Problems {
	p := &validate.Problems{}
	if s.Name == "" {
		p.Field("name", "required")
	}
	if s.PrefixTokens < 0 {
		p.Field("prefix_tokens", "must not be negative")
	}
	s.Prompt.validate(p, "prompt")
	s.Output.validate(p, "output")
	// A shared prefix longer than the prompt it is a prefix of describes no request.
	if s.Prompt.Mean > 0 && s.PrefixTokens > s.Prompt.Mean {
		p.Field("prefix_tokens",
			"%d exceeds the mean prompt length %d, so no prompt contains the prefix",
			s.PrefixTokens, s.Prompt.Mean)
	}
	return p
}

func (d Distribution) validate(p *validate.Problems, at string) {
	if d.Mean < 1 {
		p.Field(at+".tokens", "must be positive")
	}
	if d.StdDev < 0 {
		p.Field(at+".tokens_stdev", "must not be negative")
	}
	if d.Min < 0 {
		p.Field(at+".tokens_min", "must not be negative")
	}
	if d.Max > 0 && d.Min > d.Max {
		p.Field(at+".tokens_min", "%d exceeds tokens_max %d", d.Min, d.Max)
	}
	// A mean outside the declared bounds means one of the three is wrong, and which
	// one is not guessable from here.
	if d.Max > 0 && d.Mean > d.Max {
		p.Field(at+".tokens", "mean %d exceeds tokens_max %d", d.Mean, d.Max)
	}
	if d.Min > 0 && d.Mean > 0 && d.Mean < d.Min {
		p.Field(at+".tokens", "mean %d is below tokens_min %d", d.Mean, d.Min)
	}
}
