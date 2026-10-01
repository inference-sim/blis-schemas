package workload

import "github.com/inference-sim/blis-schemas/internal/validate"

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
