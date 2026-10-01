package workload

import "testing"

func chatbot() *Shape {
	return &Shape{Name: "chatbot",
		Prompt: Distribution{Mean: 256, StdDev: 100, Min: 2, Max: 800},
		Output: Distribution{Mean: 256, StdDev: 100, Min: 1, Max: 1024}}
}

// agentic is the long-context shape the published report corpus exercises: a
// request carrying tens of thousands of input tokens against a few hundred output.
func agentic() *Shape {
	return &Shape{Name: "agentic", PrefixTokens: 4096,
		Prompt: Distribution{Mean: 52393, Min: 8000, Max: 262144},
		Output: Distribution{Mean: 486, Min: 1, Max: 2048}}
}

func TestValidShapesPass(t *testing.T) {
	for _, s := range []*Shape{chatbot(), agentic()} {
		if p := s.Validate(); !p.OK() {
			t.Fatalf("%s rejected:\n%s", s.Name, p.Error())
		}
	}
}

func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Shape)
	}{
		{"no name", func(s *Shape) { s.Name = "" }},
		{"negative prefix", func(s *Shape) { s.PrefixTokens = -1 }},
		{"prefix longer than the prompt", func(s *Shape) { s.PrefixTokens = 9999 }},
		{"zero prompt mean", func(s *Shape) { s.Prompt.Mean = 0 }},
		{"zero output mean", func(s *Shape) { s.Output.Mean = 0 }},
		{"min above max", func(s *Shape) { s.Prompt.Min = 900; s.Prompt.Max = 800 }},
		{"mean above max", func(s *Shape) { s.Prompt.Max = 10 }},
		{"mean below min", func(s *Shape) { s.Prompt.Min = 9000 }},
		{"negative stdev", func(s *Shape) { s.Prompt.StdDev = -1 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := chatbot()
			tc.mutate(s)
			if p := s.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}
