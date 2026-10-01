// Package workload describes traffic shape: how long prompts are, how long
// completions are, and how much prefix requests share.
//
// A workload is a distribution rather than a pair of means, because the cost model
// is a function of an actual batch and a mean hides the shape that matters. Two
// corpora with the same mean input length — one tightly clustered, one spanning
// 52,000 to 93,000 tokens — saturate the same deployment in different ways and at
// different concurrencies.
package workload

// Shape is one traffic class. Token counts are per request.
type Shape struct {
	Name string `yaml:"name"`

	// PrefixTokens is the shared prefix length, which becomes the cached fraction a
	// cost model consumes. Zero means no sharing.
	PrefixTokens int `yaml:"prefix_tokens"`

	Prompt Distribution `yaml:"prompt"`
	Output Distribution `yaml:"output"`
}

// Distribution is a token-count distribution. Mean and StdDev describe the bulk;
// Min and Max bound it, and they matter because a scheduler's behaviour at the
// tail is what sets a knee.
type Distribution struct {
	Mean   int `yaml:"tokens"`
	StdDev int `yaml:"tokens_stdev,omitempty"`
	Min    int `yaml:"tokens_min,omitempty"`
	Max    int `yaml:"tokens_max,omitempty"`
}
