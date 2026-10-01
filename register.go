package blisschemas

import (
	"github.com/inference-sim/blis-schemas/rules"
	v0_29 "github.com/inference-sim/blis-schemas/rules/v0_29"
)

// Packs are registered here rather than in each pack's init function. An init-time
// side effect makes registration depend on which packages a build happens to
// import, so a caller that imported only spec/ would silently get no rule checking.
// Registering explicitly at one site makes the set of known versions a fact about
// this file.
func init() {
	rules.Register(v0_29.Pack())
}
