// Command validate-catalog loads every derived ModelGraph in a blis-catalog checkout
// and validates it against the schema.
//
// The catalog's own gate checks that each entry is structurally a catalog entry. This
// one checks that the derived graph is a graph a cost model can price: that its nodes
// carry the parameters their ops need, that its DAG is acyclic, and that its stack
// names layer kinds it declares. A deriver bug produces a well-formed YAML file that
// fails here.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	blisschemas "github.com/inference-sim/blis-schemas"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: validate-catalog <blis-catalog root>")
		os.Exit(2)
	}
	root := os.Args[1]
	paths, err := filepath.Glob(filepath.Join(root, "models", "*", "graph.yaml"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "globbing: %v\n", err)
		os.Exit(2)
	}
	if len(paths) == 0 {
		fmt.Fprintf(os.Stderr, "%s: no models/*/graph.yaml found\n", root)
		os.Exit(2)
	}
	sort.Strings(paths)

	var failed int
	for _, path := range paths {
		name := filepath.Base(filepath.Dir(path))
		g, err := blisschemas.LoadModelGraph(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", name, err)
			failed++
			continue
		}
		if g.Name != name {
			fmt.Fprintf(os.Stderr, "%s: graph names itself %q\n", name, g.Name)
			failed++
		}
		p := g.Validate()
		for _, problem := range p.All() {
			fmt.Fprintf(os.Stderr, "%s: %s\n", name, problem)
		}
		if !p.OK() {
			failed++
			continue
		}
		fmt.Printf("%-46s %3d layers, %d kind(s), %s\n", name,
			g.Stack.Layers(), len(g.LayerKinds), g.Global.WeightDType)
	}
	if failed > 0 {
		fmt.Fprintf(os.Stderr, "\n%d of %d graph(s) failed\n", failed, len(paths))
		os.Exit(1)
	}
	fmt.Printf("\nall %d graph(s) validate\n", len(paths))
}
