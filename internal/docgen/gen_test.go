package main

import (
	"bytes"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// TestGeneratedDocsAreCurrent regenerates every file in memory and compares it with what is
// committed, in both directions: a file whose content changed, and a file docgen no longer
// writes (a removed vocabulary), are both stale pages a reader would trust.
func TestGeneratedDocsAreCurrent(t *testing.T) {
	root := filepath.Join("..", "..")
	files, err := generate(root)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{}
	for _, f := range files {
		want[f.path] = true
		got, err := os.ReadFile(filepath.Join(root, f.path))
		if err != nil {
			t.Errorf("%s: not committed (%v); run `go run ./internal/docgen` from the repository root", f.path, err)
			continue
		}
		if !bytes.Equal(got, f.content) {
			t.Errorf("%s is stale; run `go run ./internal/docgen` from the repository root", f.path)
		}
	}
	err = filepath.WalkDir(filepath.Join(root, outDir), func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if !want[filepath.ToSlash(rel)] {
			t.Errorf("%s is committed but docgen no longer writes it; run `go run ./internal/docgen`, which removes it", rel)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestVersionLess(t *testing.T) {
	for _, c := range []struct {
		a, b string
		less bool
	}{
		{"0.29.0", "0.100.0", true},
		{"0.100.0", "0.29.0", false},
		{"0.29.0", "0.29.1", true},
		{"0.29.0", "0.29.0", false},
		{"1.0.0", "0.99.9", false},
	} {
		if got := versionLess(c.a, c.b); got != c.less {
			t.Errorf("versionLess(%q, %q) = %v, want %v", c.a, c.b, got, c.less)
		}
	}
}
