package validate

import (
	"fmt"
	"reflect"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// A custom UnmarshalYAML receives a yaml.Node, and Node.Decode does NOT honour the
// decoder's KnownFields setting. So every type that implements UnmarshalYAML loses
// strict field checking unless it re-establishes it, which is a silent loss: the type
// still decodes, and a misspelled field leaves a zero value nothing reports.
//
// RejectUnknownKeys restores the check. It compares a mapping node's keys against the
// yaml tags a target type declares, and is called by each custom unmarshaller before
// decoding.

// RejectUnknownKeys reports an error naming any key in node that target does not
// declare. Keys matching an ignore predicate are skipped, which is how the catalog's
// underscore-prefixed provenance narrative is allowed through.
func RejectUnknownKeys(node *yaml.Node, target any, ignore func(string) bool) error {
	if node.Kind != yaml.MappingNode {
		return nil
	}
	known := yamlFieldNames(reflect.TypeOf(target))
	var unknown []string
	for i := 0; i+1 < len(node.Content); i += 2 {
		key := node.Content[i].Value
		if ignore != nil && ignore(key) {
			continue
		}
		if !known[key] {
			unknown = append(unknown, key)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	allowed := make([]string, 0, len(known))
	for k := range known {
		allowed = append(allowed, k)
	}
	sort.Strings(allowed)
	return fmt.Errorf("unknown field(s) %v; this type declares %v",
		unknown, allowed)
}

// IgnoreUnderscored reports whether a key is one of the catalog's provenance
// narrative keys, which are deliberate content rather than stray fields.
func IgnoreUnderscored(key string) bool { return strings.HasPrefix(key, "_") }

func yamlFieldNames(t reflect.Type) map[string]bool {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	out := map[string]bool{}
	if t.Kind() != reflect.Struct {
		return out
	}
	for i := 0; i < t.NumField(); i++ {
		tag := t.Field(i).Tag.Get("yaml")
		if tag == "-" {
			continue
		}
		name := strings.Split(tag, ",")[0]
		if name == "" {
			// An untagged field decodes under its lowercased name.
			name = strings.ToLower(t.Field(i).Name)
		}
		out[name] = true
	}
	return out
}
