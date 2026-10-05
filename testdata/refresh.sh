#!/usr/bin/env bash
# Refresh the vendored catalog and registry fixtures.
#
# These fixtures are a SNAPSHOT, deliberately. The schema's tests must not depend on
# another repository's current state: blis-catalog and blis-registry move
# independently, so a test pinned to whatever their main branch holds today fails for
# reasons that have nothing to do with a schema change. Vendoring makes "does the
# schema accept the committed artifacts" a question with one answer per schema commit,
# and makes updating that answer a reviewable diff.
#
# Only the files the schema actually reads are copied. The vendor config.json files
# that sit beside each graph.yaml are 8MB and the schema never looks at them.
#
# Usage:  testdata/refresh.sh [path-to-blis-catalog] [path-to-blis-registry]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
catalog="${1:-$here/../../blis-catalog}"
registry="${2:-$here/../../blis-registry}"

for repo in "$catalog" "$registry"; do
  [ -d "$repo/.git" ] || { echo "not a git checkout: $repo" >&2; exit 1; }
done

rm -rf "$here/catalog" "$here/registry"
mkdir -p "$here/catalog/models" "$here/registry/coefficients"

for dir in "$catalog"/models/*/; do
  name="$(basename "$dir")"
  [ -f "$dir/graph.yaml" ] || continue
  mkdir -p "$here/catalog/models/$name"
  cp "$dir/graph.yaml" "$here/catalog/models/$name/"
done

for ns in hardware networks devices workloads; do
  [ -d "$catalog/$ns" ] || continue
  mkdir -p "$here/catalog/$ns"
  cp "$catalog/$ns"/*.yaml "$here/catalog/$ns/"
done

cp "$registry"/coefficients/*.yaml "$here/registry/coefficients/"

git -C "$catalog" rev-parse HEAD > "$here/catalog/REVISION"
git -C "$registry" rev-parse HEAD > "$here/registry/REVISION"

echo "refreshed from:"
echo "  catalog  $(cat "$here/catalog/REVISION")"
echo "  registry $(cat "$here/registry/REVISION")"
echo "$(find "$here" -name '*.yaml' | wc -l | tr -d ' ') yaml files vendored"
echo
echo "Review the diff: a change here is a change in what the schema is known to accept."
