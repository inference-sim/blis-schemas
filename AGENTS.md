# Agent instructions for blis-schemas

These rules apply to every AI-assisted change to this repository. Human contributors
follow the same rules; they are written here so an agent reads them first.

## Cross-repository rules (all five BLIS repositories)

blis-schemas, blis-catalog, blis-registry, blis-latency-kernel and inference-sim form one
system. A change that touches more than one of them follows
[Changes across the repositories](https://github.com/inference-sim/blis-schemas/blob/main/docs/contributing/cross-repo-changes.md),
the one canonical description of the process. In short:

1. **Public artifacts.** Never write internal project, customer or colleague names in a
   commit, code, comment, document, issue or pull request. Refer to private measurements by
   what they are ("internal deployment measurements"). Run the name scan before every push.
2. **Data.** Never commit or redistribute third-party data (measurements, traces, benchmark
   dumps). Commit derived values only, each with its provenance: source URL plus commit or
   sha256.
3. **Versions.** Pin dependencies by release tag, or by the exact pushed head (Go
   pseudo-version, full SHA) marked `TEMPORARY` with its pull request. Never commit a
   `replace` in `go.mod`. Push and merge in dependency order: schemas, then catalog, registry
   and kernel, then inference-sim. A statement about engine behaviour cites
   `<path>:<lines>@<release>`; data from another engine version is used only for a code path a
   recorded diff shows unchanged.
4. **Apples to apples.** Compare the same quantity, by the same definition, on the same points,
   request set and window. Gates and fit / held-out splits are written down before the data
   they judge, and never relaxed afterwards. A fix lands only if its held-out evidence does not
   worsen; figures come from a script, never typed by hand. Seed comparisons use the mean over
   the same seeds with the pooled standard error.
5. **Tests.** Every behaviour change has a test that fails without it. Mutation-check it:
   revert the fix, see the test fail, restore, and say so in the commit message.
6. **Shared tooling.** Use the shared validation tooling rather than ad-hoc checks: the
   blis-schemas validators (`validate-catalog`, `validate-registry`), the pin check on
   immutable snapshots, the name scan, the push gate (every SHA in a pull request body equals
   its branch head), and `git merge-tree --write-tree origin/main HEAD` before each push.
7. **Pull requests.** Fill in `.github/pull_request_template.md` in full, including the exact
   pinned SHAs, the duplicate-PR check and the AI-assistance statement.

## This repository

blis-schemas is the root of the dependency graph: push it first, and any change to its head
means repinning all four consumers.

- Tests: `go test ./...` (includes the docs checks), and
  `VLLM_SOURCE=<vllm checkout at the release> go test -run MatchSource ./rules/...` for rules
  packs. See [Contributing](docs/contributing/index.md).
- A new key needs its validator, a negative test per failure mode, a reference-table row and
  a release. Before changing a type the catalog or registry uses, run `validate-catalog` and
  `validate-registry` against current checkouts of both.
- Regenerate `docs/generated/` with `go run ./internal/docgen` after a vocabulary or pack change.
