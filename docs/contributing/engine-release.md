# A new engine release

<p class="lede">Supporting a new engine release changes no schema. It means writing a new
rules pack: the release's accepted values and the rules that read them, registered under
the release's version string.</p>

A pack is found by exact match on the scenario's `engine_version`. A scenario that names
`0.29.1` gets no rules until a pack is registered under `0.29.1`, even though a `0.29.0`
pack exists.

## Steps

1. **Copy the latest pack.** Create a package under `rules/`, such as `rules/v0_31/`, from
   the newest existing pack, and set its `Version` constant to the release's exact version
   string, such as `"0.31.0"`.
2. **Change what moved.** Update each constant that differs in the new release: backend
   names, cache dtypes, graph modes, speculative methods, connectors, quantization methods,
   thresholds. Change a rule's body only if the engine's behavior changed. Add or remove a
   rule if a check appeared or went away. Every rule states, in its `Because` text, what
   breaks without it.
3. **Register it.** Add `rules.Register(v0_31.Pack())` to the `init` function in the root
   package's `register.go`. Packs are registered in that one file rather than from each
   pack's own `init`, so the set of known versions does not depend on which pack packages a
   build happens to import.
4. **Check it against the source.** Carry over `source_test.go` and adjust it to the new
   release's file layout. Run it against a checkout of the release:

    ```sh
    VLLM_SOURCE=/path/to/vllm-0.31.0 go test -run MatchSource ./rules/v0_31/
    ```

    It fails on any difference in either direction: a value the pack accepts that the engine
    does not, or one the engine accepts that the pack does not. It does not cover the
    speculative-method lists, which need review by hand.

5. **Add its CI job.** Copy the `constants-against-engine` job in
   `.github/workflows/ci.yaml`, pinned to the new release's tag. Leave the existing job's tag
   alone: the old pack still needs checking against its own release.
6. **Regenerate the documentation.** Run `go run ./internal/docgen` from the repository
   root. The new pack appears in [Rules packs](../reference/rules.md). If the pack adds a
   field to `rules.Pack`, docgen fails until the field has a label in its `packFields` map.
7. **Leave the old pack alone.** A scenario pinned to the old version must keep meaning what
   it meant. Correct it only where it misdescribes its own release.

## What does not change

Nothing in `spec/` or `kernel/` changes. A field check holds in every release; a check that
holds in some releases and not others belongs in a pack. The refusal of prefill-context
parallelism combined with data parallelism is an example. vLLM 0.29.0 refuses it and later
releases support it, so it is the rule `pcp-excludes-data-parallelism` in the 0.29.0 pack,
not a field check.
