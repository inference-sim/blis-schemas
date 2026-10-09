# Rules packs

<p class="lede">A rules pack holds what one engine release accepts and what it refuses. A
scenario's <code>engine_version</code> selects the pack; a version with no pack is reported
as unchecked rather than assumed fine.</p>

Rules run after the field layer passes and read the scenario, the deployment and the model
graph together. Each problem names the rule that reported it. A rule that reports an
*error* describes a deployment the engine would refuse at startup, or one a cost model
would price with the wrong constants. A rule that reports a *warning* describes one the
engine would run differently from what the document suggests, or a name the engine may
accept from an out-of-tree plugin.

One problem comes from no pack:

`engine-version-known`
:   *warning.* No pack is registered for the scenario's `engine_version`. Field validation
    passed, but no version-specific rule ran.

Everything below is generated from the registered packs. Each rule's text is the reason
its source gives for it.

--8<-- "docs/generated/rules.md"

The constants in each pack are transcribed from the engine's source. A test re-derives most
of them from a checkout of the engine and fails on any difference in either direction; CI
runs it against the tagged release. The speculative-method lists and the all-reduce backend
names are not re-derived and rest on review alone. See [A new engine release](../contributing/engine-release.md).

Source: [`rules`](https://github.com/inference-sim/blis-schemas/tree/main/rules).
