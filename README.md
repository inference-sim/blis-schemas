# blis-schemas

Go schemas and validators for the BLIS cost model: what a model is, what hardware can do,
what a coefficient claims, what a deployment specifies, what a measured run recorded, and
the interface a cost model implements.

**Documentation: <https://inference-sim.github.io/blis-schemas/>**

This repository holds the shapes of documents and the checks on them, and no data. The data
lives in [blis-catalog](https://github.com/inference-sim/blis-catalog) (declared facts) and
[blis-registry](https://github.com/inference-sim/blis-registry) (coefficient sets), which run
this module's validators in CI.
[blis-latency-kernel](https://github.com/inference-sim/blis-latency-kernel) implements the
`Kernel` interface defined here, and [inference-sim](https://github.com/inference-sim/inference-sim)
calls it in pull request [#1851](https://github.com/inference-sim/inference-sim/pull/1851).

## Use

```sh
go get github.com/inference-sim/blis-schemas@latest
```

```go
s, err := blisschemas.LoadScenario("scenario.yaml")
if err != nil {
    return err
}
d, err := blisschemas.LoadDeployment("deployment.yaml")
if err != nil {
    return err
}

rep := blisschemas.Validate(blisschemas.Bundle{Scenario: s, Deployment: d})
for _, p := range rep.Problems() {
    fmt.Println(p) // field problems first, then rule problems
}
```

Validate a catalog or a registry checkout:

```sh
go run github.com/inference-sim/blis-schemas/cmd/validate-catalog@latest  /path/to/blis-catalog
go run github.com/inference-sim/blis-schemas/cmd/validate-registry@latest /path/to/blis-registry
```

## Develop

```sh
go test ./...                                              # schemas, validators, rules, docs
VLLM_SOURCE=/path/to/vllm go test -run MatchSource ./rules/... # pack constants against the engine
```

How changes are organized, how an engine release is added, and how the documentation is
built and published: see Contributing in the [documentation](https://inference-sim.github.io/blis-schemas/).

## License

Apache 2.0. See [LICENSE](LICENSE).
