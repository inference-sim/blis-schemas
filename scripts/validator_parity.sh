#!/usr/bin/env bash
# Cross-validator parity: the Go schema and the Python registry validator must reach
# the same verdict on the same document.
#
# Two validators for one artifact is a liability unless they agree, and they have
# disagreed: both carried a coefficient-uniqueness rule keyed on the entry name alone,
# which rejected the per-SKU registry outright, and they were fixed independently. This
# script is how that class of divergence is caught rather than discovered later.
#
# Each probe mutates a committed set, asks both validators, and restores. A DISAGREE line
# is the finding; the exit status is not, since some probes expect a rejection.
#
# Run from anywhere:
#     bash scripts/validator_parity.sh

V=/Users/sri/Documents/Projects/vllm/.venv/bin/python
REG=/Users/sri/Documents/Projects/blis-registry
F=$REG/coefficients/cost-model-primitives.yaml
cp $F /tmp/parity.bk
probe() {
  cd $REG && $V validator/validate.py >/dev/null 2>&1 && py=accept || py=reject
  cd /Users/sri/Documents/Projects/blis-schemas
  go test -count=1 -run TestEveryCommittedCoefficientSetLoadsAndValidates . >/dev/null 2>&1 && go=accept || go=reject
  printf "%-40s python=%-7s go=%-7s %s\n" "$1" "$py" "$go" \
    "$([ "$py" = "$go" ] && echo AGREE || echo '*** DISAGREE ***')"
}
mutate() { $V - "$@"; }
probe "as committed (expect accept)"
# A genuine same-(name,scope) duplicate: append a complete second entry.
mutate <<'PY'
p='/Users/sri/Documents/Projects/blis-registry/coefficients/cost-model-primitives.yaml'
s=open(p).read()
dup = """  - gemm_eps_max_fp8:
      value: 0.5
      units: dimensionless
      method: measured
      fitted: true
      scope: {hardware: [h200]}
      sources:
        - {kind: model, cite: "x", role: primary}
      rationale: >
        A second entry at the same name and scope.
"""
open(p,'w').write(s.rstrip('\n')+'\n'+dup)
PY
probe "duplicate at the same scope"
cp /tmp/parity.bk $F
# Same name, DIFFERENT scope: both must accept.
mutate <<'PY'
p='/Users/sri/Documents/Projects/blis-registry/coefficients/cost-model-primitives.yaml'
s=open(p).read()
dup = """  - gemm_eps_max_fp8:
      value: 0.5
      units: dimensionless
      method: measured
      fitted: true
      scope: {hardware: [b200]}
      sources:
        - {kind: model, cite: "x", role: primary}
      rationale: >
        The same quantity measured on a different part.
"""
open(p,'w').write(s.rstrip('\n')+'\n'+dup)
PY
probe "same name, different scope"
cp /tmp/parity.bk $F
for m in "units: bytes_per_rank|units: gigabytes" "method: measured|method: guessed" "kind: model|kind: hearsay" "role: primary|role: decisive"; do
  before="${m%%|*}"; after="${m##*|}"
  mutate <<PY
p='/Users/sri/Documents/Projects/blis-registry/coefficients/cost-model-primitives.yaml'
s=open(p).read(); open(p,'w').write(s.replace('$before','$after',1))
PY
  probe "out-of-vocabulary: $after"
  cp /tmp/parity.bk $F
done
cp /tmp/parity.bk $F
cd $REG && $V validator/validate.py >/dev/null 2>&1 && echo "restored and valid" || echo "RESTORE FAILED"
