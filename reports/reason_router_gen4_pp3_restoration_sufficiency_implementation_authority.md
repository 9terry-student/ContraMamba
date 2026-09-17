# Gen4 PP3 Restoration Sufficiency Implementation Authority

## Status

`IMPLEMENTATION_AUTHORIZED_NO_SCIENTIFIC_EXECUTION`

This document authorizes only bounded implementation and CPU-only validation
for the already frozen PP3 restoration-sufficiency design.

Scientific model execution remains unauthorized.

## Frozen upstream chain

Design:

`854bcd5585512155776c34e46846817c54b75cf6`

Static preparation:

`0f907574f1ca25ec12e35573a83e1b499ed1b53b`

Frozen population:

`xg1_fact_901..xg1_fact_1200`

Population size:

- source pairs: `300`
- rows: `1800`

Frozen static identities:

- source SHA256:
  `2c700452d818531c46a8ffd473eb6d64d8af29f3a9284da8371f5c9eb2610c21`
- rows SHA256:
  `7ec2ea86f35562394244f6df6e8b098ea3ba8a9bf358868fc614f5029746241c`
- structural manifest SHA256:
  `a1c6957d7d48fb93b53f7b93bb5378caf0bbdc0adb96c438f6aa56a779ae9922`
- tokenizer anchor manifest SHA256:
  `dc4f2cd4ca2806249467407c7c980411d8fa02051418f9a8b625c1b1c4756253`
- tokenizer eligibility summary SHA256:
  `0fb8da67687f223b6c72ea7bc946542e6543bb85c050163056696a8e945e5a89`
- static preparation manifest SHA256:
  `19fce1d109b8ab2d8e7faddd9ea5b2f5571e0ea5f1c2c30927a4d7a10e0e061c`

Tokenizer eligibility:

`PASS_300_OF_300`

## Authorized implementation scope

Create exactly two new files:

1. `scripts/reason_router_gen4_pp3_restoration_sufficiency_fast_cuda.py`
2. `tests/test_reason_router_gen4_pp3_restoration_sufficiency_fast_cuda.py`

No existing tracked file may be modified.

The existing PP3 necessity fast-CUDA runner may be reused structurally, but the
new implementation must have independent frozen constants, schemas, artifact
names, validation, and endpoint algebra for restoration sufficiency.

## Frozen conditions

Condition order must be exactly:

1. `pp3_neutralized`
2. `pp3_restored`
3. `pp5_replacement`

Let native strong-channel state be `h`.

Let:

`a = <h, pp3_plus>`

`b = <h, pp3_minus>`

`c3 = a*pp3_plus + b*pp3_minus`

`c5 = a*pp5_plus + b*pp5_minus`

The direct final-state corrections applied inside the layer-17 in-projection
hook must be exactly:

### pp3_neutralized

`delta_B = -c3`

Final state:

`B = h - c3`

### pp3_restored

`delta_R3 = 0`

Final state:

`R3 = h`

This is algebraically identical to restoring `c3` onto the common background
`B`.

### pp5_replacement

`delta_R5 = -c3 + c5`

Final state:

`R5 = h - c3 + c5`

No extra model forward may be used to obtain `h`, `a`, or `b`.

## Required intervention audits

For every branch intervention, the implementation must retain enough raw audit
information to verify:

- native PP3 coefficients `a`, `b`;
- PP3 component norm `||c3||`;
- transferred PP5 component norm `||c5||`;
- restoration-addition norm equality;
- direct final-state correction norm;
- PP3-neutralized residual coordinates;
- R3 native-state restoration identity;
- R5 construction identity;
- probe correction norm;
- applied cast residual;
- gate-half exact preservation;
- non-strong-channel exact preservation;
- all non-target-token exact preservation.

The frozen runtime cast tolerance remains unchanged.

## Frozen susceptibility measurement

Use exactly the already frozen five XG2 directions followed by the five XG4
directions.

Finite-difference epsilon:

`0.025`

For each condition:

`E_XG2 = mean_j J(v_XG2,j)^2`

`E_XG4 = mean_j J(v_XG4,j)^2`

`Q = E_XG2 - E_XG4`

## Frozen raw endpoint algebra

Per pair, raw artifact fields must include:

- `Q_B`
- `Q_R3`
- `Q_R5`
- `S3 = Q_R3 - Q_B`
- `S5 = Q_R5 - Q_B`
- `D_SUF = S3 - S5`

The implementation must validate the exact algebraic identity:

`D_SUF = Q_R3 - Q_R5`

No statistical inference is permitted inside the runner.

## Exact forward budget

Directions per condition:

`10`

Model forwards per direction:

`4`

Model forwards per condition:

`40`

Model forwards per pair:

`120`

Pairs:

`300`

Total scientific model forwards:

`36000`

Baseline model forwards:

`0`

## Raw artifact boundary

The runner must emit exactly four raw output files:

1. `pp3_restoration_sufficiency_items.jsonl`
2. `pp3_restoration_sufficiency_summary.json`
3. `artifact_manifest.json`
4. `SHA256SUMS.txt`

The summary must record:

- exact execution HEAD;
- this implementation-authority commit;
- static-preparation freeze commit;
- population bounds;
- frozen direction and condition order;
- exact forward budget;
- representative checkpoint SHA256;
- `primary_inference_executed = false`;
- `multiplicity_correction_executed = false`;
- `training_executed = false`;
- `backward_executed = false`;
- `task_heads_executed = false`;
- `logits_read = false`;
- `scientific_conclusion = null`.

## Required CPU-only validation

Tests must cover at minimum:

- frozen constants and population `901..1200`;
- exact condition order and 36000-forward budget;
- B/R3/R5 correction formulas;
- same native PP3 coefficients used for R3/R5 restoration comparison;
- `||c3|| = ||c5||`;
- R3 final state equals native `h`;
- R5 final state equals `h-c3+c5`;
- layer/token/channel hook isolation;
- endpoint identities including `D_SUF = Q_R3-Q_R5`;
- provenance mutation rejection;
- artifact checksum/manifest mutation rejection;
- exact forward-budget validation;
- absence of statistical inference code.

Validation must not:

- load the representative checkpoint;
- execute the scientific model;
- use CUDA;
- train;
- call backward;
- inspect task-head logits;
- perform the primary statistical test.

## Explicitly unauthorized

This authority does not authorize:

- Kaggle scientific execution;
- checkpoint-backed model inference;
- the 36000-forward observation;
- primary inference;
- scientific interpretation;
- rescue tests;
- subgroup analysis;
- endpoint changes.

A separate execution freeze is required after implementation has been validated
and frozen.
