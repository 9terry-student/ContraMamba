# Gen4 PP3-Excluded Residual Aggregate Restoration Sufficiency — Implementation Authority

## Status

`IMPLEMENTATION_ONLY_NO_SCIENTIFIC_MODEL_EXECUTION`

This document authorizes only the bounded implementation of the already frozen
aggregate residual restoration-sufficiency experiment.

It does not authorize scientific model execution, checkpoint-backed scientific
evaluation, Kaggle GPU execution, confirmatory inference, training, backward,
task-head evaluation, logits analysis, or scientific conclusion.

## Frozen upstream chain

Prospective design commit:

`14d71b742488dd088ca22c416962522235d8d67b`

Design blob:

`1c82c90b6286e9923a2df51c5679ae080b8a85a9`

Static-preparation freeze commit:

`7744ddabe05e0180428d3e753a4453dea531ef6a`

Static-preparation result:

`PASS_PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_SUFFICIENCY_STATIC_PREPARATION`

Fresh population:

`xg1_fact_2401..xg1_fact_2700`

No implementation choice may change the population, residual subspace,
conditions, endpoint, hypothesis, positive gates, or forward budget.

## Frozen static identities

Data root:

`data/reason_router_gen4_xg1_residual_aggregate_restoration_sufficiency_v1`

Static root:

`reports/reason_router_gen4_pp3_excluded_residual_aggregate_restoration_sufficiency_static_preparation_14d71b7`

Frozen data SHA256:

- source:
  `eb77056732740f501066026a3b65ea3522cd203916828f9d8d56a6e746c79a87`
- six-cell rows:
  `de175b9817e6b589f4580247775adf760ece7a5b929c59dc28bdad6a7fe763e7`
- structural manifest:
  `85ff2f3f47c9b104f7e20529e520f5d5b294a14398764dba2b0bc7c3e7a1a0b8`

Frozen static-report SHA256:

- geometry manifest:
  `d0d43655e3fe5f432a7878ed46a526c6bde4ba220b78d75aedafad4a256bc2e6`
- tokenizer anchor manifest:
  `d57f55af9ebe5ccb83982c58fb803952cd37ec4d9f40ad1cf2ffc8da2c47fc0f`
- tokenizer eligibility summary:
  `7052785940b10699eeffad4dea5bc4b0877d856ae3adbdd5844ec77e956d0527`
- preparation manifest:
  `b58ca182fe66b91017ce008e56ea8bd68d4d595b8a265ac4a6958afe481ced02`

Frozen static-preparation script SHA256:

`449180e53d61c35766b7b121ce2adf79474ffe3878c3ea255973c21319012606`

Static preparation guarantees:

- XG1 `2401..2700`: `300` pairs / `1800` rows;
- byte-regeneration identity through XG1 `001..2400`;
- pair, claim, evidence, and `(claim,evidence)` overlap counts: all `0`;
- tokenizer eligibility: `PASS_300_OF_300`;
- endpoint basis identity: `PASS`;
- all residual vectors reproduced exactly;
- scientific model forwards: `0`;
- checkpoint loads: `0`;
- GPU used: `False`;
- scientific outcomes observed: `False`.

## Frozen geometry

Residual subspace:

`R = P1 ⊕ P2 ⊕ P4 ⊕ P5`

For each branch-local native state `h`:

`a_k = <h,p_k+>`

`b_k = <h,p_k->`

`c_k = a_k p_k+ + b_k p_k-`

`r_k = -b_k p_k+ + a_k p_k-`

Aggregate components:

`c_R = sum_{k in {1,2,4,5}} c_k`

`r_R = sum_{k in {1,2,4,5}} r_k`

Static audit established:

`||c_R|| = ||r_R||`

and:

`<c_R,r_R> = 0`

within frozen numerical tolerance.

PP3 remains untouched.

No response-guided weighting, plane selection, sign selection, alternative
rotation, or XG2-template weighting is permitted.

## Authorized implementation delta

Create exactly:

1. `scripts/reason_router_gen4_pp3_excluded_residual_aggregate_restoration_sufficiency_fast_cuda.py`
2. `tests/test_reason_router_gen4_pp3_excluded_residual_aggregate_restoration_sufficiency_fast_cuda.py`

No other tracked file may change during implementation.

## Frozen reuse templates

Primary three-condition / two-GPU runtime template:

`scripts/reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda.py`

Frozen blob:

`be859d754fc3d5f42e6e40925a9f83fd64dc48b8`

Corresponding test template:

`tests/test_reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda.py`

Frozen blob:

`42c416746d700ff828d922d50cd796508b844a57`

Restoration-semantics reference:

`scripts/reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_fast_cuda.py`

Frozen blob:

`c208c01d3cff7fa80b44d94444df42b6cd0227be`

Corresponding restoration test reference:

`tests/test_reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_fast_cuda.py`

Frozen blob:

`266fe544e11215ba624407c0fab0d9c899fc6060`

Do not modify any of these four existing files.

Reuse unchanged machinery for:

- model and checkpoint loading;
- tokenizer and static authentication;
- frozen runtime/kernel authentication;
- XG2/XG4 basis reconstruction and signed-probe semantics;
- layer-17 target-token strong-channel intervention;
- branch-local coefficient capture;
- independent two-GPU multiprocessing;
- artifact checksums and manifest generation;
- canonical shard merge;
- raw no-inference boundary.

## Frozen conditions

Condition order must be exactly:

1. `native`
2. `residual_neutralized`
3. `residual_quarter_turn_replacement`

### Native

`delta_native = 0`

### Complete residual-neutralized background

`B = h - c_R`

`delta_B = -c_R`

All P1/P2/P4/P5 coordinates must be zero within frozen runtime tolerance.

PP3 coordinates must remain invariant within frozen runtime cast tolerance.

### Aggregate restoration replacement

Exact native restoration is the shared native condition:

`R_native = B + c_R = h`

Matched replacement is:

`C = B + r_R`

Therefore:

`C = h - c_R + r_R`

and the correction applied relative to native `h` is exactly:

`delta_C = -c_R + r_R`

This is mandatory.

The old aggregate-necessity control:

`delta = -r_R`

must not be reused as the restoration replacement.

The implementation must validate:

- native residual coefficients;
- `c_R` norm;
- `r_R` norm;
- equality of restoration/replacement addition norms;
- `<c_R,r_R>`;
- all residual coordinates after neutralization;
- all replacement coordinates `(-b_k,a_k)`;
- PP3 coordinate invariance;
- applied-cast correction residual.

## Frozen endpoint

For each condition retain exactly:

`Q = E_XG2 - E_XG4`

with the frozen ten XG2/XG4 directions and:

`epsilon = 0.025`

Per item:

`Q0 = Q(native)`

`Q_B = Q(residual_neutralized)`

`Q_C = Q(residual_quarter_turn_replacement)`

Exact aggregate native-restoration gain:

`S_R = Q0 - Q_B`

Matched aggregate replacement gain:

`S_C = Q_C - Q_B`

Canonical primary endpoint:

`D_RES_SUF = Q0 - Q_C`

The runner must compute and serialize the canonical endpoint directly as:

`Q0 - Q_C`

It may serialize `S_R` and `S_C`.

Validation must require:

- `S_R == Q0 - Q_B`
- `S_C == Q_C - Q_B`
- `D_RES_SUF == Q0 - Q_C`

Do not require bitwise equality between the canonical subtraction and the
algebraically equivalent cancellation expression:

`(Q0-Q_B) - (Q_C-Q_B)`

The equivalent expanded expression may be checked only within the frozen
numerical tolerance.

## Confirmatory boundary

Future confirmatory hypothesis:

`H0: mean(D_RES_SUF) <= 0`

`H1: mean(D_RES_SUF) > 0`

Future test:

- one-sample Student t-test;
- one-sided greater;
- `N = 300`;
- `df = 299`;
- alpha `0.05`;
- multiplicity correction: none;
- raw confirmatory p-value count: exactly `1`.

The raw runner must not compute:

- a t statistic;
- a p-value;
- a positive/negative scientific label;
- individual-plane inference;
- interaction inference;
- additive-decomposition inference.

## Exact forward budget

Directions per condition:

`10`

Scientific forwards per direction:

`4`

Scientific forwards per condition:

`40`

Conditions per pair:

`3`

Scientific forwards per pair:

`120`

Pairs:

`300`

Total scientific forwards:

`36000`

Baseline forwards:

`0`

No extra forward may be added to capture native coefficients or hidden states.

## Frozen two-GPU split

GPU 0:

`xg1_fact_2401..xg1_fact_2550`

- pairs: `150`
- forwards: `18000`

GPU 1:

`xg1_fact_2551..xg1_fact_2700`

- pairs: `150`
- forwards: `18000`

Requirements:

- multiprocessing spawn;
- no DDP;
- no NCCL;
- one model/checkpoint load per worker;
- no hard-coded `cuda:0`;
- canonical pair-order merge;
- fail closed on missing, duplicated, reordered, or cross-shard records.

## Raw artifact contract

The runner must produce:

- exactly `300` canonical item rows;
- exact population and pair order;
- exact three-condition order;
- exact ten-direction order;
- `Q0`, `Q_B`, `Q_C`;
- `S_R`, `S_C`, `D_RES_SUF`;
- residual coefficient and aggregate intervention audits;
- per-item scientific forward count `120`;
- total scientific forward count `36000`;
- baseline forward count `0`;
- exact checkpoint/runtime/kernel provenance;
- two-shard metadata;
- artifact manifest and checksums.

Raw boundary fields must include:

`primary_inference_executed = false`

`multiplicity_correction_executed = false`

`training_executed = false`

`backward_executed = false`

`task_heads_executed = false`

`logits_read = false`

`scientific_conclusion = null`

## Required tests

Tests must execute:

- scientific model forwards: `0`;
- checkpoint-backed scientific evaluation: `0`;
- GPU scientific execution: `0`;
- statistical inference: `0`.

At minimum verify:

1. exact design/static identities;
2. exact population `2401..2700`;
3. exact 150/150 shards and `18000/18000` budgets;
4. exact total budget `36000`;
5. exact condition order;
6. exact residual-vector/static authentication;
7. aggregate `c_R` construction;
8. aggregate `r_R` construction;
9. neutralization `-c_R`;
10. restoration replacement `-c_R+r_R`;
11. regression that old necessity correction `-r_R` is not the replacement;
12. aggregate restoration/replacement norm equality;
13. aggregate `c_R·r_R = 0` within tolerance;
14. neutralized residual-coordinate zeroing;
15. simultaneous replacement coordinates `(-b_k,a_k)`;
16. PP3 invariance;
17. canonical `S_R`, `S_C`, `D_RES_SUF` identities;
18. regression that canonical `D_RES_SUF` is stored as `Q0-Q_C`;
19. three-condition forward accounting;
20. artifact round-trip validation;
21. malformed order/hash/budget/provenance rejection;
22. rejection of inference/scientific-conclusion boundary violations;
23. no statistical inference machinery in runner source;
24. device-generic two-GPU structure and no hard-coded `cuda:0`.

Tests may use synthetic tensors, monkeypatching, and static source inspection.

They must not load the scientific checkpoint or execute scientific model
forwards.

## Required implementation validation

Before implementation freeze require:

- `python -m py_compile` PASS for runner and test;
- targeted pytest PASS;
- `git diff --check` PASS;
- exact changed-file scope of the two authorized files only;
- scientific model forward count `0`;
- checkpoint-backed scientific execution `0`;
- GPU scientific execution `0`;
- primary inference `False`.

## Current boundary

Implementation authorized after this authority is committed: `YES`

Scientific model execution: `NO`

Kaggle scientific execution: `NO`

Confirmatory inference: `NO`

Training/backward: `NO`

Implementation completion does not authorize scientific execution.

A later explicit execution freeze is required after exact implementation blobs
are known.

Commit/push: manual only.