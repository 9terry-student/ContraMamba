# Gen4 PP3-Excluded Residual Aggregate Necessity — Implementation Authority

## Status

`IMPLEMENTATION_ONLY_NO_SCIENTIFIC_MODEL_EXECUTION`

This document authorizes a bounded implementation of the already frozen
aggregate residual necessity design.

It does not authorize scientific model execution, Kaggle GPU execution,
checkpoint-backed evaluation, confirmatory inference, or scientific conclusion.

## Upstream frozen authority

Prospective design commit:

`c4518d20f4417ca9f057fbd4940c28539e4ffb2c`

Design path:

`reports/reason_router_gen4_pp3_excluded_residual_aggregate_necessity_design.md`

Static-preparation freeze commit:

`4a7698264488e811370bdf071c3cde73735757e0`

Static preparation result:

`PASS_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_STATIC_PREPARATION`

Fresh population:

`xg1_fact_1501..xg1_fact_1800`

No implementation choice may change the frozen scientific question, population,
plane set, endpoint, hypothesis, positive-label gates, or forward budget.

## Authorized implementation delta

Create exactly:

1. `scripts/reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda.py`
2. `tests/test_reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda.py`

No other tracked file may be modified by the implementation phase.

## Frozen static inputs

Data root:

`data/reason_router_gen4_xg1_residual_aggregate_necessity_v1`

Static root:

`reports/reason_router_gen4_pp3_excluded_residual_aggregate_necessity_static_preparation_c4518d2`

Frozen data identities:

- source SHA256:
  `b8b2186fb4bdf5d9781efb9ab6eb56eb8a3d51052618ab7cf7d257aa6e72df21`
- rows SHA256:
  `16d9646cab9fa10d0241db2eeecf15164a04eb80740b873c2db6e8f02acab578`
- structural manifest SHA256:
  `3a161b933c87a00360fc0a1ed86f60d145f539220f64f5f29685d8df4fc37c58`
- tokenizer anchor manifest SHA256:
  `d4049107c3465fcfd027dcf9b1309630232bf53d2acc3881328f0b2b0853affd`

Frozen residual vector identities:

- P1+:
  `209da6bf007c0eadcd78db648835e6ee174d0290726acb61731007eff436aef1`
- P1-:
  `b6470bbec5a586f34e87d6e87f32c7f1778af7a55622d60508d68c6679c1ab26`
- P2+:
  `a0f48476f77e9e1876adc9919ba61a3c4e6ac894001f789216245d2ff3c945ea`
- P2-:
  `b48683f584ab31e31d542fc9b20327d19c0019cca02a41c97db0418d2bd69a78`
- P4+:
  `494f7b5de31673d53b368341d7960781767f948ed32afbae32b64b06a68f5cd8`
- P4-:
  `5583cb0cb6ab6fe0a2abae926ed56e3dd800d524d0539acbc54ced9ca8dea079`
- P5+:
  `7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2`
- P5-:
  `311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855`

Frozen PP3 authentication identities:

- PP3+:
  `66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`
- PP3-:
  `ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

The runtime must authenticate all required static files before any model load or
scientific forward.

## Frozen residual coordinate

Residual planes:

`R = {P1, P2, P4, P5}`

For each branch-local native strong-channel state `h`, for each residual plane
`Pk`:

`a_k = <h, pk_plus>`

`b_k = <h, pk_minus>`

`c_k = a_k pk_plus + b_k pk_minus`

Complete residual component:

`c_R = sum_k c_k`

Quarter-turn component:

`r_k = -b_k pk_plus + a_k pk_minus`

`r_R = sum_k r_k`

No outcome-dependent weighting, sign choice, selection, or template weighting is
authorized.

## Frozen conditions

Condition order must be exactly:

1. `native`
2. `residual_neutralized`
3. `quarter_turn_control`

### Native

`delta_0 = 0`

### Residual neutralization

`delta_R = -c_R`

The runtime must verify, before the signed probe is added:

- all eight P1/P2/P4/P5 post-condition projections are zero within frozen
  numerical tolerance;
- PP3+ and PP3- coefficients are unchanged within frozen runtime tolerance.

### Quarter-turn matched control

`delta_C = -r_R`

The runtime must verify:

- `||delta_R||_2 = ||delta_C||_2` within frozen numerical tolerance;
- `<delta_R, delta_C> = 0` within frozen numerical tolerance;
- PP3+ and PP3- coefficients are unchanged within frozen runtime tolerance.

Treatment/control coefficients must come from the same branch-local native state.

## Signed susceptibility semantics

Reuse the validated Gen4 signed-probe semantics without modification.

For each condition, each frozen XG2/XG4 basis direction, and each orientation:

- target-plus branch receives `+ orientation * epsilon * v`;
- target-minus branch receives `- orientation * epsilon * v`.

Frozen epsilon:

`0.025`

Frozen broad endpoint:

`Q = E_XG2 - E_XG4`

No task heads, logits, training, or backward pass are allowed.

No extra baseline/capture model forward is allowed. Native coefficients must be
read inside the same intervention hook from the state already present in the
scientific forward.

## Frozen per-pair endpoint

For every pair:

- `Q0 = Q(native)`
- `QR = Q(residual_neutralized)`
- `QC = Q(quarter_turn_control)`
- `A_R = Q0 - QR`
- `A_C = Q0 - QC`
- `D_RES_NEC = A_R - A_C = QC - QR`

The runner must validate the algebraic identity exactly from serialized values.

The raw runner must not compute a p-value and must not assign the final
scientific label.

## Exact forward budget

Per condition:

- 10 basis directions
- 4 scientific forwards per direction
- 40 scientific forwards per pair

Across 3 conditions:

- 120 scientific forwards per pair

Across 300 pairs:

- exactly `36000` scientific model forwards
- baseline model forwards: `0`

Any implementation requiring an additional scientific model forward is invalid.

## Two-GPU execution architecture

The implementation must use the already validated explicit two-worker,
two-device architecture used by the PP3-excluded residual-template transport
runner.

No DDP and no NCCL.

Exact sharding:

### GPU 0

- pair range: `xg1_fact_1501..xg1_fact_1650`
- pair count: `150`
- scientific forwards: `18000`

### GPU 1

- pair range: `xg1_fact_1651..xg1_fact_1800`
- pair count: `150`
- scientific forwards: `18000`

Total:

`36000`

Canonical merge order is source-pair order `1501..1800`.

Each worker loads the same model/checkpoint once.

Device handling must remain generic:

- explicit `torch.cuda.set_device(gpu_id)`
- device-specific `.to(device)`
- device-specific synchronization
- no literal `"cuda:0"` in implementation logic

## Runtime/kernel contract

Reuse the validated frozen runtime and kernel compatibility path already used by
the current Gen4 two-GPU runner.

The implementation must not alter:

- model identity;
- checkpoint identity;
- tokenizer identity;
- kernel scientific revision identities;
- kernel binary SHA identities;
- compatibility whitelist;
- frozen layer/token/strong-channel intervention coordinate;
- XG2/XG4 basis identities.

Kernel/runtime compatibility failure must occur before scientific forwards.

## Raw artifact contract

The implementation must produce a fail-closed raw artifact containing:

- 300 canonical item rows;
- exact condition order;
- exact direction order;
- per-condition Q values;
- `Q0`, `QR`, `QC`, `A_R`, `A_C`, `D_RES_NEC`;
- intervention audit fields sufficient to validate:
  - native residual coefficients;
  - treatment correction L2;
  - control correction L2;
  - treatment/control L2 mismatch;
  - treatment/control dot product;
  - residual post-neutralization projections;
  - PP3 coefficient drift under treatment;
  - PP3 coefficient drift under control;
  - applied-correction cast residual;
- per-item scientific forward count `120`;
- baseline forward count `0`;
- execution provenance;
- shard metadata;
- exact checkpoint SHA;
- raw artifact checksums.

Raw summary fields must include:

- `scientific_model_forward_count_this_run = 36000`
- `baseline_model_forward_count_this_run = 0`
- `gpu_count = 2`
- `primary_inference_executed = False`
- `multiplicity_correction_executed = False`
- `training_executed = False`
- `backward_executed = False`
- `task_heads_executed = False`
- `logits_read = False`
- `scientific_conclusion = None`

## Required implementation tests

The test file must cover, without scientific model execution:

1. exact pair range and shard coverage;
2. exact 150/150 shard sizes and 18000/18000 forward budgets;
3. total 36000-forward arithmetic;
4. residual vector file authentication;
5. condition order and direction order;
6. synthetic residual-neutralization coefficient math;
7. synthetic blockwise quarter-turn coefficient math;
8. treatment/control L2 equality;
9. treatment/control orthogonality;
10. PP3 invariance under both residual corrections;
11. endpoint identities:
    - `A_R = Q0 - QR`
    - `A_C = Q0 - QC`
    - `D_RES_NEC = QC - QR`;
12. no extra baseline forward accounting;
13. raw-artifact validation rejects malformed budgets/provenance/order;
14. raw-artifact boundary forbids primary inference and scientific conclusion;
15. device-generic two-GPU binding and absence of hard-coded `"cuda:0"`;
16. fail-closed behavior for static hash/provenance mismatch.

Tests may use synthetic tensors, monkeypatching, and static source inspection.
They must not load the scientific checkpoint or execute a scientific model
forward.

## Prohibited implementation changes

Do not:

- alter the prospective design;
- change the residual plane set;
- add individual-plane endpoints or tests;
- use the XG2-like template as a hidden-state intervention direction;
- add response-guided weights;
- add rescue controls;
- change epsilon;
- change the layer/token/channel coordinate;
- change the model/checkpoint/tokenizer/kernel identities;
- add training/backward/task-head/logit logic;
- calculate the confirmatory t-test in the raw runner;
- authorize or launch Kaggle execution.

## Validation required before implementation freeze

At minimum:

- Python syntax/compile validation;
- targeted unit tests for the new runner;
- `git diff --check`;
- exact changed-file scope of two files only.

A later implementation freeze may be created only after those checks pass.

Scientific execution requires a separate later execution freeze.

## Current boundary

Implementation authorized: `YES`

Scientific model execution authorized: `NO`

Kaggle GPU execution authorized: `NO`

Confirmatory inference authorized now: `NO`

Commit/push of implementation authority itself: manual only.
