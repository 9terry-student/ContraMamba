# Gen4 PP3-Excluded Residual Individual-Plane Restoration Sufficiency Localization — Implementation Authority

## Status

`IMPLEMENTATION_ONLY_NO_SCIENTIFIC_MODEL_EXECUTION`

This document authorizes only bounded implementation of the already frozen
individual residual-plane restoration-sufficiency localization design.

It does not authorize scientific model execution, Kaggle execution, checkpoint-
backed evaluation, confirmatory inference, Holm correction, training, backward,
task-head evaluation, or scientific conclusion.

## Upstream frozen authority

Prospective design commit:

`8c0ff6dbad77ed876fc1481b3b53c3fd47a27d3b`

Design path:

`reports/reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_design.md`

Design blob:

`6c1cd31ad852505eefc095011d184699c53a5e4b`

Static-preparation freeze commit:

`4a1d5871fad17a34951bc433a283c95e682a081b`

Static result:

`PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_RESTORATION_SUFFICIENCY_LOCALIZATION_STATIC_PREPARATION`

Fresh population:

`xg1_fact_2101..xg1_fact_2400`

No implementation choice may change the frozen population, residual-plane
family, intervention geometry, endpoint, hypotheses, multiplicity procedure,
positive gates, condition count, or forward budget.

## Authorized implementation delta

Create exactly:

1. `scripts/reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_fast_cuda.py`
2. `tests/test_reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_fast_cuda.py`

No other tracked file may be added or modified during implementation.

## Frozen static inputs

Data root:

`data/reason_router_gen4_xg1_residual_individual_plane_restoration_sufficiency_v1`

Static-preparation root:

`reports/reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_static_preparation_8c0ff6d`

Frozen data SHA256:

- source facts:
  `d2cc7276254bb96c32ea54af3c3fb768cbe005dcd7cb329512232df18629ea81`
- six-cell rows:
  `56957588464df337ad65509e377b534e49c264ddaac766e3131633dd5e2ac087`
- structural manifest:
  `fd37beb2eb1d30c7af81d6f771ddfee731e47ccdb56b32ddf84cc57a922836e1`

Frozen static-report SHA256:

- geometry manifest:
  `9e63a5ecc7cd172cd07335b36a637bf2ec6ba03569ebf02b9a6cce80f2fab51f`
- preparation manifest:
  `afe37f32950724c2a997a233dba9ff456705d646ab01f5f69487ecaeef136574`
- tokenizer anchor manifest:
  `bf2c0a2a82ecff43fa8bc1b9beb31e214563d027d50a3a24df4a1d488da9115f`
- tokenizer eligibility summary:
  `9c9ba1862837a3e7bbc0010636a9eafc1f5c8b438141dc04be2e902045ed6905`

Frozen static-preparation script SHA256:

`a946ba85b2d765b83edb0fcc7d146a8a6ee93a859620e623f56a873cece3c5dd`

Frozen endpoint basis-plan SHA256:

- XG2:
  `b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c`
- XG4:
  `792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f`

Frozen tokenizer contract:

- revision:
  `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`
- tokenizers:
  `0.22.2`
- eligibility:
  `PASS_300_OF_300`

## Frozen residual geometry

Residual plane order:

`[P1, P2, P4, P5]`

Frozen residual-vector SHA256:

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

Frozen PP3 audit vectors:

- PP3+:
  `66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`
- PP3-:
  `ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

## Reuse boundary

Reuse the already validated individual-plane necessity implementation as the
primary runtime template:

`scripts/reason_router_gen4_pp3_excluded_residual_individual_plane_necessity_localization_fast_cuda.py`

Frozen reuse blob:

`9cc0f41b24953d94f870aa3e07b993fafd3d83ea`

Reuse its test structure from:

`tests/test_reason_router_gen4_pp3_excluded_residual_individual_plane_necessity_localization_fast_cuda.py`

Frozen test blob:

`f81c9a97fc4579773e521261708042722e0b1cd6`

Do not modify either existing necessity file.

Reuse unchanged machinery for:

- XG2/XG4 susceptibility directions;
- finite-difference epsilon `0.025`;
- target-token / layer-17 hook semantics;
- branch-local coefficient capture;
- frozen checkpoint authentication;
- runtime provenance;
- two-GPU independent-shard execution;
- device-generic worker selection;
- model/checkpoint loading;
- raw artifact checksums and manifest;
- canonical pair-order merge;
- no-inference raw boundary.

## Frozen conditions

Exactly nine conditions:

1. `native`
2. `p1_neutralized`
3. `p1_quarter_turn_replacement`
4. `p2_neutralized`
5. `p2_quarter_turn_replacement`
6. `p4_neutralized`
7. `p4_quarter_turn_replacement`
8. `p5_neutralized`
9. `p5_quarter_turn_replacement`

The native condition is evaluated once per pair and is the exact native
restoration state for all four plane tests.

## Branch-local geometry

For every scientific forward branch and plane `Pk`:

`a_k = <h, p_k+>`

`b_k = <h, p_k->`

`c_k = a_k p_k+ + b_k p_k-`

`r_k = -b_k p_k+ + a_k p_k-`

Coefficients must be obtained from that branch's native captured state before
the condition correction and before the susceptibility probe correction.

No coefficient may be reused from another branch, direction, sign, pair, plane,
or condition.

## Frozen corrections

Native:

`delta_native = 0`

Neutralized background for plane `Pk`:

`B_k = h - c_k`

`delta_B,k = -c_k`

Matched quarter-turn replacement:

`C_k = B_k + r_k`

`C_k = h - c_k + r_k`

`delta_C,k = -c_k + r_k`

The replacement correction is therefore not the old necessity control
`-r_k`.

This distinction is mandatory.

The implementation must audit:

- native component norm;
- quarter-turn component norm;
- equality of restoration and replacement addition norms;
- `<c_k,r_k> = 0`;
- neutralized target-plane residual;
- replacement target-plane coordinates `(-b_k,a_k)`;
- PP3 coordinate invariance;
- non-target residual-plane coordinate invariance;
- applied-cast correction residual.

## Frozen endpoint

For every condition:

`Q = E_XG2 - E_XG4`

Shared exact-restoration endpoint:

`Q0 = Q(native)`

For plane `Pk`:

`Q_B,k = Q(pk_neutralized)`

`Q_C,k = Q(pk_quarter_turn_replacement)`

Exact native restoration gain:

`S_k = Q0 - Q_B,k`

Matched replacement gain:

`S_C,k = Q_C,k - Q_B,k`

Canonical primary contrast:

`D_SUF,k = Q0 - Q_C,k`

The raw runner must compute `D_SUF,k` directly using:

`Q0 - Q_C,k`

It may also store `S_k` and `S_C,k`.

Stored-operation validation must require:

- `S_k == Q0 - Q_B,k`
- `S_C,k == Q_C,k - Q_B,k`
- `D_SUF,k == Q0 - Q_C,k`

Do not require bitwise equality between alternative cancellation paths.

## Confirmatory boundary

Planned later confirmatory family:

`{P1,P2,P4,P5}`

Exactly four future raw confirmatory p-values.

Planned test:

- one-sample Student t-test;
- one-sided greater;
- `N = 300`;
- `df = 299`.

Multiplicity:

`Holm step-down`

Familywise alpha:

`0.05`

The raw runner must not compute any t statistic, p-value, Holm decision,
supported-plane set, ranking, or scientific label.

## Frozen forward budget

Directions per condition:

`10`

Scientific forwards per direction:

`4`

Scientific forwards per condition:

`40`

Conditions per pair:

`9`

Scientific forwards per pair:

`360`

Pairs:

`300`

Total scientific forwards:

`108000`

Baseline forwards:

`0`

No extra forward may be added solely to obtain the native state or
coefficients.

## Two-GPU execution structure

Implementation must support the prospectively frozen independent shards:

GPU 0:

`xg1_fact_2101..xg1_fact_2250`

- pairs: `150`
- scientific forwards: `54000`

GPU 1:

`xg1_fact_2251..xg1_fact_2400`

- pairs: `150`
- scientific forwards: `54000`

Requirements:

- multiprocessing spawn;
- no DDP;
- no NCCL;
- one model/checkpoint load per worker;
- no hard-coded `cuda:0`;
- canonical merge by prospective pair index;
- fail closed on missing, duplicated, reordered, or cross-shard records.

## Raw artifact boundary

Raw output must preserve at least:

- exact execution HEAD;
- design commit;
- static-preparation commit;
- implementation-authority commit;
- exact population and pair order;
- condition order;
- direction order;
- epsilon;
- all nine condition observations;
- `Q0`;
- per-plane `Q_B`, `Q_C`, `S`, `S_C`, `D_SUF`;
- branch-local intervention audits;
- exact scientific-forward accounting;
- checkpoint and runtime provenance.

Raw boundary fields:

`primary_inference_executed = false`

`multiplicity_correction_executed = false`

`training_executed = false`

`backward_executed = false`

`task_heads_executed = false`

`logits_read = false`

`scientific_conclusion = null`

## Tests

Tests must execute:

- scientific model forwards: `0`;
- checkpoint-backed scientific evaluation: `0`;
- GPU scientific execution: `0`;
- primary inference: `0`.

At minimum test:

- exact frozen commits and static hashes;
- exact population and two-shard split;
- exact condition order;
- residual-vector authentication;
- neutralization math;
- quarter-turn replacement math `-c_k+r_k`;
- restoration-addition norm match;
- component orthogonality;
- PP3 invariance;
- non-target-plane invariance;
- replacement coordinates `(-b_k,a_k)`;
- canonical `S`, `S_C`, `D_SUF` stored identities;
- regression that `D_SUF` is stored as `Q0-Q_C`;
- exact nine-condition forward accounting;
- artifact round-trip validation;
- rejection of endpoint/order/hash/budget mutation;
- rejection of inference/conclusion boundary violations;
- no statistical inference machinery in runner source;
- device-generic two-GPU execution structure.

## Implementation validation

Before implementation freeze require:

- `python -m py_compile` PASS for runner and test;
- targeted pytest PASS;
- `git diff --check` PASS;
- exactly the two authorized implementation files changed;
- scientific model forward count `0`;
- checkpoint-backed scientific execution `0`;
- primary inference `False`.

## Scientific execution authority

Implementation completion does not authorize scientific execution.

A later explicit execution freeze is required after implementation is committed
and its exact Git blob identities are known.

## Commit / push

Implementation-authority freeze may be manually committed and pushed.

Implementation is a later separate commit.

No scientific execution is authorized by this document.