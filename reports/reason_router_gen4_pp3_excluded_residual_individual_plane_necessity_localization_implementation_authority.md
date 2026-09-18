# Gen4 PP3-Excluded Residual Individual-Plane Necessity Localization — Implementation Authority

## Status

`IMPLEMENTATION_ONLY_NO_SCIENTIFIC_MODEL_EXECUTION`

This document authorizes only the bounded implementation of the already frozen
individual-plane necessity-localization design.

It does not authorize scientific model execution, Kaggle execution, checkpoint-
backed evaluation, confirmatory inference, Holm correction, or scientific
conclusion.

## Upstream frozen authority

Prospective design commit:

`d046a8e03e7522a72dfbd08cc9129b769cd5686a`

Design path:

`reports/reason_router_gen4_pp3_excluded_residual_individual_plane_necessity_localization_design.md`

Design blob:

`191e3a74647afb9a007a85bfdb7cefa8d79c9010`

Static-preparation freeze commit:

`ab9a6ebbc95bec20e8682b9538365f53167f108e`

Static-preparation result:

`PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_LOCALIZATION_STATIC_PREPARATION`

Fresh population:

`xg1_fact_1801..xg1_fact_2100`

No implementation choice may change the frozen population, residual plane
family, condition family, endpoint, hypotheses, multiplicity procedure,
positive gates, or forward budget.

## Authorized implementation delta

Create exactly:

1. `scripts/reason_router_gen4_pp3_excluded_residual_individual_plane_necessity_localization_fast_cuda.py`
2. `tests/test_reason_router_gen4_pp3_excluded_residual_individual_plane_necessity_localization_fast_cuda.py`

No other tracked file may be modified during implementation.

## Frozen static inputs

Data root:

`data/reason_router_gen4_xg1_residual_individual_plane_necessity_v1`

Static-preparation root:

`reports/reason_router_gen4_pp3_excluded_residual_individual_plane_necessity_localization_static_preparation_d046a8e`

Frozen data identities:

- source SHA256:
  `18381b4d31bf6b5b5975edf29dd23ad9a4cfbece23cf8c119cda4fb2539fcc9f`
- rows SHA256:
  `f3b67944c05e0198f9f33a3b544ddee344a3741ceccb00fe2a74191f177b18b5`
- structural manifest SHA256:
  `5cacd0639ce8dbb635bc2f895b81612482219859ff4e36a0768617350fd9e51d`
- tokenizer anchor manifest SHA256:
  `26b8762065063cc266286a00b394ddfc7088688643a8a4fed5b5b9825da22789`
- tokenizer eligibility summary SHA256:
  `25f95dff5d8929be216ea2b9961a76136ae7506fc6b4afb35670ec95354b35c7`
- geometry manifest SHA256:
  `dead284a5bb8a21becc10d8542018a7de3f1cf2af6870f2cfda2990104d215ec`
- preparation manifest SHA256:
  `d1ba40bd13382b512fa26d5f446b16581a11fe925764bfc9fd970fe7278107b0`

Frozen residual vector identities remain the exact vectors frozen by the prior
aggregate-residual static preparation:

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

Frozen PP3 identities used only for invariance auditing:

- PP3+:
  `66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`
- PP3-:
  `ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

The runner must authenticate all required static files, manifests, vector bytes,
branch/HEAD identity, and this implementation-authority blob before any model
load or scientific forward.

## Reuse boundary

The implementation should minimize new runtime surface.

Reuse the already validated aggregate-residual runner:

`scripts/reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda.py`

for:

- frozen XG2/XG4 susceptibility directions;
- signed probe semantics;
- epsilon `0.025`;
- target-token / layer-17 hook semantics;
- checkpoint authentication;
- frozen model/runtime provenance;
- static-file hash helpers;
- JSON/JSONL canonicalization;
- two-GPU spawn architecture;
- device-generic runtime gate;
- fast Mamba capture construction;
- checkpoint/model load pattern;
- artifact checksum/manifest pattern.

The new implementation may wrap or reuse these validated helpers where their
semantics are unchanged.

Do not fork or modify the existing aggregate runner.

## Frozen plane family

Plane order:

`[P1, P2, P4, P5]`

All four planes are implemented symmetrically.

No plane may receive special-case logic based on prior scientific outcomes.

No plane may be omitted, promoted, down-weighted, or reordered based on
outcomes.

## Frozen conditions

Exactly nine conditions:

1. `native`
2. `p1_neutralized`
3. `p1_quarter_turn_control`
4. `p2_neutralized`
5. `p2_quarter_turn_control`
6. `p4_neutralized`
7. `p4_quarter_turn_control`
8. `p5_neutralized`
9. `p5_quarter_turn_control`

The native condition is evaluated once per pair and shared across all four
plane contrasts.

## Branch-local plane coefficients

For each scientific forward branch and each plane `Pk`, obtain native
branch-local coefficients from the captured native hidden state:

`a_k = <h, p_k+>`

`b_k = <h, p_k->`

`c_k = a_k p_k+ + b_k p_k-`

`r_k = -b_k p_k+ + a_k p_k-`

The implementation must not reuse coefficients from another branch, probe sign,
direction, pair, or condition.

## Condition corrections

For `native`:

`delta = 0`

For `pk_neutralized`:

`delta_N,k = -c_k`

For `pk_quarter_turn_control`:

`delta_C,k = -r_k`

The implementation must audit for every intervention branch:

- treatment/control norm equality for the same native `(a_k,b_k)`;
- treatment/control orthogonality;
- PP3 coefficient invariance;
- no direct projection onto other frozen residual planes beyond tolerance;
- target-plane neutralization under the neutralized condition;
- applied correction cast residual.

The correction must be computed from the branch-local native state before the
probe correction is applied.

## Frozen endpoint

Use exactly the established broad susceptibility endpoint.

For each condition:

`Q = E_XG2 - E_XG4`

where each family energy is the mean squared directional susceptibility across
its five frozen directions.

For each plane `Pk`:

`Q0 = Q(native)`

`QN,k = Q(pk_neutralized)`

`QC,k = Q(pk_quarter_turn_control)`

`A_N,k = Q0 - QN,k`

`A_C,k = Q0 - QC,k`

Canonical primary per-plane causal contrast:

`D_k = QC,k - QN,k`

The implementation may store `A_N,k` and `A_C,k`, but `D_k` must be computed
directly as `QC,k - QN,k`.

Do not introduce an exact-equality assertion between the cancellation path

`(Q0 - QN,k) - (Q0 - QC,k)`

and the canonical path

`QC,k - QN,k`.

The prior aggregate-run endpoint correction established why such bitwise
equality is not a valid IEEE-754 invariant.

## Endpoint validation

For each item and each plane, validation must require exact stored identities:

- `A_N,k == Q0 - QN,k`
- `A_C,k == Q0 - QC,k`
- `D_k == QC,k - QN,k`

No tolerance is needed for these stored-operation identities because validation
recomputes the same canonical operations.

## Frozen scientific forward budget

Ten directions per condition:

- five XG2;
- five XG4.

Four forwards per direction.

Therefore:

`40 forwards / condition / pair`

Nine conditions:

`360 forwards / pair`

300 pairs:

`108000 scientific model forwards`

Baseline forwards:

`0`

Any raw run whose accounting differs is invalid.

## Two-GPU execution structure

Implementation must support exactly two independent pair shards:

GPU 0:

`xg1_fact_1801..xg1_fact_1950`

Pair count:

`150`

Scientific forward budget:

`54000`

GPU 1:

`xg1_fact_1951..xg1_fact_2100`

Pair count:

`150`

Scientific forward budget:

`54000`

Requirements:

- multiprocessing spawn;
- no DDP;
- no NCCL;
- one model/checkpoint load per worker;
- device selected by worker `gpu_id`;
- no hard-coded `"cuda:0"`;
- canonical merge by pair index;
- fail closed on missing, duplicated, reordered, or cross-shard pair records.

## Raw item schema

Each raw item must contain at least:

- family key;
- source pair ID;
- pair index;
- frozen anchor indices;
- epsilon;
- frozen condition order;
- frozen direction order;
- all nine condition observations;
- `Q0`;
- per plane P1/P2/P4/P5:
  - `QN`;
  - `QC`;
  - `A_N`;
  - `A_C`;
  - `D`;
- branch-local native plane coefficients in intervention audits;
- treatment/control correction norm and dot-product audits;
- PP3 pre/post coefficient and drift audits;
- other-plane projection audits;
- target-plane post-neutralization audit;
- applied correction cast-residual audit;
- exact model-forward accounting.

Raw item boundary:

`primary_inference_executed = false`

`multiplicity_correction_executed = false`

`scientific_conclusion = null`

## Raw summary schema

Summary must include:

- design commit;
- static-preparation freeze commit;
- implementation-authority commit;
- execution HEAD;
- exact pair interval and count;
- epsilon;
- plane order;
- condition order;
- direction order;
- per-direction / per-condition / per-pair / total forward counts;
- exact two-shard metadata;
- primary endpoint definition for all four planes;
- planned raw confirmatory p-value count: `4`;
- planned multiplicity method:
  `Holm step-down across exactly P1,P2,P4,P5`;
- planned familywise alpha: `0.05`.

Raw boundary fields must be:

`primary_inference_executed = false`

`multiplicity_correction_executed = false`

`training_executed = false`

`backward_executed = false`

`task_heads_executed = false`

`logits_read = false`

`scientific_conclusion = null`

## Statistical boundary

The runner must not import or invoke statistical inference machinery.

Specifically prohibited in the raw runner:

- SciPy hypothesis tests;
- Student t p-values;
- Holm correction;
- adjusted p-values;
- supported-plane decisions;
- family-level label assignment;
- plane ranking.

Exactly four raw confirmatory p-values and the Holm procedure belong to a later
CPU-only inference step after raw artifact import and provenance validation.

## Tests

Tests must execute no scientific model forwards and no checkpoint-backed
scientific evaluation.

At minimum test:

- frozen commits, paths, condition order, plane order, shard split, and budget;
- frozen static-input authentication;
- residual-vector exact identities and orthonormality;
- plane-specific treatment neutralization;
- plane-specific quarter-turn control;
- equal treatment/control norm;
- treatment/control orthogonality;
- PP3 invariance;
- no direct other-plane projection;
- canonical per-plane endpoint identity;
- cancellation-sensitive regression ensuring `D_k` uses `QC-QN`;
- exact nine-condition forward accounting;
- exact two-shard merge;
- artifact round-trip validation;
- rejection of endpoint mutation;
- rejection of provenance/hash/order/budget mutation;
- rejection of inference/conclusion boundary violations;
- absence of statistical inference in runner source;
- device-generic two-GPU implementation.

## Implementation validation

Before implementation freeze, require:

- `python -m py_compile` PASS for runner and test;
- targeted pytest PASS;
- `git diff --check` PASS;
- exactly the two authorized implementation files changed;
- scientific model forward count `0`;
- checkpoint-backed scientific execution `0`;
- primary inference `False`.

## Scientific execution authority

Implementation completion does not authorize scientific execution.

A later explicit execution freeze is required after the implementation files
are committed and their exact Git blob identities are known.

## Commit / push

Implementation-authority freeze may be manually committed and pushed.

Implementation itself is a later separate commit after local validation.

No automated commit or push is authorized.
