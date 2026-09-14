# Gen4 × K Large-Correction Regime Prospective Validation Design

## Status

**PROSPECTIVE DESIGN — OUTCOME BLIND**

This starts a new scientific question after the closed Gen4 × K transport bridge.

It does not reopen, reinterpret, or replace the closed result:

`DIRECTIONAL_ALIGNMENT_CAUSAL_TRANSPORT_NOT_ESTABLISHED`

Discovery evidence is frozen at:

- bridge closure: `ad375c88385eef507e151f95717e951919a6a3fe`
- static decomposition: `4d68f06d5bfaf9bc680e625000062e36b29c0b30`
- tail localization: `41f4678d09f7c779a8edc2eec6cc08a9effd9f41`

No validation-cohort causal outcome has been observed when this design is frozen.

## New scientific question

Does **large pre-intervention directional correction demand** prospectively identify a regime in which the frozen directional-alignment intervention produces an adverse `R_ALIGN` sign reversal at the Gen4 layer-17 `POST4_PATH_EFFICIENCY` endpoint?

This is narrower than the closed transport claim.

## Prospective discriminator

The discriminator is fixed before validation outcomes:

`alignment_shift_abs = abs(reference_C - target_C)`

where `target_C` and `reference_C` are the same pre-intervention baseline geometry quantities used by the closed bridge.

Discovery-cohort threshold:

`T = 0.11228626366380845`

This is exactly the frozen discovery cohort q75 of `alignment_shift_abs`.

Validation classification:

- `LARGE` iff `alignment_shift_abs >= T`
- `SMALL` iff `alignment_shift_abs < T`

The threshold must not be re-estimated, optimized, quantile-matched, or tuned on the validation cohort.

## Why this discriminator

The frozen post-hoc discovery evidence showed:

- `R_ALIGN` bottom-10: 100% in discovery `alignment_shift_abs` Q4
- bottom-20: 100% in Q4
- bottom-30: 23/30 in Q4, 3.0667× enrichment
- bottom-30 `alignment_shift_abs` mean was +1.133 SD above population
- `R_ALIGN` association:
  - Pearson `-0.3902745349616349`
  - Spearman `-0.22514516827964756`

`target_C` was also associated with the adverse tail, but `alignment_shift_abs` is selected because it is more directly tied to the demanded directional correction and had the stronger enrichment/standardized contrast.

No alternative discriminator is tested in the prospective inferential family.

## Independent synthetic holdout cohort

The discovery cohort used generator source-pair indices `0:300`.

The validation cohort is frozen as generator source-pair indices:

`300:600`

That corresponds to exactly:

`generated_fact_301` through `generated_fact_600`

Source generator:

`scripts/build_controlled_v5.py`

Frozen generator authority:

`91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea`

Frozen generator blob:

`baee23a9f71333125f4a8735c2c92d20cab7eb4f`

The generator is deterministic and supports arbitrary `num_pairs` by extending `_generated_fact_template`.

The six-cell rows are produced by the already-frozen materializer:

`scripts/materialize_reason_router_gen4_six_cell_contrast.py`

materializer blob:

`6a0f5bf58614cdff0dad5f78c7d4bd86d507cd3b`

The validation cohort must satisfy before tokenizer/model execution:

- 300 unique source pair IDs;
- exact IDs 301–600;
- zero source-pair overlap with discovery 1–300;
- zero rendered claim-string overlap with discovery;
- six complete cells per pair;
- 1800 rows total;
- no labels, logits, predictions, or outcome fields.

This is an independent **synthetic holdout from the same frozen generator family**, not an external real-world validation dataset.

## Tokenizer / anchor gate

After structural materialization, before causal execution:

- use the exact frozen tokenizer revision;
- derive anchors without using model outcomes;
- require all 300 pairs to satisfy the required `A_IDENTITY/A_NAME` and POST4 eligibility contract;
- do not drop individual pairs after inspecting geometry or response;
- if the complete fixed cohort cannot satisfy the anchor contract, stop and redesign before causal outcomes are produced.

## Frozen causal mapping

Keep the closed bridge intervention semantics unchanged:

- target pair: `C2_NAME - C0_SHAM`
- reference geometry pair: `C5_TITLE_NAME - C1_TITLE`
- roles: layer `15 → 16 → 17`
- intervention coordinate: own-branch `A_IDENTITY + 2`
- target branches: `A_IDENTITY == A_NAME`
- intervention: layer-17 x branch, post-`in_proj`, pre-depthwise-conv
- gate branch untouched
- fixed strong set: 395 channels
- endpoint: layer-17 `POST4_PATH_EFFICIENCY`

No layer, endpoint, token-offset, channel, checkpoint, or threshold scan is allowed.

## Prospective causal budget

Magnitude intervention is not required for this question.

Per validation pair:

- baseline C0/C1/C2/C5: 4 forwards
- alignment C0/C2: 2 forwards

Total:

- 6 forwards/pair
- 300 pairs
- 1800 scientific GPU forwards

A bounded CPU-slow ↔ CUDA-fast equivalence gate may run before the full study and is not part of the 1800-forward scientific budget.

## Pre-intervention classification timing

`alignment_shift_abs` is computed only from baseline geometry.

Regime membership is determined before reading alignment-intervention endpoint responses.

The full fixed cohort may be run in one execution artifact, but the analysis code must compute and freeze each pair's regime membership from baseline geometry before accessing `R_ALIGN`.

No response-dependent pair inclusion or exclusion is allowed.

## Minimum group-size stop condition

Before inferential testing:

- require `n_LARGE >= 30`
- require `n_SMALL >= 30`

If either condition fails:

`PROSPECTIVE_REGIME_TEST = BLOCKED_INSUFFICIENT_FIXED_GROUP_SIZE`

Do not alter `T` to repair group size.

## Prospective inferential family

Exactly two one-sided tests:

### H1 — adverse LARGE-regime response

`H1: mean(R_ALIGN | LARGE) < 0`

Test:

one-sample Student t-test, one-sided less-than-zero.

### H2 — LARGE-vs-SMALL heterogeneity

`H2: mean(R_ALIGN | LARGE) < mean(R_ALIGN | SMALL)`

Test:

independent two-sample Welch t-test, one-sided LARGE < SMALL.

Multiplicity:

- Holm correction over exactly these two tests
- family alpha = 0.05

Prospective regime support requires:

1. provenance/anchor/runtime/manipulation gates PASS;
2. fixed group-size gates PASS;
3. H1 Holm-adjusted rejection in the pre-specified direction;
4. H2 Holm-adjusted rejection in the pre-specified direction.

If either inferential hypothesis fails, the prospective adverse-regime claim is not established.

## Descriptive quantities

The study may additionally report, without adding tests:

- `n_LARGE`, `n_SMALL`
- means/medians/SDs of `R_ALIGN` in each group
- positive/negative response fractions
- fixed-threshold group geometry summaries
- overall 300-pair `R_ALIGN`

No extra p-values or post-hoc subgroup scans are allowed.

## Backend

Preferred execution backend is the validated Gen4 fast-CUDA path documented in:

`reports/reason_router_gen4_fast_cuda_backend_runbook.md`

Use the exact known-good runtime/kernel identities.

Before full causal execution:

1. structural cohort materialization PASS;
2. tokenizer/anchor gate PASS;
3. bounded model equivalence gate PASS on the new cohort;
4. commit/push exact runner;
5. use normal `cm run save/run → collect → import` provenance workflow.

Do not repeat the earlier manual-execution provenance shortcut.

## Claim boundary

A positive result would support only:

> In the fixed independent synthetic holdout cohort, the pre-specified large directional-correction-demand regime predicts an adverse causal response to the frozen alignment intervention at the frozen layer-17 POST4 path-efficiency endpoint.

It would not establish:

- behavioral mediation;
- output failure or hallucination causation;
- universality;
- general Mamba instability;
- validity outside this generator family;
- superiority of magnitude intervention;
- reopening of the original population-level bridge claim.

## Immediate phase boundary

The current phase authorizes only:

1. outcome-blind materialization of source pairs 301–600;
2. structural validation and provenance freeze.

Tokenizer execution, model execution, causal intervention, and inferential testing remain outside this immediate phase until the materialized holdout is frozen.
