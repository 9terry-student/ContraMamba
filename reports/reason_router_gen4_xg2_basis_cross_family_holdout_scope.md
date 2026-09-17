# Gen4-K XG2-Basis Cross-Family Fresh-Index Holdout — Prospective Scope

Base HEAD: `88de940d1b24b4b02c670a9ae67a2046aa082b38`

Branch: `gen4-k-xg2-basis-holdout`

## Purpose

The completed 301..600 family-subspace experiment did not establish the
pre-specified symmetric own-versus-cross hypothesis.

It did reveal a distinct qualitative pattern that may motivate a new
prospective question: on both observed families, the frozen XG2 basis showed
greater squared local directional sensitivity than the frozen XG4 basis.

That prior result is used only to choose the direction of the new hypothesis.
No prior effect size, t statistic, p-value, subgroup, tail, basis-wise result,
or alternative threshold is used to tune this holdout.

This document freezes the new question before any model-derived response on
pair ids 601..900 is observed.

## Holdout population

Families:

- XG2
- XG4

Exact pair ids:

- XG2: `xg2_fact_601` through `xg2_fact_900`
- XG4: `xg4_fact_601` through `xg4_fact_900`

Exactly 300 pairs per family in ascending pair-id order.

Source construction must be a mechanical range extension of the frozen
synthetic family generator. The absolute source index remains:

`i = pair_number - 1`

Pinned source generator:

`scripts/build_reason_router_gen4_generator_family_prevalence_cohorts.py`

Frozen blob:

`4505acc99db0627733592694da290a007d281421`

Pinned prior absolute-index holdout-builder semantics:

`scripts/build_reason_router_gen4_xg2_xg4_fresh_response_holdouts.py`

Frozen blob:

`911846cf0b3304caaab0399535bd9f3de7f8249e`

After this scope is frozen, pair replacement, range shifting, eligibility-based
substitution, subgroup selection, or exclusion based on model response is not
allowed. Structural ineligibility must block the affected execution rather
than cause replacement by another pair.

## Holdout interpretation boundary

The frozen XG2/XG4 source generators reuse finite vocabularies and deterministic
schedules. Their non-ID template combinations are periodic rather than a new
independent template distribution.

The 601..900 block therefore constitutes a prospective same-generator
fresh-index holdout. It is not an independent generator, independent template
family, or external-data replication.

A positive result may establish stability across a previously unobserved
index block under the same frozen synthetic generator. It must not be described
as template-independent or generator-independent generalization.

## Frozen subspaces

Do not construct a new basis from 601..900.

Both comparison subspaces are frozen from the already completed 301..600
Phase-1 direction plans.

Frozen Phase-1 artifact root:

`reports/reason_router_gen4_xg2_xg4_fresh_response_restartable_phase1_d217bad_r3/`

XG2 `alignment_delta_h.pt` SHA256:

`b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c`

XG4 `alignment_delta_h.pt` SHA256:

`792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f`

For each family `f`, normalize every frozen direction row:

`u_fi = d_fi / ||d_fi||_2`

Construct the uncentered sign-invariant second moment over the original
300 rows:

`M_f = (1/300) * sum_i u_fi u_fi^T`

Use float64 CPU linear algebra and the five eigenvectors corresponding to the
five largest eigenvalues.

The two frozen comparison bases are:

`U_xg2 = [v_xg2,1, ..., v_xg2,5]`

`U_xg4 = [v_xg4,1, ..., v_xg4,5]`

Subspace dimension is fixed:

`k = 5`

No dimension sweep, response-guided rotation, sign search, basis replacement,
new-basis fitting on 601..900, weighting, or basis-wise selection is allowed.

Finite-difference basis semantics remain those corrected and frozen at:

`594cb45bdd8740b2dfd63c4780f766f1d0b375bc`

## Fixed local probe

Use exactly:

`epsilon = 0.025`

For each unit basis direction `v`, preserve the previously frozen intervention
semantics:

forward probe input delta:

`delta_h = +2 * epsilon * v`

reverse probe input delta:

`delta_h = -2 * epsilon * v`

with the runtime applying the corresponding half-delta to the plus/minus
branches.

For holdout pair `i`:

`F_i(+epsilon; v) = PE(tp, +epsilon*v) - PE(tm, -epsilon*v)`

`F_i(-epsilon; v) = PE(tp, -epsilon*v) - PE(tm, +epsilon*v)`

`J_i(v) = [F_i(+epsilon; v) - F_i(-epsilon; v)] / (2*epsilon)`

All values must be finite.

No new baseline model forward is part of the scientific endpoint.

## Primary endpoint

For every holdout pair, regardless of source family:

`E_XG2_i = (1/5) * sum_j J_i(v_xg2,j)^2`

`E_XG4_i = (1/5) * sum_j J_i(v_xg4,j)^2`

Define the single paired endpoint:

`Q_i = E_XG2_i - E_XG4_i`

The prospective directional hypothesis is:

`mean(Q) > 0`

for each source family independently.

Thus the same frozen XG2-versus-XG4 basis ordering is tested in:

1. XG2 holdout pairs 601..900
2. XG4 holdout pairs 601..900

## Primary confirmatory rule

For each family independently, use a one-sample Student t-test on its 300
paired `Q_i` values with the one-sided alternative:

`H1: mean(Q) > 0`

The multiplicity family contains exactly two tests:

- XG2
- XG4

Apply Holm correction at:

`alpha = 0.05`

Assign:

`XG2_BASIS_CROSS_FAMILY_SENSITIVITY_SUPPORTED_ON_FRESH_INDEX_HOLDOUT`

only if both families have positive mean `Q` and both corresponding hypotheses
are rejected after Holm correction.

Otherwise assign:

`XG2_BASIS_CROSS_FAMILY_SENSITIVITY_NOT_ESTABLISHED`

There is no rescue rule.

Allowed descriptive outputs without additional hypothesis tests are mean,
median, and SD of `E_XG2`, `E_XG4`, and `Q`; mean absolute directional
derivative for each frozen basis; and finite-value / execution-budget audits.

No opposite-direction test, subgroup test, tail test, per-basis-direction
significance test, epsilon search, dimension search, alternative multiplicity
family, or posthoc threshold is allowed.

## Scientific forward budget

Each basis contains five directions.

Each direction requires four scientific forwards per pair.

Two bases therefore require:

- 40 scientific forwards per pair
- 12,000 scientific forwards per family
- 24,000 scientific forwards total
- 0 new baseline scientific forwards

No training, backward pass, task head, or logits are part of this experiment.

## Outcome-blindness boundary

Before this scope freeze, no 601..900 model-derived response may be inspected.

After this scope freeze, the following outcome-blind preparation is allowed:

- deterministic construction of exact 601..900 source rows;
- source/schema/range/order validation;
- tokenizer/anchor eligibility validation;
- structural manifests and hashes;
- implementation and static tests for one dedicated holdout runner.

Those preparation artifacts must not contain or inspect scientific response
quantities such as path-efficiency outcomes, finite-difference `J`, basis
energies, `Q`, logits, predictions, labels, or scientific p-values.

Tokenizer or structural failure may block execution. It may not be used to
select a more favorable replacement range.

## Implementation boundary

Once this scope itself is committed, implementation may add only what is
needed for this prospective holdout:

- one outcome-blind 601..900 holdout/eligibility preparation path if required;
- one scientific holdout runner;
- dedicated static/unit tests;
- deterministic validation of the two already frozen 301..600 bases;
- fail-closed provenance, range, hash, finite-value, and forward-budget checks.

The implementation must authenticate the exact intended branch/HEAD and fail
closed on dirty worktree, frozen-input drift, pair-range/order drift,
structural artifact drift, output collision, non-finite values, wrong
subspace dimension, wrong epsilon, or scientific forward-budget mismatch.

Not authorized by this scope:

- Kaggle scientific execution;
- model forward execution on the 601..900 scientific endpoint;
- confirmatory t-tests or Holm correction on 601..900 outcomes;
- scientific interpretation of 601..900 outcomes;
- training or backward;
- any rescue or exploratory adaptation.

Scientific execution requires a separate minimal execution freeze after the
implementation and outcome-blind structural preparation have passed their
static validation.

## Claim boundary

A positive holdout result would support the proposition that, under the same
frozen XG2/XG4 synthetic generators and the same model/runtime geometry, the
previously identified XG2-basis sensitivity ordering persists into the
pre-specified unseen index block 601..900 for both source families.

It would not establish:

- universal or generator-independent sensitivity geometry;
- template-independent replication;
- transport to XG3 or another generator family;
- transport to another model/checkpoint/layer/offset;
- superiority of `k = 5`;
- robustness to another epsilon;
- that every XG2 basis direction is individually causal;
- a common signed adverse direction.

A negative result would falsify the frozen cross-family fresh-index
generalization criterion above, without establishing absence of all shared or
generator-specific geometry.
