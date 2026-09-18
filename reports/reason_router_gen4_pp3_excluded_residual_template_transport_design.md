# Gen4 PP3-Excluded Residual Template Transport — Prospective Design

## Status

`PROSPECTIVE_DESIGN_FROZEN_BEFORE_FRESH_XG1_1201_1500_OBSERVATION`

This document freezes exactly one new scientific question.

No fresh XG1 `1201..1500` model response has been observed at the time of this
design.

This design does not authorize scientific execution by itself.

## Motivation

The committed exploratory static characterization at:

`82957bdb57cd83367293753c3b0596722810f7d3`

showed that the PP3-excluded residual profile was nearly identical across two
non-overlapping XG1 cohorts and was much more similar to the frozen XG2 residual
profile than to the frozen XG4 residual profile.

Those XG1 observations are hypothesis-generating only.

They are not used to construct either confirmatory template.

## Frozen template source

Templates are constructed exclusively from the already frozen XG2 and XG4
`601..900` residual mean net vectors in:

`reports/reason_router_gen4_pp3_excluded_residual_static_analysis_7a6c30f.json`

Artifact commit:

`82957bdb57cd83367293753c3b0596722810f7d3`

Artifact SHA256:

`712be029cbb52b30a41e9322c58978392f0c61bb76eefa15ffb8340c91899dc6`

Residual plane order is fixed as:

`[P1, P2, P4, P5]`

Frozen XG2 residual mean net vector:

`r_XG2 = [1.5920587453157877e-08, 2.0711288247409458e-08, -8.756974184315863e-10, 2.9916623498998595e-08]`

Its Euclidean norm:

`||r_XG2|| = 3.972648704920197e-08`

Frozen unit XG2 template:

`u_XG2 = [0.4007549782451175, 0.5213470856800944, -0.022043162722820656, 0.7530649126348957]`

Frozen XG4 residual mean net vector:

`r_XG4 = [1.8670022911077672e-08, -2.177503237973276e-08, 1.1605587525048473e-07, 4.28622819921435e-08]`

Its Euclidean norm:

`||r_XG4|| = 1.2699946137038802e-07`

Frozen unit XG4 template:

`u_XG4 = [0.14700867790791192, -0.17145767505443896, 0.9138296650882098, 0.337499714799086]`

Frozen template cosine:

`<u_XG2, u_XG4> = 0.2035409975407082`

No XG1 outcome contributes to either template.

## Fresh confirmatory population

The prospective population is:

`xg1_fact_1201..xg1_fact_1500`

Required pair count:

`N = 300`

The static preparation stage must prove zero overlap with all previously used
XG1 populations `001..1200` under all of:

- pair ID;
- claim text;
- evidence text;
- `(claim, evidence)` row identity.

Any overlap blocks the experiment.

No outcome-dependent filtering is allowed.

## Frozen response geometry

Use exactly the existing frozen XG2 and XG4 five-dimensional bases.

For each fresh XG1 pair, observe native-state signed directional Jacobians along
the same ten frozen directions:

`xg2_0..xg2_4, xg4_0..xg4_4`

Use the already frozen finite-difference semantics:

- epsilon: `0.025`
- two signed orientations per direction;
- two branch forwards per signed orientation;
- four scientific model forwards per direction;
- ten directions per pair;
- forty scientific model forwards per pair.

For `N=300`, the exact scientific model-forward budget is:

`12000`

No baseline model forward is added.

No training, backward pass, task-head evaluation, or logits read is allowed.

## Frozen per-item decomposition

For each item, reconstruct the exact five principal-plane contributions to:

`Q_i = (1/5) g_i^T (P_XG2 - P_XG4) g_i`

using the same deterministic principal-plane decomposition already used in the
committed localization/static characterization.

The five plane net contributions must reconstruct stored `Q_i` to numerical
precision.

Then discard only the PP3 coordinate and define the four-dimensional residual
net vector:

`r_i = [q_i,P1, q_i,P2, q_i,P4, q_i,P5]`

Plane order is fixed as `[P1,P2,P4,P5]`.

No residual plane is selected, reweighted, rotated, sign-flipped, or removed.

## Frozen primary endpoint

For every item, require:

`||r_i||_2 > 0`

If any item has exactly zero residual norm, the confirmatory protocol is
invalid and inference must not be executed. The item must not be filtered.

Define:

`C_XG2_i = <r_i, u_XG2> / ||r_i||_2`

`C_XG4_i = <r_i, u_XG4> / ||r_i||_2`

Primary per-item endpoint:

`D_TEMPLATE_i = C_XG2_i - C_XG4_i`

Positive values mean the item's PP3-excluded residual orientation is more
cosine-aligned with the frozen XG2 residual template than with the frozen XG4
residual template.

## Exactly one confirmatory inference

Hypotheses:

`H0: mean(D_TEMPLATE) <= 0`

`H1: mean(D_TEMPLATE) > 0`

Test:

- one-sample Student t-test;
- one-sided alternative: greater;
- N: `300`;
- df: `299`;
- alpha: `0.05`;
- exactly one confirmatory p-value;
- no multiplicity correction.

No other inferential test is allowed.

## Frozen positive-label gates

The positive label requires all of:

1. `mean(C_XG2) > 0`
2. `mean(D_TEMPLATE) > 0`
3. one-sided primary `p < 0.05`

If and only if all gates pass, assign:

`PP3_EXCLUDED_XG2_LIKE_RESIDUAL_TEMPLATE_TRANSPORT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Otherwise assign:

`PP3_EXCLUDED_XG2_LIKE_RESIDUAL_TEMPLATE_TRANSPORT_NOT_ESTABLISHED_ON_FRESH_XG1_HOLDOUT`

## Allowed descriptive outputs

The following may be reported descriptively without additional inference:

- mean and SD of `C_XG2`;
- mean and SD of `C_XG4`;
- mean and SD of `D_TEMPLATE`;
- fraction of items with `D_TEMPLATE > 0`;
- mean residual net vector in `[P1,P2,P4,P5]`;
- residual effective absolute-net plane count;
- maximum absolute Q reconstruction residual.

These are not additional hypotheses.

## Forbidden adaptations

After this design is frozen, do not perform:

- selection of P1/P2/P4/P5 based on fresh outcomes;
- alternative template construction;
- template averaging with prior XG1 cohorts;
- response-guided template rotation;
- sign rescue;
- epsilon sweep;
- layer sweep;
- token-position sweep;
- checkpoint sweep;
- subgroup analysis;
- alternative tails;
- alternative primary metrics;
- removal of low-norm items;
- additional p-values;
- outcome-dependent filtering.

A command or implementation change that alters the scientific endpoint requires
a new prospective design and new fresh population.

## Interpretation boundary

A positive result would establish only that, on fresh XG1 examples, the
PP3-excluded local susceptibility residual is more aligned with the frozen XG2
residual template than with the frozen XG4 residual template.

It would support prospective transport of an XG2-like secondary residual
geometry.

It would not by itself establish:

- causality of the residual template;
- necessity or sufficiency of P1/P2/P4/P5;
- that XG1 and XG2 are globally mechanistically identical;
- universal family-conditioned residual geometry;
- behavioral or downstream-task effects;
- architecture-wide universality.

A negative result would mean that the exploratory XG1 residual resemblance did
not prospectively transport under this frozen test. No rescue analysis follows.
