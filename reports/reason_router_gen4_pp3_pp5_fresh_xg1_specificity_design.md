# Gen4 PP3-vs-PP5 Fresh-XG1 Specificity Design

## Status

`PROSPECTIVE_SPECIFICITY_DESIGN_OUTCOME_BLIND_ON_FRESH_XG1`

This design is frozen after the validated PP3 transport result on
`xg1_fact_001..xg1_fact_300`, but before any model response is observed for the
fresh population `xg1_fact_301..xg1_fact_600`.

No response from `xg1_fact_301..xg1_fact_600` may be inspected before this
design, the fresh structural cohort, the probe geometry, and execution scope
are frozen.

## Prior evidence boundary

Current synthesis commit:

`510f29b79d5c1ba89fe49524a2f5d5b363aa5514`

Completed PP3 XG1 transport result:

`7ae74c656922663569d84a7e060308a77c450eab`

Completed projector-contrast localization:

`83b8cf7e73eb68343cc4c97e3f0698df4b587408`

The completed XG1 result on `001..300` remains valid evidence for PP3 transport,
but those responses are not part of the confirmatory specificity sample.

## Scientific question

On a fresh population from the same independently constructed XG1 generator
family, does the frozen PP3 projector-contrast plane carry more positive local
susceptibility contrast than the frozen principal plane having the maximum
projector separation?

This is a specificity test, not another unconstrained search for a responsive
direction.

## Response-blind matched control

The matched control is frozen as principal pair 5, PP5.

Selection rule:

`PP5 = argmax_i sin(theta_i), i in {1,2,3,4,5}`

This rule depends only on the already-frozen XG2/XG4 projector geometry.

Frozen values:

- PP3: `sin(theta_3) = 0.98692852916688512`
- PP5: `sin(theta_5) = 0.99986792842854511`

Thus PP5 has the greatest projector-separation magnitude of the five frozen
principal planes.

PP5 is not selected because of any XG1 response.

Historical XG2/XG4 responses along components of PP5 already exist as part of
the completed localization evidence, but they are not the selection criterion
for this control. The control is determined solely by the maximum frozen
projector-separation rule above.

No alternative PP1/PP2/PP4/PP5 selection after fresh response observation is
allowed.

## Fresh XG1 population

The confirmatory specificity population is exactly:

`xg1_fact_301..xg1_fact_600`

with:

- 300 source pairs;
- 6 structural rows per pair;
- 1800 rows total;
- generator family `xg1_independent_structured_records_v1`;
- the same six-cell masked-slot-substitution semantics;
- the same deterministic renderer and lexical inventories;
- no response-dependent inclusion or exclusion.

The previously observed `xg1_fact_001..xg1_fact_300` population is excluded
from the primary specificity test.

## Generator-continuation identity contract

The frozen original XG1 builder must not be modified.

A new structural-only continuation builder must implement the same deterministic
index formula for arbitrary positive XG1 source indices.

Before it may emit `301..600`, it must regenerate `001..300` using that
generalized formula and prove exact byte identity against the already-frozen:

`data/reason_router_gen4_xg1_cross_generator_v1/structured_source_facts.jsonl`

Failure of byte identity blocks the fresh cohort.

The fresh cohort must also satisfy:

- exact IDs `xg1_fact_301..xg1_fact_600`;
- exactly six complete cells per pair;
- exact frozen cell order and masks;
- zero duplicate row IDs;
- zero exact claim overlap with XG1 `001..300`;
- zero exact evidence overlap with XG1 `001..300`;
- no labels, predictions, logits, model geometry, endpoint values, or response
  fields;
- deterministic byte-identical regeneration.

This continuation deliberately reuses the XG1 lexical inventories because the
objective is a fresh-index holdout from the same generator family, not a new
generator family.

## Tokenizer gate

After structural freeze, all 300 fresh pairs must pass the same frozen
tokenizer/anchor contract used by the completed PP3 XG1 transport study.

No pair may be dropped or replaced.

Any failure closes the study as blocked before model execution.

## Frozen probe geometry

PP3 and PP5 are reconstructed deterministically from the exact same frozen
XG2/XG4 projector geometry.

For each plane `k` in `{3,5}`:

- `PPk+` is the positive projector-contrast eigenmode;
- `PPk-` is the negative projector-contrast eigenmode;
- both are unit vectors in ambient dimension 395;
- the two modes are orthogonal;
- signs use the frozen deterministic canonical convention;
- no XG1 response may affect vector construction or sign.

The existing frozen PP3 vectors remain unchanged.

PP5 vectors must be materialized and hashed before any fresh XG1 model
execution.

## Finite-difference protocol

Unchanged from the completed PP3 transport experiment:

- epsilon: `0.025`;
- representative checkpoint:
  seed180 `G3-GROUP-D-HALF`;
- checkpoint SHA256:
  `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`;
- tokenizer/model revision:
  `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`;
- same layer, intervention site, token contract, path-efficiency endpoint, and
  validated fast-CUDA runtime;
- no training, backward pass, task heads, or logits.

For a unit direction `w`:

`J_i(w) = [F_i(+epsilon; w) - F_i(-epsilon; w)] / (2 epsilon)`

## Primary endpoints

For each fresh pair:

`C_PP3_i = (s3 / 5) * (J_PP3_PLUS_i^2 - J_PP3_MINUS_i^2)`

`C_PP5_i = (s5 / 5) * (J_PP5_PLUS_i^2 - J_PP5_MINUS_i^2)`

with:

- `s3 = 0.98692852916688512`
- `s5 = 0.99986792842854511`

The single primary specificity endpoint is:

`D_SPEC_i = C_PP3_i - C_PP5_i`

## Confirmatory hypothesis

Exactly one primary inferential hypothesis is allowed:

`H1: mean(D_SPEC) > 0`

Test:

- one-sample Student t-test;
- one-sided;
- `N = 300`;
- `alpha = 0.05`;
- no multiplicity correction.

No separate p-value is computed for PP3, PP5, signed modes, subgroups, tails,
or any other plane.

## Decision rule

The positive result label is:

`PP3_SPECIFICITY_OVER_MAX_SEPARATION_PP5_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

and requires all of:

1. structural/provenance/runtime/completeness gates PASS;
2. exact 300-pair fresh population;
3. exact frozen forward budget;
4. `mean(C_PP3) > 0`;
5. `mean(D_SPEC) > 0`;
6. one-sided primary p-value `< 0.05`.

Otherwise:

`PP3_SPECIFICITY_OVER_MAX_SEPARATION_PP5_NOT_ESTABLISHED_ON_FRESH_XG1_HOLDOUT`

No rescue analysis changes the decision.

## Scientific forward budget

Directions per pair, in fixed order:

1. PP3+
2. PP3-
3. PP5+
4. PP5-

Each direction uses exactly four model forwards.

Therefore:

- 16 scientific model forwards per pair;
- 300 pairs;
- 4800 scientific model forwards total;
- 0 baseline model forwards.

## Descriptive quantities

Without additional hypothesis tests, the study may report:

- mean/SD of `C_PP3`;
- mean/SD of `C_PP5`;
- mean/SD of `D_SPEC`;
- fraction `D_SPEC > 0`;
- signed J means, mean absolute J, SD, and sign fractions for PP3+ and PP5+.

These are descriptive only.

## Prohibited adaptations

No:

- PP1/PP2/PP4 rescue;
- alternative control selection;
- response-guided direction rotation;
- sign changes after response observation;
- epsilon sweep;
- checkpoint sweep;
- layer or token sweep;
- subgroup or tail mining;
- pair dropping or replacement;
- extra p-values;
- baseline rerun unless separately frozen and scientifically required.

## Interpretation boundary

A positive result would support:

PP3 carries specificity beyond the frozen principal plane with maximal
projector separation on a fresh XG1 holdout.

It would specifically weaken the explanation that PP3 transport is merely a
consequence of using a highly separated projector-contrast plane.

It would not establish:

- PP3 dominance over every possible direction;
- PP3 as the sole native-Mamba mechanism;
- necessity or sufficiency for downstream behavior;
- arbitrary-generator universality;
- benchmark state of the art.

## Immediate phase boundary

This commit authorizes only structural/static preparation:

1. implement a new XG1 continuation builder without modifying the frozen
   original builder;
2. prove byte-identical regeneration of XG1 `001..300`;
3. materialize exact fresh XG1 `301..600`;
4. validate and freeze the fresh structural cohort;
5. deterministically reconstruct and freeze PP5 geometry.

No tokenizer execution, checkpoint load, model forward, CUDA scientific
execution, or inferential testing is authorized by this design commit.
