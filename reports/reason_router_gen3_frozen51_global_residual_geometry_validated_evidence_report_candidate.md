# Generation-3 Frozen-51 Global Residual Geometry Validated-Evidence Analysis

VERDICT = PASS_CANDIDATE

PHASE = GEN3_FROZEN51_GLOBAL_RESIDUAL_GEOMETRY_VALIDATED_EVIDENCE_ANALYSIS

AUTHORITY_INPUT_PAIRWISE = 4594f2d58e073610e63e679c97f4f19aea1db92e

SCIENTIFIC_EXECUTION = NOT_PERFORMED

TRAINING = NOT_PERFORMED

EVALUATION = NOT_PERFORMED

INFERENCE = NOT_PERFORMED

CHECKPOINT_LOADING = NOT_PERFORMED

IMPLEMENTATION_CHANGE = NONE

K_SERIES_MIXING = NO

## 1. Purpose

This report records a read-only row-level analysis of the already admitted
Generation-3 historical evidence.

It does not introduce a new intervention.

Its purpose is to characterize the 51 D1 A0-correct break occurrences that
were absent from every single-edge Gen3 condition.

The frozen first-pass decomposition is preserved exactly:

- D1 A0-correct breaks total = 111
- reproduced by at least one single-edge condition = 60
- absent from every single-edge condition = 51

This report calls the final population FROZEN51.

## 2. Evidence Sources

Canonical A0 references are the frozen Seed8192 revised-split primary N=3
exports.

Seed180 replacement_r1:

SHA256 =
5c9722ac0f75c411b3d744a29beec0c35d2f2809331f257a5e1d5ea81e6cf75d

Seed181:

SHA256 =
789d02f9092ce6b051d0ca435272c9e93a3962183dbb0a4d4dbb20cebf2ac3fe

Seed182:

SHA256 =
029ec6ae31df2f5ca9526d1e631496f7ee272967a6f5e08684f29aa09ad490d4

D1 and the ten single-edge conditions are the same admitted prediction exports
used by the frozen Gen3 first-pass validated-evidence analysis.

The two reproducible positive pairwise conditions G7+G10 and G5+G10 are the
same admitted prediction exports used by the frozen Gen3 pairwise
validated-evidence analysis at commit
4594f2d58e073610e63e679c97f4f19aea1db92e.

No replacement endpoint or newly trained artifact was used.

### 2.1 Immutable Source Manifest and Geometry Schema Contract

The complete source identity set for this analysis is recorded in the companion manifest:

Manifest path:
reports/reason_router_gen3_frozen51_global_residual_geometry_evidence_manifest.json

Manifest SHA256:
3ec5e37d92a347f169a6a5f8902f75b055ecb47b7d301909253b6112fdd08203

Manifest byte count:
19036

The manifest contains exactly 42 source prediction exports:

- 3 canonical A0 exports
- 3 historical D1 / GLOBAL-HALF exports
- 30 Gen3 single-edge exports
- 6 exports from the two admitted reproducible positive pairwise conditions

For every source export the manifest records:

- source-root identity
- exact repository-relative path
- training seed
- arm identity
- byte count
- SHA256
- row count
- unique stable-ID count

All 42 exports were re-read while constructing the manifest.
Every export contained exactly 720 prediction rows and 720 unique stable IDs.
Within each seed, all D1, single-edge, and selected pairwise exports matched the canonical A0 stable-ID universe and gold labels exactly.

The immutable upstream evidence bindings are:

- first-pass validated-evidence authority commit = 28d7fa2fcc0d286a0cd16723c302a45f214b2901
- D1 seed182 residual-localization authority commit = a8e68bb6fb111242628dda2c290983096ca120ca
- pairwise validated-evidence authority commit = 4594f2d58e073610e63e679c97f4f19aea1db92e

SUPPORT-minus-NOT_ENTITLED final margin was calculated from support_ne_margin_active, verified row-wise as support_logit - ne_logit; no probability-margin substitute was used.

The manifest records the same geometry-field contract and a row-wise equivalence tolerance of 1e-6.

## 3. FROZEN51 Definition

For each training seed separately:

D1_BREAK =
rows correct under matched canonical A0 and wrong under D1.

SINGLE_UNION =
union of A0-correct break sets from G1 through G10.

FROZEN51 =
D1_BREAK minus SINGLE_UNION.

Recomputation yielded:

- seed180 D1 breaks = 37
- seed180 FROZEN51 occurrences = 21

- seed181 D1 breaks = 23
- seed181 FROZEN51 occurrences = 7

- seed182 D1 breaks = 51
- seed182 FROZEN51 occurrences = 23

Total FROZEN51 occurrences = 51.

## 4. Class and Error-Transition Structure

Gold-label distribution across the 51 occurrences:

- NOT_ENTITLED = 50
- REFUTE = 1

D1 error transitions:

- NOT_ENTITLED to SUPPORT = 50
- REFUTE to SUPPORT = 1

Therefore 50 of the 51 global-only break occurrences share one final-task
transition: NOT_ENTITLED to SUPPORT.

This is a descriptive row-level fact.

It is not by itself a causal mechanism claim.

## 5. Intervention and Primary-Reason Structure

Intervention distribution:

- predicate_swap = 13
- role_swap = 10
- title_name_swap = 10
- location_swap = 8
- entity_swap = 7
- event_swap = 2
- paraphrase = 1

Primary-reason target distribution:

- FRAME = 37
- PREDICATE = 13
- AUTHORIZED = 1

Thus the 50 NOT_ENTITLED to SUPPORT occurrences divide exactly into:

- FRAME target = 37
- PREDICATE target = 13

The remaining one occurrence is the REFUTE to SUPPORT AUTHORIZED-target row.

## 6. Cross-Seed Recurrence

The 51 occurrences correspond to 43 unique stable IDs.

Stable IDs recurring in at least two of three training seeds = 6.

Stable IDs recurring in all three seeds = 2.

The recurrent IDs are:

2 of 3:
clinic_expansion__event_swap
gold = NOT_ENTITLED
primary reason = FRAME

2 of 3:
clinic_expansion__predicate_swap
gold = NOT_ENTITLED
primary reason = PREDICATE

2 of 3:
generated_fact_045__role_swap
gold = NOT_ENTITLED
primary reason = FRAME

3 of 3:
generated_fact_181__role_swap
gold = NOT_ENTITLED
primary reason = FRAME

2 of 3:
generated_fact_193__predicate_swap
gold = NOT_ENTITLED
primary reason = PREDICATE

3 of 3:
generated_fact_258__title_name_swap
gold = NOT_ENTITLED
primary reason = FRAME

All six recurrent stable IDs are NOT_ENTITLED to SUPPORT failures under D1.

## 7. Relation to the Reproducible Pairwise Signals

The frozen pairwise analysis identified:

- G7+G10
- G5+G10

as reproducible pairwise non-additive signals.

Their overlap with FROZEN51 was recomputed.

G7+G10:

- seed180 overlap = 0
- seed181 overlap = 0
- seed182 overlap = 0
- total occurrence overlap = 0

G5+G10:

- seed180 overlap = 0
- seed181 overlap = 0
- seed182 overlap = 0
- total occurrence overlap = 0

Therefore the two admitted pairwise-positive signals do not account for the
FROZEN51 population.

This does not prove that all possible pairwise mechanisms are irrelevant.

It establishes only the zero-overlap result for the two reproducible positive
pairs already admitted by the frozen pairwise analysis.

## 8. Common A0-to-D1 Geometry of the 50 NOT_ENTITLED-to-SUPPORT Occurrences

Across all 50 NOT_ENTITLED to SUPPORT FROZEN51 occurrences:

q_AUTHORIZED delta from A0 to D1:

- positive = 50 of 50
- negative = 0
- mean = +0.163756
- median = +0.161642
- minimum = +0.004919
- maximum = +0.309739

Entitlement-probability delta from A0 to D1:

- positive = 50 of 50
- negative = 0
- mean = +0.163756
- median = +0.161642
- minimum = +0.004919
- maximum = +0.309738

SUPPORT-minus-NOT_ENTITLED final-margin delta:

- positive = 50 of 50
- negative = 0
- mean = +0.811884
- median = +0.804380
- minimum = +0.082287
- maximum = +1.430767

For all 50 occurrences:

- A0 SUPPORT-minus-NOT_ENTITLED margin is negative
- D1 SUPPORT-minus-NOT_ENTITLED margin is positive

Therefore every one of these 50 rows crosses the same final
NOT_ENTITLED-versus-SUPPORT decision boundary between matched A0 and D1.

## 9. Polarity-Margin Control

Polarity-margin delta does not have a common sign.

Across the same 50 occurrences:

- positive delta = 27
- negative delta = 23
- zero = 0
- mean = -0.153786
- median = +0.139684
- minimum = -0.913635
- maximum = +0.323715

Therefore a uniform polarity-margin direction is not a shared geometric
property of the FROZEN51 NOT_ENTITLED-to-SUPPORT population.

This does not establish that polarity is irrelevant.

It rejects only a description requiring one common polarity-margin direction
across all 50 occurrences.

## 10. FRAME-Target Geometry

For the 37 FRAME-target NOT_ENTITLED-to-SUPPORT occurrences:

frame probability delta:

- positive = 37 of 37
- mean = +0.215085
- median = +0.216504

q_FRAME delta:

- negative = 37 of 37
- mean = -0.215085
- median = -0.216503

q_AUTHORIZED delta:

- positive = 37 of 37
- mean = +0.168267
- median = +0.175994

SUPPORT-minus-NOT_ENTITLED margin delta:

- positive = 37 of 37
- mean = +0.825129
- median = +0.804110

The observed FRAME-target geometry is therefore internally coherent in these
exported quantities.

This is a descriptive association and is not a causal gradient-path claim.

## 11. PREDICATE-Target Geometry

For the 13 PREDICATE-target NOT_ENTITLED-to-SUPPORT occurrences:

predicate-coverage probability delta:

- positive = 13 of 13
- mean = +0.185819
- median = +0.191672

q_AUTHORIZED delta:

- positive = 13 of 13
- mean = +0.150917
- median = +0.152440

SUPPORT-minus-NOT_ENTITLED margin delta:

- positive = 13 of 13
- mean = +0.774188
- median = +0.804650

q_PREDICATE delta is not sign-uniform:

- positive = 8
- negative = 5
- mean = +0.013802
- median = +0.011025

Therefore the PREDICATE-target population does not support a universal
description based on monotonic q_PREDICATE reduction.

## 12. Recurrent-Row Geometry

All six recurrent FROZEN51 stable IDs retain the same qualitative final
transition when they occur:

- A0 prediction = NOT_ENTITLED
- D1 prediction = SUPPORT

For every recurrent occurrence inspected:

- q_AUTHORIZED increases from matched A0 to D1
- SUPPORT-minus-NOT_ENTITLED margin crosses from negative to positive

The recurrent population contains both FRAME-target and PREDICATE-target rows.

Thus cross-seed recurrence is not confined to a single target-reason family.

## 13. Bounded Interpretation

The strongest common description supported by the exported evidence is:

GLOBAL_CUMULATIVE_AUTHORIZATION_ASSOCIATED_RESIDUAL

This label means only that:

- the rows are absent from every observed single-edge break set
- nearly all are NOT_ENTITLED to SUPPORT transitions
- q_AUTHORIZED and entitlement increase in all 50 such rows
- the final SUPPORT-minus-NOT_ENTITLED margin crosses upward in all 50
- the two already admitted reproducible positive pairwise conditions have
  zero overlap with these rows

The evidence does not establish that simultaneous attenuation is necessary.

The evidence does not establish that simultaneous attenuation is causally
sufficient.

The evidence does not identify a unique edge set.

The evidence does not identify a native Mamba recurrent-state mechanism.

The evidence does not establish parameter-level ownership.

## 14. Falsification Consequence

A future design may test whether attenuation across multiple topology groups is
associated with reproducible authorization-side and final
SUPPORT-versus-NOT_ENTITLED boundary movement in FROZEN51 rows.

Such a future design must not presume that multi-group attenuation is
necessary or causally sufficient.

A future proper-subset condition that reproducibly explains the recurrent
FROZEN51 rows would weaken a global-only interpretation.

A future result in which the historical authorization/final-boundary geometry
does not recur would weaken the present descriptive interpretation.

## 15. Research Boundary

This report is validated-evidence analysis only.

It authorizes no:

- implementation change
- grouped multi-edge arm
- training
- evaluation
- Kaggle execution
- checkpoint loading
- commit or push by itself

A separate scientific-design authority is required before any new intervention
is specified.

END_OF_GEN3_FROZEN51_GLOBAL_RESIDUAL_GEOMETRY_REPORT