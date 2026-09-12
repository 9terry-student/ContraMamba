# Generation-3 Grouped-Factorial Validated-Evidence Analysis

VERDICT = PASS_CANDIDATE

PHASE = GEN3_GROUPED_FACTORIAL_VALIDATED_EVIDENCE_ANALYSIS

GROUPED_EXECUTION_AUTHORITY =
dbd2746cef80b6d3e1c2c428afb96004887827b8

GROUPED_IMPLEMENTATION_COMMIT =
3e0e9a435068c552abf20f3a74e0c3eccca344a3

FROZEN51_EVIDENCE_AUTHORITY =
792b7a7e9bdb7fa957fa3ecda6ee9c6e37ce0f96

PAIRWISE_VALIDATED_EVIDENCE =
4594f2d58e073610e63e679c97f4f19aea1db92e

GROUPED_MATRIX = COMPLETE_18_OF_18

GROUPED_PROVENANCE_ADMISSION = PASS_18_OF_18

HISTORICAL_EXACT_SOURCE_RECONCILIATION = PASS_36_OF_36

FROZEN51_RECONSTRUCTION = PASS

PRIMARY_GEN3_INTERPRETATION =
DISTRIBUTED_OWNERSHIP_DEPENDENCE

MACRO_GROUP_SUFFICIENT = NOT_SUPPORTED

CROSS_GROUP_CUMULATIVE_INTERACTION = SUPPORTED

REPRODUCIBLE_GROUP_INTERACTION = U+D

RECURRENT_NEW_FROZEN51_UD =
generated_fact_181__role_swap

GLOBAL_ONLY_CUMULATIVE_THRESHOLD = REJECTED

MIXED_GROUP_AND_GLOBAL_COMPONENT = SUPPORTED

PRIMARY_GROUPED_INTERPRETATION =
MIXED_GROUP_AND_GLOBAL_COMPONENT

GROUPED_EXPLANATION_OF_FROZEN51 = PARTIAL

HIGHER_ORDER_OR_GLOBAL_CUMULATIVE_COMPONENT =
REMAINS_UNRESOLVED_FOR_4_OF_6_RECURRENT_IDS

TRAINING_EVALUATION_INFERENCE = NOT_PERFORMED_BY_THIS_ANALYSIS

CHECKPOINT_LOADING = NOT_PERFORMED

K_SERIES_MIXING = NO

NEW_EXPERIMENT = NO


## 1. Scope

This report analyzes only the completed and provenance-admitted
Generation-3 grouped-factorial matrix.

The matrix contains six prespecified proper-subset macro-group conditions:

- U
- Q
- D
- U+Q
- U+D
- Q+D

evaluated at frozen training seeds 180, 181, and 182, for exactly
18 scientific runs.

Historical U+Q+D is the already-frozen D1 / GLOBAL-HALF endpoint and was not
rerun.

No lambda sweep, arbitrary N-edge search, group reselection, implementation
change, model change, additional seed, Gen4 execution, or K-series mixing was
performed.


## 2. Evidence admission

All 18 grouped runs executed from scientific source commit:

3e0e9a435068c552abf20f3a74e0c3eccca344a3

The imported grouped matrix passed provenance admission for:

- exact source commit;
- exact independent run identity;
- training seed;
- split_seed = 8192;
- reason_router_mode = explicit_product;
- gradient_ownership_mode = edge_specific;
- reason_loss_weight = 0;
- freeze_encoder = true;
- frame_downstream_gradient_mode = joint;
- exact canonical ten-edge lambda map;
- artifact path scope;
- artifact SHA256.

The imported matrix contained exactly:

18 runs x 5 scoped artifacts = 90 artifacts.

Therefore:

GROUPED_PROVENANCE_ADMISSION = PASS_18_OF_18

GROUPED_MATRIX = COMPLETE_18_OF_18


## 3. Historical reference reconciliation

The frozen FROZEN51 source manifest was read from commit:

792b7a7e9bdb7fa957fa3ecda6ee9c6e37ce0f96

Manifest SHA256:

3ec5e37d92a347f169a6a5f8902f75b055ecb47b7d301909253b6112fdd08203

The exact historical bytes required for grouped analysis were independently
recovered and SHA256-authenticated:

- 3 canonical A0 prediction exports;
- 3 historical D1 / GLOBAL-HALF prediction exports;
- 30 Generation-3 single-edge prediction exports.

Total exact historical sources:

36

The frozen decomposition reproduced exactly:

- seed180 D1 A0-correct breaks = 37
- seed181 D1 A0-correct breaks = 23
- seed182 D1 A0-correct breaks = 51

FROZEN51 occurrences:

- seed180 = 21
- seed181 = 7
- seed182 = 23
- total = 51

The 51 occurrences correspond to:

- 43 unique stable IDs;
- 6 stable IDs recurrent in at least two frozen training seeds.

The six frozen recurrent IDs are:

- clinic_expansion__event_swap
- clinic_expansion__predicate_swap
- generated_fact_045__role_swap
- generated_fact_181__role_swap
- generated_fact_193__predicate_swap
- generated_fact_258__title_name_swap


## 4. Frozen grouped interpretation rule

For each seed and grouped condition X:

A0_BREAK_X =
rows correct under matched canonical A0 and wrong under X.

The primary grouped residual is:

A0_BREAK_X intersect FROZEN51.

Cross-seed recurrence means that the same stable ID occurs in at least two of
the three frozen training seeds.

For each two-group condition XY:

NEW_FROZEN51_XY =
(A0_BREAK_XY intersect FROZEN51)
minus
(A0_BREAK_X union A0_BREAK_Y).

A cross-group cumulative interaction requires recurrent NEW_FROZEN51_XY stable
IDs beyond the constituent single-group union.

Aggregate accuracy or three-seed means alone do not determine the scientific
category.


## 5. Secondary aggregate measurements

Three-seed means are reported only as secondary context.

| Arm | Mean Accuracy | Mean Macro-F1 | Mean NOT_ENTITLED F1 | Mean REFUTE F1 | Mean SUPPORT F1 |
|---|---:|---:|---:|---:|---:|
| U | 0.893519 | 0.794199 | 0.931976 | 0.998117 | 0.452505 |
| Q | 0.898611 | 0.799385 | 0.935464 | 0.998117 | 0.464575 |
| D | 0.884259 | 0.789535 | 0.924953 | 1.000000 | 0.443651 |
| U+Q | 0.884259 | 0.790039 | 0.925333 | 0.998117 | 0.446668 |
| U+D | 0.883333 | 0.785620 | 0.925471 | 0.994329 | 0.437060 |
| Q+D | 0.882870 | 0.787868 | 0.924390 | 0.998117 | 0.441096 |

These measurements are not the primary grouped endpoint.


## 6. Single macro-group result

FROZEN51 overlap counts by seed:

U:

- seed180 = 1
- seed181 = 0
- seed182 = 2

Q:

- seed180 = 0
- seed181 = 0
- seed182 = 1

D:

- seed180 = 0
- seed181 = 2
- seed182 = 0

No FROZEN51 stable ID recurred across at least two seeds under U alone.

No FROZEN51 stable ID recurred across at least two seeds under Q alone.

No FROZEN51 stable ID recurred across at least two seeds under D alone.

Therefore:

MACRO_GROUP_SUFFICIENT = NOT_SUPPORTED

The grouped evidence does not support an intervention-level description in
which any one of U, Q, or D alone reproducibly recovers the frozen recurrent
GLOBAL-HALF residual.


## 7. U+Q result

FROZEN51 overlap counts:

- seed180 = 5
- seed181 = 3
- seed182 = 3

One FROZEN51 stable ID recurred under U+Q:

clinic_expansion__event_swap

It occurred under U+Q at:

- seed180
- seed182

For both recurrent occurrences:

- q_AUTHORIZED increased;
- entitlement probability increased;
- SUPPORT-minus-NOT_ENTITLED final margin increased;
- the final margin crossed from negative under matched A0 to positive under
  U+Q.

Thus U+Q reproduces one recurrent historical FROZEN51 phenotype together with
the frozen authorization/final-boundary geometry.

However the prespecified NEW_FROZEN51_UQ sets were:

- seed180 = 4
- seed181 = 3
- seed182 = 1

and no NEW_FROZEN51_UQ stable ID recurred in at least two seeds.

Therefore:

UQ_RECURRENT_FROZEN51 = YES

UQ_RECURRENT_NEW_FROZEN51 = NO

U+Q does not independently satisfy the stronger frozen
CROSS_GROUP_CUMULATIVE_INTERACTION criterion.


## 8. U+D result

FROZEN51 overlap counts:

- seed180 = 4
- seed181 = 3
- seed182 = 2

One FROZEN51 stable ID recurred under U+D:

generated_fact_181__role_swap

It occurred under U+D at:

- seed180
- seed181

The prespecified NEW_FROZEN51_UD sets were:

seed180:

- clinic_expansion__event_swap
- generated_fact_181__role_swap
- generated_fact_195__role_swap

seed181:

- generated_fact_181__role_swap
- jazz_archive__role_swap

seed182:

- generated_fact_167__role_swap

Therefore:

NEW_FROZEN51_UD_COUNTS = 3, 2, 1

The stable ID:

generated_fact_181__role_swap

is NEW_FROZEN51_UD in both seed180 and seed181.

It is therefore recurrent beyond the constituent U-alone and D-alone break
union.

For both recurrent occurrences:

- q_AUTHORIZED increased;
- entitlement probability increased;
- SUPPORT-minus-NOT_ENTITLED final margin increased;
- the final margin crossed from negative under matched A0 to positive under
  U+D.

Exact recurrent geometry:

seed180:

- delta q_AUTHORIZED = +0.183299
- delta entitlement probability = +0.183299
- delta SUPPORT-minus-NOT_ENTITLED margin = +0.880042
- A0 margin = -0.602158
- U+D margin = +0.277884

seed181:

- delta q_AUTHORIZED = +0.025301
- delta entitlement probability = +0.025301
- delta SUPPORT-minus-NOT_ENTITLED margin = +0.136298
- A0 margin = -0.098136
- U+D margin = +0.038162

Therefore U+D satisfies the prespecified cross-group cumulative interaction
criterion.

CROSS_GROUP_CUMULATIVE_INTERACTION = SUPPORTED

REPRODUCIBLE_GROUP_INTERACTION = U+D

RECURRENT_NEW_FROZEN51_UD =
generated_fact_181__role_swap


## 9. Q+D result

FROZEN51 overlap counts:

- seed180 = 3
- seed181 = 2
- seed182 = 0

No FROZEN51 stable ID recurred across at least two seeds under Q+D.

The prespecified NEW_FROZEN51_QD counts were:

- seed180 = 3
- seed181 = 0
- seed182 = 0

No NEW_FROZEN51_QD stable ID recurred.

Therefore Q+D does not satisfy the frozen cross-group cumulative interaction
criterion.


## 10. Recurrent FROZEN51 decomposition

The six historically recurrent FROZEN51 stable IDs decompose as follows.

### clinic_expansion__event_swap

Historical D1 occurrence seeds:

- 180
- 182

Grouped occurrences:

- U: seed182
- Q: seed182
- U+Q: seed180, seed182
- U+D: seed180

It is recurrently recovered by the proper subset U+Q.

### clinic_expansion__predicate_swap

Historical D1 occurrence seeds:

- 180
- 182

Grouped occurrence:

- U+Q: seed182 only

It is not recurrently recovered by any proper grouped subset.

### generated_fact_045__role_swap

Historical D1 occurrence seeds:

- 180
- 181

Grouped occurrences:

- D: seed181
- U+Q: seed181
- Q+D: seed181

It is not recurrently recovered by any proper grouped subset.

### generated_fact_181__role_swap

Historical D1 occurrence seeds:

- 180
- 181
- 182

Grouped occurrences:

- U+Q: seed180
- U+D: seed180, seed181
- Q+D: seed180

It is recurrently recovered by U+D and is recurrent NEW_FROZEN51_UD.

### generated_fact_193__predicate_swap

Historical D1 occurrence seeds:

- 180
- 182

No proper grouped subset reproduces this stable ID.

### generated_fact_258__title_name_swap

Historical D1 occurrence seeds:

- 180
- 181
- 182

Grouped occurrence:

- Q+D: seed180 only

It is not recurrently recovered by any proper grouped subset.

Therefore the historically recurrent population decomposes into:

PROPER_SUBSET_RECOVERED_RECURRENT = 2_OF_6

- clinic_expansion__event_swap
- generated_fact_181__role_swap

PROPER_SUBSET_UNRECOVERED_RECURRENT = 4_OF_6

- clinic_expansion__predicate_swap
- generated_fact_045__role_swap
- generated_fact_193__predicate_swap
- generated_fact_258__title_name_swap


## 11. Frozen category decision

The grouped evidence does not support MACRO_GROUP_SUFFICIENT because no single
macro-group produces a recurrent FROZEN51 stable ID.

The grouped evidence supports CROSS_GROUP_CUMULATIVE_INTERACTION because U+D
produces recurrent NEW_FROZEN51_UD beyond the constituent U and D break union.

The grouped evidence rejects GLOBAL_ONLY_CUMULATIVE_THRESHOLD as a complete
description because proper grouped subsets reproducibly recover part of the
historical recurrent FROZEN51 phenotype.

However proper grouped subsets do not reproducibly recover the entire
historical recurrent residual.

Four of the six historically recurrent FROZEN51 stable IDs remain unrecovered
by every proper grouped subset at the frozen recurrence threshold.

Therefore the prespecified category supported by the complete grouped matrix
is:

MIXED_GROUP_AND_GLOBAL_COMPONENT

This means that:

1. part of the historical GLOBAL-HALF residual can be reproduced by proper
   grouped interventions;
2. a reproducible U+D cumulative interaction exists;
3. another recurrent component remains restricted to the historical all-group
   D1 phenotype under the tested proper-subset matrix.

Thus:

PRIMARY_GROUPED_INTERPRETATION =
MIXED_GROUP_AND_GLOBAL_COMPONENT

GROUPED_EXPLANATION_OF_FROZEN51 = PARTIAL

HIGHER_ORDER_OR_GLOBAL_CUMULATIVE_COMPONENT =
REMAINS_UNRESOLVED_FOR_4_OF_6_RECURRENT_IDS


## 12. Relation to earlier Generation-3 evidence

The earlier Generation-3 single-edge matrix rejected single-edge
localization.

The frozen pairwise matrix established reproducible pairwise nonadditivity for:

- G7 + G10
- G5 + G10

but those two admitted pairwise-positive signals had zero FROZEN51 overlap.

The grouped evidence now shows that the unresolved global residual is not
purely all-group-only.

A proper two-group U+D intervention reproduces one recurrent historical
FROZEN51 stable ID as a new cumulative interaction, while U+Q reproduces
another recurrent historical stable ID without satisfying the stricter
NEW_FROZEN51_UQ interaction criterion.

This partially resolves the earlier higher-order/global residual without
eliminating it.


## 13. Claim boundary

The grouped evidence supports a bounded descriptive intervention result.

It does not establish:

- necessity of U, Q, or D;
- causal sufficiency of any macro-group;
- unique edge causation;
- unique pair causation;
- parameter-level ownership;
- gradient orthogonality;
- a native Mamba recurrent-state mechanism;
- polarity irrelevance;
- an optimal lambda;
- production readiness.

The fixed 0.5 attenuation remains a prespecified scientific probe and is not
an optimized value.

The remaining 4-of-6 recurrent residual does not itself authorize an arbitrary
higher-order search, lambda sweep, Gen4 execution, or K-series mixing.


## 14. Companion validated analysis artifact

Companion file:

reports/reason_router_gen3_grouped_factorial_validated_evidence_analysis.json

Extraction-source SHA256:

503c02ed05977fe221bae6994d2da6624e2a132da3dc0be79139e34f1961e50d

Canonical repository LF SHA256:

d24122372a82f2bf9fb83a20b03a0f0ee4216c59f2baaa598bc9f2743617601c

The extraction-source JSON and canonical repository JSON were verified to
parse to the identical JSON object. The byte-level difference is line-ending
normalization only.

The companion artifact retains:

- all 18 grouped run metrics;
- exact A0-correct break IDs;
- exact A0-wrong repair IDs;
- exact FROZEN51 overlap IDs;
- exact historical D1 overlap IDs;
- per-occurrence frozen residual geometry;
- cross-seed recurrence;
- exact NEW_FROZEN51_UQ;
- exact NEW_FROZEN51_UD;
- exact NEW_FROZEN51_QD;
- the six-ID historical recurrent residual matrix;
- mechanical decision facts used by this report.

No checkpoint inference was used to construct this analysis.


## 15. Decision

GROUPED_MATRIX = COMPLETE_18_OF_18

GROUPED_PROVENANCE_ADMISSION = PASS_18_OF_18

MACRO_GROUP_SUFFICIENT = NOT_SUPPORTED

CROSS_GROUP_CUMULATIVE_INTERACTION = SUPPORTED

REPRODUCIBLE_GROUP_INTERACTION = U+D

GLOBAL_ONLY_CUMULATIVE_THRESHOLD = REJECTED

MIXED_GROUP_AND_GLOBAL_COMPONENT = SUPPORTED

PRIMARY_GROUPED_INTERPRETATION =
MIXED_GROUP_AND_GLOBAL_COMPONENT

NEW_TRAINING_OR_EVALUATION = NOT_AUTHORIZED_BY_THIS_REPORT

KAGGLE = NOT_REQUIRED

NEXT_ACTION =
FREEZE_THIS_VALIDATED_EVIDENCE_ANALYSIS_AFTER_EXACT_FILE_REVIEW

END_OF_GEN3_GROUPED_FACTORIAL_VALIDATED_EVIDENCE_REPORT
