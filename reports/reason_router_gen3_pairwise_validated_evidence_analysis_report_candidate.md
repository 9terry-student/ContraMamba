# Generation-3 Pairwise Interaction Validated-Evidence Analysis

VERDICT = PASS

PHASE = GEN3_PAIRWISE_VALIDATED_EVIDENCE_ANALYSIS

PAIRWISE_MATRIX = COMPLETE_36_OF_36

PAIRWISE_PROVENANCE_ADMISSION = PASS_36_OF_36

PRIMARY_GEN3_INTERPRETATION = DISTRIBUTED_OWNERSHIP_DEPENDENCE

PAIRWISE_NONADDITIVE_COMPONENT = SUPPORTED

REPRODUCIBLE_PAIRWISE_NONADDITIVITY = G7+G10 AND G5+G10

COMMON_INTERACTION_PARTICIPANT = G10 / Q_TO_D

SINGLE_EDGE_LOCALIZATION = REJECTED

PAIRWISE_EXPLANATION_OF_GLOBAL_HALF = INCOMPLETE

FROZEN_51_GLOBAL_ONLY_BREAKS_EXPLAINED_BY_REPRODUCIBLE_PAIRWISE_SIGNAL = NO

HIGHER_ORDER_OR_GLOBAL_CUMULATIVE_COMPONENT = REMAINS_UNRESOLVED

TRAINING_EVALUATION_INFERENCE = NOT_PERFORMED_BY_THIS_ANALYSIS

CHECKPOINT_LOADING = NOT_PERFORMED

K_SERIES_MIXING = NO

## 1. Scope

This report analyzes only the completed frozen Generation-3 pairwise matrix.

The matrix contains 12 topology-selected pairwise arms evaluated at training
seeds 180, 181, and 182, for 36 scientific runs total.

All pairwise runs used scientific implementation commit
d62c375e2d730f040717422d3951199b71dc688e.

The frozen execution coordinate was:

- reason_router_mode = explicit_product
- gradient_ownership_mode = edge_specific
- exactly two named edge lambdas = 0.5
- all other eight edge lambdas = 1.0
- reason_loss_weight = 0
- split_seed = 8192
- freeze_encoder = true
- frame_downstream_gradient_mode = joint

All 36 runs completed execution, collection, local import, and provenance
admission.

All 36 run provenances reconciled to the exact scientific implementation
commit, training seed, split seed, pair identity, and ten-edge lambda map.

No lambda sweep, pair replacement, adaptive ownership, three-edge
intervention, architecture change, K-series mixing, or checkpoint-based
inference was performed.

## 2. Canonical A0 Reference Reconciliation

The canonical Seed8192 revised-split A0 primary N=3 references were recovered
from the previously validated scientific-evidence worktree and independently
checked against their frozen exact SHA256 identities.

Seed180 replacement_r1:

SHA256:
5c9722ac0f75c411b3d744a29beec0c35d2f2809331f257a5e1d5ea81e6cf75d

Seed181:

SHA256:
789d02f9092ce6b051d0ca435272c9e93a3962183dbb0a4d4dbb20cebf2ac3fe

Seed182:

SHA256:
029ec6ae31df2f5ca9526d1e631496f7ee272967a6f5e08684f29aa09ad490d4

Each canonical A0 prediction export contains exactly 720 matched stable IDs.

The three-seed canonical A0 aggregate reproduced the frozen first-pass result:

- mean accuracy = 0.903240741
- mean macro-F1 = 0.803636394

CANONICAL_A0_RECONCILIATION = PASS

## 3. Frozen First-Pass Reproduction

Before any pairwise interpretation, the historical D1 and single-edge
row-level decomposition was recomputed against the recovered canonical A0.

The frozen values reproduced exactly:

- seed180 D1 A0-correct breaks = 37
- seed181 D1 A0-correct breaks = 23
- seed182 D1 A0-correct breaks = 51
- total D1 A0-correct breaks = 111
- D1 breaks seen in at least one single-edge arm = 60
- D1 breaks absent from every single-edge arm = 51

All frozen individual single-edge break totals also reproduced.

FROZEN_FIRST_PASS_ROW_ANALYSIS = PASS

## 4. Pairwise Interpretation Rule

The frozen pairwise design defines a reproducible pairwise non-additive signal
as D1-direction A0-correct breakage that is not present in the union of the two
constituent single-edge break sets, recurs across seeds, and is accompanied by
a consistent task-quality or class-specific degradation direction.

No conclusion is based only on one seed, one class, one stable ID, or aggregate
score ranking.

## 5. G7+G10 Result

G7+G10 corresponds to F_TO_D plus Q_TO_D.

CATEGORY = PAIRWISE_NONADDITIVE_SIGNAL

Observed three-seed summary:

- mean delta accuracy = -0.013889
- mean delta macro-F1 = -0.015690
- accuracy degraded in all three seeds
- macro-F1 degraded in all three seeds
- NOT_ENTITLED F1 degraded in all three seeds
- SUPPORT F1 degraded in all three seeds

New D1-direction A0-correct breaks beyond the constituent single-edge union:

- seed180 = 2
- seed181 = 9
- seed182 = 0

One new stable ID recurred in two of three seeds:

jazz_archive__predicate_swap

Therefore G7+G10 satisfies the frozen pairwise non-additivity criterion.

## 6. G5+G10 Result

G5+G10 corresponds to P_TO_Q plus Q_TO_D.

CATEGORY = PAIRWISE_NONADDITIVE_SIGNAL

Observed three-seed summary:

- mean delta accuracy = -0.011574
- mean delta macro-F1 = -0.005894
- accuracy degraded in all three seeds
- macro-F1 degraded in all three seeds
- NOT_ENTITLED F1 degraded in all three seeds
- SUPPORT F1 degraded in all three seeds

New D1-direction A0-correct breaks beyond the constituent single-edge union:

- seed180 = 0
- seed181 = 6
- seed182 = 9

Three new stable IDs recurred in two of three seeds:

- generated_fact_056__predicate_swap
- generated_fact_091__predicate_swap
- generated_fact_144__predicate_swap

Therefore G5+G10 also satisfies the frozen pairwise non-additivity criterion.

## 7. Remaining Pairwise Arms

No other frozen pair produced a new D1-direction stable ID outside its
constituent single-edge break union that recurred in at least two seeds.

The recurrent-new-D1 counts for the remaining pairs are all zero:

- G4+G5 = 0
- G4+G6 = 0
- G5+G6 = 0
- G7+G8 = 0
- G7+G9 = 0
- G8+G9 = 0
- G8+G10 = 0
- G9+G10 = 0
- G4+G10 = 0
- G6+G10 = 0

Several of these conditions contain one-seed degradation, substantial
constituent-row repair, or seed-heterogeneous effects.

Those observations do not satisfy the frozen reproducible non-additivity
criterion.

No pair is promoted to PAIRWISE_CUMULATIVE_ONLY from the completed evidence.

## 8. Relation to the Frozen 51 GLOBAL-HALF-Only Breaks

The two reproducible pairwise non-additive signals do not reproduce the
original frozen 51 GLOBAL-HALF-only break occurrences.

For G7+G10:

- seed180 Frozen51Overlap = 0
- seed181 Frozen51Overlap = 0
- seed182 Frozen51Overlap = 0

For G5+G10:

- seed180 Frozen51Overlap = 0
- seed181 Frozen51Overlap = 0
- seed182 Frozen51Overlap = 0

Therefore:

PAIRWISE_NONADDITIVITY_EXISTS = YES

PAIRWISE_NONADDITIVITY_EXPLAINS_THE_FROZEN_51_RESIDUAL = NO

The still-unexplained GLOBAL-HALF component remains scientifically distinct
from the two detected reproducible pairwise signals.

## 9. Structural Interpretation

Both reproducible positive pairs contain G10, corresponding to Q_TO_D.

The two positive contexts are:

- G7+G10 = F_TO_D plus Q_TO_D
- G5+G10 = P_TO_Q plus Q_TO_D

The bounded conclusion is that Q_TO_D participates in two reproducible
controlled interaction contexts.

This does not establish that G10 alone is the causal mechanism.

The G10 single-edge intervention was weak in the first-pass matrix, while the
positive pairwise effects occur only with distinct partner edges.

Therefore this evidence supports interaction-context participation rather than
single-edge ownership localization.

## 10. Relation to GLOBAL-HALF Magnitude

Historical D1 GLOBAL-HALF mean delta accuracy was approximately -0.039352.

The two reproducible pairwise effects were smaller:

- G7+G10 mean delta accuracy = -0.013889
- G5+G10 mean delta accuracy = -0.011574

Neither pair reproduces the full GLOBAL-HALF phenotype.

## 11. Final Generation-3 Interpretation

The completed Generation-3 evidence supports the following bounded result.

PRIMARY = DISTRIBUTED_OWNERSHIP_DEPENDENCE

PAIRWISE_COMPONENT = SUPPORTED

REPRODUCIBLE_PAIRWISE_NONADDITIVITY = G7+G10 AND G5+G10

COMMON_INTERACTION_PARTICIPANT = G10 / Q_TO_D

SINGLE_EDGE_LOCALIZATION = REJECTED

PAIRWISE_EXPLANATION_OF_GLOBAL_HALF = INCOMPLETE

HIGHER_ORDER_OR_GLOBAL_CUMULATIVE_COMPONENT = REMAINS_UNRESOLVED

The evidence does not establish parameter-level orthogonality, unique
parameter ownership, a native Mamba recurrent-state mechanism, an optimal
lambda, or production readiness.

## 12. Research Boundary

The current pairwise execution authority explicitly excludes three-edge or
higher-order attenuation.

This report therefore does not authorize additional Gen3 training,
evaluation, or Kaggle execution.

The scientifically valid next phase is static design work only after this
validated-evidence report is frozen.

That later design should define a bounded falsification test for the remaining
GLOBAL-HALF residual.

An exhaustive combinatorial three-edge sweep is neither implied nor
authorized.

END_OF_GEN3_PAIRWISE_VALIDATED_EVIDENCE_REPORT