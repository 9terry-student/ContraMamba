# Generation-4 Within-Item Boundary Susceptibility
# Independent Static-Falsification Verification Report

STATUS = PASS_CANDIDATE

PHASE =
GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY_INDEPENDENT_VERIFICATION

FALSIFICATION_AUTHORITY =
d941fd7e9c41c9680918831619a2f74db21c25f1

PARENT_VALIDATED_STATIC_EVIDENCE =
5388460bf0ac2ed1f8489e35c6942f6044da3df0

PRIMARY_FALSIFICATION_VERDICT =
NOT_SUPPORTED

## 1. Verification scope

The verification independently re-read the frozen Generation-4 primary
population artifact and reconstructed the prespecified stable_id-level
within-item falsification.

No raw A0 or D1 source was reinterpreted.

No training, model inference, checkpoint loading, new seed, new Gen3 arm,
Kaggle execution, or GPU execution was performed.

No confidence metric, margin definition, eligibility rule, seed subset,
effect threshold, or statistical test was changed.

## 2. Frozen input verification

FROZEN_PRIMARY_POPULATION =
PASS

ARTIFACT_INTEGRITY =
PASS

Frozen primary population SHA256:

eb5aaacf38be224461801cab5e0629e46a9d375d7efaeadb72b2dbf3c5d56f3e

Frozen population row count:

1590

PRIMARY_INPUT_RECONSTRUCTION =
PASS

## 3. Within-item effect reconstruction

Primary unit:

stable_id

Primary effect:

delta_i =
mean_flip_d_A0 - mean_nonflip_d_A0

Prespecified direction:

delta_i < 0

Independent reconstruction:

N_DISCORDANT =
37

N_NONZERO =
37

N_ZERO_TIES =
0

K_NEGATIVE =
17

K_POSITIVE =
20

WITHIN_ITEM_EFFECT_RECONSTRUCTION =
PASS

## 4. Primary statistical recomputation

WITHIN_ITEM_CONCORDANCE =
0.4594594594594595

MEDIAN_DELTA_I =
0.02173464000225067

EXACT_ONE_SIDED_SIGN_TEST_P =
0.7443121092073852

The prespecified feasibility gate passes:

N_DISCORDANT >= 15

and:

N_NONZERO >= 15

However, the directional endpoint does not.

WITHIN_ITEM_CONCORDANCE is below 0.5 and therefore also below the frozen
minimum effect threshold of 0.70.

The median delta_i is positive rather than negative.

PRIMARY_STATISTICAL_RECOMPUTATION =
PASS

## 5. Prespecified pairwise diagnostics

180_vs_181:

- discordant = 14
- flip-seed smaller d_A0 = 6
- concordance = 0.428571428571429
- median pairwise delta = 0.041359409689903

180_vs_182:

- discordant = 24
- flip-seed smaller d_A0 = 13
- concordance = 0.541666666666667
- median pairwise delta = -0.088360048830509

181_vs_182:

- discordant = 29
- flip-seed smaller d_A0 = 11
- concordance = 0.379310344827586
- median pairwise delta = 0.081723630428314

No seed pair provides a basis for overriding the primary stable_id-level
verdict.

SECONDARY_PAIRWISE_RECOMPUTATION =
PASS

## 6. Frozen verdict verification

The frozen authority requires:

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY =
NOT_SUPPORTED

when feasibility passes and:

WITHIN_ITEM_CONCORDANCE <= 0.5

Observed:

0.4594594594594595

Therefore:

FROZEN_VERDICT_RULE =
PASS

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY =
NOT_SUPPORTED

## 7. Scientific interpretation

The previous cross-sectional Generation-4 result remains valid:

baseline A0 SUPPORT / NOT_ENTITLED boundary proximity strongly predicts D1
supportward susceptibility across examples.

The within-item falsification shows that this relationship does not persist
when stable item identity is held fixed across frozen training seeds.

Therefore the evidence does not support treating seed-to-seed changes in
A0 boundary proximity as the causal vulnerability responsible for D1
supportward failure.

The data are instead consistent with the cross-sectional boundary signal
capturing an item-level latent vulnerability, difficulty dimension, or other
stable item-associated structure.

This verification does not identify what that latent item-level factor is.

Accordingly:

SCIENTIFIC_INTERPRETATION =
CROSS_SECTIONAL_PREDICTION_DOES_NOT_PERSIST_WITHIN_ITEM

CAUSAL_BOUNDARY_INTERVENTION_JUSTIFIED =
NO

SCIENTIFIC_CLAIM_BOUNDARY =
PREDICTIVE_BETWEEN_ITEM_NOT_WITHIN_ITEM_CAUSAL

## 8. Prohibited rescue interpretations

This negative result does not authorize:

- redefining d_A0;
- switching to probability confidence;
- choosing a favorable seed pair;
- removing seed182;
- restricting to FROZEN51;
- restricting to recurrent residual IDs;
- lowering the 0.70 concordance threshold;
- adding seeds to rescue the hypothesis;
- launching a boundary causal intervention as though the within-item test
  were positive.

## 9. Verification verdict

FROZEN_PRIMARY_POPULATION =
PASS

ARTIFACT_INTEGRITY =
PASS

WITHIN_ITEM_EFFECT_RECONSTRUCTION =
PASS

PRIMARY_STATISTICAL_RECOMPUTATION =
PASS

SECONDARY_PAIRWISE_RECOMPUTATION =
PASS

INDEPENDENT_VERIFY =
PASS

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY =
NOT_SUPPORTED

CAUSAL_BOUNDARY_INTERVENTION_JUSTIFIED =
NO

END_OF_GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY_INDEPENDENT_VERIFICATION
