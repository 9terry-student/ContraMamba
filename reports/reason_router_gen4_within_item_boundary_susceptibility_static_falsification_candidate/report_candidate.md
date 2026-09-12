# Generation-4 Within-Item Boundary Susceptibility
# Read-Only Static Falsification

STATUS = CANDIDATE

AUTHORITY = d941fd7e9c41c9680918831619a2f74db21c25f1
PARENT_VALIDATED_STATIC_EVIDENCE = 5388460bf0ac2ed1f8489e35c6942f6044da3df0

TRAINING = NO
MODEL_FORWARD = NO
CHECKPOINT_LOADING = NO
NEW_INFERENCE = NO
KAGGLE = NO
GPU = NO

## Primary feasibility and endpoint

N_DISCORDANT = 37
N_NONZERO = 37
N_ZERO_TIES = 0
K_NEGATIVE = 17
K_POSITIVE = 20
FEASIBILITY_GATE = PASS
WITHIN_ITEM_CONCORDANCE = 0.459459459
MEDIAN_DELTA_I = 0.021734640
EXACT_ONE_SIDED_SIGN_TEST_P = 0.744312109207

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY = NOT_SUPPORTED

## Population description

ROW_COUNT = 1590
UNIQUE_STABLE_IDS = 538
SINGLE_OBSERVATION_STABLE_IDS = 4
MULTI_OBSERVATION_STABLE_IDS = 534
CONCORDANT_ALL_NONFLIP_STABLE_IDS = 476
CONCORDANT_ALL_FLIP_STABLE_IDS = 21

## Prespecified secondary seed-pair audit

| Pair | Discordant | Flip-seed smaller d_A0 | Concordance | Median pairwise delta |
|---|---:|---:|---:|---:|
| 180_vs_181 | 14 | 6 | 0.428571 | 0.041359410 |
| 180_vs_182 | 24 | 13 | 0.541667 | -0.088360049 |
| 181_vs_182 | 29 | 11 | 0.379310 | 0.081723630 |

## Secondary magnitude description

MEAN_DELTA_I = 0.021443351
Q1_DELTA_I = -0.154838458
Q3_DELTA_I = 0.201952189
MINIMUM_DELTA_I = -0.554249883
MAXIMUM_DELTA_I = 0.469449893

## Interpretation boundary

This falsification conditions the association on stable_id across frozen training seeds.

A SUPPORTED result reduces stable item identity as a sufficient explanation of the prior cross-sectional Gen4 association.

It does not establish that boundary proximity causally produces D1 failure or that manipulating the boundary would prevent failure.

SCIENTIFIC_CLAIM_BOUNDARY = ITEM_CONTROLLED_PREDICTIVE_NOT_CAUSAL

END_OF_GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY_STATIC_FALSIFICATION
