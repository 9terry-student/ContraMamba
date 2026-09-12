# Generation-4 Leave-One-Seed-Out Stable-Item Boundary Susceptibility
# Read-Only Static Audit

STATUS = CANDIDATE

EXECUTION_AUTHORITY = 183b4a92be08e4a94ca19b3d43ccfdf1e29b57f1
DESIGN_AUTHORITY = cfe948ca20d74cbdd59baeb0a73c0a8c14fc0f74

TARGET_SEED_MARGIN_IN_PREDICTOR = NO
OTHER_SEED_D1_OUTCOME_IN_PREDICTOR = NO
TRAINING = NO
MODEL_FORWARD = NO
CHECKPOINT_LOADING = NO
NEW_INFERENCE = NO
KAGGLE = NO
GPU = NO

## Primary LOO results

| Target seed | Eligible | Positive | Negative | Gate | AUC | Raw p | Holm p |
|---:|---:|---:|---:|---|---:|---:|---:|
| 180 | 534 | 34 | 500 | PASS | 0.959412 | 0.000010 | 0.000030 |
| 181 | 529 | 23 | 506 | PASS | 0.973019 | 0.000010 | 0.000030 |
| 182 | 523 | 46 | 477 | PASS | 0.986464 | 0.000010 | 0.000030 |

COUNT_GATE_ALL_SEEDS = PASS

GEN4_STABLE_ITEM_BOUNDARY_COMPONENT = SUPPORTED

## Secondary predictor stability

| Pair | Common IDs | Pearson | Spearman |
|---|---:|---:|---:|
| 180_vs_181 | 529 | 0.930822 | 0.970763 |
| 180_vs_182 | 523 | 0.761431 | 0.888484 |
| 181_vs_182 | 518 | 0.725206 | 0.853170 |

## Interpretation boundary

This audit tests whether an A0 boundary component measured only in other training seeds predicts D1 SUPPORT susceptibility in the held-out seed.

A positive result supports a stable item-associated vulnerability signal, not a causal role for the boundary itself.

END_OF_GEN4_LEAVE_ONE_SEED_OUT_STABLE_ITEM_BOUNDARY_SUSCEPTIBILITY_STATIC_AUDIT
