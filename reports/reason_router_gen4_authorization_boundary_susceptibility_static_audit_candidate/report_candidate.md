# Generation-4 Authorization Boundary Susceptibility
# Read-Only Static Audit

STATUS = CANDIDATE

EXECUTION_AUTHORITY = 96ccb359c579515293c24f7544d128b0591343d3
DESIGN_AUTHORITY = 1fec5e47bcae970b5ec6a23e3724087fd3f10029

TRAINING = NO
MODEL_FORWARD = NO
CHECKPOINT_LOADING = NO
KAGGLE = NO
GPU = NO

## Primary population and endpoints

| Seed | Population | D1 SUPPORT flip | Non-flip | Count gate | AUC | Raw p | Holm p |
|---:|---:|---:|---:|---|---:|---:|---:|
| 180 | 537 | 37 | 500 | PASS | 0.996378 | 0.000010 | 0.000030 |
| 181 | 529 | 23 | 506 | PASS | 0.977659 | 0.000010 | 0.000030 |
| 182 | 524 | 47 | 477 | PASS | 0.882377 | 0.000010 | 0.000030 |

COUNT_GATE_ALL_SEEDS = PASS

GEN4_BOUNDARY_SUSCEPTIBILITY = SUPPORTED

## Fixed five-bin descriptive analysis

### Seed 180

| Bin | N | D1 SUPPORT flips | Flip rate | Median d_A0 |
|---:|---:|---:|---:|---:|
| 1 | 108 | 37 | 0.342593 | 0.646334 |
| 2 | 108 | 0 | 0.000000 | 0.814717 |
| 3 | 107 | 0 | 0.000000 | 0.897843 |
| 4 | 107 | 0 | 0.000000 | 1.016361 |
| 5 | 107 | 0 | 0.000000 | 1.030928 |

### Seed 181

| Bin | N | D1 SUPPORT flips | Flip rate | Median d_A0 |
|---:|---:|---:|---:|---:|
| 1 | 106 | 23 | 0.216981 | 0.525897 |
| 2 | 106 | 0 | 0.000000 | 0.763949 |
| 3 | 106 | 0 | 0.000000 | 0.860372 |
| 4 | 106 | 0 | 0.000000 | 1.013623 |
| 5 | 105 | 0 | 0.000000 | 1.026844 |

### Seed 182

| Bin | N | D1 SUPPORT flips | Flip rate | Median d_A0 |
|---:|---:|---:|---:|---:|
| 1 | 105 | 33 | 0.314286 | 0.459121 |
| 2 | 105 | 10 | 0.095238 | 0.768573 |
| 3 | 105 | 3 | 0.028571 | 0.904328 |
| 4 | 105 | 1 | 0.009524 | 1.020724 |
| 5 | 104 | 0 | 0.000000 | 1.032748 |

## Interpretation boundary

This audit tests predictive boundary susceptibility only. It does not establish that A0 boundary proximity causes D1 failure.

No new training, model forward pass, checkpoint loading, Kaggle execution, GPU execution, seed addition, or Gen3 arm was used.

END_OF_GEN4_AUTHORIZATION_BOUNDARY_SUSCEPTIBILITY_READ_ONLY_STATIC_AUDIT
