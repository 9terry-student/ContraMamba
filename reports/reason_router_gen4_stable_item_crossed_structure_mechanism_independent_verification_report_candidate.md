# Generation-4 Stable-Item Crossed-Structure Mechanism
# Independent Verification Report

STATUS = PASS_CANDIDATE

PHASE =
GEN4_STABLE_ITEM_CROSSED_STRUCTURE_MECHANISM_INDEPENDENT_VERIFICATION

DESIGN_AUTHORITY =
702d5304e2b8214d6b4b497c72f163d52e8c9862

EXECUTION_AUTHORITY =
905c7246fc03da072a1999e4db940262a36a6cab

PARENT_VALIDATED_EVIDENCE =
857a64cd7521d5ccb30d331a57b0b0f73aa7f39d

## 1. Verification scope

The independent verifier reconstructed the complete crossed-structure analysis
from the frozen Generation-4 primary population.

It independently reconstructed:

- leave-one-seed-out cell margins;
- common primary populations;
- BASE predictors;
- OPERATOR predictors;
- ADDITIVE predictors;
- CELL_RESIDUAL predictors;
- all twelve ROC AUC endpoints;
- the exact frozen single-RNG permutation stream;
- twelve-endpoint Holm correction;
- the four component verdicts;
- prespecified secondary variance summaries.

No training, model forward, checkpoint loading, new inference, Kaggle
execution, or GPU execution occurred.

## 2. Frozen provenance and leakage controls

FROZEN_PRIMARY_POPULATION =
PASS

Frozen population SHA256:

eb5aaacf38be224461801cab5e0629e46a9d375d7efaeadb72b2dbf3c5d56f3e

ARTIFACT_INTEGRITY =
PASS

COMMON_POPULATION_RECONSTRUCTION =
PASS

CROSSED_STRUCTURE_ROW_RECONSTRUCTION =
PASS

TARGET_SEED_MARGIN_LEAKAGE =
NO

D1_OUTCOME_PREDICTOR_LEAKAGE =
NO

## 3. Common primary populations

Seed 180:

LOO eligible =
534

Common primary population =
534

Positive outcomes =
34

Negative outcomes =
500

Seed 181:

LOO eligible =
529

Common primary population =
523

Positive outcomes =
23

Negative outcomes =
500

Seed 182:

LOO eligible =
523

Common primary population =
523

Positive outcomes =
46

Negative outcomes =
477

COUNT_GATE_ALL_SEEDS =
PASS

## 4. Independently reconstructed primary endpoints

BASE_AUC_180 =
0.4518235294117647

BASE_AUC_181 =
0.45304347826086955

BASE_AUC_182 =
0.49822258681979764

OPERATOR_AUC_180 =
0.7448823529411764

OPERATOR_AUC_181 =
0.7569565217391304

OPERATOR_AUC_182 =
0.7217209005560113

ADDITIVE_AUC_180 =
0.7770588235294118

ADDITIVE_AUC_181 =
0.800695652173913

ADDITIVE_AUC_182 =
0.7653814602132896

RESIDUAL_AUC_180 =
0.9476470588235294

RESIDUAL_AUC_181 =
0.9669565217391304

RESIDUAL_AUC_182 =
0.9717436879044754

DIRECT_PAIRWISE_12_AUC_VERIFY =
PASS

## 5. Permutation and multiplicity verification

Frozen permutation RNG seed:

1882018564

Permutations per endpoint:

100000

The verifier reproduced the exact single RNG stream in the frozen endpoint
order.

For every OPERATOR, ADDITIVE, and RESIDUAL endpoint:

permutation exceed count =
0

raw permutation p =
0.00000999990000099999

For the BASE endpoints:

BASE_AUC_180 exceed count =
82498

BASE_AUC_180 raw p =
0.8249817501824982

BASE_AUC_181 exceed count =
77655

BASE_AUC_181 raw p =
0.7765522344776552

BASE_AUC_182 exceed count =
51922

BASE_AUC_182 raw p =
0.5192248077519225

PERMUTATION_STREAM_REPRODUCIBILITY =
PASS

HOLM_12_ENDPOINT_VERIFY =
PASS

For every OPERATOR, ADDITIVE, and RESIDUAL endpoint:

Holm-adjusted p =
0.00011999880001199988

BASE endpoint Holm-adjusted p values are:

1.0
1.0
1.0

## 6. Frozen component verdicts

BASE_COMPONENT =
NOT_SUPPORTED

OPERATOR_COMPONENT =
SUPPORTED

ADDITIVE_COMPONENT =
SUPPORTED

CELL_RESIDUAL_COMPONENT =
SUPPORTED

FROZEN_COMPONENT_VERDICT_RULE =
PASS

## 7. Secondary descriptive verification

Seed 180:

variance m =
0.04644488204536851

variance additive =
0.02386831749855984

variance residual =
0.022319069088623328

Seed 181:

variance m =
0.03498905630588017

variance additive =
0.018576212523823382

variance residual =
0.015722656175999203

Seed 182:

variance m =
0.03362150137458006

variance additive =
0.016696427589854873

variance residual =
0.01793222806310339

SECONDARY_DESCRIPTIVE_RECOMPUTATION =
PASS

## 8. Scientific interpretation

The validated pattern rejects a shared base-item component as a sufficient
predictive structural explanation.

BASE_COMPONENT =
NOT_SUPPORTED

In contrast, perturbation-operator structure carries reproducible held-out
susceptibility information:

OPERATOR_COMPONENT =
SUPPORTED

The prespecified additive base-plus-operator construction also predicts
susceptibility:

ADDITIVE_COMPONENT =
SUPPORTED

Most importantly, stable cell-specific deviation remaining after removal of
the additive crossed structure retains strong predictive information:

CELL_RESIDUAL_COMPONENT =
SUPPORTED

The residual AUCs are:

0.9476470588235294
0.9669565217391304
0.9717436879044754

The permitted structural interpretation is therefore:

stable vulnerability contains an operator-associated component and substantial
base-by-operator cell-specific structure beyond the prespecified additive
decomposition.

This result does not authorize selecting the numerically strongest component
as the unique mechanism.

## 9. Claim boundary

The evidence supports:

OPERATOR_ASSOCIATED_STRUCTURE =
SUPPORTED

ADDITIVE_CROSSED_STRUCTURE =
SUPPORTED

CELL_SPECIFIC_STRUCTURE_BEYOND_ADDITIVE =
SUPPORTED

It does not establish:

- that operator identity causes D1 susceptibility;
- that the additive factorization is a neural mechanism;
- that cell residual corresponds to one circuit, edge, gradient path, or
  parameter group;
- that manipulating any component changes D1 outcome;
- that CELL_RESIDUAL is the unique causal mechanism.

Therefore:

CAUSAL_MECHANISM_ESTABLISHED =
NO

SCIENTIFIC_CLAIM_BOUNDARY =
STRUCTURAL_PREDICTIVE_NOT_CAUSAL

## 10. Research consequence

The shared BASE-only hypothesis is not authorized for rescue.

The next mechanism-design stage should focus on independently motivated
structure capable of explaining:

1. why perturbation operators contribute reproducible susceptibility;
2. why substantial stable base-by-operator cell-specific deviation remains
   after additive decomposition.

The following are prohibited as rescue strategies:

- choosing only a favorable operator;
- selecting only high-residual cells after outcome access;
- returning to FROZEN51-only analysis;
- returning to prior Gen3 residual IDs;
- dropping seed182;
- changing component signs;
- changing context thresholds;
- redefining stable_id grouping;
- declaring the largest AUC component uniquely causal.

## 11. Verification verdict

FROZEN_PRIMARY_POPULATION =
PASS

ARTIFACT_INTEGRITY =
PASS

COMMON_POPULATION_RECONSTRUCTION =
PASS

CROSSED_STRUCTURE_ROW_RECONSTRUCTION =
PASS

TARGET_SEED_MARGIN_LEAKAGE =
NO

D1_OUTCOME_PREDICTOR_LEAKAGE =
NO

DIRECT_PAIRWISE_12_AUC_VERIFY =
PASS

PERMUTATION_STREAM_REPRODUCIBILITY =
PASS

HOLM_12_ENDPOINT_VERIFY =
PASS

FROZEN_COMPONENT_VERDICT_RULE =
PASS

SECONDARY_DESCRIPTIVE_RECOMPUTATION =
PASS

BASE_COMPONENT =
NOT_SUPPORTED

OPERATOR_COMPONENT =
SUPPORTED

ADDITIVE_COMPONENT =
SUPPORTED

CELL_RESIDUAL_COMPONENT =
SUPPORTED

SCIENTIFIC_INTERPRETATION =
OPERATOR_AND_CELL_SPECIFIC_STRUCTURE_SUPPORTED

CAUSAL_MECHANISM_ESTABLISHED =
NO

SCIENTIFIC_CLAIM_BOUNDARY =
STRUCTURAL_PREDICTIVE_NOT_CAUSAL

INDEPENDENT_VERIFY =
PASS

END_OF_GEN4_STABLE_ITEM_CROSSED_STRUCTURE_MECHANISM_INDEPENDENT_VERIFICATION
