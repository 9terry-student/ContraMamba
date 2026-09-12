# Generation-4 Leave-One-Seed-Out Stable-Item Boundary Susceptibility
# Independent Verification Report

STATUS = PASS_CANDIDATE

PHASE =
GEN4_LEAVE_ONE_SEED_OUT_STABLE_ITEM_BOUNDARY_SUSCEPTIBILITY_INDEPENDENT_VERIFICATION

DESIGN_AUTHORITY =
cfe948ca20d74cbdd59baeb0a73c0a8c14fc0f74

EXECUTION_AUTHORITY =
183b4a92be08e4a94ca19b3d43ccfdf1e29b57f1

PARENT_CROSS_SECTIONAL_EVIDENCE =
5388460bf0ac2ed1f8489e35c6942f6044da3df0

PARENT_WITHIN_ITEM_FALSIFICATION_EVIDENCE =
c81eba59cb11a0ed845371ea4ef37dabc05ec9ad

## 1. Verification scope

The independent verifier reconstructed the prespecified leave-one-seed-out
predictors directly from the frozen Generation-4 primary population.

No training, model forward, checkpoint loading, new inference, Kaggle
execution, GPU execution, new seed, or Gen3 combinatorial execution occurred.

The target seed's own A0 boundary margin was excluded from every predictor.

Other-seed D1 outcomes were excluded from every predictor.

## 2. Frozen input and artifact integrity

FROZEN_PRIMARY_POPULATION =
PASS

Frozen primary population SHA256:

eb5aaacf38be224461801cab5e0629e46a9d375d7efaeadb72b2dbf3c5d56f3e

ARTIFACT_INTEGRITY =
PASS

LOO_HELDOUT_ROW_RECONSTRUCTION =
PASS

TARGET_SEED_MARGIN_LEAKAGE =
NO

OTHER_SEED_D1_OUTCOME_LEAKAGE =
NO

## 3. Independently reconstructed primary results

### Seed 180

Eligible held-out rows:

534

Positive D1 SUPPORT outcomes:

34

Negative outcomes:

500

LOO_AUC_180:

0.9594117647058824

Permutation exceed count:

0

Raw permutation p:

0.00000999990000099999

Holm-adjusted p:

0.00002999970000299997

### Seed 181

Eligible held-out rows:

529

Positive D1 SUPPORT outcomes:

23

Negative outcomes:

506

LOO_AUC_181:

0.9730194191441829

Permutation exceed count:

0

Raw permutation p:

0.00000999990000099999

Holm-adjusted p:

0.00002999970000299997

### Seed 182

Eligible held-out rows:

523

Positive D1 SUPPORT outcomes:

46

Negative outcomes:

477

LOO_AUC_182:

0.9864643150123051

Permutation exceed count:

0

Raw permutation p:

0.00000999990000099999

Holm-adjusted p:

0.00002999970000299997

DIRECT_PAIRWISE_LOO_AUC_VERIFY =
PASS

PERMUTATION_REPRODUCIBILITY =
PASS

HOLM_VERIFY =
PASS

## 4. Predictor stability

180_vs_181:

- common stable IDs = 529
- Pearson = 0.930822110782140
- Spearman = 0.970762904214236

180_vs_182:

- common stable IDs = 523
- Pearson = 0.761430620139402
- Spearman = 0.888484121899175

181_vs_182:

- common stable IDs = 518
- Pearson = 0.725205639366636
- Spearman = 0.853169788725071

PREDICTOR_STABILITY_RECOMPUTATION =
PASS

## 5. Frozen support rule

Every target seed satisfies:

- minimum positive count >= 10;
- minimum negative count >= 30;
- LOO AUC > 0.5;
- LOO AUC >= 0.70;
- Holm-adjusted p < 0.05.

The frozen population provenance passes.

Target-seed margin leakage is absent.

Other-seed D1 outcome leakage is absent.

Therefore:

FROZEN_VERDICT_RULE =
PASS

GEN4_STABLE_ITEM_BOUNDARY_COMPONENT =
SUPPORTED

## 6. Joint scientific interpretation

The earlier Generation-4 cross-sectional audit established that same-seed A0
authorization-boundary proximity strongly predicts D1 supportward
susceptibility.

The subsequent within-item falsification showed that seed-to-seed variation
of the same item's A0 boundary margin does not track seed-to-seed D1 outcome:

WITHIN_ITEM_CONCORDANCE =
0.4594594594594595

and:

GEN4_WITHIN_ITEM_BOUNDARY_SUSCEPTIBILITY =
NOT_SUPPORTED

The present leave-one-seed-out audit now establishes a different result:

an item's boundary position measured only in OTHER training seeds strongly
predicts its D1 susceptibility in a held-out target seed.

The three held-out AUCs are:

0.9594117647058824
0.9730194191441829
0.9864643150123051

This pattern supports a stable item-associated vulnerability interpretation.

The evidence is most consistent with the A0 authorization-boundary margin
acting as a strong observable marker of stable item-associated structure,
rather than seed-specific boundary fluctuation itself being the causal
mechanism.

## 7. Claim boundary

The validated evidence supports:

STABLE_ITEM_ASSOCIATED_VULNERABILITY =
SUPPORTED

It does not establish:

- that A0 boundary position causes D1 failure;
- that changing the boundary would change the D1 outcome;
- the latent item property responsible for the stable geometry;
- a unique edge, group, ownership path, or gradient mechanism;
- a universal mechanism shared by every susceptible example.

Therefore:

BOUNDARY_CAUSALITY_ESTABLISHED =
NO

SCIENTIFIC_CLAIM_BOUNDARY =
STABLE_ITEM_ASSOCIATED_PREDICTIVE_NOT_CAUSAL

## 8. Research consequence

Direct boundary causal intervention is not justified by this result.

The next justified research phase is:

GEN4_STABLE_ITEM_VULNERABILITY_MECHANISM_DESIGN

That phase should seek independently motivated item-level structural
properties capable of explaining both:

1. the strong cross-seed stability of A0 boundary geometry; and
2. held-out D1 susceptibility.

It must not rescue the result through:

- margin redefinition;
- confidence substitution;
- seed removal;
- favorable subgroup selection;
- FROZEN51-only restriction;
- residual-only restriction;
- renewed Gen3 combinatorial search.

## 9. Verification verdict

FROZEN_PRIMARY_POPULATION =
PASS

ARTIFACT_INTEGRITY =
PASS

LOO_HELDOUT_ROW_RECONSTRUCTION =
PASS

TARGET_SEED_MARGIN_LEAKAGE =
NO

OTHER_SEED_D1_OUTCOME_LEAKAGE =
NO

DIRECT_PAIRWISE_LOO_AUC_VERIFY =
PASS

PERMUTATION_REPRODUCIBILITY =
PASS

HOLM_VERIFY =
PASS

PREDICTOR_STABILITY_RECOMPUTATION =
PASS

GEN4_STABLE_ITEM_BOUNDARY_COMPONENT =
SUPPORTED

SCIENTIFIC_INTERPRETATION =
STABLE_ITEM_ASSOCIATED_VULNERABILITY_SUPPORTED

BOUNDARY_CAUSALITY_ESTABLISHED =
NO

INDEPENDENT_VERIFY =
PASS

END_OF_GEN4_LOO_STABLE_ITEM_BOUNDARY_SUSCEPTIBILITY_INDEPENDENT_VERIFICATION
