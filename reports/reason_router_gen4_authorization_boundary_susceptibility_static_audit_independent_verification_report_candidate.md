# Generation-4 Authorization Boundary Susceptibility
# Independent Static-Audit Verification Report

STATUS = PASS_CANDIDATE

PHASE =
GEN4_BOUNDARY_SUSCEPTIBILITY_INDEPENDENT_STATIC_VERIFICATION

DESIGN_AUTHORITY =
1fec5e47bcae970b5ec6a23e3724087fd3f10029

EXECUTION_MANIFEST_AUTHORITY =
96ccb359c579515293c24f7544d128b0591343d3

PRIMARY_STATIC_AUDIT_VERDICT =
SUPPORTED

## 1. Verification scope

The verification independently re-read the exact frozen historical A0/D1
prediction sources, reconstructed the prespecified primary population, and
recomputed the prespecified Generation-4 statistics.

No scientific artifact was modified during verification.

No training, model forward pass, checkpoint loading, Kaggle execution, GPU
execution, new seed, new Gen3 arm, margin tuning, or hypothesis modification
was performed.

## 2. Byte and provenance verification

BYTE_IDENTITY =
PASS

SOURCE_PROVENANCE =
PASS

ARTIFACT_INTEGRITY =
PASS

PRIMARY_POPULATION_RECONSTRUCTION =
PASS

Exact evidence artifact SHA256 values:

- primary_population.jsonl:
  eb5aaacf38be224461801cab5e0629e46a9d375d7efaeadb72b2dbf3c5d56f3e

- analysis.json:
  e511dd4cab87b4aad5d4fa22a683ca70c18882ba3228919852df6343c323830b

- report_candidate.md:
  bca352ccc1b234f9d697b61bdd815afb16a70045f75a28236da9bffbbb68b9ec

- artifact_sha256_manifest.json:
  1cc6f6188e35f131aeaf9c87c2c52a01969e5383f1d04e437f805c70b68cd4ed

## 3. Independent primary-population reconstruction

| Seed | N | D1 SUPPORT flips | Non-flips |
|---:|---:|---:|---:|
| 180 | 537 | 37 | 500 |
| 181 | 529 | 23 | 506 |
| 182 | 524 | 47 | 477 |

All three prespecified minimum-count gates pass.

## 4. Independent AUC recomputation

The verification used the direct pairwise definition:

P(score_positive > score_negative)
+ 0.5 * P(score_positive = score_negative)

where the prespecified susceptibility score is:

-d_A0

and:

d_A0 =
z_A0(NOT_ENTITLED) - z_A0(SUPPORT)

Results:

| Seed | AUC |
|---:|---:|
| 180 | 0.996378378378378 |
| 181 | 0.977659391648049 |
| 182 | 0.882376555600161 |

DIRECT_PAIRWISE_AUC_VERIFY =
PASS

All three AUC values exceed both:

- chance direction: 0.5
- frozen minimum effect threshold: 0.60

## 5. Independent permutation verification

Frozen permutation count:

100000

Frozen RNG seed:

535584327

For all three seeds:

PERMUTATION_EXCEED_COUNT =
0

Raw one-sided Monte-Carlo p-value for each seed:

0.00000999990000099999

PERMUTATION_REPRODUCIBILITY =
PASS

## 6. Multiplicity verification

Prespecified method:

HOLM_FWER_ALPHA_0.05

Independent Holm recomputation:

PASS

Holm-adjusted p-value for every seed:

0.00002999970000299997

HOLM_VERIFY =
PASS

## 7. Secondary fixed-bin verification

The prespecified deterministic five-bin descriptive analysis was independently
recomputed from the reconstructed primary population.

FIVE_BIN_VERIFY =
PASS

The secondary analysis did not alter the primary verdict.

## 8. Frozen verdict-rule verification

The prespecified support rule requires, for all three seeds:

1. AUC > 0.5;
2. AUC >= 0.60;
3. Holm-adjusted p < 0.05;
4. at least 10 positive D1 SUPPORT flips;
5. at least 30 negative rows;
6. exact provenance and joins.

Every requirement passes.

FROZEN_VERDICT_RULE =
PASS

GEN4_BOUNDARY_SUSCEPTIBILITY =
SUPPORTED

## 9. Scientific interpretation boundary

The validated result supports the bounded claim:

Baseline A0 SUPPORT / NOT_ENTITLED decision-boundary proximity is a strong,
reproducible predictor of D1 supportward susceptibility across frozen training
seeds 180, 181, and 182.

The result does NOT establish that baseline boundary proximity causes D1
failure.

It also does not establish:

- one edge or macro-group as the unique cause;
- U+D as a universal causal mechanism;
- a unique parameter-ownership mechanism;
- a unique gradient-path mechanism;
- one homogeneous mechanism for every residual error.

Accordingly:

SCIENTIFIC_CLAIM_BOUNDARY =
PREDICTIVE_NOT_CAUSAL

## 10. Verification verdict

CODE_CORRECTNESS =
PASS

SOURCE_PROVENANCE =
PASS

ARTIFACT_INTEGRITY =
PASS

PRIMARY_POPULATION_RECONSTRUCTION =
PASS

STATISTICAL_RECOMPUTATION =
PASS

INDEPENDENT_VERIFY =
PASS

GEN4_BOUNDARY_SUSCEPTIBILITY =
SUPPORTED

END_OF_GEN4_BOUNDARY_SUSCEPTIBILITY_INDEPENDENT_STATIC_VERIFICATION
