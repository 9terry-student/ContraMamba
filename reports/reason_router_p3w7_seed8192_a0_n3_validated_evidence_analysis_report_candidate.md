# Seed8192 Revised-Split A0 N=3 Validated-Evidence Analysis Report Candidate

## Verdict and boundary

This is a **report-only validated-evidence analysis candidate** for the completed Seed8192 revised-split A0 primary N=3.

Primary membership:

1. seed180 replacement R1
2. seed181
3. seed182

All three members completed authorized execution, authentic-wrapper verification, fixed-path v3 handoff validation, and standard local import/audit before this analysis.

This report:

- does not authorize training or evaluation;
- does not authorize calibration execution;
- does not authorize A1/A2/A3;
- does not modify any scientific artifact;
- does not treat historical seed180 r2 as a primary N=3 member;
- does not infer causal benefit from A0 alone.

The evidence supports the following bounded conclusion:

> The Seed8192 revised-split A0 baseline and its dominant failure structure are reproducible across seeds 180/181/182. The dominant stable defect is upstream FRAME overblocking of valid SUPPORT and systematic FRAME capture of PREDICATE-target rows. This is sufficient to justify authoring a later revised reason-loss calibration authority, but not to claim that reason-specific supervision will improve the system.

Candidate verdict:

```text
A0_N3_EXECUTION_PROVENANCE_VALIDITY = PASS
A0_N3_CROSS_SEED_REPRODUCIBILITY = PASS
STABLE_FAILURE_STRUCTURE_IDENTIFIED = PASS
READY_FOR_REVISED_REASON_LOSS_CALIBRATION_AUTHORITY_AUTHORING = YES
CALIBRATION_EXECUTION_AUTHORIZED = NO
A1_A2_A3_EXECUTION_AUTHORIZED = NO
SCIENTIFIC_CAUSAL_CLAIM = NONE
```

## Governing execution authority

Frozen execution authority commit:

```text
55debe94f0d19d16a334395e8561901fed6b52fa
```

Execution-authority report:

```text
reports/reason_router_p3w7_seed8192_a0_n3_clean_replacement_execution_authority_spec_candidate.md
```

The primary N=3 source directories are exactly:

```text
reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0
reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed181/A0
reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed182/A0
```

Historical seed180 r2 is excluded from every aggregate and row-level comparison in this report.

## Analysis evidence set

This analysis was recomputed from the following imported JSON evidence only:

| Member | Artifact | Bytes | SHA256 |
| --- | --- | ---: | --- |
| seed180 replacement R1 | `clean_dev_predictions.json` | 4,840,320 | `5c9722ac0f75c411b3d744a29beec0c35d2f2809331f257a5e1d5ea81e6cf75d` |
| seed180 replacement R1 | `run_provenance.json` | 69,444 | `a758538e93e6e52ca261cb593285c298344808a3626eed7d9b9664e29a6c1a3d` |
| seed180 replacement R1 | `training_report.json` | 306,325 | `2cdf0925e3a0ef1b925f6b00ac4b2095d18a113896a437ded77622f5134b2013` |
| seed181 | `clean_dev_predictions.json` | 4,838,584 | `789d02f9092ce6b051d0ca435272c9e93a3962183dbb0a4d4dbb20cebf2ac3fe` |
| seed181 | `run_provenance.json` | 69,133 | `82f6a511f9b8228c91d3419cc8872f7371c0785a01aa730904d43a8f04f6a98b` |
| seed181 | `training_report.json` | 306,101 | `0068aec52a9afb4bd8e79d711ac666a5257186ab142c8c72da890fe64c2c45e8` |
| seed182 | `clean_dev_predictions.json` | 4,841,685 | `029ec6ae31df2f5ca9526d1e631496f7ee272967a6f5e08684f29aa09ad490d4` |
| seed182 | `run_provenance.json` | 69,143 | `934acab332773b4127ffa5c68b09a8bded168ab96ac5b18c29018e3e78b77c66` |
| seed182 | `training_report.json` | 306,108 | `ef60c457a28be8ed91e57ef8d9be4d89150b0e93fe90a045eb12a6ea78262a22` |

The imported `training_report_predictions.jsonl` and `selected_checkpoint.pt` artifacts were not needed for the numerical recomputations in this report. Their execution/import validity remains governed by the completed per-run verification and import audits.

## Provenance and contract reconciliation

All three primary members bind to the same frozen implementation and split contract.

| Field | seed180 | seed181 | seed182 |
| --- | --- | --- | --- |
| Git commit | `55debe94f0d19d16a334395e8561901fed6b52fa` | same | same |
| Trainer SHA256 | `9792f95df934b8b78cffe07bb7613a35984dba79d56ee7ea719d533dd7117d87` | same | same |
| Training seed | 180 | 181 | 182 |
| Split seed | 8192 | 8192 | 8192 |
| Train rows | 2880 | 2880 | 2880 |
| Dev rows | 720 | 720 | 720 |
| Architecture | `v6b_minimal` | same | same |
| Backbone | Mamba | same | same |
| Encoder | frozen | frozen | frozen |
| Reason-router arm | A0 | A0 | A0 |
| Router mode | `explicit_product` | same | same |
| Gradient ownership | `joint` | same | same |
| Reason-loss weight | 0.0 | 0.0 | 0.0 |
| A0 reference required | false | false | false |
| A0 reference joined rows | 0 | 0 | 0 |
| Time-swap use | false | false | false |
| Clean-dev-only checkpoint selection | true | true | true |

The three `clean_dev_predictions.json` files contain the same 720 `stable_id` values with the same gold labels and intervention ordering, enabling exact row-level cross-seed comparison.

## Per-seed primary metrics

| Metric | seed180 replacement R1 | seed181 | seed182 |
| --- | ---: | ---: | ---: |
| Selected epoch | 20 | 17 | 20 |
| Final macro-F1 | 0.8048257041 | 0.7927344803 | 0.8133489983 |
| Final accuracy | 0.9097222686 | 0.8986111283 | 0.9013888836 |
| Frame accuracy | 0.8319444656 | 0.8250000477 | 0.8388888836 |
| Predicate accuracy | 0.7500000000 | 0.7458333373 | 0.7847222686 |
| Sufficiency accuracy | 1.0000000000 | 1.0000000000 | 1.0000000000 |
| Polarity accuracy, entitled | 1.0000000000 | 1.0000000000 | 1.0000000000 |
| NOT_ENTITLED F1 | 0.9429323968 | 0.9354553492 | 0.9365504915 |
| REFUTE F1 | 1.0000000000 | 1.0000000000 | 1.0000000000 |
| SUPPORT F1 | 0.4715447154 | 0.4427480916 | 0.5034965035 |

Prediction distributions:

| Seed | NOT_ENTITLED | REFUTE | SUPPORT |
| --- | ---: | ---: | ---: |
| Gold | 540 | 89 | 91 |
| seed180 | 599 | 89 | 32 |
| seed181 | 591 | 89 | 40 |
| seed182 | 579 | 89 | 52 |

## N=3 descriptive aggregates

These are descriptive statistics only. N=3 is not used for population-level significance claims.

| Metric | Mean | Population SD | Sample SD | Min | Max | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Final macro-F1 | 0.8036363942 | 0.0084577551 | 0.0103585922 | 0.7927344803 | 0.8133489983 | 0.0206145181 |
| Final accuracy | 0.9032407602 | 0.0047213306 | 0.0057824255 | 0.8986111283 | 0.9097222686 | 0.0111111403 |
| NOT_ENTITLED F1 | 0.9383127459 | 0.0032970406 | 0.0040380336 | 0.9354553492 | 0.9429323968 | 0.0074770476 |
| REFUTE F1 | 1.0000000000 | 0.0000000000 | 0.0000000000 | 1.0000000000 | 1.0000000000 | 0.0000000000 |
| SUPPORT F1 | 0.4725964368 | 0.0248115830 | 0.0303878590 | 0.4427480916 | 0.5034965035 | 0.0607484119 |

Aggregate performance is therefore comparatively stable across these three seeds. The dominant scientific issue is the stable location of the errors rather than large seed-to-seed variance in aggregate score.

## Exact confusion structure

Row-level recomputation gives:

### seed180 replacement R1

```text
gold SUPPORT:
  SUPPORT       29
  NOT_ENTITLED  62

gold NOT_ENTITLED:
  NOT_ENTITLED  537
  SUPPORT       3

gold REFUTE:
  REFUTE        89
```

### seed181

```text
gold SUPPORT:
  SUPPORT       29
  NOT_ENTITLED  62

gold NOT_ENTITLED:
  NOT_ENTITLED  529
  SUPPORT       11

gold REFUTE:
  REFUTE        89
```

### seed182

```text
gold SUPPORT:
  SUPPORT       36
  NOT_ENTITLED  55

gold NOT_ENTITLED:
  NOT_ENTITLED  524
  SUPPORT       16

gold REFUTE:
  REFUTE        89
```

Every final classification error in all three runs lies on the SUPPORT versus NOT_ENTITLED boundary. REFUTE is 89/89 correct in every seed and has zero cross-class confusion.

SUPPORT recall is:

```text
seed180 = 29 / 91 = 0.3186813187
seed181 = 29 / 91 = 0.3186813187
seed182 = 36 / 91 = 0.3956043956
mean              = 0.3443223443
```

The corresponding increase in SUPPORT recovery in seed182 is accompanied by more NOT_ENTITLED to SUPPORT false positives. This is evidence of a precision/recall tradeoff, not evidence that a simple global output shift solves the mechanism defect.

## Cross-seed row-level stability

Across the same 720 dev rows:

```text
identical prediction in all 3 seeds = 691 / 720
seed disagreement                  = 29 / 720

correct in all 3 seeds             = 635
wrong in all 3 seeds               = 56
correct in exactly 1 seed          = 12
correct in exactly 2 seeds         = 17
```

Of the 56 rows wrong in all three seeds:

```text
gold SUPPORT       = 54
gold NOT_ENTITLED  = 2
gold REFUTE        = 0
```

Therefore 54/91 gold SUPPORT rows, approximately 59.34%, are wrong in all three seeds. The dominant SUPPORT failure is not primarily seed stochasticity.

## Reason-ownership structure

The reason-supervision-eligible dev population contains 360 rows in every seed with the same target distribution:

```text
FRAME        186
PREDICATE     31
SUFFICIENCY   62
AUTHORIZED    81
```

Using the largest explicit-product `q_*` mass as the dominant A0 reason:

| Target reason | seed180 dominant target match | seed181 | seed182 |
| --- | ---: | ---: | ---: |
| FRAME | 185 / 186 | 184 / 186 | 185 / 186 |
| PREDICATE | 0 / 31 | 0 / 31 | 0 / 31 |
| SUFFICIENCY | 62 / 62 | 62 / 62 | 62 / 62 |
| AUTHORIZED | 42 / 81 | 39 / 81 | 49 / 81 |

For PREDICATE-target rows, mean q masses are:

| Seed | q_FRAME | q_PREDICATE | q_SUFFICIENCY | q_AUTHORIZED |
| --- | ---: | ---: | ---: | ---: |
| seed180 | 0.801379 | 0.130701 | 0.000439 | 0.067481 |
| seed181 | 0.752007 | 0.164090 | 0.000720 | 0.083183 |
| seed182 | 0.687774 | 0.209212 | 0.000848 | 0.102166 |

All 31 PREDICATE-target rows are captured by FRAME as the dominant q mass in all three seeds. Predicate-specific ownership therefore fails reproducibly even when the final class prediction can still be correct.

## SUPPORT under-entitlement is FRAME overblocking

For gold SUPPORT rows incorrectly predicted as NOT_ENTITLED:

```text
seed180: 62 / 62 dominant blocker = FRAME
seed181: 62 / 62 dominant blocker = FRAME
seed182: 55 / 55 dominant blocker = FRAME
```

Mean frame probability on these failed SUPPORT rows:

```text
seed180 = 0.171771
seed181 = 0.201882
seed182 = 0.225312
```

Mean frame probability on correctly recovered SUPPORT rows:

```text
seed180 = 0.735836
seed181 = 0.753980
seed182 = 0.792547
```

Mean sufficiency probability on failed SUPPORT rows remains approximately 0.99 in all seeds, while entitled polarity accuracy is exactly 1.0 in all seeds.

The primary SUPPORT failure is therefore not a downstream sufficiency or polarity failure. It is upstream FRAME overblocking that removes otherwise valid SUPPORT entitlement.

## Intervention-level stability

Final accuracy by intervention:

| Intervention | seed180 | seed181 | seed182 |
| --- | ---: | ---: | ---: |
| entity_swap | 1.0000 | 0.9833 | 1.0000 |
| event_swap | 1.0000 | 1.0000 | 0.9833 |
| evidence_deletion | 1.0000 | 1.0000 | 1.0000 |
| evidence_truncation | 1.0000 | 1.0000 | 1.0000 |
| irrelevant_evidence | 1.0000 | 1.0000 | 1.0000 |
| location_swap | 1.0000 | 0.9667 | 0.8833 |
| none | 0.4833 | 0.5000 | 0.5167 |
| paraphrase | 0.9667 | 0.9500 | 1.0000 |
| polarity_flip | 0.5167 | 0.5167 | 0.5667 |
| predicate_swap | 1.0000 | 0.9833 | 0.9167 |
| role_swap | 0.9667 | 0.9333 | 0.9500 |
| title_name_swap | 0.9833 | 0.9500 | 1.0000 |

Stable high-performing families include evidence deletion, evidence truncation, irrelevant evidence, and paraphrase.

The most persistent low-performing families are `none` and `polarity_flip`, specifically because valid SUPPORT rows are frequently rejected as NOT_ENTITLED.

For gold SUPPORT within `none`:

```text
seed180: 0 / 31 correct
seed181: 1 / 31 correct
seed182: 2 / 31 correct
```

For gold SUPPORT within `polarity_flip`:

```text
seed180: 0 / 29 correct
seed181: 0 / 29 correct
seed182: 3 / 29 correct
```

This is consistent with the FRAME overblocking interpretation.

## Pairwise diagnostic synthesis

Checks passing in all three seeds:

```text
deletion_sufficiency_lower:
  1.0000 / 1.0000 / 1.0000

truncation_sufficiency_lower:
  1.0000 / 1.0000 / 1.0000
```

Checks failing in all three seeds:

```text
entity_frame_lower:
  0.7833 / 0.8000 / 0.9500

event_frame_lower:
  0.9333 / 0.9000 / 0.9000

paraphrase_preserved:
  0.4667 / 0.4667 / 0.4833

polarity_flip_preserved_and_reversed:
  0.0000 / 0.0000 / 0.0000

predicate_disentangled:
  0.0333 / 0.0833 / 0.2000
```

The sufficiency mechanism is robust across all seeds. The stable failures concern frame preservation and predicate disentanglement.

High paraphrase final accuracy does not contradict failed `paraphrase_preserved`: the pair-level preservation diagnostic can fail because the corresponding `none` SUPPORT row is rejected while the paraphrase row is accepted.

Likewise, strong final accuracy on `predicate_swap` does not establish predicate ownership. Many predicate mismatches are classified correctly as NOT_ENTITLED while FRAME, rather than PREDICATE, dominates the explicit-product failure mass.

## Audit warning disposition

`aux_to_ce_loss_ratio_weighted` is reproducibly greater than 2:

```text
seed180 = 2.0676
seed181 = 2.1152
seed182 = 2.1153
```

This is a stable training characteristic and should remain visible in later calibration analysis. It is not an execution/provenance invalidator under the frozen A0 authority and is not promoted to one by this report.

## Scientific interpretation

### Established by validated evidence

The validated N=3 supports all of the following:

1. The A0 baseline aggregate behavior is reproducible across seeds 180/181/182.
2. REFUTE behavior is saturated and stable on the clean dev split.
3. Sufficiency classification is saturated and stable.
4. Entitled polarity classification is saturated and stable.
5. SUPPORT under-entitlement is the dominant final-label weakness.
6. Valid SUPPORT is systematically converted to NOT_ENTITLED by FRAME overblocking.
7. PREDICATE-target rows are systematically captured by FRAME rather than PREDICATE under the A0 explicit-product decomposition.
8. The dominant failure structure persists at row level across all three seeds and is not adequately explained by seed stochasticity.

### Not established

This report does **not** establish:

1. that reason-specific supervision will correct the defect;
2. that nonzero reason-loss weight improves any metric;
3. that A1, A2, or A3 is superior to A0;
4. that a causal mechanism has been proven;
5. that any calibration value is acceptable;
6. that a later factorial execution is authorized;
7. population-level statistical significance from N=3.

## Required acceptance structure for a later calibration authority

A later revised reason-loss calibration authority should not accept a candidate merely because macro-F1 rises.

At minimum, acceptance should jointly track:

1. SUPPORT recall and F1 recovery;
2. reduction of AUTHORIZED-to-FRAME leakage;
3. recovery of PREDICATE reason ownership;
4. NOT_ENTITLED-to-SUPPORT false-positive control;
5. preservation of currently stable sufficiency performance;
6. preservation of saturated REFUTE behavior;
7. preservation of entitled polarity behavior;
8. cross-seed consistency rather than a single best seed;
9. provenance-equivalent data, split, trainer, and checkpoint selection unless a separately authorized change says otherwise.

This guards against accepting a simple operating-point shift that increases SUPPORT recall by converting excessive NOT_ENTITLED rows into false SUPPORT.

## Next authority boundary

The completed validated-evidence analysis supports **authoring** a revised reason-loss calibration authority.

It does not itself authorize the calibration execution.

The next report family may define bounded candidate reason-loss settings and explicit acceptance/rejection criteria, but any actual training/evaluation requires a separately frozen and activated execution authority following the normal provenance workflow.

No A1/A2/A3 factorial execution should begin before the revised reason-loss calibration phase is authorized, executed, validated, and interpreted.

## Candidate status

This file is a report-only candidate and must undergo independent verification before freeze/commit/push.

Required independent verification should confirm:

- exact primary N=3 membership;
- evidence artifact SHA256 identities;
- all provenance bindings;
- all reported per-seed metrics;
- all N=3 aggregate recomputations;
- confusion matrices;
- row-level agreement/error-overlap counts;
- dominant q-mass reason analysis;
- intervention-level counts;
- pairwise diagnostic statuses;
- warning disposition;
- non-claim boundaries;
- no repository mutation other than this single candidate file.

```text
PASS_READY_FOR_INDEPENDENT_SEED8192_A0_N3_VALIDATED_EVIDENCE_ANALYSIS_VERIFICATION
```
