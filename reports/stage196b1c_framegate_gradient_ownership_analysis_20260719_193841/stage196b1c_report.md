# Stage196-B1-C FrameGate gradient-ownership analysis

## Executive decision

`STAGE196B1C_MIXED_GRADIENT_OWNERSHIP_EFFECT`

No new causal promotion is authorized; the gradient-ownership effect is mixed.

## Source and provenance closure

All six runs close to training runtime commit `9835cbbf86d83aca0964821669e63f7f6deb1c59`, frozen Mamba `state-spaces/mamba-130m-hf`, `v6b_minimal`, CUDA, 20 epochs, split seed 174, frozen encoder/A_log, no shared-encoder gradient path, and no external, bridge, margin, SWA, calibration, threshold search, or state-capsule activity. Analysis commit: `9c7bb278a90c603a9fab95e5fe969f79b1bd1929`. These are independent provenance roles; equality is not required.

Resolved alignment key: `id`, cross-validated against trajectory `source_row_id`; `dev_position` is used only as a certified stable position. Resolved schema is recorded in the JSON report.

## Exact six-run matrix

1. `seed183_joint`
2. `seed183_frame_local_only`
3. `seed184_joint`
4. `seed184_frame_local_only`
5. `seed185_joint`
6. `seed185_frame_local_only`

## Selected-checkpoint metrics

| Run | Best epoch | Accuracy | Macro-F1 | SUPPORT recall | False NE | False entitlement | Polarity errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| `seed183_joint` | 20 | 0.901389 | 0.848658 | 0.629213 | 33 | 38 | 0 |
| `seed183_frame_local_only` | 20 | 0.898611 | 0.783811 | 0.303371 | 61 | 10 | 2 |
| `seed184_joint` | 18 | 0.883333 | 0.829354 | 0.617978 | 34 | 50 | 0 |
| `seed184_frame_local_only` | 13 | 0.883333 | 0.783221 | 0.359551 | 56 | 27 | 1 |
| `seed185_joint` | 20 | 0.898611 | 0.849795 | 0.662921 | 30 | 43 | 0 |
| `seed185_frame_local_only` | 20 | 0.831944 | 0.777466 | 0.561798 | 39 | 82 | 0 |

Selected-best outputs above remain distinct from the fixed tail-three analysis; best-epoch movement is not decision evidence by itself.

## Tail-three persistent SUPPORT failures

Persistent means gold SUPPORT and NOT_ENTITLED at each of epochs 18, 19, and 20.

| Run | Persistent SUPPORT→NE | FrameGate failures | Stable SUPPORT | Unstable SUPPORT |
|---|---:|---:|---:|---:|
| `seed183_joint` | 26 | 26 | 52 | 11 |
| `seed183_frame_local_only` | 31 | 31 | 26 | 32 |
| `seed184_joint` | 34 | 34 | 11 | 44 |
| `seed184_frame_local_only` | 50 | 50 | 18 | 21 |
| `seed185_joint` | 30 | 30 | 17 | 42 |
| `seed185_frame_local_only` | 39 | 34 | 35 | 15 |

## FrameGate localization

Frozen Stage196-A native channel threshold 0.5 is reused for frame, predicate, sufficiency, and entitlement aggregation. Polarity pass is the exported SUPPORT-facing margin sign (>= 0); no searched threshold is used. Row-level buckets are in `stage196b1c_tail3_persistent_rows.csv`.

## Stage196-A recurrent-position effects

Separate loaded sets: baseline recurrent=22, intervention recurrent=19, common recurrent=19, universal all-six=10. Selected and tail-three frame effects and rescue/harm flags are in `stage196b1c_recurrent_position_effects.csv`.

## Stable-correct SUPPORT preservation

| Seed | Baseline stable-correct | Preserved | Changed to NE | Changed to REFUTE | Unstable | Preservation rate |
|---:|---:|---:|---:|---:|---:|---:|
| 183 | 52 | 23 | 11 | 0 | 18 | 0.442308 |
| 184 | 11 | 5 | 2 | 0 | 4 | 0.454545 |
| 185 | 17 | 14 | 0 | 0 | 3 | 0.823529 |

## False-entitlement and polarity safety

Selected-checkpoint paired safety deltas are shown below. Negative error-count delta is improvement.

## Twenty-epoch trajectory comparison

`stage196b1c_epoch_trajectory.csv` contains exactly 120 rows (3 seeds × 2 arms × 20 epochs), selected-epoch markers, frame trajectories, and tail-membership precursors. No trainer-selected epoch substitutes for epochs 18–20.

## Paired-seed direction table

| Metric | Seed 183 | Seed 184 | Seed 185 | Mean | Median | + / 0 / − |
|---|---:|---:|---:|---:|---:|---:|
| selected_accuracy | -0.00277778 | 0 | -0.0666667 | -0.0231481 | -0.00277778 | 0 / 1 / 2 |
| selected_macro_f1 | -0.0648479 | -0.0461335 | -0.0723289 | -0.0611035 | -0.0648479 | 0 / 0 / 3 |
| selected_support_recall | -0.325843 | -0.258427 | -0.101124 | -0.228464 | -0.258427 | 0 / 0 / 3 |
| selected_false_not_entitled_count | 28 | 22 | 9 | 19.6667 | 22 | 3 / 0 / 0 |
| selected_false_entitlement_count | -28 | -23 | 39 | -4 | -23 | 1 / 0 / 2 |
| selected_polarity_error_count | 2 | 1 | 0 | 1 | 1 | 2 / 1 / 0 |
| persistent_stable_support_negative_count | 5 | 16 | 9 | 10 | 9 | 3 / 0 / 0 |
| framegate_failure_count_among_persistent_support_negatives | 5 | 16 | 4 | 8.33333 | 5 | 3 / 0 / 0 |
| stable_correct_support_preservation | -29 | -6 | -3 | -12.6667 | -6 | 0 / 0 / 3 |
| mean_frame_probability_stage196a_common_recurrent | -0.136243 | 0.140103 | 0.147282 | 0.0503805 | 0.140103 | 2 / 0 / 1 |
| mean_frame_probability_stage196a_universal_all_six | -0.100975 | 0.151992 | 0.171927 | 0.0743148 | 0.151992 | 2 / 0 / 1 |

## Decision-rule evaluation

Supports requirements: `{"common_recurrent_frame_probability_higher_at_least_two_not_lower_any": false, "false_entitlement_not_increased_in_at_least_two": true, "framegate_failures_lower_at_least_two_not_higher_any": false, "persistent_lower_at_least_two_not_higher_any": false, "polarity_error_not_increased_in_at_least_two": false, "stable_correct_preservation_not_reduced_in_at_least_two": false}`

Mixed conditions: `{"framegate_failure_reduction_with_multiseed_safety_degradation": false, "material_seed_direction_conflict": true, "one_seed_dominates_primary_signature": false, "persistent_rescue_without_coherent_recurrent_frame_probability": false, "selected_checkpoint_and_tail_three_disagree": false}`

## Authorized causal claim

No new causal promotion is authorized; the gradient-ownership effect is mixed.

## Prohibited claims

- unfrozen encoder behavior
- external/OOD improvement
- production readiness
- contrastive-loss necessity
- architecture superiority
- complete mechanistic explanation

## Remaining uncertainty

The smallest unresolved causal question is whether seed-specific persistent rescue and common-position FrameGate probability move together without entitlement or polarity harm.

## Recommended next stage

`STAGE196B2_NO_PROMOTION_TARGETED_CAUSAL_FOLLOWUP`

This recommendation does not automatically authorize retraining, a new loss, an architecture change, or external evaluation.
