# Stage20-C Predicate Bias Calibration Summary

Stage20-C is prediction-level predicate bias calibration, not an end-to-end trained model.

## Per-seed summary

| seed | selected_alpha | valid_guarded_alpha_found | calibration_pred_base_false_entitled | calibration_pred_adjusted_false_entitled | heldout_pred_base_false_entitled | heldout_pred_adjusted_false_entitled | heldout_controls_preserved | heldout_detector_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.250 | False | 29.000 | 3.000 | 25.000 | 1.000 | True | 0.980 |
| 2 | 1.500 | True | 44.000 | 3.000 | 37.000 | 1.000 | True | 0.980 |
| 3 | 0.750 | True | 24.000 | 2.000 | 21.000 | 1.000 | True | 0.980 |

## Group metrics (selected alpha)

| seed | split | stage15_probe_type | n | base_accuracy | adjusted_accuracy | base_false_entitled_count | adjusted_false_entitled_count | changed_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | calibration | frame_location_mismatch | 13 | 0.769 | 0.769 | 3 | 3 | 0 |
| 1 | calibration | frame_role_mismatch | 8 | 1.000 | 1.000 | 0 | 0 | 0 |
| 1 | calibration | predicate_mismatch | 51 | 0.431 | 0.941 | 29 | 3 | 26 |
| 1 | calibration | sufficiency_control | 56 | 1.000 | 1.000 | 0 | 0 | 0 |
| 1 | calibration | surface_control | 42 | 0.976 | 0.976 | 0 | 0 | 0 |
| 1 | calibration | temporal_erased | 50 | 1.000 | 1.000 | 0 | 0 | 0 |
| 1 | calibration | temporal_mismatch | 50 | 1.000 | 1.000 | 0 | 0 | 0 |
| 1 | heldout | frame_location_mismatch | 7 | 1.000 | 1.000 | 0 | 0 | 0 |
| 1 | heldout | frame_role_mismatch | 12 | 0.917 | 0.917 | 1 | 1 | 0 |
| 1 | heldout | predicate_mismatch | 49 | 0.490 | 0.980 | 25 | 1 | 24 |
| 1 | heldout | sufficiency_control | 44 | 1.000 | 1.000 | 0 | 0 | 0 |
| 1 | heldout | surface_control | 58 | 1.000 | 1.000 | 0 | 0 | 0 |
| 1 | heldout | temporal_erased | 50 | 1.000 | 1.000 | 0 | 0 | 0 |
| 1 | heldout | temporal_mismatch | 50 | 1.000 | 1.000 | 0 | 0 | 0 |
| 2 | calibration | frame_location_mismatch | 13 | 0.615 | 0.615 | 5 | 5 | 0 |
| 2 | calibration | frame_role_mismatch | 8 | 1.000 | 1.000 | 0 | 0 | 0 |
| 2 | calibration | predicate_mismatch | 51 | 0.137 | 0.941 | 44 | 3 | 41 |
| 2 | calibration | sufficiency_control | 56 | 1.000 | 1.000 | 0 | 0 | 0 |
| 2 | calibration | surface_control | 42 | 0.976 | 0.976 | 0 | 0 | 0 |
| 2 | calibration | temporal_erased | 50 | 1.000 | 1.000 | 0 | 0 | 0 |
| 2 | calibration | temporal_mismatch | 50 | 1.000 | 1.000 | 0 | 0 | 0 |
| 2 | heldout | frame_location_mismatch | 7 | 0.857 | 0.857 | 1 | 1 | 0 |
| 2 | heldout | frame_role_mismatch | 12 | 1.000 | 1.000 | 0 | 0 | 0 |
| 2 | heldout | predicate_mismatch | 49 | 0.245 | 0.980 | 37 | 1 | 36 |
| 2 | heldout | sufficiency_control | 44 | 1.000 | 1.000 | 0 | 0 | 0 |
| 2 | heldout | surface_control | 58 | 1.000 | 1.000 | 0 | 0 | 0 |
| 2 | heldout | temporal_erased | 50 | 1.000 | 1.000 | 0 | 0 | 0 |
| 2 | heldout | temporal_mismatch | 50 | 1.000 | 1.000 | 0 | 0 | 0 |
| 3 | calibration | frame_location_mismatch | 13 | 0.154 | 0.154 | 11 | 11 | 0 |
| 3 | calibration | frame_role_mismatch | 8 | 0.750 | 0.750 | 2 | 2 | 0 |
| 3 | calibration | predicate_mismatch | 51 | 0.529 | 0.961 | 24 | 2 | 22 |
| 3 | calibration | sufficiency_control | 56 | 1.000 | 1.000 | 0 | 0 | 0 |
| 3 | calibration | surface_control | 42 | 0.905 | 0.905 | 0 | 0 | 0 |
| 3 | calibration | temporal_erased | 50 | 1.000 | 1.000 | 0 | 0 | 0 |
| 3 | calibration | temporal_mismatch | 50 | 0.980 | 0.980 | 1 | 1 | 0 |
| 3 | heldout | frame_location_mismatch | 7 | 0.714 | 0.714 | 2 | 2 | 0 |
| 3 | heldout | frame_role_mismatch | 12 | 0.833 | 0.833 | 2 | 2 | 0 |
| 3 | heldout | predicate_mismatch | 49 | 0.571 | 0.980 | 21 | 1 | 20 |
| 3 | heldout | sufficiency_control | 44 | 1.000 | 1.000 | 0 | 0 | 0 |
| 3 | heldout | surface_control | 58 | 0.966 | 0.966 | 0 | 0 | 0 |
| 3 | heldout | temporal_erased | 50 | 0.980 | 0.980 | 0 | 0 | 0 |
| 3 | heldout | temporal_mismatch | 50 | 0.960 | 0.960 | 2 | 2 | 0 |

## Interpretation

- seed=1: selected_alpha=1.25, valid=False
-   WARNING: no valid guarded alpha found for seed=1; best-accuracy alpha used.
-   Detector recall bottleneck: 98.0% of heldout predicate_mismatch flagged; residual false-entitled reflects unflagged examples.
-   Heldout predicate false-entitled: 25.0 -> 1.0 (96.0% reduction).
- seed=2: selected_alpha=1.5, valid=True
-   Detector recall bottleneck: 98.0% of heldout predicate_mismatch flagged; residual false-entitled reflects unflagged examples.
-   Heldout predicate false-entitled: 37.0 -> 1.0 (97.3% reduction).
- seed=3: selected_alpha=0.75, valid=True
-   Detector recall bottleneck: 98.0% of heldout predicate_mismatch flagged; residual false-entitled reflects unflagged examples.
-   Heldout predicate false-entitled: 21.0 -> 1.0 (95.2% reduction).
