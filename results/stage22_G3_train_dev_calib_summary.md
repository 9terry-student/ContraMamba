# Stage22 G3 train_dev controlled-only calibration

| seed | shift | penalty | acc | macro-F1 | surface FNE | temporal-erased FNE | frame-loc FE | frame-role FE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 2.0000 | 0.7037 | 0.3921 | 0.8500 | 0.6800 | 0.1500 | 0.2000 |
| 2 | 0.0000 | 0.5000 | 0.7648 | 0.4714 | 0.6200 | 0.5300 | 0.3000 | 0.3000 |
| 3 | 0.0000 | 2.0000 | 0.6611 | 0.3219 | 0.9700 | 0.8300 | 0.0000 | 0.1500 |

## Aggregate

| metric | mean | sd | values |
|---|---:|---:|---|
| selected_shift | 0.0000 | 0.0000 | [0.0, 0.0, 0.0] |
| selected_frame_penalty | 1.5000 | 0.8660 | [2.0, 0.5, 2.0] |
| objective_score | 0.9400 | 0.0173 | [0.95, 0.9199999999999999, 0.95] |
| pres_accept_rate | None | None | [] |
| frame_false_entitled_rate | None | None | [] |
| calibration_pres_record_count | 20.0000 | 0.0000 | [20, 20, 20] |
| calibration_frame_record_count | 50.0000 | 0.0000 | [50, 50, 50] |
| calibration_unflagged_count | 120.0000 | 0.0000 | [120, 120, 120] |
| ood_overall_acc | 0.7099 | 0.0521 | [0.7037037014961243, 0.7648147940635681, 0.6611111164093018] |
| ood_macro_f1 | 0.3951 | 0.0748 | [0.3921247132244063, 0.4713581690008731, 0.32194553601484655] |
| ood_surface_fne | 0.8133 | 0.1779 | [0.85, 0.62, 0.97] |
| ood_temporal_erased_fne | 0.6800 | 0.1500 | [0.68, 0.53, 0.83] |
| ood_frame_loc_fe | 0.1500 | 0.1500 | [0.15, 0.3, 0.0] |
| ood_frame_role_fe | 0.2167 | 0.0764 | [0.2, 0.3, 0.15] |
| ood_predicate_mismatch_acc | 1.0000 | 0.0000 | [1.0, 1.0, 1.0] |
| ood_temporal_mismatch_acc | 1.0000 | 0.0000 | [1.0, 1.0, 1.0] |
| ood_sufficiency_control_acc | 1.0000 | 0.0000 | [1.0, 1.0, 1.0] |