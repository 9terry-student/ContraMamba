# Stage22 G2 dev-calibrated selective NE shift

## Interpretation

Controlled-dev calibration selected zero shift across all three seeds. G2 therefore does not reproduce the OOD-tuned G1 improvement as a non-leaky deployable fix.

## Per-seed results

| seed | shift | acc | macro-F1 | surface FNE | temporal-erased FNE | frame-loc FE | frame-role FE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.7037 | 0.3921 | 0.8500 | 0.6800 | 0.1500 | 0.2000 |
| 2 | 0.0 | 0.7648 | 0.4714 | 0.6200 | 0.5300 | 0.3000 | 0.3000 |
| 3 | 0.0 | 0.6611 | 0.3219 | 0.9700 | 0.8300 | 0.0000 | 0.1500 |

## Aggregate

| metric | mean | sd | values |
|---|---:|---:|---|
| selected_shift | 0.0000 | 0.0000 | [0.0, 0.0, 0.0] |
| dev_objective_score | 0.7500 | 0.0000 | [0.75, 0.75, 0.75] |
| dev_pres_accept_rate | 0.7500 | 0.0000 | [0.75, 0.75, 0.75] |
| dev_frame_false_entitled_rate | 0.0000 | 0.0000 | [0.0, 0.0, 0.0] |
| ood_overall_acc | 0.7099 | 0.0521 | [0.7037037014961243, 0.7648147940635681, 0.6611111164093018] |
| ood_macro_f1 | 0.3951 | 0.0748 | [0.3921247132244063, 0.4713581690008731, 0.32194553601484655] |
| ood_surface_fne | 0.8133 | 0.1779 | [0.85, 0.62, 0.97] |
| ood_temporal_erased_fne | 0.6800 | 0.1500 | [0.68, 0.53, 0.83] |
| ood_frame_loc_fe | 0.1500 | 0.1500 | [0.15, 0.3, 0.0] |
| ood_frame_role_fe | 0.2167 | 0.0764 | [0.2, 0.3, 0.15] |
| ood_surface_acc | 0.1867 | 0.1779 | [0.15000000596046448, 0.3799999952316284, 0.029999999329447746] |
| ood_temporal_erased_acc | 0.3200 | 0.1500 | [0.3199999928474426, 0.4699999988079071, 0.17000000178813934] |
| ood_frame_loc_acc | 0.8500 | 0.1500 | [0.8500000238418579, 0.699999988079071, 1.0] |
| ood_frame_role_acc | 0.7833 | 0.0764 | [0.800000011920929, 0.699999988079071, 0.8500000238418579] |
| ood_predicate_mismatch_acc | 1.0000 | 0.0000 | [1.0, 1.0, 1.0] |
| ood_temporal_mismatch_acc | 1.0000 | 0.0000 | [1.0, 1.0, 1.0] |
| ood_sufficiency_control_acc | 1.0000 | 0.0000 | [1.0, 1.0, 1.0] |

## Claim status

- G1 OOD-tuned shift remains diagnostic / upper-bound.
- G2 controlled-dev calibration selected shift=0.0 across seeds.
- Therefore current non-leaky calibration does not recover the OOD-tuned improvement.