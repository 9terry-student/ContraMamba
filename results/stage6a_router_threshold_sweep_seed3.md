# Stage 6A Router Threshold Sweep: Seed 3

## CLASSIFICATION

| threshold | system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 |
|---:|---|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 |
| 0.3 | conservative_strict_router | 0.908 | 0.871 | 0.937 | 1.000 | 0.676 |
| 0.3 | dual_auditor_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 |
| 0.4 | conservative_balanced_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 |
| 0.4 | conservative_strict_router | 0.908 | 0.871 | 0.937 | 1.000 | 0.676 |
| 0.4 | dual_auditor_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 |
| 0.5 | conservative_balanced_router | 0.923 | 0.885 | 0.948 | 1.000 | 0.706 |
| 0.5 | conservative_strict_router | 0.908 | 0.869 | 0.938 | 0.984 | 0.685 |
| 0.5 | dual_auditor_router | 0.919 | 0.879 | 0.946 | 0.984 | 0.706 |
| 0.6 | conservative_balanced_router | 0.915 | 0.870 | 0.944 | 1.000 | 0.667 |
| 0.6 | conservative_strict_router | 0.900 | 0.846 | 0.934 | 0.968 | 0.636 |
| 0.6 | dual_auditor_router | 0.904 | 0.850 | 0.936 | 0.968 | 0.646 |
| 0.7 | conservative_balanced_router | 0.912 | 0.864 | 0.941 | 0.984 | 0.667 |
| 0.7 | conservative_strict_router | 0.900 | 0.842 | 0.934 | 0.968 | 0.625 |
| 0.7 | dual_auditor_router | 0.904 | 0.846 | 0.937 | 0.968 | 0.635 |

## INTERNAL_FAITHFULNESS

| threshold | system | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.3 | conservative_strict_router | 78.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 |
| 0.3 | dual_auditor_router | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.4 | conservative_balanced_router | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.4 | conservative_strict_router | 78.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 |
| 0.4 | dual_auditor_router | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.5 | conservative_balanced_router | 72.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.5 | conservative_strict_router | 76.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 |
| 0.5 | dual_auditor_router | 71.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.6 | conservative_balanced_router | 70.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 |
| 0.6 | conservative_strict_router | 68.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 |
| 0.6 | dual_auditor_router | 67.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 |
| 0.7 | conservative_balanced_router | 69.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 |
| 0.7 | conservative_strict_router | 66.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 |
| 0.7 | dual_auditor_router | 65.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 |

## PAIRWISE_CONSISTENCY

| threshold | system | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.3 | conservative_strict_router | 0.850 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.3 | dual_auditor_router | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | conservative_balanced_router | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | conservative_strict_router | 0.850 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | dual_auditor_router | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | conservative_balanced_router | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | conservative_strict_router | 0.850 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | dual_auditor_router | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | conservative_balanced_router | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | conservative_strict_router | 0.950 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | dual_auditor_router | 0.950 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | conservative_balanced_router | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | conservative_strict_router | 1.000 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | dual_auditor_router | 1.000 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
