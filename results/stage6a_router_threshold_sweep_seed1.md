# Stage 6A Router Threshold Sweep: Seed 1

## CLASSIFICATION

| threshold | system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 |
|---:|---|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 0.912 | 0.878 | 0.940 | 1.000 | 0.693 |
| 0.3 | conservative_strict_router | 0.912 | 0.878 | 0.940 | 1.000 | 0.693 |
| 0.3 | dual_auditor_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 |
| 0.4 | conservative_balanced_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 |
| 0.4 | conservative_strict_router | 0.912 | 0.878 | 0.940 | 1.000 | 0.693 |
| 0.4 | dual_auditor_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 |
| 0.5 | conservative_balanced_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 |
| 0.5 | conservative_strict_router | 0.908 | 0.871 | 0.938 | 1.000 | 0.676 |
| 0.5 | dual_auditor_router | 0.912 | 0.875 | 0.941 | 1.000 | 0.685 |
| 0.6 | conservative_balanced_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 |
| 0.6 | conservative_strict_router | 0.908 | 0.868 | 0.938 | 1.000 | 0.667 |
| 0.6 | dual_auditor_router | 0.912 | 0.872 | 0.941 | 1.000 | 0.676 |
| 0.7 | conservative_balanced_router | 0.912 | 0.875 | 0.941 | 1.000 | 0.685 |
| 0.7 | conservative_strict_router | 0.904 | 0.852 | 0.937 | 0.983 | 0.636 |
| 0.7 | dual_auditor_router | 0.904 | 0.852 | 0.937 | 0.983 | 0.636 |

## INTERNAL_FAITHFULNESS

| threshold | system | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 75.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.3 | conservative_strict_router | 75.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.3 | dual_auditor_router | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.4 | conservative_balanced_router | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.4 | conservative_strict_router | 75.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.4 | dual_auditor_router | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.5 | conservative_balanced_router | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.5 | conservative_strict_router | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.5 | dual_auditor_router | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.6 | conservative_balanced_router | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.6 | conservative_strict_router | 72.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.6 | dual_auditor_router | 71.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.7 | conservative_balanced_router | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.7 | conservative_strict_router | 65.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 |
| 0.7 | dual_auditor_router | 65.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 |

## PAIRWISE_CONSISTENCY

| threshold | system | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.3 | conservative_strict_router | 0.800 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.3 | dual_auditor_router | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | conservative_balanced_router | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | conservative_strict_router | 0.800 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | dual_auditor_router | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | conservative_balanced_router | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | conservative_strict_router | 0.750 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | dual_auditor_router | 0.750 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | conservative_balanced_router | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | conservative_strict_router | 0.700 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | dual_auditor_router | 0.700 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | conservative_balanced_router | 0.850 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | conservative_strict_router | 0.850 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | dual_auditor_router | 0.850 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
