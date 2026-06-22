# Stage 6A Router Threshold Sweep: Seed 1

## CLASSIFICATION

| threshold | system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 |
|---:|---|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 0.926 | 0.901 | 0.949 | 1.000 | 0.752 |
| 0.3 | conservative_strict_router | 0.924 | 0.899 | 0.948 | 1.000 | 0.749 |
| 0.3 | dual_auditor_router | 0.926 | 0.901 | 0.949 | 1.000 | 0.752 |
| 0.4 | conservative_balanced_router | 0.926 | 0.901 | 0.949 | 1.000 | 0.752 |
| 0.4 | conservative_strict_router | 0.924 | 0.899 | 0.948 | 1.000 | 0.749 |
| 0.4 | dual_auditor_router | 0.926 | 0.901 | 0.949 | 1.000 | 0.752 |
| 0.5 | conservative_balanced_router | 0.926 | 0.901 | 0.949 | 1.000 | 0.752 |
| 0.5 | conservative_strict_router | 0.924 | 0.899 | 0.948 | 1.000 | 0.749 |
| 0.5 | dual_auditor_router | 0.926 | 0.901 | 0.949 | 1.000 | 0.752 |
| 0.6 | conservative_balanced_router | 0.926 | 0.901 | 0.949 | 1.000 | 0.752 |
| 0.6 | conservative_strict_router | 0.927 | 0.902 | 0.950 | 1.000 | 0.755 |
| 0.6 | dual_auditor_router | 0.928 | 0.903 | 0.951 | 1.000 | 0.759 |
| 0.7 | conservative_balanced_router | 0.928 | 0.903 | 0.951 | 1.000 | 0.759 |
| 0.7 | conservative_strict_router | 0.928 | 0.903 | 0.951 | 1.000 | 0.757 |
| 0.7 | dual_auditor_router | 0.929 | 0.904 | 0.952 | 1.000 | 0.760 |

## INTERNAL_FAITHFULNESS

| threshold | system | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 234.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.3 | conservative_strict_router | 235.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.3 | dual_auditor_router | 234.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.4 | conservative_balanced_router | 234.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.4 | conservative_strict_router | 235.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.4 | dual_auditor_router | 234.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.5 | conservative_balanced_router | 234.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.5 | conservative_strict_router | 235.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.5 | dual_auditor_router | 234.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.6 | conservative_balanced_router | 234.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.6 | conservative_strict_router | 233.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.6 | dual_auditor_router | 232.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.7 | conservative_balanced_router | 232.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.7 | conservative_strict_router | 230.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.7 | dual_auditor_router | 229.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |

## PAIRWISE_CONSISTENCY

| threshold | system | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.3 | conservative_strict_router | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.3 | dual_auditor_router | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.4 | conservative_balanced_router | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.4 | conservative_strict_router | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.4 | dual_auditor_router | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.5 | conservative_balanced_router | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.5 | conservative_strict_router | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.5 | dual_auditor_router | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.6 | conservative_balanced_router | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.6 | conservative_strict_router | 0.967 | 0.983 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 |
| 0.6 | dual_auditor_router | 0.967 | 0.983 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 |
| 0.7 | conservative_balanced_router | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 |
| 0.7 | conservative_strict_router | 0.950 | 0.983 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 |
| 0.7 | dual_auditor_router | 0.950 | 0.983 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 |
