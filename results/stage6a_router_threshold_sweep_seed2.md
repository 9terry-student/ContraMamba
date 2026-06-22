# Stage 6A Router Threshold Sweep: Seed 2

## CLASSIFICATION

| threshold | system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 |
|---:|---|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 0.892 | 0.869 | 0.926 | 1.000 | 0.682 |
| 0.3 | conservative_strict_router | 0.888 | 0.863 | 0.923 | 1.000 | 0.667 |
| 0.3 | dual_auditor_router | 0.888 | 0.861 | 0.923 | 1.000 | 0.659 |
| 0.4 | conservative_balanced_router | 0.896 | 0.873 | 0.928 | 1.000 | 0.690 |
| 0.4 | conservative_strict_router | 0.885 | 0.855 | 0.921 | 1.000 | 0.643 |
| 0.4 | dual_auditor_router | 0.885 | 0.855 | 0.921 | 1.000 | 0.643 |
| 0.5 | conservative_balanced_router | 0.896 | 0.868 | 0.929 | 1.000 | 0.675 |
| 0.5 | conservative_strict_router | 0.885 | 0.852 | 0.921 | 1.000 | 0.634 |
| 0.5 | dual_auditor_router | 0.892 | 0.859 | 0.927 | 1.000 | 0.650 |
| 0.6 | conservative_balanced_router | 0.896 | 0.868 | 0.929 | 1.000 | 0.675 |
| 0.6 | conservative_strict_router | 0.881 | 0.842 | 0.919 | 0.982 | 0.625 |
| 0.6 | dual_auditor_router | 0.885 | 0.846 | 0.922 | 0.982 | 0.633 |
| 0.7 | conservative_balanced_router | 0.892 | 0.862 | 0.927 | 1.000 | 0.659 |
| 0.7 | conservative_strict_router | 0.877 | 0.818 | 0.919 | 0.923 | 0.611 |
| 0.7 | dual_auditor_router | 0.877 | 0.818 | 0.919 | 0.923 | 0.611 |

## INTERNAL_FAITHFULNESS

| threshold | system | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 84.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 |
| 0.3 | conservative_strict_router | 83.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.3 | dual_auditor_router | 81.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 |
| 0.4 | conservative_balanced_router | 83.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 |
| 0.4 | conservative_strict_router | 80.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 |
| 0.4 | dual_auditor_router | 80.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 |
| 0.5 | conservative_balanced_router | 79.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 |
| 0.5 | conservative_strict_router | 78.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 |
| 0.5 | dual_auditor_router | 76.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 |
| 0.6 | conservative_balanced_router | 79.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 |
| 0.6 | conservative_strict_router | 75.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 |
| 0.6 | dual_auditor_router | 74.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 |
| 0.7 | conservative_balanced_router | 78.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 |
| 0.7 | conservative_strict_router | 64.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.7 | dual_auditor_router | 64.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 |

## PAIRWISE_CONSISTENCY

| threshold | system | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.3 | conservative_strict_router | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 |
| 0.3 | dual_auditor_router | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | conservative_balanced_router | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | conservative_strict_router | 0.850 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | dual_auditor_router | 0.850 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | conservative_balanced_router | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | conservative_strict_router | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | dual_auditor_router | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | conservative_balanced_router | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | conservative_strict_router | 0.750 | 0.950 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | dual_auditor_router | 0.750 | 0.950 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | conservative_balanced_router | 0.750 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | conservative_strict_router | 0.650 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | dual_auditor_router | 0.650 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
