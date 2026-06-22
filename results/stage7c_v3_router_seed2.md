# Stage 6A Router Threshold Sweep: Seed 2

## CLASSIFICATION

| threshold | system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 |
|---:|---|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.3 | conservative_strict_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.3 | dual_auditor_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.4 | conservative_balanced_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.4 | conservative_strict_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.4 | dual_auditor_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.5 | conservative_balanced_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.5 | conservative_strict_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.5 | dual_auditor_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.6 | conservative_balanced_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.6 | conservative_strict_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.6 | dual_auditor_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.7 | conservative_balanced_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.7 | conservative_strict_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |
| 0.7 | dual_auditor_router | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 |

## INTERNAL_FAITHFULNESS

| threshold | system | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.3 | conservative_strict_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.3 | dual_auditor_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.4 | conservative_balanced_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.4 | conservative_strict_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.4 | dual_auditor_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.5 | conservative_balanced_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.5 | conservative_strict_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.5 | dual_auditor_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.6 | conservative_balanced_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.6 | conservative_strict_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.6 | dual_auditor_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.7 | conservative_balanced_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.7 | conservative_strict_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.7 | dual_auditor_router | 233.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |

## PAIRWISE_CONSISTENCY

| threshold | system | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.3 | conservative_strict_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.3 | dual_auditor_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | conservative_balanced_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | conservative_strict_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | dual_auditor_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | conservative_balanced_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | conservative_strict_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | dual_auditor_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | conservative_balanced_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | conservative_strict_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | dual_auditor_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | conservative_balanced_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | conservative_strict_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | dual_auditor_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
