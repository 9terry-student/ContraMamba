# Stage 6A Router Threshold Sweep: Seed 3

## CLASSIFICATION

| threshold | system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 |
|---:|---|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 0.928 | 0.904 | 0.951 | 1.000 | 0.761 |
| 0.3 | conservative_strict_router | 0.927 | 0.902 | 0.950 | 0.994 | 0.761 |
| 0.3 | dual_auditor_router | 0.928 | 0.904 | 0.951 | 1.000 | 0.761 |
| 0.4 | conservative_balanced_router | 0.928 | 0.904 | 0.951 | 1.000 | 0.761 |
| 0.4 | conservative_strict_router | 0.927 | 0.902 | 0.950 | 0.994 | 0.761 |
| 0.4 | dual_auditor_router | 0.928 | 0.904 | 0.951 | 1.000 | 0.761 |
| 0.5 | conservative_balanced_router | 0.931 | 0.907 | 0.953 | 1.000 | 0.767 |
| 0.5 | conservative_strict_router | 0.927 | 0.902 | 0.950 | 0.994 | 0.761 |
| 0.5 | dual_auditor_router | 0.931 | 0.907 | 0.953 | 1.000 | 0.767 |
| 0.6 | conservative_balanced_router | 0.929 | 0.905 | 0.952 | 1.000 | 0.762 |
| 0.6 | conservative_strict_router | 0.927 | 0.902 | 0.950 | 0.994 | 0.761 |
| 0.6 | dual_auditor_router | 0.929 | 0.905 | 0.952 | 1.000 | 0.762 |
| 0.7 | conservative_balanced_router | 0.929 | 0.905 | 0.952 | 1.000 | 0.762 |
| 0.7 | conservative_strict_router | 0.931 | 0.907 | 0.953 | 1.000 | 0.767 |
| 0.7 | dual_auditor_router | 0.929 | 0.905 | 0.952 | 1.000 | 0.762 |

## INTERNAL_FAITHFULNESS

| threshold | system | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 234.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.3 | conservative_strict_router | 235.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.3 | dual_auditor_router | 234.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.4 | conservative_balanced_router | 234.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.4 | conservative_strict_router | 235.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.4 | dual_auditor_router | 234.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.5 | conservative_balanced_router | 232.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.5 | conservative_strict_router | 235.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.5 | dual_auditor_router | 232.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.6 | conservative_balanced_router | 231.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.6 | conservative_strict_router | 235.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.6 | dual_auditor_router | 231.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.7 | conservative_balanced_router | 231.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |
| 0.7 | conservative_strict_router | 232.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0.7 | dual_auditor_router | 231.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 |

## PAIRWISE_CONSISTENCY

| threshold | system | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 0.983 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.3 | conservative_strict_router | 0.983 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.3 | dual_auditor_router | 0.983 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.4 | conservative_balanced_router | 0.983 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.4 | conservative_strict_router | 0.983 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.4 | dual_auditor_router | 0.983 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.5 | conservative_balanced_router | 0.983 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.5 | conservative_strict_router | 0.983 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.5 | dual_auditor_router | 0.983 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.6 | conservative_balanced_router | 1.000 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.6 | conservative_strict_router | 0.983 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.6 | dual_auditor_router | 1.000 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.7 | conservative_balanced_router | 1.000 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.7 | conservative_strict_router | 0.983 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| 0.7 | dual_auditor_router | 1.000 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |

## ROUTER_COST_OF_CONSERVATISM

| threshold | system | classifier_entitled_count | routed_entitled_count | downgraded_count | downgrade_rate_among_classifier_entitled | downgraded_gold_support_count | support_recall_pre_router | support_recall_post_router | support_recall_drop | support_precision_pre_router | support_precision_post_router | support_precision_gain | false_support_removed_count | false_refute_removed_count | retained_violation_rate | pre_router_candidate_gate_fail_rate | pre_router_candidate_gate_fail_count |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | conservative_balanced_router | 235.000 | 234.000 | 1.000 | 0.004 | 0.000 | 0.989 | 0.989 | 0.000 | 0.618 | 0.618 | 0.000 | 0.000 | 1.000 | 0.000 | 0.004 | 1.000 |
| 0.3 | conservative_strict_router | 235.000 | 235.000 | 0.000 | 0.000 | 0.000 | 0.989 | 0.989 | 0.000 | 0.618 | 0.618 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.3 | dual_auditor_router | 235.000 | 234.000 | 1.000 | 0.004 | 0.000 | 0.989 | 0.989 | 0.000 | 0.618 | 0.618 | 0.000 | 0.000 | 1.000 | 0.000 | 0.004 | 1.000 |
| 0.4 | conservative_balanced_router | 235.000 | 234.000 | 1.000 | 0.004 | 0.000 | 0.989 | 0.989 | 0.000 | 0.618 | 0.618 | 0.000 | 0.000 | 1.000 | 0.000 | 0.004 | 1.000 |
| 0.4 | conservative_strict_router | 235.000 | 235.000 | 0.000 | 0.000 | 0.000 | 0.989 | 0.989 | 0.000 | 0.618 | 0.618 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.4 | dual_auditor_router | 235.000 | 234.000 | 1.000 | 0.004 | 0.000 | 0.989 | 0.989 | 0.000 | 0.618 | 0.618 | 0.000 | 0.000 | 1.000 | 0.000 | 0.004 | 1.000 |
| 0.5 | conservative_balanced_router | 235.000 | 232.000 | 3.000 | 0.013 | 0.000 | 0.989 | 0.989 | 0.000 | 0.618 | 0.627 | 0.009 | 2.000 | 1.000 | 0.000 | 0.013 | 3.000 |
| 0.5 | conservative_strict_router | 235.000 | 235.000 | 0.000 | 0.000 | 0.000 | 0.989 | 0.989 | 0.000 | 0.618 | 0.618 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.5 | dual_auditor_router | 235.000 | 232.000 | 3.000 | 0.013 | 0.000 | 0.989 | 0.989 | 0.000 | 0.618 | 0.627 | 0.009 | 2.000 | 1.000 | 0.000 | 0.013 | 3.000 |
| 0.6 | conservative_balanced_router | 235.000 | 231.000 | 4.000 | 0.017 | 1.000 | 0.989 | 0.978 | 0.011 | 0.618 | 0.624 | 0.006 | 2.000 | 1.000 | 0.000 | 0.017 | 4.000 |
| 0.6 | conservative_strict_router | 235.000 | 235.000 | 0.000 | 0.000 | 0.000 | 0.989 | 0.989 | 0.000 | 0.618 | 0.618 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.6 | dual_auditor_router | 235.000 | 231.000 | 4.000 | 0.017 | 1.000 | 0.989 | 0.978 | 0.011 | 0.618 | 0.624 | 0.006 | 2.000 | 1.000 | 0.000 | 0.017 | 4.000 |
| 0.7 | conservative_balanced_router | 235.000 | 231.000 | 4.000 | 0.017 | 1.000 | 0.989 | 0.978 | 0.011 | 0.618 | 0.624 | 0.006 | 2.000 | 1.000 | 0.000 | 0.017 | 4.000 |
| 0.7 | conservative_strict_router | 235.000 | 232.000 | 3.000 | 0.013 | 0.000 | 0.989 | 0.989 | 0.000 | 0.618 | 0.627 | 0.009 | 2.000 | 1.000 | 0.000 | 0.013 | 3.000 |
| 0.7 | dual_auditor_router | 235.000 | 231.000 | 4.000 | 0.017 | 1.000 | 0.989 | 0.978 | 0.011 | 0.618 | 0.624 | 0.006 | 2.000 | 1.000 | 0.000 | 0.017 | 4.000 |
