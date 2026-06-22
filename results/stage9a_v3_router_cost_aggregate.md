# Stage 9A Router Cost of Conservatism

## THRESHOLD 0.3

| system | final_accuracy | final_macro_f1 | classifier_entitled_count | routed_entitled_count | downgraded_count | downgrade_rate_among_classifier_entitled | downgraded_gold_support_count | support_recall_pre_router | support_recall_post_router | support_recall_drop | support_precision_pre_router | support_precision_post_router | support_precision_gain | false_support_removed_count | false_refute_removed_count | retained_violation_rate | pre_router_candidate_gate_fail_rate | pre_router_candidate_gate_fail_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 234.333 +/- 1.155 | 233.667 +/- 0.577 | 0.667 +/- 0.577 | 0.003 +/- 0.002 | 0.000 +/- 0.000 | 0.989 +/- 0.011 | 0.989 +/- 0.011 | 0.000 +/- 0.000 | 0.619 +/- 0.013 | 0.620 +/- 0.011 | 0.001 +/- 0.002 | 0.333 +/- 0.577 | 0.333 +/- 0.577 | 0.000 +/- 0.000 | 0.003 +/- 0.002 | 0.667 +/- 0.577 |
| conservative_strict_router | 0.928 +/- 0.004 | 0.903 +/- 0.005 | 234.333 +/- 1.155 | 234.333 +/- 1.155 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.989 +/- 0.011 | 0.989 +/- 0.011 | 0.000 +/- 0.000 | 0.619 +/- 0.013 | 0.619 +/- 0.013 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| dual_auditor_router | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 234.333 +/- 1.155 | 233.667 +/- 0.577 | 0.667 +/- 0.577 | 0.003 +/- 0.002 | 0.000 +/- 0.000 | 0.989 +/- 0.011 | 0.989 +/- 0.011 | 0.000 +/- 0.000 | 0.619 +/- 0.013 | 0.620 +/- 0.011 | 0.001 +/- 0.002 | 0.333 +/- 0.577 | 0.333 +/- 0.577 | 0.000 +/- 0.000 | 0.003 +/- 0.002 | 0.667 +/- 0.577 |

## THRESHOLD 0.4

| system | final_accuracy | final_macro_f1 | classifier_entitled_count | routed_entitled_count | downgraded_count | downgrade_rate_among_classifier_entitled | downgraded_gold_support_count | support_recall_pre_router | support_recall_post_router | support_recall_drop | support_precision_pre_router | support_precision_post_router | support_precision_gain | false_support_removed_count | false_refute_removed_count | retained_violation_rate | pre_router_candidate_gate_fail_rate | pre_router_candidate_gate_fail_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 234.333 +/- 1.155 | 233.667 +/- 0.577 | 0.667 +/- 0.577 | 0.003 +/- 0.002 | 0.000 +/- 0.000 | 0.989 +/- 0.011 | 0.989 +/- 0.011 | 0.000 +/- 0.000 | 0.619 +/- 0.013 | 0.620 +/- 0.011 | 0.001 +/- 0.002 | 0.333 +/- 0.577 | 0.333 +/- 0.577 | 0.000 +/- 0.000 | 0.003 +/- 0.002 | 0.667 +/- 0.577 |
| conservative_strict_router | 0.928 +/- 0.004 | 0.903 +/- 0.005 | 234.333 +/- 1.155 | 234.333 +/- 1.155 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.989 +/- 0.011 | 0.989 +/- 0.011 | 0.000 +/- 0.000 | 0.619 +/- 0.013 | 0.619 +/- 0.013 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| dual_auditor_router | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 234.333 +/- 1.155 | 233.667 +/- 0.577 | 0.667 +/- 0.577 | 0.003 +/- 0.002 | 0.000 +/- 0.000 | 0.989 +/- 0.011 | 0.989 +/- 0.011 | 0.000 +/- 0.000 | 0.619 +/- 0.013 | 0.620 +/- 0.011 | 0.001 +/- 0.002 | 0.333 +/- 0.577 | 0.333 +/- 0.577 | 0.000 +/- 0.000 | 0.003 +/- 0.002 | 0.667 +/- 0.577 |

## THRESHOLD 0.5

| system | final_accuracy | final_macro_f1 | classifier_entitled_count | routed_entitled_count | downgraded_count | downgrade_rate_among_classifier_entitled | downgraded_gold_support_count | support_recall_pre_router | support_recall_post_router | support_recall_drop | support_precision_pre_router | support_precision_post_router | support_precision_gain | false_support_removed_count | false_refute_removed_count | retained_violation_rate | pre_router_candidate_gate_fail_rate | pre_router_candidate_gate_fail_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.929 +/- 0.003 | 0.906 +/- 0.005 | 234.333 +/- 1.155 | 233.000 +/- 1.000 | 1.333 +/- 1.528 | 0.006 +/- 0.007 | 0.000 +/- 0.000 | 0.989 +/- 0.011 | 0.989 +/- 0.011 | 0.000 +/- 0.000 | 0.619 +/- 0.013 | 0.623 +/- 0.011 | 0.004 +/- 0.004 | 1.000 +/- 1.000 | 0.333 +/- 0.577 | 0.000 +/- 0.000 | 0.006 +/- 0.007 | 1.333 +/- 1.528 |
| conservative_strict_router | 0.928 +/- 0.004 | 0.903 +/- 0.005 | 234.333 +/- 1.155 | 234.333 +/- 1.155 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.989 +/- 0.011 | 0.989 +/- 0.011 | 0.000 +/- 0.000 | 0.619 +/- 0.013 | 0.619 +/- 0.013 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| dual_auditor_router | 0.929 +/- 0.003 | 0.906 +/- 0.005 | 234.333 +/- 1.155 | 233.000 +/- 1.000 | 1.333 +/- 1.528 | 0.006 +/- 0.007 | 0.000 +/- 0.000 | 0.989 +/- 0.011 | 0.989 +/- 0.011 | 0.000 +/- 0.000 | 0.619 +/- 0.013 | 0.623 +/- 0.011 | 0.004 +/- 0.004 | 1.000 +/- 1.000 | 0.333 +/- 0.577 | 0.000 +/- 0.000 | 0.006 +/- 0.007 | 1.333 +/- 1.528 |

## THRESHOLD 0.6

| system | final_accuracy | final_macro_f1 | classifier_entitled_count | routed_entitled_count | downgraded_count | downgrade_rate_among_classifier_entitled | downgraded_gold_support_count | support_recall_pre_router | support_recall_post_router | support_recall_drop | support_precision_pre_router | support_precision_post_router | support_precision_gain | false_support_removed_count | false_refute_removed_count | retained_violation_rate | pre_router_candidate_gate_fail_rate | pre_router_candidate_gate_fail_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 234.333 +/- 1.155 | 232.667 +/- 1.528 | 1.667 +/- 2.082 | 0.007 +/- 0.009 | 0.333 +/- 0.577 | 0.989 +/- 0.011 | 0.985 +/- 0.013 | 0.004 +/- 0.006 | 0.619 +/- 0.013 | 0.622 +/- 0.011 | 0.003 +/- 0.003 | 1.000 +/- 1.000 | 0.333 +/- 0.577 | 0.000 +/- 0.000 | 0.007 +/- 0.009 | 1.667 +/- 2.082 |
| conservative_strict_router | 0.929 +/- 0.003 | 0.904 +/- 0.004 | 234.333 +/- 1.155 | 233.667 +/- 1.155 | 0.667 +/- 1.155 | 0.003 +/- 0.005 | 0.000 +/- 0.000 | 0.989 +/- 0.011 | 0.989 +/- 0.011 | 0.000 +/- 0.000 | 0.619 +/- 0.013 | 0.622 +/- 0.009 | 0.003 +/- 0.005 | 0.667 +/- 1.155 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.003 +/- 0.005 | 0.667 +/- 1.155 |
| dual_auditor_router | 0.930 +/- 0.002 | 0.906 +/- 0.003 | 234.333 +/- 1.155 | 232.000 +/- 1.000 | 2.333 +/- 2.082 | 0.010 +/- 0.009 | 0.333 +/- 0.577 | 0.989 +/- 0.011 | 0.985 +/- 0.013 | 0.004 +/- 0.006 | 0.619 +/- 0.013 | 0.625 +/- 0.006 | 0.006 +/- 0.006 | 1.667 +/- 1.528 | 0.333 +/- 0.577 | 0.000 +/- 0.000 | 0.010 +/- 0.009 | 2.333 +/- 2.082 |

## THRESHOLD 0.7

| system | final_accuracy | final_macro_f1 | classifier_entitled_count | routed_entitled_count | downgraded_count | downgrade_rate_among_classifier_entitled | downgraded_gold_support_count | support_recall_pre_router | support_recall_post_router | support_recall_drop | support_precision_pre_router | support_precision_post_router | support_precision_gain | false_support_removed_count | false_refute_removed_count | retained_violation_rate | pre_router_candidate_gate_fail_rate | pre_router_candidate_gate_fail_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.930 +/- 0.002 | 0.906 +/- 0.003 | 234.333 +/- 1.155 | 232.000 +/- 1.000 | 2.333 +/- 2.082 | 0.010 +/- 0.009 | 0.333 +/- 0.577 | 0.989 +/- 0.011 | 0.985 +/- 0.013 | 0.004 +/- 0.006 | 0.619 +/- 0.013 | 0.625 +/- 0.006 | 0.006 +/- 0.006 | 1.667 +/- 1.528 | 0.333 +/- 0.577 | 0.000 +/- 0.000 | 0.010 +/- 0.009 | 2.333 +/- 2.082 |
| conservative_strict_router | 0.930 +/- 0.002 | 0.906 +/- 0.003 | 234.333 +/- 1.155 | 231.667 +/- 1.528 | 2.667 +/- 2.517 | 0.011 +/- 0.011 | 0.333 +/- 0.577 | 0.989 +/- 0.011 | 0.985 +/- 0.017 | 0.004 +/- 0.006 | 0.619 +/- 0.013 | 0.627 +/- 0.005 | 0.008 +/- 0.007 | 2.000 +/- 2.000 | 0.333 +/- 0.577 | 0.000 +/- 0.000 | 0.011 +/- 0.011 | 2.667 +/- 2.517 |
| dual_auditor_router | 0.930 +/- 0.001 | 0.906 +/- 0.003 | 234.333 +/- 1.155 | 231.000 +/- 2.000 | 3.333 +/- 3.055 | 0.014 +/- 0.013 | 0.667 +/- 0.577 | 0.989 +/- 0.011 | 0.981 +/- 0.017 | 0.007 +/- 0.006 | 0.619 +/- 0.013 | 0.627 +/- 0.004 | 0.008 +/- 0.010 | 2.333 +/- 2.517 | 0.333 +/- 0.577 | 0.000 +/- 0.000 | 0.014 +/- 0.013 | 3.333 +/- 3.055 |

## INTERPRETATION

Post-routing retained-output gate violations are an enforced invariant for correctly implemented conservative routers. The empirical tradeoff is described by pre-router gate failures, downgrade rates, SUPPORT recall loss, precision gain, and removed false entitled predictions.
