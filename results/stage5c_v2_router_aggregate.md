# Stage 5C v2 Router Aggregate

## ROUTER_CLASSIFICATION_AGGREGATE

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 |
|---|---:|---:|---:|---:|---:|
| classifier_only | 0.897 ± 0.012 | 0.870 ± 0.006 | 0.929 ± 0.009 | 1.000 ± 0.000 | 0.680 ± 0.010 |
| balanced_only | 0.876 ± 0.026 | 0.850 ± 0.017 | 0.912 ± 0.020 | 0.995 ± 0.009 | 0.644 ± 0.028 |
| strict_only | 0.844 ± 0.014 | 0.820 ± 0.014 | 0.889 ± 0.011 | 1.000 ± 0.000 | 0.570 ± 0.034 |
| conservative_balanced_router | 0.912 ± 0.014 | 0.878 ± 0.009 | 0.940 ± 0.010 | 1.000 ± 0.000 | 0.694 ± 0.017 |
| conservative_strict_router | 0.900 ± 0.013 | 0.864 ± 0.011 | 0.932 ± 0.009 | 0.995 ± 0.009 | 0.665 ± 0.027 |
| dual_auditor_router | 0.908 ± 0.014 | 0.871 ± 0.010 | 0.938 ± 0.010 | 0.995 ± 0.009 | 0.680 ± 0.028 |

## ROUTER_PAIRWISE_AGGREGATE

| system | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|
| classifier_only | 0.917 ± 0.058 | 0.867 ± 0.076 | 0.933 ± 0.076 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.967 ± 0.029 | 0.983 ± 0.029 |
| balanced_only | 0.933 ± 0.029 | 0.917 ± 0.029 | 0.967 ± 0.029 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.950 ± 0.050 | 0.883 ± 0.104 |
| strict_only | 0.850 ± 0.000 | 0.833 ± 0.058 | 0.933 ± 0.058 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.967 ± 0.029 | 0.867 ± 0.029 |
| conservative_balanced_router | 0.833 ± 0.058 | 0.883 ± 0.058 | 0.883 ± 0.058 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| conservative_strict_router | 0.800 ± 0.050 | 0.867 ± 0.076 | 0.900 ± 0.050 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| dual_auditor_router | 0.817 ± 0.076 | 0.883 ± 0.058 | 0.883 ± 0.058 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## ROUTER_INTERNAL_FAITHFULNESS_AGGREGATE

| system | entitled_output_gate_violation_rate | entitled_output_count | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad |
|---|---:|---:|---:|---:|---:|---:|---:|
| classifier_only | 0.222 ± 0.066 | 83.333 ± 5.033 | 18.333 ± 4.509 | 0.933 ± 0.076 | 0.683 ± 0.208 | 0.250 ± 0.150 | 5.000 ± 3.000 |
| balanced_only | 0.126 ± 0.013 | 89.667 ± 7.767 | 11.333 ± 2.082 | 0.967 ± 0.029 | 0.900 ± 0.050 | 0.067 ± 0.029 | 1.333 ± 0.577 |
| strict_only | 0.125 ± 0.054 | 94.667 ± 3.512 | 12.000 ± 5.568 | 0.933 ± 0.058 | 0.917 ± 0.029 | 0.017 ± 0.029 | 0.333 ± 0.577 |
| conservative_balanced_router | 0.000 ± 0.000 | 75.000 ± 3.606 | 0.000 ± 0.000 | 0.883 ± 0.058 | 0.883 ± 0.058 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| conservative_strict_router | 0.000 ± 0.000 | 76.000 ± 2.000 | 0.000 ± 0.000 | 0.900 ± 0.050 | 0.900 ± 0.050 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| dual_auditor_router | 0.000 ± 0.000 | 73.333 ± 2.517 | 0.000 ± 0.000 | 0.883 ± 0.058 | 0.883 ± 0.058 | 0.000 ± 0.000 | 0.000 ± 0.000 |

## INTERPRETATION

- The conservative balanced-auditor router achieves the best final-label performance among evaluated systems.
- The classifier-only system remains a strong label predictor, but shows a larger output/internal faithfulness gap.
- Conservative routers eliminate entitled-output gate violations by construction while preserving competitive macro-F1.
- Balanced-only has strong output-level pairwise consistency, but lower final-label macro-F1 than the conservative balanced router.
