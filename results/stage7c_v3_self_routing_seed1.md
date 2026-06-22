# Stage 6B Single-Model Self-Routing

## THRESHOLD 0.3

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.924 | 0.899 | 0.948 | 1.000 | 0.749 | 235.000 | 0.183 | 43.000 | 0.983 | 0.733 | 0.250 | 15.000 | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| self_routed_classifier | 0.941 | 0.907 | 0.961 | 1.000 | 0.760 | 192.000 | 0.000 | 0.000 | 0.733 | 0.733 | 0.000 | 0.000 | 0.850 | 0.833 | 0.733 | 1.000 | 1.000 | 1.000 | 0.983 |
| raw_balanced_only | 0.892 | 0.869 | 0.925 | 1.000 | 0.682 | 264.000 | 0.011 | 3.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.933 | 0.900 |
| self_routed_balanced | 0.896 | 0.872 | 0.928 | 1.000 | 0.690 | 261.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.933 | 0.933 |
| raw_strict_only | 0.821 | 0.810 | 0.868 | 1.000 | 0.562 | 320.000 | 0.022 | 7.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.933 | 1.000 | 1.000 | 1.000 | 0.917 | 0.883 |
| self_routed_strict | 0.829 | 0.817 | 0.875 | 1.000 | 0.575 | 313.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.933 | 1.000 | 1.000 | 1.000 | 0.917 | 0.900 |

## THRESHOLD 0.4

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.924 | 0.899 | 0.948 | 1.000 | 0.749 | 235.000 | 0.272 | 64.000 | 0.983 | 0.550 | 0.433 | 26.000 | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| self_routed_classifier | 0.937 | 0.891 | 0.959 | 1.000 | 0.713 | 171.000 | 0.000 | 0.000 | 0.550 | 0.550 | 0.000 | 0.000 | 0.733 | 0.733 | 0.550 | 1.000 | 1.000 | 1.000 | 0.983 |
| raw_balanced_only | 0.892 | 0.869 | 0.925 | 1.000 | 0.682 | 264.000 | 0.030 | 8.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.933 | 0.900 |
| self_routed_balanced | 0.903 | 0.879 | 0.932 | 1.000 | 0.703 | 256.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 | 0.933 |
| raw_strict_only | 0.821 | 0.810 | 0.868 | 1.000 | 0.562 | 320.000 | 0.081 | 26.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.933 | 1.000 | 1.000 | 1.000 | 0.917 | 0.883 |
| self_routed_strict | 0.854 | 0.836 | 0.895 | 1.000 | 0.612 | 294.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.950 | 1.000 | 1.000 | 1.000 | 0.917 | 0.900 |

## THRESHOLD 0.5

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.924 | 0.899 | 0.948 | 1.000 | 0.749 | 235.000 | 0.366 | 86.000 | 0.983 | 0.467 | 0.517 | 31.000 | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| self_routed_classifier | 0.937 | 0.877 | 0.960 | 1.000 | 0.671 | 149.000 | 0.000 | 0.000 | 0.467 | 0.467 | 0.000 | 0.000 | 0.817 | 0.717 | 0.467 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.892 | 0.869 | 0.925 | 1.000 | 0.682 | 264.000 | 0.034 | 9.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.933 | 0.900 |
| self_routed_balanced | 0.904 | 0.880 | 0.933 | 1.000 | 0.706 | 255.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 | 0.933 |
| raw_strict_only | 0.821 | 0.810 | 0.868 | 1.000 | 0.562 | 320.000 | 0.125 | 40.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.933 | 1.000 | 1.000 | 1.000 | 0.917 | 0.883 |
| self_routed_strict | 0.872 | 0.851 | 0.909 | 1.000 | 0.643 | 280.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.950 | 1.000 | 1.000 | 1.000 | 0.933 | 0.917 |

## THRESHOLD 0.6

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.924 | 0.899 | 0.948 | 1.000 | 0.749 | 235.000 | 0.421 | 99.000 | 0.983 | 0.333 | 0.650 | 39.000 | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| self_routed_classifier | 0.931 | 0.853 | 0.957 | 1.000 | 0.603 | 136.000 | 0.000 | 0.000 | 0.333 | 0.333 | 0.000 | 0.000 | 0.750 | 0.667 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.892 | 0.869 | 0.925 | 1.000 | 0.682 | 264.000 | 0.042 | 11.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.933 | 0.900 |
| self_routed_balanced | 0.906 | 0.882 | 0.935 | 1.000 | 0.711 | 253.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 | 0.933 |
| raw_strict_only | 0.821 | 0.810 | 0.868 | 1.000 | 0.562 | 320.000 | 0.163 | 52.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.933 | 1.000 | 1.000 | 1.000 | 0.917 | 0.883 |
| self_routed_strict | 0.887 | 0.864 | 0.921 | 1.000 | 0.672 | 268.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.967 | 1.000 | 1.000 | 1.000 | 0.967 | 0.950 |

## THRESHOLD 0.7

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.924 | 0.899 | 0.948 | 1.000 | 0.749 | 235.000 | 0.511 | 120.000 | 0.983 | 0.133 | 0.850 | 51.000 | 0.967 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| self_routed_classifier | 0.917 | 0.794 | 0.949 | 1.000 | 0.435 | 115.000 | 0.000 | 0.000 | 0.133 | 0.133 | 0.000 | 0.000 | 0.733 | 0.550 | 0.133 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.892 | 0.869 | 0.925 | 1.000 | 0.682 | 264.000 | 0.053 | 14.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.933 | 0.900 |
| self_routed_balanced | 0.910 | 0.886 | 0.938 | 1.000 | 0.720 | 250.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 | 0.967 |
| raw_strict_only | 0.821 | 0.810 | 0.868 | 1.000 | 0.562 | 320.000 | 0.206 | 66.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.933 | 1.000 | 1.000 | 1.000 | 0.917 | 0.883 |
| self_routed_strict | 0.900 | 0.875 | 0.931 | 1.000 | 0.693 | 254.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 | 0.967 |
