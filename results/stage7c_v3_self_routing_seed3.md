# Stage 6B Single-Model Self-Routing

## THRESHOLD 0.3

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.927 | 0.902 | 0.950 | 0.994 | 0.761 | 235.000 | 0.038 | 9.000 | 1.000 | 0.983 | 0.017 | 1.000 | 0.983 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| self_routed_classifier | 0.936 | 0.911 | 0.957 | 0.994 | 0.782 | 226.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 | 0.967 | 0.950 | 0.983 | 1.000 | 1.000 | 1.000 | 0.967 |
| raw_balanced_only | 0.905 | 0.880 | 0.934 | 0.994 | 0.711 | 254.000 | 0.004 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 |
| self_routed_balanced | 0.906 | 0.881 | 0.935 | 0.994 | 0.714 | 253.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 |
| raw_strict_only | 0.851 | 0.834 | 0.893 | 1.000 | 0.608 | 296.000 | 0.041 | 12.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 | 0.983 | 0.917 |
| self_routed_strict | 0.867 | 0.846 | 0.905 | 1.000 | 0.634 | 284.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.933 | 1.000 | 1.000 | 1.000 | 0.983 | 0.917 |

## THRESHOLD 0.4

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.927 | 0.902 | 0.950 | 0.994 | 0.761 | 235.000 | 0.077 | 18.000 | 1.000 | 0.967 | 0.033 | 2.000 | 0.983 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| self_routed_classifier | 0.945 | 0.922 | 0.963 | 1.000 | 0.802 | 217.000 | 0.000 | 0.000 | 0.967 | 0.967 | 0.000 | 0.000 | 0.950 | 0.950 | 0.967 | 1.000 | 1.000 | 1.000 | 0.967 |
| raw_balanced_only | 0.905 | 0.880 | 0.934 | 0.994 | 0.711 | 254.000 | 0.016 | 4.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 |
| self_routed_balanced | 0.910 | 0.885 | 0.938 | 0.994 | 0.723 | 250.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.933 |
| raw_strict_only | 0.851 | 0.834 | 0.893 | 1.000 | 0.608 | 296.000 | 0.078 | 23.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 | 0.983 | 0.917 |
| self_routed_strict | 0.881 | 0.858 | 0.916 | 1.000 | 0.659 | 273.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.950 | 1.000 | 1.000 | 1.000 | 0.983 | 0.933 |

## THRESHOLD 0.5

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.927 | 0.902 | 0.950 | 0.994 | 0.761 | 235.000 | 0.115 | 27.000 | 1.000 | 0.933 | 0.067 | 4.000 | 0.983 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| self_routed_classifier | 0.949 | 0.925 | 0.966 | 1.000 | 0.808 | 208.000 | 0.000 | 0.000 | 0.933 | 0.933 | 0.000 | 0.000 | 0.967 | 0.933 | 0.933 | 1.000 | 1.000 | 1.000 | 0.967 |
| raw_balanced_only | 0.905 | 0.880 | 0.934 | 0.994 | 0.711 | 254.000 | 0.028 | 7.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 |
| self_routed_balanced | 0.914 | 0.890 | 0.941 | 1.000 | 0.729 | 247.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.950 |
| raw_strict_only | 0.851 | 0.834 | 0.893 | 1.000 | 0.608 | 296.000 | 0.091 | 27.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 | 0.983 | 0.917 |
| self_routed_strict | 0.886 | 0.863 | 0.920 | 1.000 | 0.669 | 269.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.967 | 1.000 | 1.000 | 1.000 | 0.983 | 0.933 |

## THRESHOLD 0.6

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.927 | 0.902 | 0.950 | 0.994 | 0.761 | 235.000 | 0.162 | 38.000 | 1.000 | 0.900 | 0.100 | 6.000 | 0.983 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| self_routed_classifier | 0.958 | 0.935 | 0.972 | 1.000 | 0.832 | 197.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.950 | 0.917 | 0.900 | 1.000 | 1.000 | 1.000 | 0.983 |
| raw_balanced_only | 0.905 | 0.880 | 0.934 | 0.994 | 0.711 | 254.000 | 0.031 | 8.000 | 1.000 | 0.983 | 0.017 | 1.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 |
| self_routed_balanced | 0.913 | 0.888 | 0.940 | 1.000 | 0.724 | 246.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 | 0.983 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.950 |
| raw_strict_only | 0.851 | 0.834 | 0.893 | 1.000 | 0.608 | 296.000 | 0.122 | 36.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 | 0.983 | 0.917 |
| self_routed_strict | 0.897 | 0.874 | 0.929 | 1.000 | 0.692 | 260.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.933 |

## THRESHOLD 0.7

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.927 | 0.902 | 0.950 | 0.994 | 0.761 | 235.000 | 0.281 | 66.000 | 1.000 | 0.700 | 0.300 | 18.000 | 0.983 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| self_routed_classifier | 0.963 | 0.935 | 0.976 | 1.000 | 0.828 | 169.000 | 0.000 | 0.000 | 0.700 | 0.700 | 0.000 | 0.000 | 0.867 | 0.833 | 0.700 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.905 | 0.880 | 0.934 | 0.994 | 0.711 | 254.000 | 0.035 | 9.000 | 1.000 | 0.983 | 0.017 | 1.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 |
| self_routed_balanced | 0.912 | 0.886 | 0.939 | 1.000 | 0.718 | 245.000 | 0.000 | 0.000 | 0.983 | 0.983 | 0.000 | 0.000 | 1.000 | 0.967 | 0.983 | 1.000 | 1.000 | 1.000 | 0.950 |
| raw_strict_only | 0.851 | 0.834 | 0.893 | 1.000 | 0.608 | 296.000 | 0.139 | 41.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 | 0.983 | 0.917 |
| self_routed_strict | 0.904 | 0.880 | 0.933 | 1.000 | 0.706 | 255.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 | 0.950 |
