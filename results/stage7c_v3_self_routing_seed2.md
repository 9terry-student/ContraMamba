# Stage 6B Single-Model Self-Routing

## THRESHOLD 0.3

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 | 233.000 | 0.013 | 3.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_classifier | 0.936 | 0.914 | 0.957 | 1.000 | 0.784 | 230.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.915 | 0.892 | 0.942 | 1.000 | 0.734 | 246.000 | 0.008 | 2.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
| self_routed_balanced | 0.918 | 0.894 | 0.944 | 1.000 | 0.740 | 244.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
| raw_strict_only | 0.910 | 0.887 | 0.938 | 1.000 | 0.722 | 250.000 | 0.012 | 3.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 | 0.967 |
| self_routed_strict | 0.914 | 0.891 | 0.941 | 1.000 | 0.731 | 247.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 | 0.983 |

## THRESHOLD 0.4

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 | 233.000 | 0.043 | 10.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_classifier | 0.945 | 0.924 | 0.963 | 1.000 | 0.809 | 223.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.915 | 0.892 | 0.942 | 1.000 | 0.734 | 246.000 | 0.008 | 2.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
| self_routed_balanced | 0.918 | 0.894 | 0.944 | 1.000 | 0.740 | 244.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
| raw_strict_only | 0.910 | 0.887 | 0.938 | 1.000 | 0.722 | 250.000 | 0.012 | 3.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 | 0.967 |
| self_routed_strict | 0.914 | 0.891 | 0.941 | 1.000 | 0.731 | 247.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 | 0.983 |

## THRESHOLD 0.5

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 | 233.000 | 0.060 | 14.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_classifier | 0.950 | 0.930 | 0.966 | 1.000 | 0.824 | 219.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.915 | 0.892 | 0.942 | 1.000 | 0.734 | 246.000 | 0.008 | 2.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
| self_routed_balanced | 0.918 | 0.894 | 0.944 | 1.000 | 0.740 | 244.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
| raw_strict_only | 0.910 | 0.887 | 0.938 | 1.000 | 0.722 | 250.000 | 0.024 | 6.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 | 0.967 |
| self_routed_strict | 0.918 | 0.894 | 0.944 | 1.000 | 0.740 | 244.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |

## THRESHOLD 0.6

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 | 233.000 | 0.086 | 20.000 | 1.000 | 0.967 | 0.033 | 2.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_classifier | 0.953 | 0.932 | 0.968 | 1.000 | 0.828 | 213.000 | 0.000 | 0.000 | 0.967 | 0.967 | 0.000 | 0.000 | 0.967 | 0.967 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.915 | 0.892 | 0.942 | 1.000 | 0.734 | 246.000 | 0.016 | 4.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
| self_routed_balanced | 0.921 | 0.897 | 0.946 | 1.000 | 0.746 | 242.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
| raw_strict_only | 0.910 | 0.887 | 0.938 | 1.000 | 0.722 | 250.000 | 0.028 | 7.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 | 0.967 |
| self_routed_strict | 0.919 | 0.896 | 0.945 | 1.000 | 0.743 | 243.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |

## THRESHOLD 0.7

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.932 | 0.909 | 0.954 | 1.000 | 0.774 | 233.000 | 0.120 | 28.000 | 1.000 | 0.950 | 0.050 | 3.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_classifier | 0.955 | 0.934 | 0.970 | 1.000 | 0.831 | 205.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.967 | 0.967 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.915 | 0.892 | 0.942 | 1.000 | 0.734 | 246.000 | 0.020 | 5.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
| self_routed_balanced | 0.922 | 0.898 | 0.946 | 1.000 | 0.749 | 241.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_strict_only | 0.910 | 0.887 | 0.938 | 1.000 | 0.722 | 250.000 | 0.032 | 8.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.983 | 1.000 | 1.000 | 1.000 | 0.983 | 0.967 |
| self_routed_strict | 0.921 | 0.897 | 0.946 | 1.000 | 0.746 | 242.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
