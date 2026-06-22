# Stage 6B Single-Model Self-Routing

## THRESHOLD 0.3

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.900 | 0.869 | 0.931 | 1.000 | 0.675 | 84.000 | 0.083 | 7.000 | 0.950 | 0.850 | 0.100 | 2.000 | 0.950 | 0.850 | 0.950 | 1.000 | 1.000 | 0.950 | 1.000 |
| self_routed_classifier | 0.896 | 0.853 | 0.930 | 1.000 | 0.630 | 77.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.850 | 0.800 | 0.850 | 1.000 | 1.000 | 0.950 | 1.000 |
| raw_balanced_only | 0.904 | 0.870 | 0.934 | 1.000 | 0.675 | 81.000 | 0.099 | 8.000 | 0.950 | 0.850 | 0.100 | 2.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_strict_only | 0.831 | 0.803 | 0.878 | 1.000 | 0.532 | 98.000 | 0.051 | 5.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 |
| self_routed_strict | 0.850 | 0.819 | 0.894 | 1.000 | 0.562 | 93.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 |

## THRESHOLD 0.4

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.900 | 0.869 | 0.931 | 1.000 | 0.675 | 84.000 | 0.131 | 11.000 | 0.950 | 0.850 | 0.100 | 2.000 | 0.950 | 0.850 | 0.950 | 1.000 | 1.000 | 0.950 | 1.000 |
| self_routed_classifier | 0.904 | 0.858 | 0.935 | 1.000 | 0.638 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.904 | 0.870 | 0.934 | 1.000 | 0.675 | 81.000 | 0.099 | 8.000 | 0.950 | 0.850 | 0.100 | 2.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_strict_only | 0.831 | 0.803 | 0.878 | 1.000 | 0.532 | 98.000 | 0.112 | 11.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 |
| self_routed_strict | 0.873 | 0.838 | 0.912 | 1.000 | 0.602 | 87.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 |

## THRESHOLD 0.5

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.900 | 0.869 | 0.931 | 1.000 | 0.675 | 84.000 | 0.167 | 14.000 | 0.950 | 0.850 | 0.100 | 2.000 | 0.950 | 0.850 | 0.950 | 1.000 | 1.000 | 0.950 | 1.000 |
| self_routed_classifier | 0.915 | 0.870 | 0.944 | 1.000 | 0.667 | 70.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.904 | 0.870 | 0.934 | 1.000 | 0.675 | 81.000 | 0.111 | 9.000 | 0.950 | 0.850 | 0.100 | 2.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.923 | 0.885 | 0.948 | 1.000 | 0.706 | 72.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_strict_only | 0.831 | 0.803 | 0.878 | 1.000 | 0.532 | 98.000 | 0.184 | 18.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 |
| self_routed_strict | 0.892 | 0.853 | 0.926 | 0.984 | 0.649 | 80.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |

## THRESHOLD 0.6

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.900 | 0.869 | 0.931 | 1.000 | 0.675 | 84.000 | 0.190 | 16.000 | 0.950 | 0.800 | 0.150 | 3.000 | 0.950 | 0.850 | 0.950 | 1.000 | 1.000 | 0.950 | 1.000 |
| self_routed_classifier | 0.915 | 0.867 | 0.944 | 1.000 | 0.656 | 68.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.850 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.904 | 0.870 | 0.934 | 1.000 | 0.675 | 81.000 | 0.136 | 11.000 | 0.950 | 0.800 | 0.150 | 3.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.915 | 0.870 | 0.944 | 1.000 | 0.667 | 70.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_strict_only | 0.831 | 0.803 | 0.878 | 1.000 | 0.532 | 98.000 | 0.276 | 27.000 | 0.900 | 0.800 | 0.100 | 2.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 |
| self_routed_strict | 0.888 | 0.834 | 0.925 | 0.968 | 0.609 | 71.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.950 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 0.950 |

## THRESHOLD 0.7

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.900 | 0.869 | 0.931 | 1.000 | 0.675 | 84.000 | 0.250 | 21.000 | 0.950 | 0.750 | 0.200 | 4.000 | 0.950 | 0.850 | 0.950 | 1.000 | 1.000 | 0.950 | 1.000 |
| self_routed_classifier | 0.919 | 0.864 | 0.947 | 1.000 | 0.644 | 63.000 | 0.000 | 0.000 | 0.750 | 0.750 | 0.000 | 0.000 | 0.900 | 0.800 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.904 | 0.870 | 0.934 | 1.000 | 0.675 | 81.000 | 0.148 | 12.000 | 0.950 | 0.800 | 0.150 | 3.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.912 | 0.864 | 0.941 | 0.984 | 0.667 | 69.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_strict_only | 0.831 | 0.803 | 0.878 | 1.000 | 0.532 | 98.000 | 0.316 | 31.000 | 0.900 | 0.800 | 0.100 | 2.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 |
| self_routed_strict | 0.896 | 0.838 | 0.931 | 0.968 | 0.615 | 67.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 1.000 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
