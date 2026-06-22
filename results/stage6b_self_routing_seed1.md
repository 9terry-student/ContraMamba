# Stage 6B Single-Model Self-Routing

## THRESHOLD 0.3

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.908 | 0.876 | 0.937 | 1.000 | 0.692 | 78.000 | 0.115 | 9.000 | 0.850 | 0.700 | 0.150 | 3.000 | 0.850 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 0.950 |
| self_routed_classifier | 0.919 | 0.881 | 0.946 | 1.000 | 0.696 | 69.000 | 0.000 | 0.000 | 0.700 | 0.700 | 0.000 | 0.000 | 0.800 | 0.750 | 0.700 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.869 | 0.843 | 0.908 | 0.984 | 0.637 | 92.000 | 0.033 | 3.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.950 | 0.900 | 0.950 | 1.000 | 1.000 | 0.900 | 0.850 |
| self_routed_balanced | 0.873 | 0.844 | 0.911 | 0.984 | 0.636 | 89.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 0.900 | 0.850 |
| raw_strict_only | 0.858 | 0.831 | 0.900 | 1.000 | 0.593 | 91.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 0.950 | 0.850 |
| self_routed_strict | 0.858 | 0.831 | 0.900 | 1.000 | 0.593 | 91.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 0.950 | 0.850 |

## THRESHOLD 0.4

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.908 | 0.876 | 0.937 | 1.000 | 0.692 | 78.000 | 0.205 | 16.000 | 0.850 | 0.550 | 0.300 | 6.000 | 0.850 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 0.950 |
| self_routed_classifier | 0.908 | 0.851 | 0.940 | 1.000 | 0.613 | 62.000 | 0.000 | 0.000 | 0.550 | 0.550 | 0.000 | 0.000 | 0.800 | 0.750 | 0.550 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.869 | 0.843 | 0.908 | 0.984 | 0.637 | 92.000 | 0.109 | 10.000 | 0.950 | 0.900 | 0.050 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 | 1.000 | 0.900 | 0.850 |
| self_routed_balanced | 0.892 | 0.861 | 0.926 | 1.000 | 0.659 | 82.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 0.950 | 0.950 |
| raw_strict_only | 0.858 | 0.831 | 0.900 | 1.000 | 0.593 | 91.000 | 0.022 | 2.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 0.950 | 0.850 |
| self_routed_strict | 0.865 | 0.837 | 0.906 | 1.000 | 0.607 | 89.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 0.950 | 0.900 |

## THRESHOLD 0.5

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.908 | 0.876 | 0.937 | 1.000 | 0.692 | 78.000 | 0.295 | 23.000 | 0.850 | 0.450 | 0.400 | 8.000 | 0.850 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 0.950 |
| self_routed_classifier | 0.912 | 0.844 | 0.943 | 0.983 | 0.607 | 55.000 | 0.000 | 0.000 | 0.450 | 0.450 | 0.000 | 0.000 | 0.850 | 0.700 | 0.450 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.869 | 0.843 | 0.908 | 0.984 | 0.637 | 92.000 | 0.130 | 12.000 | 0.950 | 0.900 | 0.050 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 | 1.000 | 0.900 | 0.850 |
| self_routed_balanced | 0.900 | 0.869 | 0.932 | 1.000 | 0.675 | 80.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| raw_strict_only | 0.858 | 0.831 | 0.900 | 1.000 | 0.593 | 91.000 | 0.077 | 7.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 0.950 | 0.850 |
| self_routed_strict | 0.877 | 0.845 | 0.915 | 1.000 | 0.619 | 84.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.800 | 0.800 | 0.900 | 1.000 | 1.000 | 0.950 | 0.900 |

## THRESHOLD 0.6

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.908 | 0.876 | 0.937 | 1.000 | 0.692 | 78.000 | 0.372 | 29.000 | 0.850 | 0.350 | 0.500 | 10.000 | 0.850 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 0.950 |
| self_routed_classifier | 0.896 | 0.799 | 0.934 | 0.983 | 0.480 | 49.000 | 0.000 | 0.000 | 0.350 | 0.350 | 0.000 | 0.000 | 0.650 | 0.650 | 0.350 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.869 | 0.843 | 0.908 | 0.984 | 0.637 | 92.000 | 0.130 | 12.000 | 0.950 | 0.900 | 0.050 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 | 1.000 | 0.900 | 0.850 |
| self_routed_balanced | 0.900 | 0.869 | 0.932 | 1.000 | 0.675 | 80.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| raw_strict_only | 0.858 | 0.831 | 0.900 | 1.000 | 0.593 | 91.000 | 0.132 | 12.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 0.950 | 0.850 |
| self_routed_strict | 0.888 | 0.852 | 0.924 | 1.000 | 0.633 | 79.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.750 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 |

## THRESHOLD 0.7

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.908 | 0.876 | 0.937 | 1.000 | 0.692 | 78.000 | 0.500 | 39.000 | 0.850 | 0.100 | 0.750 | 15.000 | 0.850 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 0.950 |
| self_routed_classifier | 0.896 | 0.756 | 0.936 | 0.983 | 0.350 | 39.000 | 0.000 | 0.000 | 0.100 | 0.100 | 0.000 | 0.000 | 0.700 | 0.550 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.869 | 0.843 | 0.908 | 0.984 | 0.637 | 92.000 | 0.163 | 15.000 | 0.950 | 0.900 | 0.050 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 | 1.000 | 0.900 | 0.850 |
| self_routed_balanced | 0.904 | 0.870 | 0.935 | 1.000 | 0.675 | 77.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.900 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| raw_strict_only | 0.858 | 0.831 | 0.900 | 1.000 | 0.593 | 91.000 | 0.220 | 20.000 | 0.900 | 0.850 | 0.050 | 1.000 | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 0.950 | 0.850 |
| self_routed_strict | 0.888 | 0.840 | 0.925 | 0.983 | 0.611 | 71.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 0.950 |
