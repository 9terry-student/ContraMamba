# Stage 6B Single-Model Self-Routing

## THRESHOLD 0.3

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.885 | 0.864 | 0.919 | 1.000 | 0.674 | 88.000 | 0.080 | 7.000 | 1.000 | 0.950 | 0.050 | 1.000 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 |
| self_routed_classifier | 0.896 | 0.870 | 0.929 | 1.000 | 0.682 | 81.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.850 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.854 | 0.839 | 0.896 | 1.000 | 0.620 | 96.000 | 0.073 | 7.000 | 1.000 | 0.950 | 0.050 | 1.000 | 0.950 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.800 |
| self_routed_balanced | 0.873 | 0.852 | 0.911 | 1.000 | 0.645 | 89.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.850 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| raw_strict_only | 0.842 | 0.825 | 0.888 | 1.000 | 0.586 | 95.000 | 0.032 | 3.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.850 |
| self_routed_strict | 0.854 | 0.834 | 0.897 | 1.000 | 0.604 | 92.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.900 |

## THRESHOLD 0.4

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.885 | 0.864 | 0.919 | 1.000 | 0.674 | 88.000 | 0.114 | 10.000 | 1.000 | 0.950 | 0.050 | 1.000 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 |
| self_routed_classifier | 0.908 | 0.881 | 0.937 | 1.000 | 0.707 | 78.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.850 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.854 | 0.839 | 0.896 | 1.000 | 0.620 | 96.000 | 0.094 | 9.000 | 1.000 | 0.950 | 0.050 | 1.000 | 0.950 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.800 |
| self_routed_balanced | 0.881 | 0.859 | 0.917 | 1.000 | 0.659 | 87.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| raw_strict_only | 0.842 | 0.825 | 0.888 | 1.000 | 0.586 | 95.000 | 0.063 | 6.000 | 1.000 | 0.950 | 0.050 | 1.000 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.850 |
| self_routed_strict | 0.850 | 0.825 | 0.895 | 1.000 | 0.581 | 89.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.850 | 0.850 | 0.950 | 1.000 | 1.000 | 1.000 | 0.900 |

## THRESHOLD 0.5

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.885 | 0.864 | 0.919 | 1.000 | 0.674 | 88.000 | 0.205 | 18.000 | 1.000 | 0.750 | 0.250 | 5.000 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 |
| self_routed_classifier | 0.892 | 0.850 | 0.928 | 1.000 | 0.622 | 70.000 | 0.000 | 0.000 | 0.750 | 0.750 | 0.000 | 0.000 | 0.750 | 0.800 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.854 | 0.839 | 0.896 | 1.000 | 0.620 | 96.000 | 0.135 | 13.000 | 1.000 | 0.950 | 0.050 | 1.000 | 0.950 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.800 |
| self_routed_balanced | 0.881 | 0.854 | 0.918 | 1.000 | 0.644 | 83.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| raw_strict_only | 0.842 | 0.825 | 0.888 | 1.000 | 0.586 | 95.000 | 0.116 | 11.000 | 1.000 | 0.950 | 0.050 | 1.000 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.850 |
| self_routed_strict | 0.862 | 0.832 | 0.904 | 1.000 | 0.591 | 84.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |

## THRESHOLD 0.6

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.885 | 0.864 | 0.919 | 1.000 | 0.674 | 88.000 | 0.261 | 23.000 | 1.000 | 0.700 | 0.300 | 6.000 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 |
| self_routed_classifier | 0.896 | 0.847 | 0.932 | 1.000 | 0.609 | 65.000 | 0.000 | 0.000 | 0.700 | 0.700 | 0.000 | 0.000 | 0.650 | 0.750 | 0.700 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.854 | 0.839 | 0.896 | 1.000 | 0.620 | 96.000 | 0.135 | 13.000 | 1.000 | 0.950 | 0.050 | 1.000 | 0.950 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.800 |
| self_routed_balanced | 0.881 | 0.854 | 0.918 | 1.000 | 0.644 | 83.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| raw_strict_only | 0.842 | 0.825 | 0.888 | 1.000 | 0.586 | 95.000 | 0.158 | 15.000 | 1.000 | 0.900 | 0.100 | 2.000 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.850 |
| self_routed_strict | 0.862 | 0.825 | 0.905 | 0.982 | 0.588 | 80.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.750 | 0.950 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |

## THRESHOLD 0.7

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_classifier_only | 0.885 | 0.864 | 0.919 | 1.000 | 0.674 | 88.000 | 0.341 | 30.000 | 1.000 | 0.550 | 0.450 | 9.000 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 |
| self_routed_classifier | 0.900 | 0.839 | 0.935 | 1.000 | 0.581 | 58.000 | 0.000 | 0.000 | 0.550 | 0.550 | 0.000 | 0.000 | 0.650 | 0.650 | 0.550 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_balanced_only | 0.854 | 0.839 | 0.896 | 1.000 | 0.620 | 96.000 | 0.146 | 14.000 | 1.000 | 0.950 | 0.050 | 1.000 | 0.950 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.800 |
| self_routed_balanced | 0.877 | 0.848 | 0.915 | 1.000 | 0.628 | 82.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.750 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| raw_strict_only | 0.842 | 0.825 | 0.888 | 1.000 | 0.586 | 95.000 | 0.295 | 28.000 | 1.000 | 0.850 | 0.150 | 3.000 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.850 |
| self_routed_strict | 0.865 | 0.807 | 0.911 | 0.923 | 0.587 | 67.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.650 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 0.950 |
