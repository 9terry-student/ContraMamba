# Stage 6C Hybrid Expert Router Search

## THRESHOLD 0.3

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.892 | 0.869 | 0.926 | 1.000 | 0.682 | 84.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.873 | 0.852 | 0.911 | 1.000 | 0.645 | 89.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.850 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| agreement_keep_router | 0.892 | 0.869 | 0.926 | 1.000 | 0.682 | 84.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.877 | 0.855 | 0.914 | 1.000 | 0.652 | 88.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| strict_veto_balanced_router | 0.888 | 0.861 | 0.923 | 1.000 | 0.659 | 81.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.865 | 0.848 | 0.905 | 1.000 | 0.639 | 93.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 | 0.900 |
| balanced_strict_agreement_router | 0.877 | 0.855 | 0.914 | 1.000 | 0.652 | 88.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| cautious_promotion_router | 0.877 | 0.855 | 0.914 | 1.000 | 0.652 | 88.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |

## THRESHOLD 0.4

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.896 | 0.873 | 0.928 | 1.000 | 0.690 | 83.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.881 | 0.859 | 0.917 | 1.000 | 0.659 | 87.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| agreement_keep_router | 0.896 | 0.873 | 0.928 | 1.000 | 0.690 | 83.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.881 | 0.859 | 0.917 | 1.000 | 0.659 | 87.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| strict_veto_balanced_router | 0.885 | 0.855 | 0.921 | 1.000 | 0.643 | 80.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.850 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.873 | 0.854 | 0.911 | 1.000 | 0.653 | 91.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 |
| balanced_strict_agreement_router | 0.877 | 0.853 | 0.914 | 1.000 | 0.644 | 86.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.850 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| cautious_promotion_router | 0.881 | 0.859 | 0.917 | 1.000 | 0.659 | 87.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |

## THRESHOLD 0.5

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.896 | 0.868 | 0.929 | 1.000 | 0.675 | 79.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.881 | 0.854 | 0.918 | 1.000 | 0.644 | 83.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| agreement_keep_router | 0.896 | 0.868 | 0.929 | 1.000 | 0.675 | 79.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.881 | 0.854 | 0.918 | 1.000 | 0.644 | 83.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| strict_veto_balanced_router | 0.892 | 0.859 | 0.927 | 1.000 | 0.650 | 76.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.869 | 0.846 | 0.909 | 1.000 | 0.630 | 88.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.850 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 | 0.950 |
| balanced_strict_agreement_router | 0.881 | 0.851 | 0.918 | 1.000 | 0.635 | 81.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.750 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| cautious_promotion_router | 0.881 | 0.854 | 0.918 | 1.000 | 0.644 | 83.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |

## THRESHOLD 0.6

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.896 | 0.868 | 0.929 | 1.000 | 0.675 | 79.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.881 | 0.854 | 0.918 | 1.000 | 0.644 | 83.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| agreement_keep_router | 0.896 | 0.868 | 0.929 | 1.000 | 0.675 | 79.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.881 | 0.854 | 0.918 | 1.000 | 0.644 | 83.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| strict_veto_balanced_router | 0.885 | 0.846 | 0.922 | 0.982 | 0.633 | 74.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.750 | 0.950 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.873 | 0.850 | 0.912 | 1.000 | 0.637 | 87.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.850 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 | 0.950 |
| balanced_strict_agreement_router | 0.877 | 0.842 | 0.916 | 0.982 | 0.627 | 78.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.700 | 0.950 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| cautious_promotion_router | 0.885 | 0.857 | 0.921 | 1.000 | 0.651 | 82.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |

## THRESHOLD 0.7

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.892 | 0.862 | 0.927 | 1.000 | 0.659 | 78.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.750 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.877 | 0.848 | 0.915 | 1.000 | 0.628 | 82.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.750 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| agreement_keep_router | 0.892 | 0.862 | 0.927 | 1.000 | 0.659 | 78.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.750 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.877 | 0.848 | 0.915 | 1.000 | 0.628 | 82.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.750 | 0.900 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |
| strict_veto_balanced_router | 0.877 | 0.818 | 0.919 | 0.923 | 0.611 | 64.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.650 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.881 | 0.854 | 0.918 | 1.000 | 0.644 | 83.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.800 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 | 0.950 |
| balanced_strict_agreement_router | 0.869 | 0.814 | 0.913 | 0.923 | 0.605 | 68.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.600 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 0.950 |
| cautious_promotion_router | 0.881 | 0.851 | 0.918 | 1.000 | 0.635 | 81.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.750 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 |

## INTERPRETATION

- Highest per-seed macro-F1: conservative_balanced_router at threshold 0.4 (0.873).
- These are fixed controlled diagnostic rules, not a learned or final router.
