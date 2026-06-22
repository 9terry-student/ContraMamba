# Stage 6C Hybrid Expert Router Search

## THRESHOLD 0.3

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.912 | 0.878 | 0.940 | 1.000 | 0.693 | 75.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.873 | 0.844 | 0.911 | 0.984 | 0.636 | 89.000 | 0.000 | 0.000 | 0.950 | 0.950 | 0.000 | 0.000 | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 0.900 | 0.850 |
| agreement_keep_router | 0.912 | 0.878 | 0.940 | 1.000 | 0.693 | 75.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.904 | 0.873 | 0.934 | 1.000 | 0.684 | 79.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| strict_veto_balanced_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.885 | 0.857 | 0.920 | 1.000 | 0.651 | 86.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.900 | 0.850 | 0.900 | 1.000 | 1.000 | 0.950 | 0.850 |
| balanced_strict_agreement_router | 0.885 | 0.854 | 0.920 | 1.000 | 0.643 | 84.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 0.950 | 0.850 |
| cautious_promotion_router | 0.885 | 0.854 | 0.920 | 1.000 | 0.643 | 84.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 0.950 | 0.850 |

## THRESHOLD 0.4

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.892 | 0.861 | 0.926 | 1.000 | 0.659 | 82.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 0.950 | 0.950 |
| agreement_keep_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.908 | 0.876 | 0.937 | 1.000 | 0.692 | 78.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| strict_veto_balanced_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.888 | 0.860 | 0.923 | 1.000 | 0.659 | 85.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.900 | 0.850 | 0.900 | 1.000 | 1.000 | 0.950 | 0.900 |
| balanced_strict_agreement_router | 0.896 | 0.865 | 0.929 | 1.000 | 0.667 | 81.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 0.950 | 0.950 |
| cautious_promotion_router | 0.896 | 0.865 | 0.929 | 1.000 | 0.667 | 81.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 0.950 | 0.950 |

## THRESHOLD 0.5

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.900 | 0.869 | 0.932 | 1.000 | 0.675 | 80.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| agreement_keep_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.908 | 0.876 | 0.937 | 1.000 | 0.692 | 78.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| strict_veto_balanced_router | 0.912 | 0.875 | 0.941 | 1.000 | 0.685 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.750 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.896 | 0.868 | 0.928 | 1.000 | 0.675 | 83.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.900 | 0.850 | 0.900 | 1.000 | 1.000 | 0.950 | 0.900 |
| balanced_strict_agreement_router | 0.900 | 0.866 | 0.932 | 1.000 | 0.667 | 78.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.800 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| cautious_promotion_router | 0.904 | 0.873 | 0.934 | 1.000 | 0.684 | 79.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |

## THRESHOLD 0.6

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.900 | 0.869 | 0.932 | 1.000 | 0.675 | 80.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| agreement_keep_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 | 74.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.908 | 0.876 | 0.937 | 1.000 | 0.692 | 78.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| strict_veto_balanced_router | 0.912 | 0.872 | 0.941 | 1.000 | 0.676 | 71.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.700 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.896 | 0.865 | 0.929 | 1.000 | 0.667 | 81.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 |
| balanced_strict_agreement_router | 0.900 | 0.863 | 0.932 | 1.000 | 0.658 | 76.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.750 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| cautious_promotion_router | 0.904 | 0.873 | 0.934 | 1.000 | 0.684 | 79.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |

## THRESHOLD 0.7

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.912 | 0.875 | 0.941 | 1.000 | 0.685 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.850 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.904 | 0.870 | 0.935 | 1.000 | 0.675 | 77.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.900 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| agreement_keep_router | 0.912 | 0.875 | 0.941 | 1.000 | 0.685 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.850 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.904 | 0.870 | 0.935 | 1.000 | 0.675 | 77.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.900 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| strict_veto_balanced_router | 0.904 | 0.852 | 0.937 | 0.983 | 0.636 | 65.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.850 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.904 | 0.873 | 0.934 | 1.000 | 0.684 | 79.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |
| balanced_strict_agreement_router | 0.896 | 0.848 | 0.931 | 0.983 | 0.629 | 69.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 0.950 |
| cautious_promotion_router | 0.904 | 0.870 | 0.935 | 1.000 | 0.675 | 77.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.900 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 |

## INTERPRETATION

- Highest per-seed macro-F1: strict_veto_balanced_router at threshold 0.3 (0.882).
- These are fixed controlled diagnostic rules, not a learned or final router.
