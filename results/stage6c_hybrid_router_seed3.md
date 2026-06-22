# Stage 6C Hybrid Expert Router Search

## THRESHOLD 0.3

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| agreement_keep_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| strict_veto_balanced_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.900 | 0.863 | 0.932 | 1.000 | 0.658 | 80.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_strict_agreement_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| cautious_promotion_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |

## THRESHOLD 0.4

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| agreement_keep_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| strict_veto_balanced_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.900 | 0.863 | 0.932 | 1.000 | 0.658 | 80.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_strict_agreement_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| cautious_promotion_router | 0.919 | 0.880 | 0.946 | 1.000 | 0.696 | 73.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |

## THRESHOLD 0.5

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.923 | 0.885 | 0.948 | 1.000 | 0.706 | 72.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.923 | 0.885 | 0.948 | 1.000 | 0.706 | 72.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| agreement_keep_router | 0.923 | 0.885 | 0.948 | 1.000 | 0.706 | 72.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.923 | 0.885 | 0.948 | 1.000 | 0.706 | 72.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| strict_veto_balanced_router | 0.919 | 0.879 | 0.946 | 0.984 | 0.706 | 71.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.908 | 0.871 | 0.937 | 1.000 | 0.676 | 78.000 | 0.000 | 0.000 | 0.900 | 0.900 | 0.000 | 0.000 | 0.850 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_strict_agreement_router | 0.919 | 0.879 | 0.946 | 0.984 | 0.706 | 71.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| cautious_promotion_router | 0.923 | 0.885 | 0.948 | 1.000 | 0.706 | 72.000 | 0.000 | 0.000 | 0.850 | 0.850 | 0.000 | 0.000 | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |

## THRESHOLD 0.6

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.915 | 0.870 | 0.944 | 1.000 | 0.667 | 70.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.915 | 0.870 | 0.944 | 1.000 | 0.667 | 70.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| agreement_keep_router | 0.915 | 0.870 | 0.944 | 1.000 | 0.667 | 70.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.915 | 0.870 | 0.944 | 1.000 | 0.667 | 70.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| strict_veto_balanced_router | 0.904 | 0.850 | 0.936 | 0.968 | 0.646 | 67.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.950 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.912 | 0.869 | 0.941 | 1.000 | 0.667 | 73.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.850 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_strict_agreement_router | 0.904 | 0.850 | 0.936 | 0.968 | 0.646 | 67.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.950 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| cautious_promotion_router | 0.915 | 0.870 | 0.944 | 1.000 | 0.667 | 70.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |

## THRESHOLD 0.7

| system | final_accuracy | final_macro_f1 | NOT_ENTITLED_f1 | REFUTE_f1 | SUPPORT_f1 | entitled_output_count | entitled_output_gate_violation_rate | entitled_output_gate_violations | polarity_flip_output_ok | polarity_flip_internal_ok | polarity_flip_output_internal_gap | polarity_flip_output_ok_but_internal_bad | paraphrase_preserved | predicate_disentangled | polarity_flip_preserved_and_reversed | deletion_sufficiency_lower | truncation_sufficiency_lower | entity_frame_lower | event_frame_lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_balanced_router | 0.912 | 0.864 | 0.941 | 0.984 | 0.667 | 69.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| self_routed_balanced | 0.912 | 0.864 | 0.941 | 0.984 | 0.667 | 69.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| agreement_keep_router | 0.912 | 0.864 | 0.941 | 0.984 | 0.667 | 69.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_override_router | 0.912 | 0.864 | 0.941 | 0.984 | 0.667 | 69.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| strict_veto_balanced_router | 0.904 | 0.846 | 0.937 | 0.968 | 0.635 | 65.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 1.000 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_gate_verified_router | 0.912 | 0.866 | 0.941 | 1.000 | 0.657 | 71.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| balanced_strict_agreement_router | 0.904 | 0.846 | 0.937 | 0.968 | 0.635 | 65.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 1.000 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| cautious_promotion_router | 0.912 | 0.864 | 0.941 | 0.984 | 0.667 | 69.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.000 | 0.000 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |

## INTERPRETATION

- Highest per-seed macro-F1: conservative_balanced_router at threshold 0.5 (0.885).
- These are fixed controlled diagnostic rules, not a learned or final router.
