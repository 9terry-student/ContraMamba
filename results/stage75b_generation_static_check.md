# Stage75B — Generation Static Check

## Decision

`STAGE75B_GENERATION_STATIC_CHECK_READY`

## Summary

| stage                 | decision                               | jsonl                                       | report_json                                                      | report_md                                                      |   row_count | label_counts                                        | family_counts                                                                                                                                                                                                                           |   duplicate_id_count |   duplicate_claim_evidence_count |   schema_error_count | report_decision                                    | training_executed   | external_eval_executed   | recommended_next_stage                                                |
|:----------------------|:---------------------------------------|:--------------------------------------------|:-----------------------------------------------------------------|:---------------------------------------------------------------|------------:|:----------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------:|---------------------------------:|---------------------:|:---------------------------------------------------|:--------------------|:-------------------------|:----------------------------------------------------------------------|
| Stage75B_static_check | STAGE75B_GENERATION_STATIC_CHECK_READY | data/stage75_targeted_residual_bridge.jsonl | results/stage75b_targeted_residual_bridge_generation_report.json | results/stage75b_targeted_residual_bridge_generation_report.md |        1020 | {"SUPPORT": 480, "REFUTE": 460, "NOT_ENTITLED": 80} | {"support_entitlement_direct_recovery_v2": 240, "refute_entitlement_direct_recovery_v2": 220, "numeric_temporal_polarity_comparison_v2": 260, "lexical_type_polarity_disambiguation_v2": 220, "strict_ne_external_style_safety_v2": 80} |                    0 |                                0 |                    0 | STAGE75B_TARGETED_RESIDUAL_BRIDGE_GENERATION_READY | False               | False                    | Stage75C runner integration plan for Stage75 bridge train-only append |

## Label counts

| label        |   count |
|:-------------|--------:|
| SUPPORT      |     480 |
| REFUTE       |     460 |
| NOT_ENTITLED |      80 |

## Family counts

| family                                  |   count |
|:----------------------------------------|--------:|
| support_entitlement_direct_recovery_v2  |     240 |
| refute_entitlement_direct_recovery_v2   |     220 |
| numeric_temporal_polarity_comparison_v2 |     260 |
| lexical_type_polarity_disambiguation_v2 |     220 |
| strict_ne_external_style_safety_v2      |      80 |

## Family-label counts

| family                                  | label        |   count |
|:----------------------------------------|:-------------|--------:|
| lexical_type_polarity_disambiguation_v2 | SUPPORT      |     110 |
| lexical_type_polarity_disambiguation_v2 | REFUTE       |     110 |
| lexical_type_polarity_disambiguation_v2 | NOT_ENTITLED |       0 |
| numeric_temporal_polarity_comparison_v2 | SUPPORT      |     130 |
| numeric_temporal_polarity_comparison_v2 | REFUTE       |     130 |
| numeric_temporal_polarity_comparison_v2 | NOT_ENTITLED |       0 |
| refute_entitlement_direct_recovery_v2   | SUPPORT      |       0 |
| refute_entitlement_direct_recovery_v2   | REFUTE       |     220 |
| refute_entitlement_direct_recovery_v2   | NOT_ENTITLED |       0 |
| strict_ne_external_style_safety_v2      | SUPPORT      |       0 |
| strict_ne_external_style_safety_v2      | REFUTE       |       0 |
| strict_ne_external_style_safety_v2      | NOT_ENTITLED |      80 |
| support_entitlement_direct_recovery_v2  | SUPPORT      |     240 |
| support_entitlement_direct_recovery_v2  | REFUTE       |       0 |
| support_entitlement_direct_recovery_v2  | NOT_ENTITLED |       0 |

## Checks

| check                         | pass   |
|:------------------------------|:-------|
| row_count_1020                | True   |
| label_support_480             | True   |
| label_refute_460              | True   |
| label_ne_80                   | True   |
| family_support_recovery_240   | True   |
| family_refute_recovery_220    | True   |
| family_numeric_temporal_260   | True   |
| family_lexical_type_220       | True   |
| family_strict_ne_80           | True   |
| no_pair_id_field              | True   |
| all_required_fields_present   | True   |
| schema_errors_zero            | True   |
| synthetic_only_all_true       | True   |
| external_text_used_all_false  | True   |
| external_label_used_all_false | True   |
| duplicate_id_zero             | True   |
| duplicate_claim_evidence_zero | True   |
| report_decision_ready         | True   |
| report_row_count_1020         | True   |
| report_training_false         | True   |
| report_external_eval_false    | True   |

## Schema errors preview

(none)

## Examples by family

| id                                                          | family                                  | label        | claim                                                                                              | evidence                                                                                         |
|:------------------------------------------------------------|:----------------------------------------|:-------------|:---------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| stage75b_ltpd2_same_surface_wrong_type_0001_support         | lexical_type_polarity_disambiguation_v2 | SUPPORT      | Kestrel Foundation is a publishing house.                                                          | As of 2018, Kestrel Foundation is officially registered as a publishing house.                   |
| stage75b_ntpc2_more_than_fewer_than_0001_support            | numeric_temporal_polarity_comparison_v2 | SUPPORT      | Ledgerfield Foundation reported more than 18126 containers in harbor cargo throughput during 2004. | Ledgerfield Foundation recorded 23308 containers in harbor cargo throughput during 2004.         |
| stage75b_redr2_type_mismatch_refute_0001                    | refute_entitlement_direct_recovery_v2   | REFUTE       | Paper Tide is a historical drama.                                                                  | As of 1980, Paper Tide is in fact a novel, not a historical drama.                               |
| stage75b_snes2_partial_evidence_missing_decisive_field_0001 | strict_ne_external_style_safety_v2      | NOT_ENTITLED | Rosalind Faircloth won the Merrow Institute Medal in 2018.                                         | Rosalind Faircloth was nominated for the Merrow Institute Medal in 2018.                         |
| stage75b_sedr2_entity_attribute_support_0001                | support_entitlement_direct_recovery_v2  | SUPPORT      | Ledgerfield Assembly is headquartered in Ridgemarsh.                                               | Ledgerfield Assembly's headquarters has been located in Ridgemarsh since it was founded in 1983. |

## Recommended next stage

Stage75C runner integration plan for Stage75 bridge train-only append
