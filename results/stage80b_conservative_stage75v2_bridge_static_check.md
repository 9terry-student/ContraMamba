# Stage80B - Conservative Stage75v2 Bridge Static Check

## Decision

`STAGE80B_CONSERVATIVE_STAGE75V2_BRIDGE_STATIC_CHECK_READY`

## Summary

| field | value |
|---|---|
| stage | Stage80B_static_check |
| decision | STAGE80B_CONSERVATIVE_STAGE75V2_BRIDGE_STATIC_CHECK_READY |
| jsonl | C:/Users/Home1/Desktop/ContraMamba/data/stage80a_conservative_stage75v2_bridge.jsonl |
| report_json | C:/Users/Home1/Desktop/ContraMamba/results/stage80b_conservative_stage75v2_bridge_generation_report.json |
| report_md | C:/Users/Home1/Desktop/ContraMamba/results/stage80b_conservative_stage75v2_bridge_generation_report.md |
| row_count | 500 |
| label_counts | {'SUPPORT': 170, 'REFUTE': 170, 'NOT_ENTITLED': 160} |
| family_counts | {'numeric_temporal_polarity_repair_v2_conservative': 180, 'lexical_type_polarity_repair_v2_conservative': 160, 'strict_ne_false_support_safety_v2': 160} |
| duplicate_id_count | 0 |
| duplicate_claim_evidence_count | 0 |
| schema_error_count | 0 |
| report_decision | STAGE80B_CONSERVATIVE_STAGE75V2_BRIDGE_GENERATION_READY |
| training_executed | False |
| external_eval_executed | False |
| recommended_next_stage | Stage80C runner integration plan for Stage80A conservative Stage75v2 bridge |

## Label counts

| label | count |
|---|---|
| SUPPORT | 170 |
| REFUTE | 170 |
| NOT_ENTITLED | 160 |

## Family counts

| family | count |
|---|---|
| numeric_temporal_polarity_repair_v2_conservative | 180 |
| lexical_type_polarity_repair_v2_conservative | 160 |
| strict_ne_false_support_safety_v2 | 160 |

## Family-label counts

| family | label | count |
|---|---|---|
| numeric_temporal_polarity_repair_v2_conservative | SUPPORT | 90 |
| numeric_temporal_polarity_repair_v2_conservative | REFUTE | 90 |
| numeric_temporal_polarity_repair_v2_conservative | NOT_ENTITLED | 0 |
| lexical_type_polarity_repair_v2_conservative | SUPPORT | 80 |
| lexical_type_polarity_repair_v2_conservative | REFUTE | 80 |
| lexical_type_polarity_repair_v2_conservative | NOT_ENTITLED | 0 |
| strict_ne_false_support_safety_v2 | SUPPORT | 0 |
| strict_ne_false_support_safety_v2 | REFUTE | 0 |
| strict_ne_false_support_safety_v2 | NOT_ENTITLED | 160 |

## Checks

| check | pass |
|---|---|
| required_files_exist | True |
| row_count_500 | True |
| label_support_170 | True |
| label_refute_170 | True |
| label_ne_160 | True |
| family_numeric_temporal_repair_180 | True |
| family_lexical_type_repair_160 | True |
| family_strict_ne_safety_160 | True |
| all_required_fields_present | True |
| schema_errors_zero | True |
| duplicate_id_zero | True |
| duplicate_claim_evidence_zero | True |
| synthetic_only_all_true | True |
| external_text_used_all_false | True |
| external_label_used_all_false | True |
| no_pair_id_field | True |
| no_forbidden_external_source_markers | True |
| report_decision_ready | True |
| report_row_count_500 | True |
| report_training_false | True |
| report_external_eval_false | True |

## Schema errors preview

(none)

## Examples by family

| id | family | label | claim | evidence |
|---|---|---|---|---|
| stage80b_ltpr2c_same_surface_wrong_type_0001_support | lexical_type_polarity_repair_v2_conservative | SUPPORT | Everwood Alliance is a trade syndicate. | As of 2016, Everwood Alliance is officially registered as a trade syndicate. |
| stage80b_ntpr2c_more_than_fewer_than_0001_support | numeric_temporal_polarity_repair_v2_conservative | SUPPORT | Everwood Federation reported more than 10369 containers in port container throughput during 2015. | Everwood Federation recorded 10617 containers in port container throughput during 2015. |
| stage80b_snfs2_partial_evidence_missing_decisive_field_0001 | strict_ne_false_support_safety_v2 | NOT_ENTITLED | Xara Oakhurst won the Amberline Registry Prize in 2017. | Xara Oakhurst was nominated for the Amberline Registry Prize in 2017. |

## Recommended next stage

Stage80C runner integration plan for Stage80A conservative Stage75v2 bridge
