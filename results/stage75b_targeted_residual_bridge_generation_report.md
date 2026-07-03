# Stage75B: Targeted Residual Bridge Generation Report

**Decision:** `STAGE75B_TARGETED_RESIDUAL_BRIDGE_GENERATION_READY`

## Summary

| field | value |
|---|---|
| stage | Stage75B |
| decision | STAGE75B_TARGETED_RESIDUAL_BRIDGE_GENERATION_READY |
| source_stage75a | /kaggle/working/ContraMamba/results/stage75a_targeted_bridge_design_plan.json |
| source_stage74_aggregate_only | True |
| output_jsonl | /kaggle/working/ContraMamba/data/stage75_targeted_residual_bridge.jsonl |
| seed | 75075002 |
| row_count | 1020 (expected 1020) |
| duplicate_id_count | 0 |
| duplicate_claim_evidence_count | 0 |
| training_executed | False |
| external_eval_executed | False |
| recommended_next_stage | Stage75C static audit and runner integration plan |

## Label counts

| label | count | expected |
|---|---|---|
| SUPPORT | 480 | 480 |
| REFUTE | 460 | 460 |
| NOT_ENTITLED | 80 | 80 |

## Family counts

| family | count | planned | target_error |
|---|---|---|---|
| support_entitlement_direct_recovery_v2 | 240 | 240 | false_NE_on_SUPPORT |
| refute_entitlement_direct_recovery_v2 | 220 | 220 | false_NE_on_REFUTE |
| numeric_temporal_polarity_comparison_v2 | 260 | 260 | wrong_polarity_SUPPORT_to_REFUTE + wrong_polarity_REFUTE_to_SUPPORT |
| lexical_type_polarity_disambiguation_v2 | 220 | 220 | wrong_polarity_SUPPORT_to_REFUTE + wrong_polarity_REFUTE_to_SUPPORT |
| strict_ne_external_style_safety_v2 | 80 | 80 | false_SUPPORT_on_NE + false_REFUTE_on_NE |

## Family-label counts

| family | SUPPORT | REFUTE | NOT_ENTITLED |
|---|---|---|---|
| support_entitlement_direct_recovery_v2 | 240 | 0 | 0 |
| refute_entitlement_direct_recovery_v2 | 0 | 220 | 0 |
| numeric_temporal_polarity_comparison_v2 | 130 | 130 | 0 |
| lexical_type_polarity_disambiguation_v2 | 110 | 110 | 0 |
| strict_ne_external_style_safety_v2 | 0 | 0 | 80 |

## Checks

| check | passed |
|---|---|
| row_count_check | True |
| required_fields_present | True |
| duplicate_id_check | True |
| duplicate_claim_evidence_pair_check | True |
| final_label_value_check | True |
| polarity_label_value_check | True |
| axis_consistency_check | True |
| polarity_label_encoder_compatibility_check | True |
| family_and_label_count_check | True |
| no_pair_id_check | True |
| synthetic_only_check | True |
| external_text_used_check | True |
| external_label_used_check | True |
| forbidden_marker_scan | True |
| design_plan_match_check | True |

## Example rows by family

| family | subtype | final_label | target_error_type | claim | evidence |
|---|---|---|---|---|---|
| support_entitlement_direct_recovery_v2 | entity_attribute_support | SUPPORT | false_NE_on_SUPPORT | Ledgerfield Assembly is headquartered in Ridgemarsh. | Ledgerfield Assembly's headquarters has been located in Ridgemarsh since it was founded in 1983. |
| support_entitlement_direct_recovery_v2 | role_work_title_membership_support | SUPPORT | false_NE_on_SUPPORT | Faela Harrowgate is the chief archivist of Merrow Institute. | Faela Harrowgate has served as Merrow Institute's chief archivist since 2006. |
| support_entitlement_direct_recovery_v2 | date_numeric_support | SUPPORT | false_NE_on_SUPPORT | Selene Kirrin was born in 1956. | Selene Kirrin was born in 1956, in Ridgemarsh. |
| support_entitlement_direct_recovery_v2 | paraphrased_evidence_support | SUPPORT | false_NE_on_SUPPORT | Hallowgate Guild sold more than 3342 ceramic filters. | Hallowgate Guild moved 3892 units of ceramic filters, comfortably above the 3342 mark. |
| refute_entitlement_direct_recovery_v2 | type_mismatch_refute | REFUTE | false_NE_on_REFUTE | Paper Tide is a historical drama. | As of 1980, Paper Tide is in fact a novel, not a historical drama. |
| refute_entitlement_direct_recovery_v2 | wrong_date_count_refute | REFUTE | false_NE_on_REFUTE | Kestrel Brackenridge was born in 1994. | Kestrel Brackenridge was actually born in 2010, not 1994. |
| refute_entitlement_direct_recovery_v2 | exclusive_only_refute | REFUTE | false_NE_on_REFUTE | Thalia Holt works exclusively as a glassblower. | According to a 2006 directory listing, Thalia Holt works as both a glassblower and a cellist. |
| refute_entitlement_direct_recovery_v2 | wrong_entity_work_role_refute | REFUTE | false_NE_on_REFUTE | Daren Corvel is the chief archivist of Merrow Trust. | Odessa Osprey, not Daren Corvel, is the chief archivist of Merrow Trust. |
| numeric_temporal_polarity_comparison_v2 | more_than_fewer_than | SUPPORT | wrong_polarity_SUPPORT_to_REFUTE | Ledgerfield Foundation reported more than 18126 containers in harbor cargo throughput during 2004. | Ledgerfield Foundation recorded 23308 containers in harbor cargo throughput during 2004. |
| numeric_temporal_polarity_comparison_v2 | more_than_fewer_than | REFUTE | wrong_polarity_REFUTE_to_SUPPORT | Ledgerfield Foundation reported fewer than 18126 containers in harbor cargo throughput during 2004. | Ledgerfield Foundation recorded 23308 containers in harbor cargo throughput during 2004. |
| numeric_temporal_polarity_comparison_v2 | at_least_under | SUPPORT | wrong_polarity_SUPPORT_to_REFUTE | Luma Society recorded at least 12353 visitors in reading room visitor count during 2006. | Luma Society recorded 13810 visitors in reading room visitor count during 2006. |
| numeric_temporal_polarity_comparison_v2 | at_least_under | REFUTE | wrong_polarity_REFUTE_to_SUPPORT | Luma Society recorded under 12353 visitors in reading room visitor count during 2006. | Luma Society recorded 13810 visitors in reading room visitor count during 2006. |
| numeric_temporal_polarity_comparison_v2 | before_after | SUPPORT | wrong_polarity_SUPPORT_to_REFUTE | The harbor festival in Silverfen took place before 2029. | The harbor festival in Silverfen took place in 2009. |
| numeric_temporal_polarity_comparison_v2 | before_after | REFUTE | wrong_polarity_REFUTE_to_SUPPORT | The harbor festival in Silverfen took place after 2029. | The harbor festival in Silverfen took place in 2009. |
| numeric_temporal_polarity_comparison_v2 | exact_threshold_contradiction | SUPPORT | wrong_polarity_SUPPORT_to_REFUTE | Brightwater Foundation recorded exactly 325047 dollars in annual grant funding during 1997. | Brightwater Foundation recorded exactly 325047 dollars in annual grant funding during 1997. |
| numeric_temporal_polarity_comparison_v2 | exact_threshold_contradiction | REFUTE | wrong_polarity_REFUTE_to_SUPPORT | Brightwater Foundation recorded exactly 325137 dollars in annual grant funding during 1997. | Brightwater Foundation recorded exactly 325047 dollars in annual grant funding during 1997. |
| lexical_type_polarity_disambiguation_v2 | same_surface_wrong_type | SUPPORT | wrong_polarity_SUPPORT_to_REFUTE | Kestrel Foundation is a publishing house. | As of 2018, Kestrel Foundation is officially registered as a publishing house. |
| lexical_type_polarity_disambiguation_v2 | same_surface_wrong_type | REFUTE | wrong_polarity_REFUTE_to_SUPPORT | Kestrel Foundation is a trade guild. | As of 2018, Kestrel Foundation is officially registered as a publishing house. |
| lexical_type_polarity_disambiguation_v2 | person_org_place_mismatch | SUPPORT | wrong_polarity_SUPPORT_to_REFUTE | Percy Penrose is a choreographer. | Percy Penrose is a choreographer who resides in Hallowgate. |
| lexical_type_polarity_disambiguation_v2 | person_org_place_mismatch | REFUTE | wrong_polarity_REFUTE_to_SUPPORT | Percy Penrose is a city located near Hallowgate. | Percy Penrose is a choreographer who resides in Hallowgate. |
| lexical_type_polarity_disambiguation_v2 | work_title_vs_creator | SUPPORT | wrong_polarity_SUPPORT_to_REFUTE | Isolde Merrow wrote Driftwood Accord. | Driftwood Accord, a documentary film, was written by Isolde Merrow and performed by Arlo Thackeray. |
| lexical_type_polarity_disambiguation_v2 | work_title_vs_creator | REFUTE | wrong_polarity_REFUTE_to_SUPPORT | Arlo Thackeray wrote Driftwood Accord. | Driftwood Accord, a documentary film, was written by Isolde Merrow and performed by Arlo Thackeray. |
| lexical_type_polarity_disambiguation_v2 | category_membership_vs_lexical_overlap | SUPPORT | wrong_polarity_SUPPORT_to_REFUTE | Hallowgate Consortium belongs to the coastal fisheries council category. | As of 2022, Hallowgate Consortium is classified under the coastal fisheries council category within the regional registry. |
| lexical_type_polarity_disambiguation_v2 | category_membership_vs_lexical_overlap | REFUTE | wrong_polarity_REFUTE_to_SUPPORT | Hallowgate Consortium belongs to the coastal forestry council category. | As of 2022, Hallowgate Consortium is classified under the coastal fisheries council category within the regional registry. |
| strict_ne_external_style_safety_v2 | partial_evidence_missing_decisive_field | NOT_ENTITLED | false_SUPPORT_on_NE + false_REFUTE_on_NE | Rosalind Faircloth won the Merrow Institute Medal in 2018. | Rosalind Faircloth was nominated for the Merrow Institute Medal in 2018. |
| strict_ne_external_style_safety_v2 | conjunction_only_one_conjunct_supported | NOT_ENTITLED | false_SUPPORT_on_NE + false_REFUTE_on_NE | Joran Delacourt is both a cartographer and a lexicographer. | Joran Delacourt has worked as a cartographer since 2003. |
| strict_ne_external_style_safety_v2 | entity_present_predicate_absent | NOT_ENTITLED | false_SUPPORT_on_NE + false_REFUTE_on_NE | Dessa Jessup founded Ashcombe Institute. | Dessa Jessup is a longtime member of Ashcombe Institute. |
| strict_ne_external_style_safety_v2 | near_threshold_numeric_insufficiency | NOT_ENTITLED | false_SUPPORT_on_NE + false_REFUTE_on_NE | Thistlemoor Archive reported more than 9329 scans in archive digitization count. | Thistlemoor Archive reported activity in archive digitization count near 9329 scans, without confirming whether the figure exceeded that amount. |

## Leakage checks

- `synthetic_only`: True
- `no_vitaminc_text_or_labels_used`: True
- `no_stage74_example_claim_evidence_text_used`: True
- `stage74_used_as_aggregate_motivation_only`: True
- `external_metrics_used_for_threshold_tuning`: False
- `training_executed_by_this_script`: False
- `external_eval_executed_by_this_script`: False
- `forbidden_marker_scan.passed`: True
- `forbidden_marker_scan.hit_count`: 0

## Notes

- This dataset is synthetic training/diagnostic data only. It is NOT an external evaluation result and must not be reported as VitaminC or any other external-benchmark metric.
- Stage74 residual external error counts (false_NE_total=323, polarity_error_total=244, false_entitlement_total=80) were used only to set family quotas in the Stage75A design; no Stage74 residual example claim/evidence text, VitaminC text, or VitaminC labels were read or used to produce any row in this file.
- No field named 'pair_id' is emitted. Optional grouping metadata for the two polarity-comparison families uses 'bridge_pair_id' instead, so this bridge cannot be accidentally swept into the intervention pairwise-loss grouping path that keys off 'pair_id'.
- This script performs no training, no smoke run, no mini-run, no full run, no OOD or external evaluation, and does not modify scripts/train_controlled_v6b_minimal.py, existing Stage57/66 data, or any existing Stage73/74/75A report.

## Recommended next stage

- Stage75C static audit and runner integration plan
