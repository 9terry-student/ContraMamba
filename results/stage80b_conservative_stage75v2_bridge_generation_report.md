# Stage80B: Conservative Stage75v2 Bridge Generation Report

**Decision:** `STAGE80B_CONSERVATIVE_STAGE75V2_BRIDGE_GENERATION_READY`

## Summary

| field | value |
|---|---|
| stage | Stage80B |
| decision | STAGE80B_CONSERVATIVE_STAGE75V2_BRIDGE_GENERATION_READY |
| design_source | Stage80A |
| source_design_json | C:/Users/Home1/Desktop/ContraMamba/results/stage80a_conservative_stage75v2_design_plan.json |
| output_jsonl | C:/Users/Home1/Desktop/ContraMamba/data/stage80a_conservative_stage75v2_bridge.jsonl |
| seed | 80080002 |
| row_count | 500 (expected 500) |
| duplicate_id_count | 0 |
| duplicate_claim_evidence_count | 0 |
| pair_id_required | False |
| training_executed | False |
| external_eval_executed | False |
| recommended_next_stage | Stage80C runner integration plan for Stage80A conservative Stage75v2 bridge |

## Label counts

| label | count | expected |
|---|---|---|
| SUPPORT | 170 | 170 |
| REFUTE | 170 | 170 |
| NOT_ENTITLED | 160 | 160 |

## Family counts

| family | count | planned | purpose |
|---|---|---|---|
| numeric_temporal_polarity_repair_v2_conservative | 180 | 180 | Retain Stage75's useful polarity repair signal while keeping support/refute balanced. |
| lexical_type_polarity_repair_v2_conservative | 160 | 160 | Retain type/predicate polarity disambiguation but reduce volume from Stage75. |
| strict_ne_false_support_safety_v2 | 160 | 160 | Counter the observed false SUPPORT and false entitlement increase after Stage75. |

## Excluded families

- support_entitlement_direct_recovery_v2
- refute_entitlement_direct_recovery_v2

## Family-label counts

| family | SUPPORT | REFUTE | NOT_ENTITLED |
|---|---|---|---|
| numeric_temporal_polarity_repair_v2_conservative | 90 | 90 | 0 |
| lexical_type_polarity_repair_v2_conservative | 80 | 80 | 0 |
| strict_ne_false_support_safety_v2 | 0 | 0 | 160 |

## Checks

| check | passed |
|---|---|
| row_count_check | True |
| required_fields_present | True |
| duplicate_id_check | True |
| duplicate_claim_evidence_pair_check | True |
| final_label_value_check | True |
| final_label_id_consistency_check | True |
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

| family | subtype | final_label | claim | evidence |
|---|---|---|---|---|
| numeric_temporal_polarity_repair_v2_conservative | more_than_fewer_than | SUPPORT | Everwood Federation reported more than 10369 containers in port container throughput during 2015. | Everwood Federation recorded 10617 containers in port container throughput during 2015. |
| numeric_temporal_polarity_repair_v2_conservative | more_than_fewer_than | REFUTE | Everwood Federation reported fewer than 10369 containers in port container throughput during 2015. | Everwood Federation recorded 10617 containers in port container throughput during 2015. |
| numeric_temporal_polarity_repair_v2_conservative | at_least_under | SUPPORT | Everwood Bureau recorded at least 89 millimeters in monthly snowfall total during 2006. | Everwood Bureau recorded 109 millimeters in monthly snowfall total during 2006. |
| numeric_temporal_polarity_repair_v2_conservative | at_least_under | REFUTE | Everwood Bureau recorded under 89 millimeters in monthly snowfall total during 2006. | Everwood Bureau recorded 109 millimeters in monthly snowfall total during 2006. |
| numeric_temporal_polarity_repair_v2_conservative | before_after | SUPPORT | The trade conclave in Oldbarrow took place before 1945. | The trade conclave in Oldbarrow took place in 1934. |
| numeric_temporal_polarity_repair_v2_conservative | before_after | REFUTE | The trade conclave in Oldbarrow took place after 1945. | The trade conclave in Oldbarrow took place in 1934. |
| numeric_temporal_polarity_repair_v2_conservative | exact_threshold_contradiction | SUPPORT | Amberline Alliance recorded exactly 2615 kiloliters in municipal water usage during 1999. | Amberline Alliance recorded exactly 2615 kiloliters in municipal water usage during 1999. |
| numeric_temporal_polarity_repair_v2_conservative | exact_threshold_contradiction | REFUTE | Amberline Alliance recorded exactly 2705 kiloliters in municipal water usage during 1999. | Amberline Alliance recorded exactly 2615 kiloliters in municipal water usage during 1999. |
| lexical_type_polarity_repair_v2_conservative | same_surface_wrong_type | SUPPORT | Everwood Alliance is a trade syndicate. | As of 2016, Everwood Alliance is officially registered as a trade syndicate. |
| lexical_type_polarity_repair_v2_conservative | same_surface_wrong_type | REFUTE | Everwood Alliance is a registry. | As of 2016, Everwood Alliance is officially registered as a trade syndicate. |
| lexical_type_polarity_repair_v2_conservative | person_org_place_mismatch | SUPPORT | Wystan Juniper is a bookbinder. | Wystan Juniper is a bookbinder who resides in Gullwick. |
| lexical_type_polarity_repair_v2_conservative | person_org_place_mismatch | REFUTE | Wystan Juniper is a town located near Gullwick. | Wystan Juniper is a bookbinder who resides in Gullwick. |
| lexical_type_polarity_repair_v2_conservative | work_title_vs_creator | SUPPORT | Casimir Ambercross wrote Silver Ledger. | Silver Ledger, a board game expansion, was written by Casimir Ambercross and performed by Casimir Kingsley. |
| lexical_type_polarity_repair_v2_conservative | work_title_vs_creator | REFUTE | Casimir Kingsley wrote Silver Ledger. | Silver Ledger, a board game expansion, was written by Casimir Ambercross and performed by Casimir Kingsley. |
| lexical_type_polarity_repair_v2_conservative | category_membership_vs_lexical_overlap | SUPPORT | Ironbrook Union belongs to the chamber music circle category. | As of 1989, Ironbrook Union is classified under the chamber music circle category within the regional registry. |
| lexical_type_polarity_repair_v2_conservative | category_membership_vs_lexical_overlap | REFUTE | Ironbrook Union belongs to the chamber theater circle category. | As of 1989, Ironbrook Union is classified under the chamber music circle category within the regional registry. |
| strict_ne_false_support_safety_v2 | partial_evidence_missing_decisive_field | NOT_ENTITLED | Xara Oakhurst won the Amberline Registry Prize in 2017. | Xara Oakhurst was nominated for the Amberline Registry Prize in 2017. |
| strict_ne_false_support_safety_v2 | conjunction_only_one_conjunct_supported | NOT_ENTITLED | Ysolde Emberly is both a volcanologist and a clockmaker. | Ysolde Emberly has worked as a volcanologist since 1993. |
| strict_ne_false_support_safety_v2 | entity_present_predicate_absent | NOT_ENTITLED | Norah Ashworth founded Amberline Bureau. | Norah Ashworth is a longtime member of Amberline Bureau. |
| strict_ne_false_support_safety_v2 | near_threshold_numeric_insufficiency | NOT_ENTITLED | Copperfield Bureau reported more than 72 millimeters in monthly snowfall total. | Copperfield Bureau reported activity in monthly snowfall total near 72 millimeters, without confirming whether the figure exceeded that amount. |
| strict_ne_false_support_safety_v2 | related_entity_mentioned_not_proving_claim | NOT_ENTITLED | Faelan Blackwood is the founding director of Copperfield Union. | Ravel Whitlock, a colleague of Faelan Blackwood at Copperfield Union in Larkspur Reach, discussed Copperfield Union's early history in a recent interview. |

## Leakage checks

- `synthetic_only`: True
- `no_vitaminc_text_or_labels_used`: True
- `no_prior_stage_example_claim_evidence_text_used`: True
- `stage74_through_stage79_used_as_aggregate_motivation_only`: True
- `external_metrics_used_for_threshold_tuning`: False
- `training_executed_by_this_script`: False
- `external_eval_executed_by_this_script`: False
- `forbidden_marker_scan.passed`: True
- `forbidden_marker_scan.hit_count`: 0

## Notes

- This dataset is synthetic training/diagnostic data only. It is NOT an external evaluation result and must not be reported as VitaminC or any other external-benchmark metric.
- Stage77/Stage78/Stage79 aggregate diagnostics (external macro-F1 drop, reduced polarity_error_total, reduced false_refute_total, increased false_support_total, flat false_ne_total) were used only to set the Stage80A family taxonomy and quotas; no prior-stage example claim/evidence text, VitaminC text, or VitaminC labels were read or used to produce any row in this file.
- The two Stage75 'direct recovery' families (support_entitlement_direct_recovery_v2, refute_entitlement_direct_recovery_v2) are intentionally excluded from this bridge per the Stage80A design.
- No field named 'pair_id' is emitted. Optional grouping metadata for the two polarity-repair families uses 'bridge_pair_id' instead, so this bridge cannot be accidentally swept into the intervention pairwise-loss grouping path that keys off 'pair_id'.
- This script performs no training, no smoke run, no mini-run, no full run, no OOD or external evaluation, and does not modify scripts/train_controlled_v6b_minimal.py, or any existing Stage57 / Stage66 / Stage75 / Stage76 / Stage77 / Stage78 / Stage79 file.

## Recommended next stage

- Stage80C runner integration plan for Stage80A conservative Stage75v2 bridge
