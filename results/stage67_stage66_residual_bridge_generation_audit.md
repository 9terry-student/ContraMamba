# Stage67: Stage66 Residual Bridge Generation Audit

**Decision:** `STAGE67_STAGE66_RESIDUAL_BRIDGE_DATA_READY`

- Source design JSON: `/kaggle/working/ContraMamba/results/stage66_residual_bridge_expansion_design.json`
- Output JSONL: `/kaggle/working/ContraMamba/data/stage66_residual_bridge.jsonl`
- Seed: 660067
- Total rows: 720 (expected 720)

## Counts by label

| Label | Count | Expected |
|---|---|---|
| SUPPORT | 360 | 360 |
| REFUTE | 320 | 320 |
| NOT_ENTITLED | 40 | 40 |

## Polarity label encoder compatibility

- `polarity_label_encoder_compatible`: True
- `encoder_mapping_used` (POLARITY_LABEL_TO_ID keys): {'NONE': 0, 'REFUTE': 1, 'SUPPORT': 2}

| polarity_label | Count |
|---|---|
| NONE | 40 |
| REFUTE | 320 |
| SUPPORT | 360 |

## Counts by bridge family

| Family | Count |
|---|---|
| support_entitlement_recovery_bridge | 200 |
| refute_entitlement_recovery_bridge | 160 |
| polarity_disambiguation_bridge | 200 |
| numeric_temporal_comparison_bridge | 120 |
| strict_ne_frame_safety_bridge | 40 |

## Counts by bridge family x label

| Family | SUPPORT | REFUTE | NOT_ENTITLED |
|---|---|---|---|
| support_entitlement_recovery_bridge | 200 | 0 | 0 |
| refute_entitlement_recovery_bridge | 0 | 160 | 0 |
| polarity_disambiguation_bridge | 100 | 100 | 0 |
| numeric_temporal_comparison_bridge | 60 | 60 | 0 |
| strict_ne_frame_safety_bridge | 0 | 0 | 40 |

## Counts by subtype

| Subtype | Count |
|---|---|
| album_film_work_support | 33 |
| before_after_pair | 34 |
| before_after_refute | 20 |
| before_after_support | 20 |
| date_release_birth_support | 33 |
| direct_attribute_support | 34 |
| founded_released_born_date_pair | 32 |
| is_a_vs_is_not_a_type_pair | 32 |
| more_than_less_than_pair | 34 |
| number_word_equivalence_support | 20 |
| numeric_threshold_refute | 20 |
| partial_attribute_without_entitlement_ne | 10 |
| positive_review_vs_negative_review_pair | 34 |
| profession_role_support | 34 |
| quantity_paraphrase_support | 33 |
| related_entity_distractor_ne | 10 |
| review_sentiment_support | 33 |
| same_domain_missing_predicate_ne | 10 |
| same_value_vs_conflicting_value_pair | 34 |
| wrong_creator_director_refute | 26 |
| wrong_date_refute | 27 |
| wrong_location_refute | 27 |
| wrong_numeric_threshold_refute | 26 |
| wrong_role_refute | 27 |
| wrong_subject_same_predicate_ne | 10 |
| wrong_type_refute | 27 |
| year_exact_match_support | 20 |
| year_mismatch_refute | 20 |

## Checks

| Check | Passed |
|---|---|
| required_fields_present | True |
| duplicate_id_check | True |
| duplicate_claim_evidence_pair_check | True |
| label_mapping_check | True |
| axis_consistency_check | True |
| polarity_label_encoder_compatibility_check | True |
| family_and_label_count_check | True |
| bridge_family_identity_check | True |
| forbidden_marker_scan | True |
| design_plan_match_check | True |

## Leakage checks

- `synthetic_only`: True
- `no_vitaminc_text_or_labels_used`: True
- `stage65_residual_samples_used_as_templates`: False
- `time_swap_used`: False
- `external_metrics_used_for_threshold_tuning`: False

## Notes

- This dataset is synthetic training/diagnostic data only. It is NOT an external evaluation result and must not be reported as VitaminC or any other external-benchmark metric.
- Stage65 residual error taxonomy/counts were used only to set family quotas in the Stage66 design; no Stage65 residual sample text, VitaminC text, or VitaminC labels were read or used to produce any row in this file.
- This dataset must not be mixed with corrupted time_swap rows from data/controlled_v5_v3.jsonl.
- Per the Stage66 design, this bridge is intended to be appended to the training split only, after a clean main split (Stage69 scope); this script performs no such integration and no training.

## Recommended next stage

- **Stage68**: static audit of generated Stage66 residual bridge — Validate schema, label balance, and non-leakage of this bridge dataset before any training uses it.
