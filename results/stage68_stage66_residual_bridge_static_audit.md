# Stage68 — Stage66 Residual Bridge Static Audit

## Decision

`STAGE68_STAGE66_RESIDUAL_BRIDGE_STATIC_AUDIT_READY`

## Summary

| stage   | decision                                           |   row_count | computed_ready   | stage67_audit_decision                     | stage66_design_decision                        |   stage57_row_count |   id_overlap_with_stage57_count |   claim_evidence_overlap_with_stage57_count | recommended_next_stage                            |
|:--------|:---------------------------------------------------|------------:|:-----------------|:-------------------------------------------|:-----------------------------------------------|--------------------:|--------------------------------:|--------------------------------------------:|:--------------------------------------------------|
| Stage68 | STAGE68_STAGE66_RESIDUAL_BRIDGE_STATIC_AUDIT_READY |         720 | True             | STAGE67_STAGE66_RESIDUAL_BRIDGE_DATA_READY | STAGE66_RESIDUAL_BRIDGE_EXPANSION_DESIGN_READY |                 520 |                               0 |                                           0 | Stage69 Stage57 + Stage66 bridge integration plan |

## Checks

| check                                  | pass   |
|:---------------------------------------|:-------|
| row_count_ok                           | True   |
| label_counts_ok                        | True   |
| family_counts_ok                       | True   |
| family_label_counts_ok                 | True   |
| required_fields_ok                     | True   |
| duplicate_ids_ok                       | True   |
| duplicate_claim_evidence_pairs_ok      | True   |
| label_mapping_ok                       | True   |
| axis_consistency_ok                    | True   |
| known_family_names_ok                  | True   |
| bridge_family_equals_family_ok         | True   |
| target_error_mapping_ok                | True   |
| generation_source_ok                   | True   |
| leakage_policy_ok                      | True   |
| forbidden_text_markers_ok              | True   |
| no_id_overlap_with_stage57             | True   |
| no_claim_evidence_overlap_with_stage57 | True   |
| stage67_audit_ready                    | True   |
| stage66_design_ready                   | True   |

## Label counts

| final_label   |   count |   expected |
|:--------------|--------:|-----------:|
| NOT_ENTITLED  |      40 |         40 |
| REFUTE        |     320 |        320 |
| SUPPORT       |     360 |        360 |

## Family counts

| stage66_family                      |   count |   expected |
|:------------------------------------|--------:|-----------:|
| numeric_temporal_comparison_bridge  |     120 |        120 |
| polarity_disambiguation_bridge      |     200 |        200 |
| refute_entitlement_recovery_bridge  |     160 |        160 |
| strict_ne_frame_safety_bridge       |      40 |         40 |
| support_entitlement_recovery_bridge |     200 |        200 |

## Family x label counts

| stage66_family                      | final_label   |   count |   expected |
|:------------------------------------|:--------------|--------:|-----------:|
| numeric_temporal_comparison_bridge  | REFUTE        |      60 |         60 |
| numeric_temporal_comparison_bridge  | SUPPORT       |      60 |         60 |
| polarity_disambiguation_bridge      | REFUTE        |     100 |        100 |
| polarity_disambiguation_bridge      | SUPPORT       |     100 |        100 |
| refute_entitlement_recovery_bridge  | REFUTE        |     160 |        160 |
| strict_ne_frame_safety_bridge       | NOT_ENTITLED  |      40 |         40 |
| support_entitlement_recovery_bridge | SUPPORT       |     200 |        200 |

## Target error counts

| stage66_target_error                   |   count |
|:---------------------------------------|--------:|
| REFUTE_SUPPORT_polarity_confusion      |     200 |
| false_NE_on_REFUTE                     |     160 |
| false_NE_on_SUPPORT                    |     200 |
| false_SUPPORT_or_REFUTE_on_NE          |      40 |
| numeric_temporal_comparative_residuals |     120 |

## Subtype counts

| stage66_subtype                          |   count |
|:-----------------------------------------|--------:|
| album_film_work_support                  |      33 |
| before_after_pair                        |      34 |
| before_after_refute                      |      20 |
| before_after_support                     |      20 |
| date_release_birth_support               |      33 |
| direct_attribute_support                 |      34 |
| founded_released_born_date_pair          |      32 |
| is_a_vs_is_not_a_type_pair               |      32 |
| more_than_less_than_pair                 |      34 |
| number_word_equivalence_support          |      20 |
| numeric_threshold_refute                 |      20 |
| partial_attribute_without_entitlement_ne |      10 |
| positive_review_vs_negative_review_pair  |      34 |
| profession_role_support                  |      34 |
| quantity_paraphrase_support              |      33 |
| related_entity_distractor_ne             |      10 |
| review_sentiment_support                 |      33 |
| same_domain_missing_predicate_ne         |      10 |
| same_value_vs_conflicting_value_pair     |      34 |
| wrong_creator_director_refute            |      26 |
| wrong_date_refute                        |      27 |
| wrong_location_refute                    |      27 |
| wrong_numeric_threshold_refute           |      26 |
| wrong_role_refute                        |      27 |
| wrong_subject_same_predicate_ne          |      10 |
| wrong_type_refute                        |      27 |
| year_exact_match_support                 |      20 |
| year_mismatch_refute                     |      20 |

## Source files

- Stage66 bridge JSONL: `data/stage66_residual_bridge.jsonl`
- Stage67 generation audit: `results/stage67_stage66_residual_bridge_generation_audit.json`
- Stage66 design: `results/stage66_residual_bridge_expansion_design.json`
- Stage57 bridge JSONL: `data/stage57_nonleaking_external_bridge.jsonl`

## Recommended next stage

Stage69 Stage57 + Stage66 bridge integration plan
