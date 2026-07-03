# Stage97A - Half-size Numeric/Date Anti-NE Bridge

## Decision

`STAGE97A_HALF_SIZE_NUMERIC_DATE_ANTI_NE_BRIDGE_READY`

## Summary

| stage    | decision                                             | stage97a_out                                             | combined_stage80a_out                                    |   stage97a_n | stage97a_label_counts                             | stage97a_family_counts                                                                                                                                          | stage97a_subtype_counts                                                                                                      | stage97a_polarity_counts                  |   combined_n | combined_label_counts                              | combined_intervention_counts                                                                             |   parent_stage95a_overlap_intentional | target                                                                                                                  | source_policy                                                                                                                                      | intended_next_stage                                                                                                                                        |   expected_stage97b_bridge_count |   expected_stage97b_final_train_count | overlap_by_file                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | checks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|:---------|:-----------------------------------------------------|:---------------------------------------------------------|:---------------------------------------------------------|-------------:|:--------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|:------------------------------------------|-------------:|:---------------------------------------------------|:---------------------------------------------------------------------------------------------------------|--------------------------------------:|:------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------:|--------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage97A | STAGE97A_HALF_SIZE_NUMERIC_DATE_ANTI_NE_BRIDGE_READY | data/stage97a_half_size_numeric_date_antine_bridge.jsonl | data/stage97a_stage92a_combined_half_antine_bridge.jsonl |           80 | {"SUPPORT": 36, "REFUTE": 24, "NOT_ENTITLED": 20} | {"explicit_sufficient_support_with_ne_distractor": 36, "explicit_sufficient_refute_with_ne_distractor": 24, "matched_missing_decisive_predicate_guardrail": 20} | {"support_numeric_date_ne_distractor": 36, "refute_numeric_date_ne_distractor": 24, "ne_missing_numeric_date_predicate": 20} | {"SUPPORT": 36, "REFUTE": 24, "NONE": 20} |          320 | {"SUPPORT": 156, "REFUTE": 84, "NOT_ENTITLED": 80} | {"stage92a_support_preserving_counter_bridge": 240, "stage97a_half_size_numeric_date_antine_bridge": 80} |                                    80 | Half-size numeric/date anti-NE ablation to reduce Stage95A clean overpressure while retaining false_NE repair pressure. | Subset from synthetic/internal Stage95A only; Stage94 diagnostic and VitaminC external rows are checked for overlap but not used as training rows. | Stage97B clean-dev frozen recovery run with Stage57 + Stage66 + combined Stage92A+Stage97A via stage80a slot; no Stage83A; no Stage88A; no external first. |                             1560 |                                  4440 | {"data/controlled_v5_v3_without_time_swap.jsonl": 0, "data/stage57_nonleaking_external_bridge.jsonl": 0, "data/stage66_residual_bridge.jsonl": 0, "data/stage83a_ne_safety_only_bridge.jsonl": 0, "data/stage88a_balanced_entitlement_recovery_bridge.jsonl": 0, "data/stage92a_support_preserving_counter_bridge.jsonl": 0, "data/stage43b1_vitaminc_validation_sample1000.jsonl": 0, "data/stage84e2_vitaminc_validation_sample1000_exact_ood_schema.jsonl": 0, "results/stage94_new_false_ne_stage73_non_ne_stage92c_ne_examples.jsonl": 0, "results/stage94_repaired_false_ne_stage73_ne_stage92c_non_ne_examples.jsonl": 0, "results/stage94_persistent_false_ne_stage73_stage92c_both_ne_examples.jsonl": 0, "results/stage94_stage92c_false_ne_all_examples.jsonl": 0} | {"stage97a_row_count_80": true, "stage97a_support_36": true, "stage97a_refute_24": true, "stage97a_not_entitled_20": true, "stage97a_pairs_unique": true, "stage97a_parent_overlap_stage95a_80_intentional": true, "stage97a_no_exact_overlap_non_parent_checked_files": true, "stage97a_source_subset_from_stage95a": true, "stage97a_no_vitaminc_source": true, "combined_row_count_320": true, "combined_contains_stage92a_240": true, "combined_contains_stage97a_80": true, "combined_pairs_unique": true, "combined_support_156": true, "combined_refute_84": true, "combined_ne_80": true, "encode_label_tensors_stage97a_ok": true, "encode_label_tensors_combined_ok": true, "encode_mamba_records_stage97a_ok": true, "encode_mamba_records_combined_ok": true} |

## Stage97A label counts

| label        |   count |
|:-------------|--------:|
| NOT_ENTITLED |      20 |
| REFUTE       |      24 |
| SUPPORT      |      36 |

## Stage97A subtype counts

| subtype                            |   count |
|:-----------------------------------|--------:|
| ne_missing_numeric_date_predicate  |      20 |
| refute_numeric_date_ne_distractor  |      24 |
| support_numeric_date_ne_distractor |      36 |

## Combined Stage92A+Stage97A label counts

| label        |   count |
|:-------------|--------:|
| NOT_ENTITLED |      80 |
| REFUTE       |      84 |
| SUPPORT      |     156 |

## Non-parent exact-overlap checks

| file                                                                        |   exact_pair_overlap |
|:----------------------------------------------------------------------------|---------------------:|
| data/controlled_v5_v3_without_time_swap.jsonl                               |                    0 |
| data/stage57_nonleaking_external_bridge.jsonl                               |                    0 |
| data/stage66_residual_bridge.jsonl                                          |                    0 |
| data/stage83a_ne_safety_only_bridge.jsonl                                   |                    0 |
| data/stage88a_balanced_entitlement_recovery_bridge.jsonl                    |                    0 |
| data/stage92a_support_preserving_counter_bridge.jsonl                       |                    0 |
| data/stage43b1_vitaminc_validation_sample1000.jsonl                         |                    0 |
| data/stage84e2_vitaminc_validation_sample1000_exact_ood_schema.jsonl        |                    0 |
| results/stage94_new_false_ne_stage73_non_ne_stage92c_ne_examples.jsonl      |                    0 |
| results/stage94_repaired_false_ne_stage73_ne_stage92c_non_ne_examples.jsonl |                    0 |
| results/stage94_persistent_false_ne_stage73_stage92c_both_ne_examples.jsonl |                    0 |
| results/stage94_stage92c_false_ne_all_examples.jsonl                        |                    0 |

## Checks

| check                                              | pass   |
|:---------------------------------------------------|:-------|
| stage97a_row_count_80                              | True   |
| stage97a_support_36                                | True   |
| stage97a_refute_24                                 | True   |
| stage97a_not_entitled_20                           | True   |
| stage97a_pairs_unique                              | True   |
| stage97a_parent_overlap_stage95a_80_intentional    | True   |
| stage97a_no_exact_overlap_non_parent_checked_files | True   |
| stage97a_source_subset_from_stage95a               | True   |
| stage97a_no_vitaminc_source                        | True   |
| combined_row_count_320                             | True   |
| combined_contains_stage92a_240                     | True   |
| combined_contains_stage97a_80                      | True   |
| combined_pairs_unique                              | True   |
| combined_support_156                               | True   |
| combined_refute_84                                 | True   |
| combined_ne_80                                     | True   |
| encode_label_tensors_stage97a_ok                   | True   |
| encode_label_tensors_combined_ok                   | True   |
| encode_mamba_records_stage97a_ok                   | True   |
| encode_mamba_records_combined_ok                   | True   |
