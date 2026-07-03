# Stage99A - SUPPORT Floor Micro Bridge

## Decision

`STAGE99A_SUPPORT_FLOOR_MICRO_BRIDGE_READY`

## Summary

| stage    | decision                                  | stage99a_out                                   | combined_stage80a_out                                               |   stage99a_n | stage99a_label_counts               | stage99a_family_counts                                                           | stage99a_subtype_counts                                                                | stage99a_polarity_counts    |   combined_n | combined_label_counts                              | combined_intervention_counts                                                                                                                        | target                                                                                                                                                       | source_policy                                                                                                                       | intended_next_stage                                                                                                                                                 |   expected_stage99b_bridge_count |   expected_stage99b_final_train_count | overlap_by_file                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | checks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|:---------|:------------------------------------------|:-----------------------------------------------|:--------------------------------------------------------------------|-------------:|:------------------------------------|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|:----------------------------|-------------:|:---------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------:|--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage99A | STAGE99A_SUPPORT_FLOOR_MICRO_BRIDGE_READY | data/stage99a_support_floor_micro_bridge.jsonl | data/stage99a_stage92a_stage97a_combined_support_floor_bridge.jsonl |           32 | {"SUPPORT": 16, "NOT_ENTITLED": 16} | {"support_floor_micro_numeric_date": 16, "support_floor_micro_ne_guardrail": 16} | {"support_floor_verified_numeric_date": 16, "ne_guardrail_missing_verified_count": 16} | {"SUPPORT": 16, "NONE": 16} |          352 | {"SUPPORT": 172, "REFUTE": 84, "NOT_ENTITLED": 96} | {"stage92a_support_preserving_counter_bridge": 240, "stage97a_half_size_numeric_date_antine_bridge": 80, "stage99a_support_floor_micro_bridge": 32} | Micro SUPPORT floor after Stage98. Restore SUPPORT recall/mass without large SUPPORT pressure or Stage95B-style clean NOT_ENTITLED-to-SUPPORT overpromotion. | Synthetic/internal Stage99A rows only. VitaminC/external and diagnostic rows are checked for overlap but not used as training rows. | Stage99B clean-dev frozen recovery run with Stage57 + Stage66 + combined Stage92A+Stage97A+Stage99A via stage80a slot; no Stage83A; no Stage88A; no external first. |                             1592 |                                  4472 | {"data/controlled_v5_v3_without_time_swap.jsonl": 0, "data/stage57_nonleaking_external_bridge.jsonl": 0, "data/stage66_residual_bridge.jsonl": 0, "data/stage83a_ne_safety_only_bridge.jsonl": 0, "data/stage88a_balanced_entitlement_recovery_bridge.jsonl": 0, "data/stage92a_support_preserving_counter_bridge.jsonl": 0, "data/stage95a_anti_ne_entitlement_preservation_bridge.jsonl": 0, "data/stage95a_stage92a_combined_support_antine_bridge.jsonl": 0, "data/stage97a_half_size_numeric_date_antine_bridge.jsonl": 0, "data/stage97a_stage92a_combined_half_antine_bridge.jsonl": 0, "data/stage43b1_vitaminc_validation_sample1000.jsonl": 0, "data/stage84e2_vitaminc_validation_sample1000_exact_ood_schema.jsonl": 0, "results/stage94_new_false_ne_stage73_non_ne_stage92c_ne_examples.jsonl": 0, "results/stage94_repaired_false_ne_stage73_ne_stage92c_non_ne_examples.jsonl": 0, "results/stage94_persistent_false_ne_stage73_stage92c_both_ne_examples.jsonl": 0, "results/stage94_stage92c_false_ne_all_examples.jsonl": 0, "results/stage96_stage92b_correct_stage95b_wrong_examples.jsonl": 0, "results/stage96_stage92b_wrong_stage95b_correct_examples.jsonl": 0, "results/stage96_stage92b_stage95b_changed_prediction_examples.jsonl": 0} | {"stage99a_row_count_32": true, "stage99a_support_16": true, "stage99a_ne_16": true, "stage99a_refute_0": true, "stage99a_pairs_unique": true, "stage99a_no_exact_overlap_checked_files": true, "stage99a_source_synthetic_internal_only": true, "stage99a_no_vitaminc_source": true, "stage99a_no_external_source": true, "combined_row_count_352": true, "combined_contains_stage92a_240": true, "combined_contains_stage97a_80": true, "combined_contains_stage99a_32": true, "combined_pairs_unique": true, "combined_support_172": true, "combined_refute_84": true, "combined_ne_96": true, "encode_label_tensors_stage99a_ok": true, "encode_label_tensors_combined_ok": true, "encode_mamba_records_stage99a_ok": true, "encode_mamba_records_combined_ok": true} |

## Stage99A label counts

| label        |   count |
|:-------------|--------:|
| NOT_ENTITLED |      16 |
| SUPPORT      |      16 |

## Stage99A family counts

| family                           |   count |
|:---------------------------------|--------:|
| support_floor_micro_ne_guardrail |      16 |
| support_floor_micro_numeric_date |      16 |

## Stage99A subtype counts

| subtype                             |   count |
|:------------------------------------|--------:|
| ne_guardrail_missing_verified_count |      16 |
| support_floor_verified_numeric_date |      16 |

## Combined Stage92A+Stage97A+Stage99A label counts

| label        |   count |
|:-------------|--------:|
| NOT_ENTITLED |      96 |
| REFUTE       |      84 |
| SUPPORT      |     172 |

## Exact-overlap checks

| file                                                                        |   exact_pair_overlap |
|:----------------------------------------------------------------------------|---------------------:|
| data/controlled_v5_v3_without_time_swap.jsonl                               |                    0 |
| data/stage57_nonleaking_external_bridge.jsonl                               |                    0 |
| data/stage66_residual_bridge.jsonl                                          |                    0 |
| data/stage83a_ne_safety_only_bridge.jsonl                                   |                    0 |
| data/stage88a_balanced_entitlement_recovery_bridge.jsonl                    |                    0 |
| data/stage92a_support_preserving_counter_bridge.jsonl                       |                    0 |
| data/stage95a_anti_ne_entitlement_preservation_bridge.jsonl                 |                    0 |
| data/stage95a_stage92a_combined_support_antine_bridge.jsonl                 |                    0 |
| data/stage97a_half_size_numeric_date_antine_bridge.jsonl                    |                    0 |
| data/stage97a_stage92a_combined_half_antine_bridge.jsonl                    |                    0 |
| data/stage43b1_vitaminc_validation_sample1000.jsonl                         |                    0 |
| data/stage84e2_vitaminc_validation_sample1000_exact_ood_schema.jsonl        |                    0 |
| results/stage94_new_false_ne_stage73_non_ne_stage92c_ne_examples.jsonl      |                    0 |
| results/stage94_repaired_false_ne_stage73_ne_stage92c_non_ne_examples.jsonl |                    0 |
| results/stage94_persistent_false_ne_stage73_stage92c_both_ne_examples.jsonl |                    0 |
| results/stage94_stage92c_false_ne_all_examples.jsonl                        |                    0 |
| results/stage96_stage92b_correct_stage95b_wrong_examples.jsonl              |                    0 |
| results/stage96_stage92b_wrong_stage95b_correct_examples.jsonl              |                    0 |
| results/stage96_stage92b_stage95b_changed_prediction_examples.jsonl         |                    0 |

## Checks

| check                                   | pass   |
|:----------------------------------------|:-------|
| stage99a_row_count_32                   | True   |
| stage99a_support_16                     | True   |
| stage99a_ne_16                          | True   |
| stage99a_refute_0                       | True   |
| stage99a_pairs_unique                   | True   |
| stage99a_no_exact_overlap_checked_files | True   |
| stage99a_source_synthetic_internal_only | True   |
| stage99a_no_vitaminc_source             | True   |
| stage99a_no_external_source             | True   |
| combined_row_count_352                  | True   |
| combined_contains_stage92a_240          | True   |
| combined_contains_stage97a_80           | True   |
| combined_contains_stage99a_32           | True   |
| combined_pairs_unique                   | True   |
| combined_support_172                    | True   |
| combined_refute_84                      | True   |
| combined_ne_96                          | True   |
| encode_label_tensors_stage99a_ok        | True   |
| encode_label_tensors_combined_ok        | True   |
| encode_mamba_records_stage99a_ok        | True   |
| encode_mamba_records_combined_ok        | True   |
