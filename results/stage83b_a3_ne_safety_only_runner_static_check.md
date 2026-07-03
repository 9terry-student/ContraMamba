# Stage83B - A3 NE-Safety-Only Runner Static Check

## Decision

`STAGE83B_A3_NE_SAFETY_ONLY_RUNNER_STATIC_CHECK_READY`

## Summary

| stage    | decision                                             | purpose                                                                                | runner                                  | stage83a_jsonl                            | loader_signature                                                                                                                  |   py_compile_returncode |   help_returncode |   stage83a_rows | stage83a_label_counts   | stage83a_family_counts                     |   loader_rows | loader_label_counts_raw                          | loader_label_counts_normalized   | loader_family_counts                       | loader_family_label_counts_raw                                                          |   stage57_rows |   stage66_rows |   combined_bridge_rows_expected |   final_train_row_count_expected |   pairwise_bridge_rows_excluded_expected | canonical_stage83c_bridge_config                               | training_executed   | external_eval_executed   | recommended_next_stage                                                     |
|:---------|:-----------------------------------------------------|:---------------------------------------------------------------------------------------|:----------------------------------------|:------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|------------------------:|------------------:|----------------:|:------------------------|:-------------------------------------------|--------------:|:-------------------------------------------------|:---------------------------------|:-------------------------------------------|:----------------------------------------------------------------------------------------|---------------:|---------------:|--------------------------------:|---------------------------------:|-----------------------------------------:|:---------------------------------------------------------------|:--------------------|:-------------------------|:---------------------------------------------------------------------------|
| Stage83B | STAGE83B_A3_NE_SAFETY_ONLY_RUNNER_STATIC_CHECK_READY | Static runner/loader check for A3 NE-safety-only bridge through Stage80A bridge flags. | scripts/train_controlled_v6b_minimal.py | data/stage83a_ne_safety_only_bridge.jsonl | (bridge_path: 'Path', existing_ids: 'set[str]') -> 'tuple[list[dict], dict[str, int], dict[str, int], dict[str, dict[str, int]]]' |                       0 |                 0 |             160 | {"NOT_ENTITLED": 160}   | {"strict_ne_false_support_safety_v2": 160} |           160 | {"REFUTE": 0, "NOT_ENTITLED": 160, "SUPPORT": 0} | {"NOT_ENTITLED": 160}            | {"strict_ne_false_support_safety_v2": 160} | {"strict_ne_false_support_safety_v2": {"REFUTE": 0, "NOT_ENTITLED": 160, "SUPPORT": 0}} |            520 |            720 |                            1400 |                             4280 |                                     1400 | Stage57 + Stage66 + Stage83A via --stage80a-bridge-train-jsonl | False               | False                    | Stage83C clean-dev run with Stage57+Stage66+Stage83A NE-safety-only bridge |

## Checks

| check                                          | pass   |
|:-----------------------------------------------|:-------|
| runner_exists                                  | True   |
| py_compile_ok                                  | True   |
| help_ok                                        | True   |
| help_contains_stage80a_jsonl_flag              | True   |
| help_contains_stage80a_mode_flag               | True   |
| stage83_decision_ready                         | True   |
| stage83_allows_a3                              | True   |
| stage83a_generation_ready                      | True   |
| stage80e_static_ready                          | True   |
| loader_function_exists                         | True   |
| loader_signature_has_existing_ids              | True   |
| loader_returns_tuple                           | True   |
| loader_accepts_stage83a_path_object            | True   |
| loader_label_counts_normalized_exact           | True   |
| loader_family_counts_exact                     | True   |
| stage83a_row_count_160                         | True   |
| stage83a_label_counts_exact                    | True   |
| stage83a_family_counts_exact                   | True   |
| stage83a_duplicate_ids_zero                    | True   |
| stage83a_duplicate_claim_evidence_zero         | True   |
| stage83a_no_pair_id                            | True   |
| stage83a_synthetic_only_true                   | True   |
| stage83a_external_text_false                   | True   |
| stage83a_external_label_false                  | True   |
| stage83a_subset_metadata_present               | True   |
| source_contains_stage80a_pairwise_exclusion    | True   |
| source_contains_final_train_row_count_expected | True   |
| source_contains_combined_bridge_metadata       | True   |
| computed_combined_bridge_1400                  | True   |
| computed_final_train_expected_4280             | True   |
| computed_pairwise_excluded_1400                | True   |
| expected_counts_match_computed                 | True   |

## Source markers

| marker                                                 | present_in_runner_source   | present_in_help   |
|:-------------------------------------------------------|:---------------------------|:------------------|
| --stage80a-bridge-train-jsonl                          | True                       | True              |
| --stage80a-bridge-train-mode                           | True                       | True              |
| load_stage80a_bridge_train_rows                        | True                       |                   |
| stage80a_bridge                                        | True                       |                   |
| stage80a_bridge_train_jsonl                            | True                       |                   |
| stage80a_bridge_train_mode                             | True                       |                   |
| stage80a_bridge_enabled                                | True                       |                   |
| stage80a_bridge_row_count                              | True                       |                   |
| stage80a_bridge_label_counts                           | True                       |                   |
| stage80a_bridge_family_counts                          | True                       |                   |
| stage80a_excluded_from_intervention_pairwise_loss      | True                       |                   |
| intervention_pairwise_loss_stage80a_row_count_excluded | True                       |                   |
| combined_bridge_row_count                              | True                       |                   |
| combined_bridge_train_only                             | True                       |                   |
| final_train_row_count_expected                         | True                       |                   |
| clean_main_train_only                                  | True                       |                   |

## Count comparison

| metric                                 |   expected |   computed | match   |
|:---------------------------------------|-----------:|-----------:|:--------|
| stage57_rows                           |        520 |        520 | True    |
| stage66_rows                           |        720 |        720 | True    |
| stage83a_rows                          |        160 |        160 | True    |
| combined_bridge_rows                   |       1400 |       1400 | True    |
| final_train_row_count_expected         |       4280 |       4280 | True    |
| pairwise_bridge_rows_excluded_expected |       1400 |       1400 | True    |

## Recommended next stage

Stage83C clean-dev run with Stage57+Stage66+Stage83A NE-safety-only bridge
