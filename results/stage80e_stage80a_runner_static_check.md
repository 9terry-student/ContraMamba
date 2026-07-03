# Stage80E - Stage80A Runner Static Check

## Decision

`STAGE80E_STAGE80A_RUNNER_STATIC_CHECK_READY`

## Summary

| stage    | decision                                    | purpose                                                                                      | runner                                  |   py_compile_returncode |   help_returncode |   stage80a_rows | stage80a_label_counts                                | stage80a_family_counts                                                                                                                                   |   stage57_rows |   stage66_rows |   combined_bridge_rows_expected |   final_train_expected |   pairwise_bridge_rows_excluded_expected | new_flags                                                         | training_executed   | external_eval_executed   | recommended_next_stage                                                        |
|:---------|:--------------------------------------------|:---------------------------------------------------------------------------------------------|:----------------------------------------|------------------------:|------------------:|----------------:|:-----------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|---------------:|---------------:|--------------------------------:|-----------------------:|-----------------------------------------:|:------------------------------------------------------------------|:--------------------|:-------------------------|:------------------------------------------------------------------------------|
| Stage80E | STAGE80E_STAGE80A_RUNNER_STATIC_CHECK_READY | Static validation of Stage80D runner patch for Stage80A conservative Stage75v2 bridge flags. | scripts/train_controlled_v6b_minimal.py |                       0 |                 0 |             500 | {"SUPPORT": 170, "REFUTE": 170, "NOT_ENTITLED": 160} | {"numeric_temporal_polarity_repair_v2_conservative": 180, "lexical_type_polarity_repair_v2_conservative": 160, "strict_ne_false_support_safety_v2": 160} |            520 |            720 |                            1740 |                   4620 |                                     1740 | ["--stage80a-bridge-train-jsonl", "--stage80a-bridge-train-mode"] | False               | False                    | Stage80F full clean-dev run with Stage57+Stage66+Stage80A conservative bridge |

## Checks

| check                                       | pass   |
|:--------------------------------------------|:-------|
| runner_exists                               | True   |
| py_compile_ok                               | True   |
| help_ok                                     | True   |
| help_contains_stage80a_jsonl_flag           | True   |
| help_contains_stage80a_mode_flag            | True   |
| source_contains_stage80a_loader             | True   |
| source_contains_stage80a_source_label       | True   |
| source_contains_stage80a_jsonl_validation   | True   |
| source_contains_stage80a_mode_validation    | True   |
| source_contains_stage80a_pairwise_exclusion | True   |
| source_contains_stage80a_excluded_bool      | True   |
| source_contains_combined_bridge_metadata    | True   |
| source_preserves_stage57_flags              | True   |
| source_preserves_stage66_flags              | True   |
| source_preserves_stage75_flags              | True   |
| stage80a_row_count_500                      | True   |
| stage80a_label_counts_exact                 | True   |
| stage80a_family_counts_exact                | True   |
| stage80a_duplicate_ids_zero                 | True   |
| stage80a_duplicate_claim_evidence_zero      | True   |
| stage80a_no_pair_id                         | True   |
| stage80a_synthetic_only_true                | True   |
| stage80a_external_text_false                | True   |
| stage80a_external_label_false               | True   |
| stage80b_generation_ready                   | True   |
| stage80b_static_ready                       | True   |
| stage80c_ready                              | True   |
| stage80c_counts_match_computed              | True   |
| combined_bridge_1740                        | True   |
| final_train_expected_4620                   | True   |
| pairwise_bridge_excluded_1740               | True   |

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
| clean_main_train_only                                  | True                       |                   |
| combined_bridge_row_count                              | True                       |                   |
| combined_bridge_train_only                             | True                       |                   |
| final_train_row_count_expected                         | True                       |                   |

## Count comparison

| metric                        |   stage80c_expected |   computed_now | match   |
|:------------------------------|--------------------:|---------------:|:--------|
| stage57_bridge_rows           |                 520 |            520 | True    |
| stage66_bridge_rows           |                 720 |            720 | True    |
| stage80a_bridge_rows          |                 500 |            500 | True    |
| combined_bridge_rows          |                1740 |           1740 | True    |
| final_train_expected          |                4620 |           4620 | True    |
| pairwise_bridge_rows_excluded |                1740 |           1740 | True    |

## Stage80A family x label counts

| family                                           |   SUPPORT |   REFUTE |   NOT_ENTITLED |   total |
|:-------------------------------------------------|----------:|---------:|---------------:|--------:|
| lexical_type_polarity_repair_v2_conservative     |        80 |       80 |              0 |     160 |
| numeric_temporal_polarity_repair_v2_conservative |        90 |       90 |              0 |     180 |
| strict_ne_false_support_safety_v2                |         0 |        0 |            160 |     160 |

## Recommended next stage

Stage80F full clean-dev run with Stage57+Stage66+Stage80A conservative bridge
