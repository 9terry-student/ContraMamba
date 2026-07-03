# Stage75E — Runner Static Check

## Decision

`STAGE75E_RUNNER_STATIC_CHECK_READY`

## Summary

| stage    | decision                           | purpose                                                           | runner                                  |   main_clean_row_count |   clean_train_estimated_row_count |   clean_dev_estimated_row_count |   stage57_row_count |   stage66_row_count |   stage75_row_count |   combined_bridge_row_count |   final_train_if_appended | stage75_label_counts                                | stage75_family_counts                                                                                                                                                                                                                   | missing_help_flags   | missing_source_tokens   | training_executed   | external_eval_executed   | recommended_next_stage                                         |
|:---------|:-----------------------------------|:------------------------------------------------------------------|:----------------------------------------|-----------------------:|----------------------------------:|--------------------------------:|--------------------:|--------------------:|--------------------:|----------------------------:|--------------------------:|:----------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------|:------------------------|:--------------------|:-------------------------|:---------------------------------------------------------------|
| Stage75E | STAGE75E_RUNNER_STATIC_CHECK_READY | Static runner check for Stage75 bridge train-only append support. | scripts/train_controlled_v6b_minimal.py |                   3600 |                              2880 |                             720 |                 520 |                 720 |                1020 |                        2260 |                      5140 | {"SUPPORT": 480, "REFUTE": 460, "NOT_ENTITLED": 80} | {"support_entitlement_direct_recovery_v2": 240, "refute_entitlement_direct_recovery_v2": 220, "numeric_temporal_polarity_comparison_v2": 260, "lexical_type_polarity_disambiguation_v2": 220, "strict_ne_external_style_safety_v2": 80} | []                   | []                      | False               | False                    | Stage75F Stage57+66+75 bridge-enabled frozen recovery full run |

## Checks

| check                                      | pass   |
|:-------------------------------------------|:-------|
| help_has_stage75_jsonl_flag                | True   |
| help_has_stage75_mode_flag                 | True   |
| help_mentions_append_train_only            | True   |
| help_mentions_none_mode                    | True   |
| source_has_all_stage75_tokens              | True   |
| source_has_no_required_pair_id_for_stage75 | True   |
| main_clean_3600                            | True   |
| clean_train_estimated_2880                 | True   |
| clean_dev_estimated_720                    | True   |
| stage57_520                                | True   |
| stage66_720                                | True   |
| stage75_1020                               | True   |
| combined_bridge_2260                       | True   |
| final_train_if_appended_5140               | True   |
| stage75_support_480                        | True   |
| stage75_refute_460                         | True   |
| stage75_ne_80                              | True   |
| stage75_no_pair_id                         | True   |
| stage75_synthetic_only                     | True   |
| stage75_external_text_false                | True   |
| stage75_external_label_false               | True   |
| stage75b_report_ready                      | True   |
| stage75b_static_ready                      | True   |
| stage75c_plan_ready                        | True   |

## Help flags

| flag                         | present_in_help   |
|:-----------------------------|:------------------|
| --stage75-bridge-train-jsonl | True              |
| --stage75-bridge-train-mode  | True              |

## Source metadata tokens

| token                                                 | present_in_source   |
|:------------------------------------------------------|:--------------------|
| stage75_bridge_train_jsonl                            | True                |
| stage75_bridge_train_mode                             | True                |
| stage75_bridge_enabled                                | True                |
| stage75_bridge_row_count                              | True                |
| stage75_bridge_label_counts                           | True                |
| stage75_bridge_family_counts                          | True                |
| stage75_bridge_train_only                             | True                |
| stage75_bridge_used_for_dev                           | True                |
| stage75_bridge_used_for_checkpoint_selection          | True                |
| stage75_excluded_from_intervention_pairwise_loss      | True                |
| intervention_pairwise_loss_stage75_row_count_excluded | True                |
| combined_bridge_row_count                             | True                |
| intervention_pairwise_loss_bridge_row_count_excluded  | True                |
| clean_main_train_only                                 | True                |

## Row counts

| scope                   |   row_count |
|:------------------------|------------:|
| main_clean              |        3600 |
| clean_train_estimated   |        2880 |
| clean_dev_estimated     |         720 |
| stage57_bridge          |         520 |
| stage66_bridge          |         720 |
| stage75_bridge          |        1020 |
| combined_bridge         |        2260 |
| final_train_if_appended |        5140 |

## Stage75 label counts

| label        |   stage75_count |
|:-------------|----------------:|
| SUPPORT      |             480 |
| REFUTE       |             460 |
| NOT_ENTITLED |              80 |

## Stage75 family counts

| family                                  |   count |
|:----------------------------------------|--------:|
| lexical_type_polarity_disambiguation_v2 |     220 |
| numeric_temporal_polarity_comparison_v2 |     260 |
| refute_entitlement_direct_recovery_v2   |     220 |
| strict_ne_external_style_safety_v2      |      80 |
| support_entitlement_direct_recovery_v2  |     240 |

## Recommended next stage

Stage75F Stage57+66+75 bridge-enabled frozen recovery full run
