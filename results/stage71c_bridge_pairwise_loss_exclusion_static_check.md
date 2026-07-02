# Stage71C — Bridge Pairwise-Loss Exclusion Static Check

## Decision

`STAGE71C_BRIDGE_PAIRWISE_LOSS_EXCLUSION_STATIC_CHECK_READY`

## Summary

| stage    | decision                                                   | help_ok   | source_ok   | heuristic_ok   | data_ok   |   main_row_count |   expected_clean_main_train_count |   expected_clean_main_dev_count |   stage57_bridge_row_count |   stage66_bridge_row_count |   combined_bridge_row_count | expected_pairwise_loss_source   | expected_bridge_rows_excluded_from_intervention_pairwise_loss   | training_executed   | external_eval_executed   | recommended_next_stage                                            |
|:---------|:-----------------------------------------------------------|:----------|:------------|:---------------|:----------|-----------------:|----------------------------------:|--------------------------------:|---------------------------:|---------------------------:|----------------------------:|:--------------------------------|:----------------------------------------------------------------|:--------------------|:-------------------------|:------------------------------------------------------------------|
| Stage71C | STAGE71C_BRIDGE_PAIRWISE_LOSS_EXCLUSION_STATIC_CHECK_READY | True      | True        | True           | True      |             3600 |                              2880 |                             720 |                        520 |                        720 |                        1240 | clean_main_train_only           | True                                                            | False               | False                    | Stage71 retry full run after pairwise-loss bridge exclusion patch |

## Help flag check

| flag                         | present_in_help   |
|:-----------------------------|:------------------|
| --stage57-bridge-train-jsonl | True              |
| --stage57-bridge-train-mode  | True              |
| --stage66-bridge-train-jsonl | True              |
| --stage66-bridge-train-mode  | True              |

## Source checks

| check                                                | pass   |
|:-----------------------------------------------------|:-------|
| intervention_pairwise_losses_call_exists             | True   |
| bridge_exclusion_report_field_present                | True   |
| stage57_exclusion_report_field_present               | True   |
| stage66_exclusion_report_field_present               | True   |
| pairwise_loss_source_report_field_present            | True   |
| pairwise_clean_main_count_report_field_present       | True   |
| pairwise_bridge_excluded_count_report_field_present  | True   |
| pairwise_stage57_excluded_count_report_field_present | True   |
| pairwise_stage66_excluded_count_report_field_present | True   |
| clean_main_train_only_literal_or_equivalent_present  | True   |
| bridge_sources_still_reported                        | True   |
| stage57_bridge_still_appended                        | True   |
| stage66_bridge_still_appended                        | True   |

## Heuristic checks

| check                                              | pass   |
|:---------------------------------------------------|:-------|
| no_obvious_old_train_records_only_pairwise_call    | True   |
| intervention_pairwise_losses_function_unchanged_ok | True   |

## Data checks

| check                                | pass   |
|:-------------------------------------|:-------|
| main_row_count_3600                  | True   |
| expected_clean_main_train_count_2880 | True   |
| expected_clean_main_dev_count_720    | True   |
| stage57_row_count_520                | True   |
| stage66_row_count_720                | True   |
| combined_bridge_row_count_1240       | True   |
| stage57_label_counts_ok              | True   |
| stage66_label_counts_ok              | True   |
| combined_label_counts_ok             | True   |
| stage68_ready                        | True   |
| stage68_encode_label_tensors_ok      | True   |
| stage69_ready                        | True   |
| stage70_ready                        | True   |
| stage70_stage66_encode_ok            | True   |
| stage71a_ready                       | True   |

## Expected report values for Stage71 retry

| field                                                 | expected_value        |
|:------------------------------------------------------|:----------------------|
| stage57_bridge_row_count                              | 520                   |
| stage66_bridge_row_count                              | 720                   |
| combined_bridge_row_count                             | 1240                  |
| intervention_pairwise_loss_clean_main_row_count       | 2880                  |
| intervention_pairwise_loss_bridge_row_count_excluded  | 1240                  |
| intervention_pairwise_loss_stage57_row_count_excluded | 520                   |
| intervention_pairwise_loss_stage66_row_count_excluded | 720                   |
| bridge_rows_excluded_from_intervention_pairwise_loss  | True                  |
| intervention_pairwise_loss_source                     | clean_main_train_only |

## Execution policy

No training, no external evaluation, no smoke run, no full run.
Only py_compile, --help, source checks, and static data/report checks were executed.
