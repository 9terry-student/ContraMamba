# Stage71 Retry2 — Stage57 + Stage66 Bridge-Enabled Frozen Recovery Run

## Decision

`STAGE71_RETRY2_STAGE57_STAGE66_BRIDGE_ENABLED_FROZEN_RECOVERY_RUN_READY`

## Summary

| stage          | decision                                                                | run_dir                                                                                   |   elapsed_min |   returncode |   best_epoch |   selected_epoch |   best_dev_acc | best_dev_accuracy   |   best_dev_macro_f1 | dev_accuracy   | dev_macro_f1   | stage44b2_decision                       | stage44_decision   | stage57_bridge_enabled   |   stage57_bridge_row_count | stage57_bridge_train_only   | stage57_used_for_dev   | stage57_used_for_checkpoint_selection   | stage66_bridge_enabled   |   stage66_bridge_row_count | stage66_bridge_train_only   | stage66_used_for_dev   | stage66_used_for_checkpoint_selection   | combined_bridge_enabled   |   combined_bridge_row_count | combined_bridge_train_only   | clean_dev_for_checkpoint_selection   | external_data_used_for_training   | external_metrics_used_for_threshold_tuning   | time_swap_used   | bridge_rows_excluded_from_intervention_pairwise_loss   | stage57_excluded_from_intervention_pairwise_loss   | stage66_excluded_from_intervention_pairwise_loss   | intervention_pairwise_loss_source   |   intervention_pairwise_loss_clean_main_row_count |   intervention_pairwise_loss_bridge_row_count_excluded |   intervention_pairwise_loss_stage57_row_count_excluded |   intervention_pairwise_loss_stage66_row_count_excluded | training_executed   | external_eval_executed   | recommended_next_stage                                           |
|:---------------|:------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|--------------:|-------------:|-------------:|-----------------:|---------------:|:--------------------|--------------------:|:---------------|:---------------|:-----------------------------------------|:-------------------|:-------------------------|---------------------------:|:----------------------------|:-----------------------|:----------------------------------------|:-------------------------|---------------------------:|:----------------------------|:-----------------------|:----------------------------------------|:--------------------------|----------------------------:|:-----------------------------|:-------------------------------------|:----------------------------------|:---------------------------------------------|:-----------------|:-------------------------------------------------------|:---------------------------------------------------|:---------------------------------------------------|:------------------------------------|--------------------------------------------------:|-------------------------------------------------------:|--------------------------------------------------------:|--------------------------------------------------------:|:--------------------|:-------------------------|:-----------------------------------------------------------------|
| Stage71_retry2 | STAGE71_RETRY2_STAGE57_STAGE66_BRIDGE_ENABLED_FROZEN_RECOVERY_RUN_READY | results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827 |         6.191 |            0 |          151 |              151 |          0.975 |                     |            0.964047 |                |                | STAGE44B2_PRIOR_AWARE_SELECTION_DISABLED |                    | True                     |                        520 | True                        | False                  | False                                   | True                     |                        720 | True                        | False                  | False                                   | True                      |                        1240 | True                         | True                                 | False                             | False                                        | False            | True                                                   | True                                               | True                                               | clean_main_train_only               |                                              2880 |                                                   1240 |                                                     520 |                                                     720 | True                | False                    | Stage72 compare Stage61 vs Stage71 retry2 clean-dev preservation |

## Metadata checks

| check                                            | pass   |
|:-------------------------------------------------|:-------|
| train_report_exists                              | True   |
| predictions_json_exists                          | True   |
| stage57_bridge_enabled_true                      | True   |
| stage57_bridge_row_count_520                     | True   |
| stage57_train_only_true                          | True   |
| stage57_used_for_dev_false                       | True   |
| stage57_used_for_checkpoint_false                | True   |
| stage66_bridge_enabled_true                      | True   |
| stage66_bridge_row_count_720                     | True   |
| stage66_train_only_true                          | True   |
| stage66_used_for_dev_false                       | True   |
| stage66_used_for_checkpoint_false                | True   |
| combined_bridge_enabled_true                     | True   |
| combined_bridge_row_count_1240                   | True   |
| combined_bridge_train_only_true                  | True   |
| clean_dev_for_checkpoint_selection_true          | True   |
| external_data_used_for_training_false            | True   |
| external_metrics_used_for_threshold_tuning_false | True   |
| time_swap_used_false                             | True   |
| bridge_rows_excluded_from_pairwise_true          | True   |
| stage57_excluded_from_pairwise_true              | True   |
| stage66_excluded_from_pairwise_true              | True   |
| pairwise_loss_source_clean_main_train_only       | True   |
| pairwise_clean_main_count_2880                   | True   |
| pairwise_bridge_excluded_count_1240              | True   |
| pairwise_stage57_excluded_count_520              | True   |
| pairwise_stage66_excluded_count_720              | True   |

## Run artifacts

- Run dir: `results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827`
- Train report: `results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_stage57_stage66_train_report.json`
- Predictions: `results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_stage57_stage66_predictions.json`
- Stdout log: `results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_stdout.log`
- Command: `results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_command.json`
- Stage44 selection report: `results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_stage44_selection_report.json`
- Stage45C recovery report JSON: `results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_stage45c_recovery_report.json`
- Stage45C recovery report MD: `results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_stage45c_recovery_report.md`

## Recommended next stage

Stage72 compare Stage61 vs Stage71 retry2 clean-dev preservation
