# Stage72 — Stage51 / Stage61 / Stage71 Retry2 Clean-Dev Preservation Comparison

## Decision

`STAGE72_CLEAN_DEV_PRESERVATION_COMPARISON_READY`

## Summary

| stage   | decision                                        | stage51_source                                                                                | stage61_source                                                                                              | stage71_retry2_source                                                                                                                      |   stage51_best_dev_acc |   stage51_best_dev_macro_f1 |   stage61_best_dev_acc |   stage61_best_dev_macro_f1 |   stage71_retry2_best_dev_acc |   stage71_retry2_best_dev_macro_f1 |   stage71_retry2_minus_stage61_acc |   stage71_retry2_minus_stage61_macro_f1 |   stage71_retry2_minus_stage51_acc |   stage71_retry2_minus_stage51_macro_f1 | metadata_checks_pass   | preservation_checks_pass   | recommended_next_stage                                    |
|:--------|:------------------------------------------------|:----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|-----------------------:|----------------------------:|-----------------------:|----------------------------:|------------------------------:|-----------------------------------:|-----------------------------------:|----------------------------------------:|-----------------------------------:|----------------------------------------:|:-----------------------|:---------------------------|:----------------------------------------------------------|
| Stage72 | STAGE72_CLEAN_DEV_PRESERVATION_COMPARISON_READY | results/stage51_frozen_recovery_run_20260702_044005/stage51_frozen_recovery_train_report.json | results/stage61_bridge_enabled_frozen_recovery_run_20260702_055007/stage61_bridge_enabled_train_report.json | results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_stage57_stage66_train_report.json |               0.973611 |                    0.962855 |               0.970833 |                    0.958808 |                         0.975 |                           0.964047 |                         0.00416666 |                              0.00523891 |                         0.00138891 |                              0.00119273 | True                   | True                       | Stage73 bridge-enabled VitaminC external diagnostic rerun |

## Run comparison

| stage                                 | source_path                                                                                                                                |   best_epoch |   selected_epoch |   best_dev_acc |   best_dev_macro_f1 |   pred_NOT_ENTITLED |   pred_REFUTE |   pred_SUPPORT | stage57_bridge_enabled   |   stage57_bridge_row_count | stage66_bridge_enabled   |   stage66_bridge_row_count | combined_bridge_enabled   |   combined_bridge_row_count | combined_bridge_train_only   | clean_dev_for_checkpoint_selection   | external_data_used_for_training   | external_metrics_used_for_threshold_tuning   | time_swap_used   | bridge_rows_excluded_from_intervention_pairwise_loss   | intervention_pairwise_loss_source   |   intervention_pairwise_loss_clean_main_row_count |   intervention_pairwise_loss_bridge_row_count_excluded |
|:--------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|-------------:|-----------------:|---------------:|--------------------:|--------------------:|--------------:|---------------:|:-------------------------|---------------------------:|:-------------------------|---------------------------:|:--------------------------|----------------------------:|:-----------------------------|:-------------------------------------|:----------------------------------|:---------------------------------------------|:-----------------|:-------------------------------------------------------|:------------------------------------|--------------------------------------------------:|-------------------------------------------------------:|
| Stage51_baseline_frozen_recovery      | results/stage51_frozen_recovery_run_20260702_044005/stage51_frozen_recovery_train_report.json                                              |           80 |               80 |       0.973611 |            0.962855 |                 525 |            86 |            109 |                          |                        nan |                          |                        nan |                           |                         nan |                              |                                      |                                   |                                              | False            |                                                        |                                     |                                               nan |                                                    nan |
| Stage61_stage57_bridge_only           | results/stage61_bridge_enabled_frozen_recovery_run_20260702_055007/stage61_bridge_enabled_train_report.json                                |          191 |              191 |       0.970833 |            0.958808 |                 523 |            87 |            110 |                          |                        nan |                          |                        nan |                           |                         nan |                              |                                      |                                   |                                              | False            |                                                        |                                     |                                               nan |                                                    nan |
| Stage71_retry2_stage57_stage66_bridge | results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_stage57_stage66_train_report.json |          151 |              151 |       0.975    |            0.964047 |                 522 |            90 |            108 | True                     |                        520 | True                     |                        720 | True                      |                        1240 | True                         | True                                 | False                             | False                                        | False            | True                                                   | clean_main_train_only               |                                              2880 |                                                   1240 |

## Deltas

| comparison                   |   delta_best_dev_acc |   delta_best_dev_macro_f1 |   delta_pred_NOT_ENTITLED |   delta_pred_REFUTE |   delta_pred_SUPPORT |
|:-----------------------------|---------------------:|--------------------------:|--------------------------:|--------------------:|---------------------:|
| Stage61_minus_Stage51        |          -0.00277776 |               -0.00404618 |                        -2 |                   1 |                    1 |
| Stage71_retry2_minus_Stage61 |           0.00416666 |                0.00523891 |                        -1 |                   3 |                   -2 |
| Stage71_retry2_minus_Stage51 |           0.00138891 |                0.00119273 |                        -3 |                   4 |                   -1 |

## Metadata checks

| check                                                  | pass   |
|:-------------------------------------------------------|:-------|
| stage71_stage57_bridge_enabled                         | True   |
| stage71_stage57_row_count_520                          | True   |
| stage71_stage66_bridge_enabled                         | True   |
| stage71_stage66_row_count_720                          | True   |
| stage71_combined_bridge_enabled                        | True   |
| stage71_combined_row_count_1240                        | True   |
| stage71_combined_train_only                            | True   |
| stage71_clean_dev_checkpoint_selection                 | True   |
| stage71_external_data_not_used_for_training            | True   |
| stage71_external_metrics_not_used_for_threshold_tuning | True   |
| stage71_time_swap_not_used                             | True   |
| stage71_bridge_rows_excluded_from_pairwise             | True   |
| stage71_pairwise_source_clean_main_train_only          | True   |
| stage71_pairwise_clean_main_count_2880                 | True   |
| stage71_pairwise_bridge_excluded_count_1240            | True   |

## Preservation checks

| check                              | pass   |
|:-----------------------------------|:-------|
| stage71_macro_preserved_vs_stage61 | True   |
| stage71_acc_preserved_vs_stage61   | True   |
| stage71_macro_preserved_vs_stage51 | True   |
| stage71_acc_preserved_vs_stage51   | True   |
| stage71_macro_nontrivial           | True   |
| stage71_acc_nontrivial             | True   |

## Per-label metrics

| stage                                 | label        |   precision |   recall |       f1 |
|:--------------------------------------|:-------------|------------:|---------:|---------:|
| Stage51_baseline_frozen_recovery      | NOT_ENTITLED |    0.99619  | 0.968519 | 0.98216  |
| Stage51_baseline_frozen_recovery      | REFUTE       |    1        | 1        | 1        |
| Stage51_baseline_frozen_recovery      | SUPPORT      |    0.844037 | 0.978723 | 0.906404 |
| Stage61_stage57_bridge_only           | NOT_ENTITLED |    0.996176 | 0.964815 | 0.980245 |
| Stage61_stage57_bridge_only           | REFUTE       |    0.988506 | 1        | 0.99422  |
| Stage61_stage57_bridge_only           | SUPPORT      |    0.836364 | 0.978723 | 0.901961 |
| Stage71_retry2_stage57_stage66_bridge | NOT_ENTITLED |    1        | 0.966667 | 0.983051 |
| Stage71_retry2_stage57_stage66_bridge | REFUTE       |    1        | 1        | 1        |
| Stage71_retry2_stage57_stage66_bridge | SUPPORT      |    0.833333 | 1        | 0.909091 |

## Interpretation

Stage71 retry2 is considered clean-dev-preserved if it does not degrade materially against both Stage61 and Stage51 under the configured tolerance and all bridge safety metadata remain valid.

## Recommended next stage

Stage73 bridge-enabled VitaminC external diagnostic rerun
