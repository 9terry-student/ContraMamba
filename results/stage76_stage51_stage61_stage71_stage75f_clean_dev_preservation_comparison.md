# Stage76 - Clean-Dev Preservation Comparison

## Decision

`STAGE76_CLEAN_DEV_PRESERVATION_COMPARISON_READY`

## Summary

| stage   | decision                                        |   stage51_best_dev_acc |   stage51_best_dev_macro_f1 |   stage61_best_dev_acc |   stage61_best_dev_macro_f1 |   stage71_retry2_best_dev_acc |   stage71_retry2_best_dev_macro_f1 |   stage75f_best_dev_acc |   stage75f_best_dev_macro_f1 |   stage75f_minus_stage71_acc |   stage75f_minus_stage71_macro_f1 |   stage75f_minus_stage51_acc |   stage75f_minus_stage51_macro_f1 |   stage75f_minus_stage61_acc |   stage75f_minus_stage61_macro_f1 | metadata_checks_pass   | preservation_checks_pass   | interpretation                                                                                                | training_executed   | external_eval_executed   | recommended_next_stage                              |
|:--------|:------------------------------------------------|-----------------------:|----------------------------:|-----------------------:|----------------------------:|------------------------------:|-----------------------------------:|------------------------:|-----------------------------:|-----------------------------:|----------------------------------:|-----------------------------:|----------------------------------:|-----------------------------:|----------------------------------:|:-----------------------|:---------------------------|:--------------------------------------------------------------------------------------------------------------|:--------------------|:-------------------------|:----------------------------------------------------|
| Stage76 | STAGE76_CLEAN_DEV_PRESERVATION_COMPARISON_READY |               0.973611 |                    0.962855 |               0.970833 |                    0.958808 |                         0.975 |                           0.964047 |                0.973611 |                     0.962205 |                  -0.00138891 |                       -0.00184226 |                            0 |                      -0.000649529 |                   0.00277776 |                        0.00339665 | True                   | True                       | Stage75F preserves clean-dev under tolerance, but does not improve over Stage71_retry2 on clean-dev macro-F1. | False               | False                    | Stage77 Stage75F VitaminC external diagnostic rerun |

## Run comparison

| stage          | source_path                                                                                                                                    |   best_epoch |   selected_epoch |   best_dev_acc |   best_dev_macro_f1 |   pred_NOT_ENTITLED |   pred_REFUTE |   pred_SUPPORT | stage57_enabled   |   stage57_row_count | stage66_enabled   |   stage66_row_count | stage75_enabled   |   stage75_row_count | combined_bridge_enabled   |   combined_bridge_row_count | combined_bridge_train_only   | clean_dev_for_checkpoint_selection   | external_data_used_for_training   | external_metrics_used_for_threshold_tuning   | time_swap_used   | bridge_pairwise_excluded   | pairwise_source       |   pairwise_clean_main_row_count |   pairwise_bridge_row_count_excluded | pairwise_stage57_excluded   | pairwise_stage66_excluded   | pairwise_stage75_excluded   |
|:---------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|-------------:|-----------------:|---------------:|--------------------:|--------------------:|--------------:|---------------:|:------------------|--------------------:|:------------------|--------------------:|:------------------|--------------------:|:--------------------------|----------------------------:|:-----------------------------|:-------------------------------------|:----------------------------------|:---------------------------------------------|:-----------------|:---------------------------|:----------------------|--------------------------------:|-------------------------------------:|:----------------------------|:----------------------------|:----------------------------|
| Stage51        | results/stage51_frozen_recovery_run_20260702_044005/stage51_frozen_recovery_train_report.json                                                  |           80 |               80 |       0.973611 |            0.962855 |                 525 |            86 |            109 | False             |                   0 | False             |                   0 | False             |                   0 | False                     |                           0 | False                        | False                                | False                             | False                                        | False            | False                      |                       |                             nan |                                  nan | False                       | False                       | False                       |
| Stage61        | results/stage61_bridge_enabled_frozen_recovery_run_20260702_055007/stage61_bridge_enabled_train_report.json                                    |          191 |              191 |       0.970833 |            0.958808 |                 523 |            87 |            110 | False             |                   0 | False             |                   0 | False             |                   0 | False                     |                           0 | False                        | False                                | False                             | False                                        | False            | False                      |                       |                             nan |                                  nan | False                       | False                       | False                       |
| Stage71_retry2 | results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_stage57_stage66_train_report.json     |          151 |              151 |       0.975    |            0.964047 |                 522 |            90 |            108 | True              |                 520 | True              |                 720 | False             |                   0 | True                      |                        1240 | True                         | True                                 | False                             | False                                        | False            | True                       | clean_main_train_only |                            2880 |                                 1240 | True                        | True                        | False                       |
| Stage75F       | results/stage75f_stage57_stage66_stage75_bridge_enabled_frozen_recovery_run_20260703_002555/stage75f_stage57_stage66_stage75_train_report.json |          184 |              184 |       0.973611 |            0.962205 |                 521 |            90 |            109 | True              |                 520 | True              |                 720 | True              |                1020 | True                      |                        2260 | True                         | True                                 | False                             | False                                        | False            | True                       | clean_main_train_only |                            2880 |                                 2260 | True                        | True                        | True                        |

## Delta comparison

| comparison                |   acc_delta |   macro_f1_delta |   pred_NE_delta |   pred_REFUTE_delta |   pred_SUPPORT_delta |
|:--------------------------|------------:|-----------------:|----------------:|--------------------:|---------------------:|
| Stage61 - Stage51         | -0.00277776 |     -0.00404618  |              -2 |                   1 |                    1 |
| Stage71_retry2 - Stage61  |  0.00416666 |      0.00523891  |              -1 |                   3 |                   -2 |
| Stage71_retry2 - Stage51  |  0.00138891 |      0.00119273  |              -3 |                   4 |                   -1 |
| Stage75F - Stage71_retry2 | -0.00138891 |     -0.00184226  |              -1 |                   0 |                    1 |
| Stage75F - Stage61        |  0.00277776 |      0.00339665  |              -2 |                   3 |                   -1 |
| Stage75F - Stage51        |  0          |     -0.000649529 |              -4 |                   4 |                    0 |

## Metadata checks

| check                                   | pass   |
|:----------------------------------------|:-------|
| stage75f_report_exists                  | True   |
| stage75f_summary_exists                 | True   |
| stage75f_stage57_enabled_520            | True   |
| stage75f_stage66_enabled_720            | True   |
| stage75f_stage75_enabled_1020           | True   |
| stage75f_combined_bridge_2260           | True   |
| stage75f_combined_train_only            | True   |
| stage75f_clean_dev_checkpoint_selection | True   |
| stage75f_external_not_training          | True   |
| stage75f_external_not_threshold_tuning  | True   |
| stage75f_time_swap_false                | True   |
| stage75f_bridge_pairwise_excluded       | True   |
| stage75f_pairwise_source_clean_main     | True   |
| stage75f_pairwise_clean_main_2880       | True   |
| stage75f_pairwise_bridge_excluded_2260  | True   |
| stage75f_pairwise_stage75_excluded      | True   |

## Preservation checks

| check                                               | pass   |
|:----------------------------------------------------|:-------|
| stage75f_acc_ge_0p97                                | True   |
| stage75f_macro_ge_0p95                              | True   |
| stage75f_acc_not_below_stage71_by_more_than_0p005   | True   |
| stage75f_macro_not_below_stage71_by_more_than_0p005 | True   |
| stage75f_acc_not_below_stage51_by_more_than_0p005   | True   |
| stage75f_macro_not_below_stage51_by_more_than_0p005 | True   |
| stage75f_acc_ge_stage61                             | True   |
| stage75f_macro_ge_stage61                           | True   |

## Per-label metrics

| stage          | label        |   precision |   recall |       f1 |
|:---------------|:-------------|------------:|---------:|---------:|
| Stage51        | NOT_ENTITLED |    0.99619  | 0.968519 | 0.98216  |
| Stage51        | REFUTE       |    1        | 1        | 1        |
| Stage51        | SUPPORT      |    0.844037 | 0.978723 | 0.906404 |
| Stage61        | NOT_ENTITLED |    0.996176 | 0.964815 | 0.980245 |
| Stage61        | REFUTE       |    0.988506 | 1        | 0.99422  |
| Stage61        | SUPPORT      |    0.836364 | 0.978723 | 0.901961 |
| Stage71_retry2 | NOT_ENTITLED |    1        | 0.966667 | 0.983051 |
| Stage71_retry2 | REFUTE       |    1        | 1        | 1        |
| Stage71_retry2 | SUPPORT      |    0.833333 | 1        | 0.909091 |
| Stage75F       | NOT_ENTITLED |    1        | 0.964815 | 0.982092 |
| Stage75F       | REFUTE       |    1        | 1        | 1        |
| Stage75F       | SUPPORT      |    0.825688 | 1        | 0.904523 |

## Interpretation

Stage75F preserves clean-dev under tolerance, but does not improve over Stage71_retry2 on clean-dev macro-F1.

## Recommended next stage

Stage77 Stage75F VitaminC external diagnostic rerun
