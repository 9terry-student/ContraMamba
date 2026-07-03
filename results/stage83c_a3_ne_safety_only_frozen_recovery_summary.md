# Stage83C - A3 NE-Safety-Only Frozen Recovery Run

## Decision

`STAGE83C_A3_NE_SAFETY_ONLY_FROZEN_RECOVERY_RUN_READY`

## Summary

| stage    | decision                                             | run_dir                                                                                      | train_report                                                                                                                                     | predictions                                                                                                                                     | stdout_log                                                                                                       |   returncode |   elapsed_min |   best_epoch |   selected_epoch |   best_dev_acc |   best_dev_macro_f1 | prediction_distribution                             |   stage57_bridge_rows |   stage66_bridge_rows |   stage75_bridge_rows |   stage83a_rows_via_stage80a_bridge |   combined_bridge_rows |   final_train_row_count_expected | pairwise_loss_source   |   pairwise_clean_main_row_count |   pairwise_bridge_rows_excluded |   pairwise_stage83a_rows_excluded |   stage71_best_dev_acc |   stage71_best_dev_macro_f1 |   stage75f_best_dev_acc |   stage75f_best_dev_macro_f1 |   stage80f_best_dev_acc |   stage80f_best_dev_macro_f1 |   stage83c_minus_stage71_acc |   stage83c_minus_stage71_macro_f1 |   stage83c_minus_stage75f_acc |   stage83c_minus_stage75f_macro_f1 |   stage83c_minus_stage80f_acc |   stage83c_minus_stage80f_macro_f1 | stage75_full_bridge_used   | stage80a_full_bridge_used   | stage83a_ne_safety_only_used   | training_executed   | external_eval_executed   | recommended_next_stage                                          |
|:---------|:-----------------------------------------------------|:---------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|-------------:|--------------:|-------------:|-----------------:|---------------:|--------------------:|:----------------------------------------------------|----------------------:|----------------------:|----------------------:|------------------------------------:|-----------------------:|---------------------------------:|:-----------------------|--------------------------------:|--------------------------------:|----------------------------------:|-----------------------:|----------------------------:|------------------------:|-----------------------------:|------------------------:|-----------------------------:|-----------------------------:|----------------------------------:|------------------------------:|-----------------------------------:|------------------------------:|-----------------------------------:|:---------------------------|:----------------------------|:-------------------------------|:--------------------|:-------------------------|:----------------------------------------------------------------|
| Stage83C | STAGE83C_A3_NE_SAFETY_ONLY_FROZEN_RECOVERY_RUN_READY | results/stage83c_stage57_stage66_stage83a_ne_safety_only_frozen_recovery_run_20260703_020152 | results/stage83c_stage57_stage66_stage83a_ne_safety_only_frozen_recovery_run_20260703_020152/stage83c_stage57_stage66_stage83a_train_report.json | results/stage83c_stage57_stage66_stage83a_ne_safety_only_frozen_recovery_run_20260703_020152/stage83c_stage57_stage66_stage83a_predictions.json | results/stage83c_stage57_stage66_stage83a_ne_safety_only_frozen_recovery_run_20260703_020152/stage83c_stdout.log |            0 |        6.5685 |          158 |              158 |       0.979167 |            0.969664 | {"NOT_ENTITLED": 525, "REFUTE": 90, "SUPPORT": 105} |                   520 |                   720 |                     0 |                                 160 |                   1400 |                             4280 | clean_main_train_only  |                            2880 |                            1400 |                               160 |                  0.975 |                    0.964047 |                0.973611 |                     0.962205 |                0.970833 |                     0.958564 |                   0.00416666 |                        0.00561689 |                    0.00555557 |                         0.00745915 |                    0.00833333 |                             0.0111 | False                      | False                       | True                           | True                | False                    | Stage83D clean-dev decision report for A3 NE-safety-only bridge |

## Metadata checks

| check                                 | pass   |
|:--------------------------------------|:-------|
| train_report_exists                   | True   |
| predictions_exists                    | True   |
| stdout_log_exists                     | True   |
| stage57_enabled_520                   | True   |
| stage66_enabled_720                   | True   |
| stage83a_via_stage80a_enabled_160     | True   |
| stage75_full_not_enabled              | True   |
| combined_bridge_1400                  | True   |
| final_train_expected_4280             | True   |
| pairwise_source_clean_main            | True   |
| pairwise_clean_main_2880              | True   |
| pairwise_bridge_excluded_1400         | True   |
| pairwise_stage83a_excluded_160        | True   |
| stage83a_excluded_from_pairwise       | True   |
| bridge_pairwise_excluded              | True   |
| combined_bridge_train_only            | True   |
| clean_dev_checkpoint_selection        | True   |
| external_data_not_training            | True   |
| external_metrics_not_threshold_tuning | True   |
| time_swap_false                       | True   |

## Preservation checks

| check                                               | pass   |
|:----------------------------------------------------|:-------|
| stage83c_acc_ge_0p97                                | True   |
| stage83c_macro_ge_0p95                              | True   |
| stage83c_acc_not_below_stage71_by_more_than_0p005   | True   |
| stage83c_macro_not_below_stage71_by_more_than_0p005 | True   |
| stage83c_macro_better_than_stage80f                 | True   |
| stage83c_acc_better_than_stage80f                   | True   |

## Per-label metrics

| label        |   precision |   recall |       f1 |
|:-------------|------------:|---------:|---------:|
| NOT_ENTITLED |    1        | 0.972222 | 0.985915 |
| REFUTE       |    1        | 1        | 1        |
| SUPPORT      |    0.857143 | 1        | 0.923077 |

## Comparison

| comparison                |   acc_delta |   macro_f1_delta |   baseline_acc |   baseline_macro_f1 |   stage83c_acc |   stage83c_macro_f1 |
|:--------------------------|------------:|-----------------:|---------------:|--------------------:|---------------:|--------------------:|
| Stage83C - Stage71_retry2 |  0.00416666 |       0.00561689 |       0.975    |            0.964047 |       0.979167 |            0.969664 |
| Stage83C - Stage75F       |  0.00555557 |       0.00745915 |       0.973611 |            0.962205 |       0.979167 |            0.969664 |
| Stage83C - Stage80F       |  0.00833333 |       0.0111     |       0.970833 |            0.958564 |       0.979167 |            0.969664 |

## Command

python scripts/train_controlled_v6b_minimal.py --data data/controlled_v5_v3_without_time_swap.jsonl --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --epochs 200 --max-length 128 --dev-ratio 0.2 --seed 1 --device cuda --flag-source controlled_heuristic --select-metric final_macro_f1 --use-intervention-loss --use-stage47-selected-recovery-config --stage47-recovery-config-path results/stage47_selected_recovery_config_check.json --stage57-bridge-train-jsonl data/stage57_nonleaking_external_bridge.jsonl --stage57-bridge-train-mode append_train_only --stage66-bridge-train-jsonl data/stage66_residual_bridge.jsonl --stage66-bridge-train-mode append_train_only --stage80a-bridge-train-jsonl data/stage83a_ne_safety_only_bridge.jsonl --stage80a-bridge-train-mode append_train_only --output-json /kaggle/working/ContraMamba/results/stage83c_stage57_stage66_stage83a_ne_safety_only_frozen_recovery_run_20260703_020152/stage83c_stage57_stage66_stage83a_train_report.json --output-predictions-json /kaggle/working/ContraMamba/results/stage83c_stage57_stage66_stage83a_ne_safety_only_frozen_recovery_run_20260703_020152/stage83c_stage57_stage66_stage83a_predictions.json

## Recommended next stage

Stage83D clean-dev decision report for A3 NE-safety-only bridge
