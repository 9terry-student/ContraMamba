# Stage80F - Stage57 + Stage66 + Stage80A Conservative Bridge Frozen Recovery Run

## Decision

`STAGE80F_STAGE57_STAGE66_STAGE80A_CONSERVATIVE_BRIDGE_FROZEN_RECOVERY_RUN_NEEDS_REVIEW`

## Summary

| stage    | decision                                                                               | run_dir                                                                                           | train_report                                                                                                                                          | predictions                                                                                                                                          | stdout_log                                                                                                            |   returncode |   elapsed_min |   best_epoch |   selected_epoch |   best_dev_acc |   best_dev_macro_f1 | prediction_distribution                             |   stage57_bridge_rows |   stage66_bridge_rows |   stage75_bridge_rows |   stage80a_bridge_rows |   combined_bridge_rows |   final_train_row_count_expected | pairwise_loss_source   |   pairwise_clean_main_row_count |   pairwise_bridge_rows_excluded |   pairwise_stage80a_rows_excluded |   stage71_best_dev_acc |   stage71_best_dev_macro_f1 |   stage75f_best_dev_acc |   stage75f_best_dev_macro_f1 |   stage80f_minus_stage71_acc |   stage80f_minus_stage71_macro_f1 |   stage80f_minus_stage75f_acc |   stage80f_minus_stage75f_macro_f1 | stage75_full_bridge_used   | training_executed   | external_eval_executed   | recommended_next_stage                                                         |
|:---------|:---------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|-------------:|--------------:|-------------:|-----------------:|---------------:|--------------------:|:----------------------------------------------------|----------------------:|----------------------:|----------------------:|-----------------------:|-----------------------:|---------------------------------:|:-----------------------|--------------------------------:|--------------------------------:|----------------------------------:|-----------------------:|----------------------------:|------------------------:|-----------------------------:|-----------------------------:|----------------------------------:|------------------------------:|-----------------------------------:|:---------------------------|:--------------------|:-------------------------|:-------------------------------------------------------------------------------|
| Stage80F | STAGE80F_STAGE57_STAGE66_STAGE80A_CONSERVATIVE_BRIDGE_FROZEN_RECOVERY_RUN_NEEDS_REVIEW | results/stage80f_stage57_stage66_stage80a_conservative_bridge_frozen_recovery_run_20260703_013106 | results/stage80f_stage57_stage66_stage80a_conservative_bridge_frozen_recovery_run_20260703_013106/stage80f_stage57_stage66_stage80a_train_report.json | results/stage80f_stage57_stage66_stage80a_conservative_bridge_frozen_recovery_run_20260703_013106/stage80f_stage57_stage66_stage80a_predictions.json | results/stage80f_stage57_stage66_stage80a_conservative_bridge_frozen_recovery_run_20260703_013106/stage80f_stdout.log |            0 |        6.9887 |          180 |              180 |       0.970833 |            0.958564 | {"NOT_ENTITLED": 519, "REFUTE": 90, "SUPPORT": 111} |                   520 |                   720 |                     0 |                    500 |                   1740 |                             4620 | clean_main_train_only  |                            2880 |                            1740 |                               500 |                  0.975 |                    0.964047 |                0.973611 |                     0.962205 |                  -0.00416666 |                       -0.00548313 |                   -0.00277776 |                        -0.00364087 | False                      | True                | False                    | Stage80G clean-dev preservation comparison for Stage71 vs Stage75F vs Stage80F |

## Metadata checks

| check                                 | pass   |
|:--------------------------------------|:-------|
| train_report_exists                   | True   |
| predictions_exists                    | True   |
| stdout_log_exists                     | True   |
| stage57_enabled_520                   | True   |
| stage66_enabled_720                   | True   |
| stage80a_enabled_500                  | True   |
| stage75_full_not_enabled              | True   |
| combined_bridge_1740                  | True   |
| final_train_expected_4620             | True   |
| pairwise_source_clean_main            | True   |
| pairwise_clean_main_2880              | True   |
| pairwise_bridge_excluded_1740         | True   |
| pairwise_stage80a_excluded_500        | True   |
| stage80a_excluded_from_pairwise       | True   |
| bridge_pairwise_excluded              | True   |
| combined_bridge_train_only            | True   |
| clean_dev_checkpoint_selection        | True   |
| external_data_not_training            | True   |
| external_metrics_not_threshold_tuning | True   |
| time_swap_false                       | True   |

## Preservation checks

| check                                                | pass   |
|:-----------------------------------------------------|:-------|
| stage80f_acc_ge_0p97                                 | True   |
| stage80f_macro_ge_0p95                               | True   |
| stage80f_acc_not_below_stage71_by_more_than_0p005    | True   |
| stage80f_macro_not_below_stage71_by_more_than_0p005  | False  |
| stage80f_acc_not_below_stage75f_by_more_than_0p005   | True   |
| stage80f_macro_not_below_stage75f_by_more_than_0p005 | True   |

## Per-label metrics

| label        |   precision |   recall |       f1 |
|:-------------|------------:|---------:|---------:|
| NOT_ENTITLED |    1        | 0.961111 | 0.98017  |
| REFUTE       |    1        | 1        | 1        |
| SUPPORT      |    0.810811 | 1        | 0.895522 |

## Comparison

| comparison                |   acc_delta |   macro_f1_delta |   baseline_acc |   baseline_macro_f1 |   stage80f_acc |   stage80f_macro_f1 |
|:--------------------------|------------:|-----------------:|---------------:|--------------------:|---------------:|--------------------:|
| Stage80F - Stage71_retry2 | -0.00416666 |      -0.00548313 |       0.975    |            0.964047 |       0.970833 |            0.958564 |
| Stage80F - Stage75F       | -0.00277776 |      -0.00364087 |       0.973611 |            0.962205 |       0.970833 |            0.958564 |

## Command

python scripts/train_controlled_v6b_minimal.py --data data/controlled_v5_v3_without_time_swap.jsonl --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --epochs 200 --max-length 128 --dev-ratio 0.2 --seed 1 --device cuda --flag-source controlled_heuristic --select-metric final_macro_f1 --use-intervention-loss --use-stage47-selected-recovery-config --stage47-recovery-config-path results/stage47_selected_recovery_config_check.json --stage57-bridge-train-jsonl data/stage57_nonleaking_external_bridge.jsonl --stage57-bridge-train-mode append_train_only --stage66-bridge-train-jsonl data/stage66_residual_bridge.jsonl --stage66-bridge-train-mode append_train_only --stage80a-bridge-train-jsonl data/stage80a_conservative_stage75v2_bridge.jsonl --stage80a-bridge-train-mode append_train_only --output-json /kaggle/working/ContraMamba/results/stage80f_stage57_stage66_stage80a_conservative_bridge_frozen_recovery_run_20260703_013106/stage80f_stage57_stage66_stage80a_train_report.json --output-predictions-json /kaggle/working/ContraMamba/results/stage80f_stage57_stage66_stage80a_conservative_bridge_frozen_recovery_run_20260703_013106/stage80f_stage57_stage66_stage80a_predictions.json

## Recommended next stage

Stage80G clean-dev preservation comparison for Stage71 vs Stage75F vs Stage80F
