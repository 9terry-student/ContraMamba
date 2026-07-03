# Stage75F - Stage57+66+75 Bridge-Enabled Frozen Recovery Run

## Decision

`STAGE75F_STAGE57_STAGE66_STAGE75_BRIDGE_ENABLED_FROZEN_RECOVERY_RUN_READY`

## Summary

| stage    | decision                                                                  | run_dir                                                                                     | train_report                                                                                                                                   | predictions                                                                                                                                   | stdout_log                                                                                                      |   returncode |   elapsed_min |   best_epoch |   selected_epoch |   best_dev_acc |   best_dev_macro_f1 |   stage57_row_count |   stage66_row_count |   stage75_row_count |   combined_bridge_row_count |   final_train_row_count_expected | pairwise_loss_source   |   pairwise_clean_main_row_count |   pairwise_bridge_row_count_excluded |   pairwise_stage75_row_count_excluded | clean_dev_for_checkpoint_selection   | external_data_used_for_training   | external_metrics_used_for_threshold_tuning   | time_swap_used   | training_executed   | external_eval_executed   | recommended_next_stage                                     |
|:---------|:--------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------|-------------:|--------------:|-------------:|-----------------:|---------------:|--------------------:|--------------------:|--------------------:|--------------------:|----------------------------:|---------------------------------:|:-----------------------|--------------------------------:|-------------------------------------:|--------------------------------------:|:-------------------------------------|:----------------------------------|:---------------------------------------------|:-----------------|:--------------------|:-------------------------|:-----------------------------------------------------------|
| Stage75F | STAGE75F_STAGE57_STAGE66_STAGE75_BRIDGE_ENABLED_FROZEN_RECOVERY_RUN_READY | results/stage75f_stage57_stage66_stage75_bridge_enabled_frozen_recovery_run_20260703_002555 | results/stage75f_stage57_stage66_stage75_bridge_enabled_frozen_recovery_run_20260703_002555/stage75f_stage57_stage66_stage75_train_report.json | results/stage75f_stage57_stage66_stage75_bridge_enabled_frozen_recovery_run_20260703_002555/stage75f_stage57_stage66_stage75_predictions.json | results/stage75f_stage57_stage66_stage75_bridge_enabled_frozen_recovery_run_20260703_002555/stage75f_stdout.log |            0 |        7.0809 |          184 |              184 |       0.973611 |            0.962205 |                 520 |                 720 |                1020 |                        2260 |                             5140 | clean_main_train_only  |                            2880 |                                 2260 |                                  1020 | True                                 | False                             | False                                        | False            | True                | False                    | Stage76 clean-dev preservation comparison vs Stage51/61/71 |

## Metadata checks

| check                          | pass   |
|:-------------------------------|:-------|
| train_report_exists            | True   |
| predictions_exists             | True   |
| stdout_log_exists              | True   |
| stage57_enabled                | True   |
| stage57_row_count_520          | True   |
| stage57_train_only             | True   |
| stage57_not_dev                | True   |
| stage57_not_checkpoint         | True   |
| stage66_enabled                | True   |
| stage66_row_count_720          | True   |
| stage66_train_only             | True   |
| stage66_not_dev                | True   |
| stage66_not_checkpoint         | True   |
| stage75_enabled                | True   |
| stage75_row_count_1020         | True   |
| stage75_train_only             | True   |
| stage75_not_dev                | True   |
| stage75_not_checkpoint         | True   |
| combined_bridge_2260           | True   |
| combined_bridge_train_only     | True   |
| bridge_pairwise_excluded       | True   |
| stage57_pairwise_excluded      | True   |
| stage66_pairwise_excluded      | True   |
| stage75_pairwise_excluded      | True   |
| pairwise_source_clean_main     | True   |
| pairwise_clean_main_2880       | True   |
| pairwise_bridge_excluded_2260  | True   |
| pairwise_stage57_excluded_520  | True   |
| pairwise_stage66_excluded_720  | True   |
| pairwise_stage75_excluded_1020 | True   |
| clean_dev_checkpoint_selection | True   |
| external_not_training          | True   |
| external_not_threshold_tuning  | True   |
| time_swap_false                | True   |

## Best dev metrics

|   final_accuracy |   final_macro_f1 |   frame_accuracy |   polarity_accuracy_entitled |   predicate_accuracy |   sufficiency_accuracy |
|-----------------:|-----------------:|-----------------:|-----------------------------:|---------------------:|-----------------------:|
|         0.973611 |         0.962205 |         0.972222 |                            1 |             0.969444 |                      1 |

## Command

python scripts/train_controlled_v6b_minimal.py --data data/controlled_v5_v3_without_time_swap.jsonl --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --epochs 200 --max-length 128 --dev-ratio 0.2 --seed 1 --device cuda --flag-source controlled_heuristic --select-metric final_macro_f1 --use-intervention-loss --use-stage47-selected-recovery-config --stage47-recovery-config-path results/stage47_selected_recovery_config_check.json --stage57-bridge-train-jsonl data/stage57_nonleaking_external_bridge.jsonl --stage57-bridge-train-mode append_train_only --stage66-bridge-train-jsonl data/stage66_residual_bridge.jsonl --stage66-bridge-train-mode append_train_only --stage75-bridge-train-jsonl data/stage75_targeted_residual_bridge.jsonl --stage75-bridge-train-mode append_train_only --output-json /kaggle/working/ContraMamba/results/stage75f_stage57_stage66_stage75_bridge_enabled_frozen_recovery_run_20260703_002555/stage75f_stage57_stage66_stage75_train_report.json --output-predictions-json /kaggle/working/ContraMamba/results/stage75f_stage57_stage66_stage75_bridge_enabled_frozen_recovery_run_20260703_002555/stage75f_stage57_stage66_stage75_predictions.json

## Recommended next stage

Stage76 clean-dev preservation comparison vs Stage51/61/71
