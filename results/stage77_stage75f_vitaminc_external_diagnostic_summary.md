# Stage77 - Stage75F VitaminC External Diagnostic

## Decision

`STAGE77_STAGE75F_VITAMINC_EXTERNAL_DIAGNOSTIC_READY`

## Summary

| stage   | decision                                            | run_dir                                                                                   | train_report                                                                                                                                | predictions                                                                                                                                | stdout_log                                                                                                   | external_output_dir                                                                                         | external_eval_name   | external_eval_jsonl                                 | external_predictions_path   |   returncode |   elapsed_min |   best_epoch |   selected_epoch |   best_dev_acc |   best_dev_macro_f1 |   external_n_records |   external_accuracy |   external_macro_f1 | external_label_counts                                | external_prediction_distribution                     |   false_SUPPORT_total |   true_SUPPORT_correct |   stage77_minus_stage73_acc |   stage77_minus_stage73_macro_f1 |   stage77_minus_stage63_acc |   stage77_minus_stage63_macro_f1 |   stage77_minus_stage53a_acc |   stage77_minus_stage53a_macro_f1 |   stage57_row_count |   stage66_row_count |   stage75_row_count |   combined_bridge_row_count | pairwise_loss_source   |   pairwise_bridge_row_count_excluded | clean_dev_for_checkpoint_selection   | external_data_used_for_training   | external_metrics_used_for_threshold_tuning   | time_swap_used   | training_executed   | external_eval_executed   | recommended_next_stage                            |
|:--------|:----------------------------------------------------|:------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:---------------------|:----------------------------------------------------|:----------------------------|-------------:|--------------:|-------------:|-----------------:|---------------:|--------------------:|---------------------:|--------------------:|--------------------:|:-----------------------------------------------------|:-----------------------------------------------------|----------------------:|-----------------------:|----------------------------:|---------------------------------:|----------------------------:|---------------------------------:|-----------------------------:|----------------------------------:|--------------------:|--------------------:|--------------------:|----------------------------:|:-----------------------|-------------------------------------:|:-------------------------------------|:----------------------------------|:---------------------------------------------|:-----------------|:--------------------|:-------------------------|:--------------------------------------------------|
| Stage77 | STAGE77_STAGE75F_VITAMINC_EXTERNAL_DIAGNOSTIC_READY | results/stage77_stage57_stage66_stage75_bridge_vitaminc_external_diag_run_20260703_004242 | results/stage77_stage57_stage66_stage75_bridge_vitaminc_external_diag_run_20260703_004242/stage77_stage57_stage66_stage75_train_report.json | results/stage77_stage57_stage66_stage75_bridge_vitaminc_external_diag_run_20260703_004242/stage77_stage57_stage66_stage75_predictions.json | results/stage77_stage57_stage66_stage75_bridge_vitaminc_external_diag_run_20260703_004242/stage77_stdout.log | results/stage77_stage57_stage66_stage75_bridge_vitaminc_external_diag_run_20260703_004242/external_vitaminc | stage77_vitaminc     | data/stage43b1_vitaminc_validation_sample1000.jsonl |                             |            0 |        7.0724 |          184 |              184 |       0.973611 |            0.962205 |                 1000 |               0.357 |              0.3182 | {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500} | {"NOT_ENTITLED": 383, "REFUTE": 182, "SUPPORT": 435} |                   196 |                    239 |                       0.004 |                           -0.008 |                       0.035 |                         0.002881 |                        0.201 |                          0.202214 |                 520 |                 720 |                1020 |                        2260 | clean_main_train_only  |                                 2260 | True                                 | False                             | False                                        | False            | True                | True                     | Stage78 residual external error audit for Stage77 |

## Metadata checks

| check                                     | pass   |
|:------------------------------------------|:-------|
| train_report_exists                       | True   |
| predictions_exists                        | True   |
| stdout_log_exists                         | True   |
| external_eval_present                     | True   |
| external_n_1000                           | True   |
| external_schema_normalized                | True   |
| external_schema_missing_final_label_fixed | True   |
| external_records_normalized_1000          | True   |
| external_probe_not_training               | True   |
| external_probe_not_checkpoint             | True   |
| external_probe_not_calibration            | True   |
| stage57_enabled_520                       | True   |
| stage66_enabled_720                       | True   |
| stage75_enabled_1020                      | True   |
| combined_bridge_2260                      | True   |
| combined_bridge_train_only                | True   |
| bridge_pairwise_excluded                  | True   |
| stage57_pairwise_excluded                 | True   |
| stage66_pairwise_excluded                 | True   |
| stage75_pairwise_excluded                 | True   |
| pairwise_source_clean_main                | True   |
| pairwise_clean_main_2880                  | True   |
| pairwise_bridge_excluded_2260             | True   |
| pairwise_stage75_excluded_1020            | True   |
| clean_dev_checkpoint_selection            | True   |
| external_data_not_training                | True   |
| external_metrics_not_threshold_tuning     | True   |
| time_swap_false                           | True   |

## External label metrics

| label        |   gold_count |   pred_count |   precision |   recall |     f1 |
|:-------------|-------------:|-------------:|------------:|---------:|-------:|
| NOT_ENTITLED |          145 |          383 |      0.154  |   0.4069 | 0.2235 |
| REFUTE       |          355 |          182 |      0.3242 |   0.1662 | 0.2197 |
| SUPPORT      |          500 |          435 |      0.5494 |   0.478  | 0.5112 |

## Comparison

| comparison              |   acc_delta |   macro_f1_delta |   baseline_acc |   baseline_macro_f1 |   stage77_acc |   stage77_macro_f1 |
|:------------------------|------------:|-----------------:|---------------:|--------------------:|--------------:|-------------------:|
| Stage77 - Stage53A      |       0.201 |         0.202214 |          0.156 |            0.115986 |         0.357 |             0.3182 |
| Stage77 - Stage63       |       0.035 |         0.002881 |          0.322 |            0.315319 |         0.357 |             0.3182 |
| Stage77 - Stage73_retry |       0.004 |        -0.008    |          0.353 |            0.3262   |         0.357 |             0.3182 |

## Command

python scripts/train_controlled_v6b_minimal.py --data data/controlled_v5_v3_without_time_swap.jsonl --backbone mamba --model-name state-spaces/mamba-130m-hf --freeze-encoder true --epochs 200 --max-length 128 --dev-ratio 0.2 --seed 1 --device cuda --flag-source controlled_heuristic --select-metric final_macro_f1 --use-intervention-loss --use-stage47-selected-recovery-config --stage47-recovery-config-path results/stage47_selected_recovery_config_check.json --stage57-bridge-train-jsonl data/stage57_nonleaking_external_bridge.jsonl --stage57-bridge-train-mode append_train_only --stage66-bridge-train-jsonl data/stage66_residual_bridge.jsonl --stage66-bridge-train-mode append_train_only --stage75-bridge-train-jsonl data/stage75_targeted_residual_bridge.jsonl --stage75-bridge-train-mode append_train_only --external-eval-jsonl data/stage43b1_vitaminc_validation_sample1000.jsonl --external-eval-name stage77_vitaminc --external-output-dir /kaggle/working/ContraMamba/results/stage77_stage57_stage66_stage75_bridge_vitaminc_external_diag_run_20260703_004242/external_vitaminc --output-json /kaggle/working/ContraMamba/results/stage77_stage57_stage66_stage75_bridge_vitaminc_external_diag_run_20260703_004242/stage77_stage57_stage66_stage75_train_report.json --output-predictions-json /kaggle/working/ContraMamba/results/stage77_stage57_stage66_stage75_bridge_vitaminc_external_diag_run_20260703_004242/stage77_stage57_stage66_stage75_predictions.json

## Recommended next stage

Stage78 residual external error audit for Stage77
