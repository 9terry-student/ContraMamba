# Stage73B2 — Stage73 Retry Embedded External Eval Summary Fix

## Decision

`STAGE73B2_STAGE73_RETRY_EMBEDDED_EXTERNAL_EVAL_SUMMARY_READY`

## Summary

| stage     | decision                                                     | purpose                                                                                             | source_run_dir                                                                          | source_train_report                                                                                                                     | source_predictions_json                                                                                                                | source_stdout_log                                                                                                |   best_epoch |   selected_epoch |   best_dev_acc |   best_dev_macro_f1 | external_eval_name     | external_eval_jsonl                                 | external_predictions_path                                                                                                                                        |   external_n_records |   external_accuracy |   external_macro_f1 | external_prediction_distribution                     | external_label_counts                                | external_per_label                                                                                                                                                                                                                                                                                              |   false_SUPPORT_total |   true_SUPPORT_correct | external_schema_normalized   | external_schema_label_field_used   | external_schema_missing_final_label_fixed   |   external_schema_records_normalized |   external_schema_records_with_added_aux_labels | bridge_rows_excluded_from_intervention_pairwise_loss   | intervention_pairwise_loss_source   |   intervention_pairwise_loss_clean_main_row_count |   intervention_pairwise_loss_bridge_row_count_excluded |   intervention_pairwise_loss_stage57_row_count_excluded |   intervention_pairwise_loss_stage66_row_count_excluded |   stage73_minus_stage63_accuracy |   stage73_minus_stage63_macro_f1 |   stage73_minus_stage53_accuracy |   stage73_minus_stage53_macro_f1 | training_executed   | external_eval_executed   | rerun_required   | recommended_next_stage                                                         |
|:----------|:-------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|-------------:|-----------------:|---------------:|--------------------:|:-----------------------|:----------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------:|--------------------:|--------------------:|:-----------------------------------------------------|:-----------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------:|-----------------------:|:-----------------------------|:-----------------------------------|:--------------------------------------------|-------------------------------------:|------------------------------------------------:|:-------------------------------------------------------|:------------------------------------|--------------------------------------------------:|-------------------------------------------------------:|--------------------------------------------------------:|--------------------------------------------------------:|---------------------------------:|---------------------------------:|---------------------------------:|---------------------------------:|:--------------------|:-------------------------|:-----------------|:-------------------------------------------------------------------------------|
| Stage73B2 | STAGE73B2_STAGE73_RETRY_EMBEDDED_EXTERNAL_EVAL_SUMMARY_READY | Correct Stage73 retry embedded external-eval summary using deep metadata lookup; no training rerun. | results/stage73_retry_stage57_stage66_bridge_vitaminc_external_diag_run_20260702_080534 | results/stage73_retry_stage57_stage66_bridge_vitaminc_external_diag_run_20260702_080534/stage73_retry_stage57_stage66_train_report.json | results/stage73_retry_stage57_stage66_bridge_vitaminc_external_diag_run_20260702_080534/stage73_retry_stage57_stage66_predictions.json | results/stage73_retry_stage57_stage66_bridge_vitaminc_external_diag_run_20260702_080534/stage73_retry_stdout.log |          151 |              151 |          0.975 |            0.964047 | stage73_retry_vitaminc | data/stage43b1_vitaminc_validation_sample1000.jsonl | results/stage73_retry_stage57_stage66_bridge_vitaminc_external_diag_run_20260702_080534/external_vitaminc/external_probe_stage73_retry_vitaminc_predictions.json |                 1000 |               0.353 |              0.3262 | {"NOT_ENTITLED": 388, "REFUTE": 219, "SUPPORT": 393} | {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500} | {"NOT_ENTITLED": {"f1": 0.2439, "precision": 0.1675, "predicted": 388, "recall": 0.4483, "support": 145}, "REFUTE": {"f1": 0.2509, "precision": 0.3288, "predicted": 219, "recall": 0.2028, "support": 355}, "SUPPORT": {"f1": 0.4838, "precision": 0.5496, "predicted": 393, "recall": 0.432, "support": 500}} |                   177 |                    216 | True                         | ["label"]                          | True                                        |                                 1000 |                                            1000 | True                                                   | clean_main_train_only               |                                              2880 |                                                   1240 |                                                     520 |                                                     720 |                            0.031 |                         0.010881 |                            0.197 |                         0.210214 | False               | False                    | False            | Stage74 residual external error audit using Stage73 retry external predictions |

## External distribution

| label        |   gold |   predicted |
|:-------------|-------:|------------:|
| NOT_ENTITLED |    145 |         388 |
| REFUTE       |    355 |         219 |
| SUPPORT      |    500 |         393 |

## External per-label metrics

| label        |     f1 |   precision |   predicted |   recall |   support |
|:-------------|-------:|------------:|------------:|---------:|----------:|
| NOT_ENTITLED | 0.2439 |      0.1675 |         388 |   0.4483 |       145 |
| REFUTE       | 0.2509 |      0.3288 |         219 |   0.2028 |       355 |
| SUPPORT      | 0.4838 |      0.5496 |         393 |   0.432  |       500 |

## Comparisons

| comparison                   |   delta_accuracy |   delta_macro_f1 |   stage73_accuracy |   baseline_accuracy |   stage73_macro_f1 |   baseline_macro_f1 |
|:-----------------------------|-----------------:|-----------------:|-------------------:|--------------------:|-------------------:|--------------------:|
| Stage73_retry_minus_Stage53A |            0.197 |         0.210214 |              0.353 |               0.156 |             0.3262 |            0.115986 |
| Stage73_retry_minus_Stage63  |            0.031 |         0.010881 |              0.353 |               0.322 |             0.3262 |            0.315319 |

## Metadata checks

| check                                              | pass   |
|:---------------------------------------------------|:-------|
| train_report_exists                                | True   |
| predictions_json_exists                            | True   |
| stdout_log_exists                                  | True   |
| external_eval_present_in_train_report              | True   |
| external_n_records_1000                            | True   |
| external_accuracy_present                          | True   |
| external_macro_f1_present                          | True   |
| external_schema_normalized_true                    | True   |
| external_schema_missing_final_label_fixed_true     | True   |
| external_schema_records_normalized_1000            | True   |
| external_schema_records_with_added_aux_labels_1000 | True   |
| external_probe_not_used_for_training               | True   |
| external_probe_not_used_for_checkpoint_selection   | True   |
| external_probe_not_used_for_calibration            | True   |
| stage57_enabled                                    | True   |
| stage57_row_count_520                              | True   |
| stage57_train_only                                 | True   |
| stage57_not_dev                                    | True   |
| stage57_not_checkpoint                             | True   |
| stage66_enabled                                    | True   |
| stage66_row_count_720                              | True   |
| stage66_train_only                                 | True   |
| stage66_not_dev                                    | True   |
| stage66_not_checkpoint                             | True   |
| combined_bridge_enabled                            | True   |
| combined_bridge_row_count_1240                     | True   |
| combined_bridge_train_only                         | True   |
| bridge_pairwise_excluded                           | True   |
| stage57_pairwise_excluded                          | True   |
| stage66_pairwise_excluded                          | True   |
| pairwise_source_clean_main                         | True   |
| pairwise_clean_main_2880                           | True   |
| pairwise_bridge_excluded_1240                      | True   |
| pairwise_stage57_excluded_520                      | True   |
| pairwise_stage66_excluded_720                      | True   |
| clean_dev_checkpoint_selection                     | True   |
| external_data_not_used_for_training                | True   |
| external_metrics_not_used_for_threshold_tuning     | True   |
| time_swap_false                                    | True   |

## Interpretation

Stage73 retry completed successfully. The earlier Stage73B post-processing used shallow top-level lookup for several pairwise-loss metadata fields, while the train report stores those fields in nested audit/config sections. This corrected summary uses deep lookup and does not rerun training or external evaluation.

## Recommended next stage

Stage74 residual external error audit using Stage73 retry external predictions
