# Stage99C - VitaminC External Diagnostic

## Decision

`STAGE99C_VITAMINC_EXTERNAL_DIAGNOSTIC_READY`

## Candidate status

`CLEAN_OK_EXTERNAL_MIXED_OR_FAILED`

## Summary

| stage    | decision                                    | candidate_status                  | promotion_candidate   | run_dir                                                                                           | bridge_config                                                                                                                                 |   stage99c_clean_best_epoch |   stage99c_clean_dev_acc |   stage99c_clean_dev_macro_f1 |   stage99b_clean_dev_acc |   stage99b_clean_dev_macro_f1 | external_eval_name                                        | external_report_storage                 | separate_external_prediction_file_expected   |   stage99c_external_n |   stage99c_external_acc |   stage99c_external_macro_f1_all3 | stage99c_external_prediction_counts                  | stage99c_external_gold_counts                        |   stage99c_support_recall |   stage99c_refute_recall |   stage99c_ne_recall |   stage99c_support_f1 |   stage99c_refute_f1 |   stage99c_ne_f1 |   stage99c_false_SUPPORT_logged |   elapsed_min | recommended_next_stage                  |
|:---------|:--------------------------------------------|:----------------------------------|:----------------------|:--------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|:----------------------------------------------------------|:----------------------------------------|:---------------------------------------------|----------------------:|------------------------:|----------------------------------:|:-----------------------------------------------------|:-----------------------------------------------------|--------------------------:|-------------------------:|---------------------:|----------------------:|---------------------:|-----------------:|--------------------------------:|--------------:|:----------------------------------------|
| Stage99C | STAGE99C_VITAMINC_EXTERNAL_DIAGNOSTIC_READY | CLEAN_OK_EXTERNAL_MIXED_OR_FAILED | False                 | results/stage99c_stage57_stage66_stage92a_stage97a_stage99a_vitaminc_external_run_20260703_071946 | Stage57 + Stage66 + combined Stage92A+Stage97A+Stage99A via stage80a bridge slot; no Stage83A; no Stage88A; VitaminC external diagnostic only |                         171 |                 0.976389 |                      0.965904 |                 0.976389 |                      0.965904 | stage84e2_vitaminc_validation_sample1000_exact_ood_schema | embedded_in_train_report_external_evals | False                                        |                  1000 |                   0.344 |                            0.3352 | {"NOT_ENTITLED": 408, "REFUTE": 314, "SUPPORT": 278} | {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500} |                     0.316 |                   0.3296 |               0.4759 |                0.4062 |               0.3498 |           0.2495 |                             120 |        6.8375 | Stage100 final external decision report |

## Comparison

| run                                        |   external_acc |   external_macro_f1_all3 |   support_recall |   refute_recall | prediction_counts                                    | status                            |
|:-------------------------------------------|---------------:|-------------------------:|-----------------:|----------------:|:-----------------------------------------------------|:----------------------------------|
| Stage73 / current primary                  |          0.353 |                 0.326179 |            0.432 |        0.202817 | {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393} | CURRENT_PRIMARY                   |
| Stage92C / near-miss                       |          0.351 |                 0.330837 |            0.402 |        0.233803 | {"REFUTE": 223, "NOT_ENTITLED": 412, "SUPPORT": 365} | NEAR_MISS                         |
| Stage97C / top-line win support-suppressed |          0.355 |                 0.343    |            0.348 |        0.3183   | {"NOT_ENTITLED": 394, "REFUTE": 296, "SUPPORT": 310} | TOPLINE_WIN_SUPPORT_SUPPRESSED    |
| Stage99C / support-floor micro             |          0.344 |                 0.3352   |            0.316 |        0.3296   | {"NOT_ENTITLED": 408, "REFUTE": 314, "SUPPORT": 278} | CLEAN_OK_EXTERNAL_MIXED_OR_FAILED |

## Delta

| comparison          |   external_acc_delta |   external_macro_delta |   support_recall_delta |   refute_recall_delta |   support_pred_count_delta |   refute_pred_count_delta |   ne_pred_count_delta |
|:--------------------|---------------------:|-----------------------:|-----------------------:|----------------------:|---------------------------:|--------------------------:|----------------------:|
| Stage99C - Stage73  |               -0.009 |             0.00902129 |                 -0.116 |              0.126783 |                       -115 |                        95 |                    20 |
| Stage99C - Stage97C |               -0.011 |            -0.0078     |                 -0.032 |              0.0113   |                        -32 |                        18 |                    14 |
| Stage99C - Stage92C |               -0.007 |             0.004363   |                 -0.086 |              0.095797 |                        -87 |                        91 |                    -4 |

## Metadata checks

| check                                                 | pass   |
|:------------------------------------------------------|:-------|
| returncode_zero                                       | True   |
| train_report_exists                                   | True   |
| clean_predictions_exists                              | True   |
| embedded_external_eval_found                          | True   |
| external_n_1000                                       | True   |
| stage57_rows_520                                      | True   |
| stage66_rows_720                                      | True   |
| stage80a_combined_stage92a_stage97a_stage99a_rows_352 | True   |
| combined_bridge_1592                                  | True   |
| final_train_4472                                      | True   |
| external_data_not_used_for_training                   | True   |
| external_metrics_not_used_for_threshold_tuning        | True   |

## Clean checks

| check                              | pass   |
|:-----------------------------------|:-------|
| clean_acc_ge_stage71_minus_0p003   | True   |
| clean_macro_ge_stage71_minus_0p003 | True   |
| clean_acc_ge_0p972                 | True   |
| clean_macro_ge_0p961               | True   |

## External checks

| check                                   | pass   |
|:----------------------------------------|:-------|
| external_acc_ge_stage73                 | False  |
| external_macro_ge_stage73               | True   |
| external_acc_ge_stage97c                | False  |
| external_macro_ge_stage97c_minus_0p003  | False  |
| support_recall_ge_stage97c              | False  |
| support_recall_ge_stage92c_minus_0p02   | False  |
| support_pred_count_ge_stage97c_plus_20  | False  |
| support_pred_count_ge_stage92c_minus_30 | False  |
| refute_recall_not_below_stage92c        | True   |

## Per-label metrics

| label        |     f1 |   precision |   predicted |   recall |   support |
|:-------------|-------:|------------:|------------:|---------:|----------:|
| NOT_ENTITLED | 0.2495 |      0.1691 |         408 |   0.4759 |       145 |
| REFUTE       | 0.3498 |      0.3726 |         314 |   0.3296 |       355 |
| SUPPORT      | 0.4062 |      0.5683 |         278 |   0.316  |       500 |

## External scalar paths

| path                                                       | value       |
|:-----------------------------------------------------------|:------------|
| external_schema_label_field_used[0]                        | final_label |
| external_schema_missing_final_label_fixed                  | False       |
| external_schema_records_with_added_aux_labels              | 0           |
| false_SUPPORT_by_intervention.vitaminc_external_validation | 120         |
| false_SUPPORT_total                                        | 120         |
| final_accuracy                                             | 0.344       |
| final_macro_f1                                             | 0.3352      |
| intervention_counts.vitaminc_external_validation           | 1000        |
| label_counts.NOT_ENTITLED                                  | 145         |
| label_counts.REFUTE                                        | 355         |
| label_counts.SUPPORT                                       | 500         |
| output_predictions_path                                    |             |
| per_label.NOT_ENTITLED.f1                                  | 0.2495      |
| per_label.NOT_ENTITLED.precision                           | 0.1691      |
| per_label.NOT_ENTITLED.predicted                           | 408         |
| per_label.NOT_ENTITLED.recall                              | 0.4759      |
| per_label.NOT_ENTITLED.support                             | 145         |
| per_label.REFUTE.f1                                        | 0.3498      |
| per_label.REFUTE.precision                                 | 0.3726      |
| per_label.REFUTE.predicted                                 | 314         |
| per_label.REFUTE.recall                                    | 0.3296      |
| per_label.REFUTE.support                                   | 355         |
| per_label.SUPPORT.f1                                       | 0.4062      |
| per_label.SUPPORT.precision                                | 0.5683      |
| per_label.SUPPORT.predicted                                | 278         |
| per_label.SUPPORT.recall                                   | 0.316       |
| per_label.SUPPORT.support                                  | 500         |
| prediction_distribution.NOT_ENTITLED                       | 408         |
| prediction_distribution.REFUTE                             | 314         |
| prediction_distribution.SUPPORT                            | 278         |
| true_SUPPORT_correct                                       | 158         |
