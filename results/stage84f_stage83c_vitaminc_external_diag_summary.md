# Stage84F - Stage83C VitaminC External Diagnostic

## Decision

`STAGE84F_STAGE83C_VITAMINC_EXTERNAL_DIAGNOSTIC_READY`

## External decision

`EXTERNAL_REGRESSION_VS_STAGE71`

## Summary

| stage    | decision                                             | external_decision              | run_dir                                                                          | ood_data                                                             |   external_n |   stage84f_clean_dev_acc |   stage84f_clean_dev_macro_f1 |   stage83c_clean_dev_acc |   stage83c_clean_dev_macro_f1 |   stage84f_external_accuracy |   stage84f_external_macro_f1_all3 | stage84f_external_prediction_counts                  | stage84f_external_gold_counts   |   stage73_external_accuracy |   stage73_external_macro_f1_all3 | stage73_external_prediction_counts                   | stage73_external_gold_counts                         |   stage84f_minus_stage73_external_acc |   stage84f_minus_stage73_external_macro_f1_all3 |   elapsed_min | recommended_next_stage           |
|:---------|:-----------------------------------------------------|:-------------------------------|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------|-------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|-----------------------------:|----------------------------------:|:-----------------------------------------------------|:--------------------------------|----------------------------:|---------------------------------:|:-----------------------------------------------------|:-----------------------------------------------------|--------------------------------------:|------------------------------------------------:|--------------:|:---------------------------------|
| Stage84F | STAGE84F_STAGE83C_VITAMINC_EXTERNAL_DIAGNOSTIC_READY | EXTERNAL_REGRESSION_VS_STAGE71 | results/stage84f_stage83c_vitaminc_external_exact_ood_schema_run_20260703_033759 | data/stage84e2_vitaminc_validation_sample1000_exact_ood_schema.jsonl |         1000 |                 0.979167 |                      0.969664 |                 0.979167 |                      0.969664 |                        0.326 |                          0.306672 | {"NOT_ENTITLED": 439, "REFUTE": 202, "SUPPORT": 359} | {}                              |                       0.353 |                           0.3262 | {"NOT_ENTITLED": 388, "REFUTE": 219, "SUPPORT": 393} | {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500} |                                -0.027 |                                      -0.0195283 |        6.9861 | Stage85 external decision report |

## Metadata checks

| check                               | pass   |
|:------------------------------------|:-------|
| returncode_zero                     | True   |
| train_report_exists                 | True   |
| ood_report_exists                   | True   |
| ood_predictions_exists              | True   |
| stage57_rows_520                    | True   |
| stage66_rows_720                    | True   |
| stage83a_rows_160                   | True   |
| combined_bridge_1400                | True   |
| final_train_4280                    | True   |
| external_metrics_parsed_or_computed | True   |

## Clean checks

| check                         | pass   |
|:------------------------------|:-------|
| clean_acc_ge_0p97             | True   |
| clean_macro_ge_0p95           | True   |
| clean_acc_close_to_stage83c   | True   |
| clean_macro_close_to_stage83c | True   |

## External comparison

| comparison                                            |   stage84f_acc |   stage84f_macro_f1_all3 |   stage73_acc |   stage73_macro_f1_all3 |   acc_delta |   macro_f1_delta | decision                       |
|:------------------------------------------------------|---------------:|-------------------------:|--------------:|------------------------:|------------:|-----------------:|:-------------------------------|
| Stage84F_Stage83C_external - Stage73_Stage71_external |          0.326 |                 0.306672 |         0.353 |                  0.3262 |      -0.027 |       -0.0195283 | EXTERNAL_REGRESSION_VS_STAGE71 |
