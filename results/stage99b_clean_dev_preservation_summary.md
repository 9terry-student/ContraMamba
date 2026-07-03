# Stage99B - Clean-dev Preservation Run

## Decision

`STAGE99B_CLEAN_DEV_PRESERVATION_READY`

## Candidate status

`ALLOW_EXTERNAL_DIAGNOSTIC`

## Summary

| stage    | decision                              | candidate_status          | run_dir                                                                                   | bridge_config                                                                                                                |   stage99b_best_epoch |   stage99b_clean_dev_acc |   stage99b_clean_dev_macro_f1 | stage99b_prediction_counts                          |   stage71_best_epoch |   stage71_clean_dev_acc |   stage71_clean_dev_macro_f1 | stage71_prediction_counts                           |   stage97b_best_epoch |   stage97b_clean_dev_acc |   stage97b_clean_dev_macro_f1 | stage97b_prediction_counts                          |   stage95b_best_epoch |   stage95b_clean_dev_acc |   stage95b_clean_dev_macro_f1 | stage95b_prediction_counts                          |   stage99b_minus_stage71_acc |   stage99b_minus_stage71_macro_f1 |   stage99b_minus_stage97b_acc |   stage99b_minus_stage97b_macro_f1 |   stage99b_minus_stage95b_acc |   stage99b_minus_stage95b_macro_f1 |   elapsed_min | recommended_next_stage                |
|:---------|:--------------------------------------|:--------------------------|:------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|----------------------:|-------------------------:|------------------------------:|:----------------------------------------------------|---------------------:|------------------------:|-----------------------------:|:----------------------------------------------------|----------------------:|-------------------------:|------------------------------:|:----------------------------------------------------|----------------------:|-------------------------:|------------------------------:|:----------------------------------------------------|-----------------------------:|----------------------------------:|------------------------------:|-----------------------------------:|------------------------------:|-----------------------------------:|--------------:|:--------------------------------------|
| Stage99B | STAGE99B_CLEAN_DEV_PRESERVATION_READY | ALLOW_EXTERNAL_DIAGNOSTIC | results/stage99b_stage57_stage66_stage92a_stage97a_stage99a_clean_dev_run_20260703_071017 | Stage57 + Stage66 + combined Stage92A+Stage97A+Stage99A via stage80a bridge slot; no Stage83A; no Stage88A; no external eval |                   171 |                 0.976389 |                      0.965904 | {"NOT_ENTITLED": 523, "REFUTE": 90, "SUPPORT": 107} |                  151 |                   0.975 |                     0.964047 | {"NOT_ENTITLED": 522, "REFUTE": 90, "SUPPORT": 108} |                   200 |                 0.976389 |                      0.965617 | {"NOT_ENTITLED": 523, "REFUTE": 91, "SUPPORT": 106} |                   171 |                 0.970833 |                      0.958564 | {"NOT_ENTITLED": 519, "REFUTE": 90, "SUPPORT": 111} |                   0.00138891 |                        0.00185712 |                  -6.87256e-08 |                         0.00028737 |                    0.00555557 |                         0.00734025 |        6.3523 | Stage99C VitaminC external diagnostic |

## Metadata checks

| check                                                 | pass   |
|:------------------------------------------------------|:-------|
| returncode_zero                                       | True   |
| train_report_exists                                   | True   |
| predictions_exists                                    | True   |
| stage57_rows_520                                      | True   |
| stage66_rows_720                                      | True   |
| stage80a_combined_stage92a_stage97a_stage99a_rows_352 | True   |
| combined_bridge_1592                                  | True   |
| final_train_4472                                      | True   |
| no_external_eval                                      | True   |

## Clean checks

| check                               | pass   |
|:------------------------------------|:-------|
| clean_acc_ge_stage71_minus_0p003    | True   |
| clean_macro_ge_stage71_minus_0p003  | True   |
| clean_acc_ge_0p972                  | True   |
| clean_macro_ge_0p961                | True   |
| clean_macro_ge_stage97b_minus_0p003 | True   |
| clean_not_stage95b_regression       | True   |
