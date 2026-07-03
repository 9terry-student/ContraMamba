# Stage95B - Clean-dev Preservation Run

## Decision

`STAGE95B_CLEAN_DEV_PRESERVATION_FAILED_OR_NEEDS_REVIEW`

## Candidate status

`REJECT_BEFORE_EXTERNAL_OR_NEEDS_REVIEW`

## Summary

| stage    | decision                                               | candidate_status                       | run_dir                                                                          | bridge_config                                                                                                       |   stage95b_best_epoch |   stage95b_clean_dev_acc |   stage95b_clean_dev_macro_f1 | stage95b_prediction_counts                          |   stage71_best_epoch |   stage71_clean_dev_acc |   stage71_clean_dev_macro_f1 | stage71_prediction_counts                           |   stage92b_best_epoch |   stage92b_clean_dev_acc |   stage92b_clean_dev_macro_f1 | stage92b_prediction_counts                          |   stage95b_minus_stage71_acc |   stage95b_minus_stage71_macro_f1 |   stage95b_minus_stage92b_acc |   stage95b_minus_stage92b_macro_f1 |   elapsed_min | recommended_next_stage              |
|:---------|:-------------------------------------------------------|:---------------------------------------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|----------------------:|-------------------------:|------------------------------:|:----------------------------------------------------|---------------------:|------------------------:|-----------------------------:|:----------------------------------------------------|----------------------:|-------------------------:|------------------------------:|:----------------------------------------------------|-----------------------------:|----------------------------------:|------------------------------:|-----------------------------------:|--------------:|:------------------------------------|
| Stage95B | STAGE95B_CLEAN_DEV_PRESERVATION_FAILED_OR_NEEDS_REVIEW | REJECT_BEFORE_EXTERNAL_OR_NEEDS_REVIEW | results/stage95b_stage57_stage66_stage92a_stage95a_clean_dev_run_20260703_054948 | Stage57 + Stage66 + combined Stage92A+Stage95A via stage80a bridge slot; no Stage83A; no Stage88A; no external eval |                   171 |                 0.970833 |                      0.958564 | {"NOT_ENTITLED": 519, "REFUTE": 90, "SUPPORT": 111} |                  151 |                   0.975 |                     0.964047 | {"NOT_ENTITLED": 522, "REFUTE": 90, "SUPPORT": 108} |                   171 |                 0.977778 |                      0.967777 | {"NOT_ENTITLED": 524, "REFUTE": 90, "SUPPORT": 106} |                  -0.00416666 |                       -0.00548313 |                   -0.00694464 |                        -0.00921288 |        6.4671 | Stop Stage95 branch before external |

## Metadata checks

| check                                        | pass   |
|:---------------------------------------------|:-------|
| returncode_zero                              | True   |
| train_report_exists                          | True   |
| predictions_exists                           | True   |
| stage57_rows_520                             | True   |
| stage66_rows_720                             | True   |
| stage80a_combined_stage92a_stage95a_rows_400 | True   |
| combined_bridge_1640                         | True   |
| final_train_4520                             | True   |
| no_external_eval                             | True   |

## Clean checks

| check                              | pass   |
|:-----------------------------------|:-------|
| clean_acc_ge_stage71_minus_0p003   | False  |
| clean_macro_ge_stage71_minus_0p003 | False  |
| clean_acc_ge_0p972                 | False  |
| clean_macro_ge_0p961               | False  |
