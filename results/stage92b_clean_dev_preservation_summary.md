# Stage92B - Clean-dev Preservation Run

## Decision

`STAGE92B_CLEAN_DEV_PRESERVATION_READY`

## Candidate status

`ALLOW_EXTERNAL_DIAGNOSTIC`

## Summary

| stage    | decision                              | candidate_status          | run_dir                                                                 | bridge_config                                                                                     |   stage92b_best_epoch |   stage92b_clean_dev_acc |   stage92b_clean_dev_macro_f1 | stage92b_prediction_counts                          |   stage71_best_epoch |   stage71_clean_dev_acc |   stage71_clean_dev_macro_f1 | stage71_prediction_counts                           |   stage92b_minus_stage71_acc |   stage92b_minus_stage71_macro_f1 |   elapsed_min | recommended_next_stage                |
|:---------|:--------------------------------------|:--------------------------|:------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|----------------------:|-------------------------:|------------------------------:|:----------------------------------------------------|---------------------:|------------------------:|-----------------------------:|:----------------------------------------------------|-----------------------------:|----------------------------------:|--------------:|:--------------------------------------|
| Stage92B | STAGE92B_CLEAN_DEV_PRESERVATION_READY | ALLOW_EXTERNAL_DIAGNOSTIC | results/stage92b_stage57_stage66_stage92a_clean_dev_run_20260703_050639 | Stage57 + Stage66 + Stage92A via stage80a bridge slot; no Stage83A; no Stage88A; no external eval |                   171 |                 0.977778 |                      0.967777 | {"NOT_ENTITLED": 524, "REFUTE": 90, "SUPPORT": 106} |                  151 |                   0.975 |                     0.964047 | {"NOT_ENTITLED": 522, "REFUTE": 90, "SUPPORT": 108} |                   0.00277776 |                        0.00372933 |        6.4947 | Stage92C VitaminC external diagnostic |

## Metadata checks

| check                               | pass   |
|:------------------------------------|:-------|
| returncode_zero                     | True   |
| train_report_exists                 | True   |
| predictions_exists                  | True   |
| stage57_rows_520                    | True   |
| stage66_rows_720                    | True   |
| stage92a_rows_240_via_stage80a_slot | True   |
| combined_bridge_1480                | True   |
| final_train_4360                    | True   |
| no_external_eval                    | True   |

## Clean checks

| check                              | pass   |
|:-----------------------------------|:-------|
| clean_acc_ge_stage71_minus_0p003   | True   |
| clean_macro_ge_stage71_minus_0p003 | True   |
| clean_acc_ge_0p972                 | True   |
| clean_macro_ge_0p961               | True   |
