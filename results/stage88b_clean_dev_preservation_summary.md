# Stage88B - Clean-dev Preservation Run

## Decision

`STAGE88B_CLEAN_DEV_PRESERVATION_READY`

## Candidate status

`ALLOW_EXTERNAL_DIAGNOSTIC`

## Summary

| stage    | decision                              | candidate_status          | run_dir                                                                 | bridge_config                                                                        |   stage88b_best_epoch |   stage88b_clean_dev_acc |   stage88b_clean_dev_macro_f1 | stage88b_prediction_counts                          |   stage71_best_epoch |   stage71_clean_dev_acc |   stage71_clean_dev_macro_f1 | stage71_prediction_counts                           |   stage88b_minus_stage71_acc |   stage88b_minus_stage71_macro_f1 |   elapsed_min | recommended_next_stage                |
|:---------|:--------------------------------------|:--------------------------|:------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|----------------------:|-------------------------:|------------------------------:|:----------------------------------------------------|---------------------:|------------------------:|-----------------------------:|:----------------------------------------------------|-----------------------------:|----------------------------------:|--------------:|:--------------------------------------|
| Stage88B | STAGE88B_CLEAN_DEV_PRESERVATION_READY | ALLOW_EXTERNAL_DIAGNOSTIC | results/stage88b_stage57_stage66_stage88a_clean_dev_run_20260703_042736 | Stage57 + Stage66 + Stage88A via stage80a bridge slot; no Stage83A; no external eval |                   171 |                 0.976389 |                      0.965904 | {"NOT_ENTITLED": 523, "REFUTE": 90, "SUPPORT": 107} |                  151 |                   0.975 |                     0.964047 | {"NOT_ENTITLED": 522, "REFUTE": 90, "SUPPORT": 108} |                   0.00138891 |                        0.00185712 |        6.8303 | Stage88C VitaminC external diagnostic |

## Metadata checks

| check                               | pass   |
|:------------------------------------|:-------|
| returncode_zero                     | True   |
| train_report_exists                 | True   |
| predictions_exists                  | True   |
| stage57_rows_520                    | True   |
| stage66_rows_720                    | True   |
| stage88a_rows_360_via_stage80a_slot | True   |
| combined_bridge_1600                | True   |
| final_train_4480                    | True   |
| no_external_eval                    | True   |

## Clean checks

| check                              | pass   |
|:-----------------------------------|:-------|
| clean_acc_ge_stage71_minus_0p003   | True   |
| clean_macro_ge_stage71_minus_0p003 | True   |
| clean_acc_ge_0p972                 | True   |
| clean_macro_ge_0p961               | True   |
